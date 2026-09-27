"""Run a paired vLLM versus Transformers fixed-batch measurement.

The output directory is append-only for one invocation. Each measured arm runs
in a fresh process; vLLM startup and warmup are excluded from timed metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_server(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as reply:
                if reply.status == 200:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy: {last}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        finally:
            process.wait(timeout=10)


def ratio(new: float | None, old: float | None) -> float | None:
    return (new / old - 1.0) if new is not None and old not in (None, 0) else None


def aggregate(stages: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    by_key = {(stage["replicate"], stage["arm"]): stage for stage in stages
              if stage.get("status") == "complete"}
    engine = []

    def output_match(left: dict[str, Any], right: dict[str, Any]) -> float | None:
        left_samples = {row["id"]: row.get("output_sha256")
                        for row in left["metrics"].get("samples", []) if row.get("ok")}
        right_samples = {row["id"]: row.get("output_sha256")
                         for row in right["metrics"].get("samples", []) if row.get("ok")}
        shared = set(left_samples) & set(right_samples)
        return (sum(left_samples[key] == right_samples[key] for key in shared) / len(shared)
                if shared else None)
    for replicate in range(1, repeats + 1):
        transformer = by_key.get((replicate, "transformers_fixed_batch"))
        primary = by_key.get((replicate, "vllm_online"))
        if transformer and primary:
            engine.append({
                "replicate": replicate,
                "requests_per_second_change": ratio(
                    primary["metrics"]["requests_per_second"],
                    transformer["metrics"]["requests_per_second"]),
                "output_tokens_per_second_change": ratio(
                    primary["metrics"]["output_tokens_per_second"],
                    transformer["metrics"]["output_tokens_per_second"]),
                "exact_output_match_fraction": output_match(primary, transformer),
            })

    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "vllm_vs_transformers": {
            "paired_replicates": engine,
            "mean_requests_per_second_change": mean(engine, "requests_per_second_change"),
            "mean_output_tokens_per_second_change": mean(
                engine, "output_tokens_per_second_change"),
            "mean_exact_output_match_fraction": mean(engine, "exact_output_match_fraction"),
            "claim_scope": "vLLM online continuous serving versus Transformers fixed batches; "
                           "both request FlashAttention and greedy decoding.",
        },
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != "asyncroll-vllm-serving-ablation-v1":
        raise ValueError("unexpected protocol_id")
    benchmark = config["benchmark"]
    if benchmark["repeats"] != len(benchmark["orders"]):
        raise ValueError("one order is required per repeat")
    required = {"transformers_fixed_batch", "vllm_online"}
    for order in benchmark["orders"]:
        if set(order) != required or len(order) != len(required):
            raise ValueError("each order must contain every arm exactly once")
    if config["transformers"]["attention_implementation"] != "flash_attention_2":
        raise ValueError("engine comparison requires FlashAttention 2 on Transformers")
    if config["vllm"]["attention_backend"] != "FLASH_ATTN":
        raise ValueError("both engine arms are intended to use FlashAttention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--plan", action="store_true",
                        help="validate and print the arm order without running a GPU")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    expected_hash = config["workload"].get("selected_workload_sha256")
    actual_hash = sha256(args.workload)
    if expected_hash and actual_hash != expected_hash:
        parser.error(f"workload sha256 mismatch: {actual_hash}")
    plan = [{"replicate": index, "order": order}
            for index, order in enumerate(config["benchmark"]["orders"], 1)]
    if args.plan:
        print(json.dumps({"protocol_id": config["protocol_id"], "plan": plan}, indent=2))
        return
    if args.output.exists():
        parser.error("--output must be a new directory")
    args.output.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "schema_version": 1, "protocol_id": config["protocol_id"],
        "status": "running", "started_utc": now(), "config": config,
        "workload_sha256": actual_hash, "stages": [],
    }
    write_json(args.output / "experiment.json", receipt)
    model = config["model"]
    benchmark = config["benchmark"]
    server = config["vllm"]
    port = int(server["port"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = (source_root + os.pathsep + environment["PYTHONPATH"]
                                 if environment.get("PYTHONPATH") else source_root)

    def common_client(output: Path) -> list[str]:
        return ["--workload", str(args.workload), "--warmup", str(benchmark["warmup_prompts"]),
                "--measured", str(benchmark["measured_prompts"]),
                "--max-tokens", str(benchmark["max_tokens"]), "--output", str(output)]

    def run_transformers(replicate: int, arm_dir: Path) -> dict[str, Any]:
        result_path = arm_dir / "metrics.json"
        command = [args.python, "-m", "asyncroll.serving_ablation", "transformers",
                   *common_client(result_path), "--model-path", str(args.model_path),
                   "--revision", model["revision"], "--batch-size",
                   str(config["transformers"]["batch_size"]), "--dtype", server["dtype"],
                   "--attention", config["transformers"]["attention_implementation"]]
        with (arm_dir / "benchmark.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment)
        if completed.returncode:
            raise RuntimeError(f"Transformers benchmark exited with {completed.returncode}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def server_command() -> list[str]:
        command = [args.vllm, "serve", str(args.model_path), "--revision", model["revision"],
                   "--served-model-name", model["served_name"], "--host", "127.0.0.1",
                   "--port", str(port), "--dtype", server["dtype"],
                   "--tensor-parallel-size", "1", "--gpu-memory-utilization",
                   str(server["gpu_memory_utilization"]), "--max-model-len",
                   str(server["max_model_len"]), "--max-num-seqs",
                   str(server["max_num_seqs"]), "--max-num-batched-tokens",
                   str(server["max_num_batched_tokens"]), "--generation-config", "vllm",
                   "--attention-backend", server["attention_backend"]]
        if server.get("chunked_prefill"):
            command.append("--enable-chunked-prefill")
        if server.get("prefix_caching"):
            command.append("--enable-prefix-caching")
        return command

    def run_vllm(replicate: int, arm_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        result_path = arm_dir / "metrics.json"
        log_path = arm_dir / "vllm.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(server_command(), stdout=log,
                                       stderr=subprocess.STDOUT, env=environment,
                                       start_new_session=True, text=True)
            try:
                wait_server(port, process, float(server["startup_timeout_seconds"]))
                command = [args.python, "-m", "asyncroll.serving_ablation", "http",
                           *common_client(result_path), "--endpoint", f"http://127.0.0.1:{port}",
                           "--model", model["served_name"], "--concurrency",
                           str(benchmark["concurrency"]), "--timeout",
                           str(benchmark["request_timeout_seconds"]), "--seed",
                           str(benchmark["seed"])]
                with (arm_dir / "benchmark.log").open("w", encoding="utf-8") as client_log:
                    completed = subprocess.run(command, stdout=client_log,
                                               stderr=subprocess.STDOUT, env=environment)
                if completed.returncode:
                    raise RuntimeError(f"HTTP benchmark exited with {completed.returncode}")
            finally:
                stop_process(process)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        evidence = {
            "requested_backend": server["attention_backend"],
            "backend_named_in_log": server["attention_backend"].lower() in log_text.lower(),
            "cuda_graph_captured": "Graph capturing finished" in log_text,
        }
        return json.loads(result_path.read_text(encoding="utf-8")), evidence

    try:
        for replicate, order in enumerate(benchmark["orders"], 1):
            for arm in order:
                arm_dir = args.output / f"replicate-{replicate}-{arm}"
                arm_dir.mkdir()
                stage: dict[str, Any] = {"replicate": replicate, "arm": arm,
                                         "started_utc": now(), "status": "running"}
                receipt["stages"].append(stage)
                write_json(args.output / "experiment.json", receipt)
                try:
                    if arm == "transformers_fixed_batch":
                        metrics = run_transformers(replicate, arm_dir)
                    else:
                        metrics, evidence = run_vllm(replicate, arm_dir)
                        stage["startup_evidence"] = evidence
                    stage.update(status="complete", finished_utc=now(), metrics=metrics)
                except Exception as exc:
                    stage.update(status="failed", finished_utc=now(),
                                 error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    write_json(args.output / "experiment.json", receipt)
        receipt["aggregate"] = aggregate(receipt["stages"], int(benchmark["repeats"]))
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_utc"] = now()
        write_json(args.output / "experiment.json", receipt)


if __name__ == "__main__":
    main()
