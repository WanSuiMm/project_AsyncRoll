"""Run the frozen LiveCodeBench qualification, screen, and gated comparison."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "asyncroll-single-5090-qwen-coder-lcb-v1":
        parser.error("unexpected protocol_id")
    args.output.mkdir(parents=True)
    receipt = {"status": "running", "protocol_id": config["protocol_id"],
               "started_utc": now(), "config": config, "stages": []}
    write_json(args.output / "experiment.json", receipt)
    server = config["server"]
    runtime = config["runtime"]
    model = config["model"]
    port = 8000

    def run_arm(stage: str, policy: str, limit: int, seed: int) -> dict:
        arm_dir = args.output / stage
        arm_dir.mkdir(parents=True)
        server_log_path = arm_dir / "vllm.log"
        command = [args.vllm, "serve", str(args.model_path), "--revision", model["revision"],
                   "--served-model-name", model["served_name"], "--host", "127.0.0.1",
                   "--port", str(port), "--dtype", server["dtype"],
                   "--tensor-parallel-size", "1", "--gpu-memory-utilization",
                   str(server["gpu_memory_utilization"]), "--max-model-len",
                   str(server["max_model_len"]), "--max-num-seqs",
                   str(server["max_num_seqs"]), "--max-num-batched-tokens",
                   str(server["max_num_batched_tokens"]), "--enable-prefix-caching",
                   "--enable-chunked-prefill"]
        if server.get("structured_outputs_backend"):
            structured = json.dumps({"backend": server["structured_outputs_backend"],
                                     "disable_any_whitespace": server["disable_any_whitespace"]},
                                    separators=(",", ":"))
            command.extend(["--structured-outputs-config", structured])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        with server_log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True, text=True)
            try:
                wait_server(port, process, 1800)
                client_dir = arm_dir / "asyncroll-run"
                client = [args.python, "-m", "asyncroll.cli", "run",
                          "--workload", str(args.workload), "--limit", str(limit),
                          "--output", str(client_dir), "--policy", policy,
                          "--backend", "vllm", "--model", model["served_name"],
                          "--endpoint", f"http://127.0.0.1:{port}",
                          "--metrics-url", f"http://127.0.0.1:{port}/metrics",
                          "--cpu-workers", str(runtime["cpu_workers"]),
                          "--max-active-trajectories", str(runtime["max_active_trajectories"]),
                          "--max-inflight-model-requests", str(runtime["max_inflight_model_requests"]),
                          "--max-turns", str(runtime["max_turns"]),
                          "--max-tokens", str(runtime["max_tokens"]),
                          "--warmup-requests", str(runtime["warmup_requests"]),
                          "--tool-timeout", str(runtime["tool_timeout_seconds"]),
                          "--worker-startup-timeout", str(runtime["worker_startup_timeout_seconds"]),
                          "--request-timeout", str(runtime["request_timeout_seconds"]),
                          "--telemetry-interval", str(runtime["telemetry_interval_seconds"]),
                          "--starvation-threshold", str(runtime["starvation_threshold"]),
                          "--aging-seconds", str(runtime["aging_seconds"]),
                          "--seed", str(seed), "--nvml-device", str(args.gpu_index),
                          "--dedicated-gpu"]
                if not server.get("structured_outputs_backend"):
                    client.append("--no-structured-output")
                with (arm_dir / "client.log").open("w", encoding="utf-8") as client_log:
                    completed = subprocess.run(client, stdout=client_log,
                                               stderr=subprocess.STDOUT, env=environment)
                if completed.returncode:
                    raise RuntimeError(f"client exited with code {completed.returncode}")
                summary = json.loads((client_dir / "summary.json").read_text(encoding="utf-8"))
            finally:
                stop_process(process)
        log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
        # Keep experiment.json as a compact control-plane receipt. The arm's
        # per-trajectory records remain in its own asyncroll-run/summary.json.
        compact_summary = {key: value for key, value in summary.items()
                           if key != "results"}
        result = {"stage": stage, "policy": policy, "limit": limit,
                  "summary": compact_summary,
                  "summary_path": str(Path(stage) / "asyncroll-run" / "summary.json"),
                  "startup": {"flash_attention": "Using FLASH_ATTN backend" in log_text,
                              "cuda_graph": "Graph capturing finished" in log_text}}
        receipt["stages"].append(result)
        write_json(args.output / "experiment.json", receipt)
        return result

    try:
        workload = config["workload"]
        base_seed = int(workload["selection_seed"])
        qualification = run_arm("qualification-fifo", "fifo",
                                int(workload["qualification_tasks"]), base_seed + 1000)
        q = qualification["summary"]
        qualified = (q["completed"] == int(workload["qualification_tasks"])
                     and q["action_valid_rate"] == 1.0
                     and q["trajectory_tool_call_rate"] == 1.0
                     and qualification["startup"]["flash_attention"]
                     and qualification["startup"]["cuda_graph"])
        receipt["qualification"] = {"pass": qualified}
        if not qualified:
            receipt["status"] = "stopped_failed_qualification"
            return
        opportunity = run_arm("opportunity-fifo", "fifo",
                              int(workload["opportunity_tasks"]), base_seed)
        summary = opportunity["summary"]
        resources = summary["resource_metrics"]
        gates = config["opportunity_gate"]
        values = {
            "client_cpu_blocked_seconds": resources["client_cpu_blocked_seconds"],
            "cpu_max_queue_depth": resources["cpu_max_queue_depth"],
            "cpu_related_idle_candidate_seconds": resources["cpu_related_idle_candidate_seconds"],
            "joint_observation_coverage_fraction": resources["joint_observation_coverage_seconds"] / summary["elapsed_seconds"],
            "tool_queue_p95_seconds": summary["phase_seconds"]["tool_queue"]["p95_seconds"],
        }
        checks = {
            "client_cpu_blocked_seconds": values["client_cpu_blocked_seconds"] >= gates["minimum_client_cpu_blocked_seconds"],
            "cpu_max_queue_depth": values["cpu_max_queue_depth"] >= gates["minimum_cpu_max_queue_depth"],
            "cpu_related_idle_candidate_seconds": values["cpu_related_idle_candidate_seconds"] is not None and values["cpu_related_idle_candidate_seconds"] >= gates["minimum_cpu_related_idle_candidate_seconds"],
            "joint_observation_coverage_fraction": values["joint_observation_coverage_fraction"] >= gates["minimum_joint_observation_coverage_fraction"],
            "tool_queue_p95_seconds": values["tool_queue_p95_seconds"] is not None and values["tool_queue_p95_seconds"] >= gates["minimum_tool_queue_p95_seconds"],
        }
        receipt["opportunity"] = {"values": values, "checks": checks,
                                  "pass": all(checks.values())}
        if not all(checks.values()):
            receipt["status"] = "stopped_no_cpu_opportunity"
            return
        for pair_index, order in enumerate(config["comparison"]["orders"], 1):
            for policy in order:
                run_arm(f"pair-{pair_index}-{policy}", policy,
                        int(workload["comparison_tasks"]), base_seed + pair_index)
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_utc"] = now()
        write_json(args.output / "experiment.json", receipt)


if __name__ == "__main__":
    main()
