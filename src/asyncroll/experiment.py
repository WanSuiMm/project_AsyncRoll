"""Fixed, single-GPU RTX 5090 experiment orchestration.

The ``plan`` command validates local inputs and prints the exact subprocess
commands without checking or touching a GPU. ``run`` owns one vLLM server and
one AsyncRoll client at a time, writes atomic receipts, and terminates only the
process groups it created. This module deliberately has no sweep or engine
matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .workload import Problem, load_jsonl


MODEL_REPOSITORY = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_REVISION = "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
TOOLMATH_REPOSITORY = "CHJ0417/ToolMATH"
TOOLMATH_REVISION = "f439a4af8dddbf061246ea9d68f6e977b18cbece"
TOOLMATH_SOURCE_SHA256 = "77cde6c200ff8dd1c4b89e24c691db8f4503b36bf9bf4a2115d703515d00866e"
POLICIES = ("fifo", "asyncroll")
PHASES = ("qualification", "opportunity", "comparison")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "experiments" / "single_5090.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = Path("experiments/single_5090.json").resolve()
ENV_MODEL = "ASYNCROLL_MODEL_PATH"
ENV_WORKLOAD = "ASYNCROLL_WORKLOAD_PATH"
ENV_TOOL_ROOT = "ASYNCROLL_TOOL_ROOT"
HEALTH_PATH = "/v1/models"


class ExperimentError(RuntimeError):
    """Configuration, preflight, or experiment gate failed."""


class AuditUnverified(ExperimentError):
    """vLLM startup logs did not establish actual backend and graph capture."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    """Durably replace a JSON receipt without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read experiment config {path}: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the fixed protocol and reject silent protocol drift."""
    if config.get("schema_version") != 1:
        raise ExperimentError("Unsupported single-GPU config schema_version")
    if not config.get("protocol_id"):
        raise ExperimentError("Config must have a protocol_id")
    model = config.get("model", {})
    workload = config.get("workload", {})
    if (model.get("repository"), model.get("revision")) != (MODEL_REPOSITORY, MODEL_REVISION):
        raise ExperimentError("Model repository and revision are frozen by this runner")
    if (workload.get("repository"), workload.get("revision")) != (
            TOOLMATH_REPOSITORY, TOOLMATH_REVISION):
        raise ExperimentError("ToolMATH repository and revision are frozen by this runner")
    if workload.get("source_sha256") != TOOLMATH_SOURCE_SHA256:
        raise ExperimentError("ToolMATH source file SHA-256 is frozen by this runner")
    if model.get("served_name") != "asyncroll-math":
        raise ExperimentError("The served model alias is fixed to asyncroll-math")
    if workload.get("qualification_tasks") != 8:
        raise ExperimentError("Qualification is fixed at exactly 8 tasks")
    if workload.get("comparison_tasks") != 32:
        raise ExperimentError("The selected fixed workload is exactly 32 tasks")
    comparison = config.get("comparison", {})
    if comparison.get("replicates") != 3:
        raise ExperimentError("Formal comparison is fixed at 3 paired replicates")
    expected_orders = [["fifo", "asyncroll"], ["asyncroll", "fifo"],
                       ["fifo", "asyncroll"]]
    if comparison.get("orders") != expected_orders:
        raise ExperimentError("Replicate order must be FIFO/AsyncRoll, AsyncRoll/FIFO, FIFO/AsyncRoll")
    hardware = config.get("hardware", {})
    if not isinstance(hardware.get("gpu_index"), int) or hardware["gpu_index"] < 0:
        raise ExperimentError("hardware.gpu_index must be a non-negative physical nvidia-smi index")
    if not hardware.get("required_name_tokens"):
        raise ExperimentError("hardware.required_name_tokens must identify the required GPU")
    server = config.get("server", {})
    if server.get("host") != "127.0.0.1":
        raise ExperimentError("vLLM must bind to IPv4 loopback on this dedicated host")
    if not isinstance(server.get("port"), int) or not 1 <= server["port"] <= 65535:
        raise ExperimentError("server.port must be in [1, 65535]")
    if server.get("tensor_parallel_size") != 1:
        raise ExperimentError("The fixed experiment uses tensor_parallel_size=1")
    if server.get("prefix_caching") is not True or server.get("chunked_prefill") is not True:
        raise ExperimentError("Prefix caching and chunked prefill must both be enabled")
    if (server.get("structured_outputs_backend") != "xgrammar"
            or server.get("disable_any_whitespace") is not True):
        raise ExperimentError(
            "Structured outputs require xgrammar with arbitrary whitespace disabled")
    if (not math.isfinite(float(server.get("gpu_memory_utilization", math.nan)))
            or not 0 < float(server["gpu_memory_utilization"]) <= 1):
        raise ExperimentError("server.gpu_memory_utilization must be in (0, 1]")
    runtime = config.get("runtime", {})
    positive_fields = (
        "cpu_workers", "max_active_trajectories", "max_inflight_model_requests",
        "max_turns", "max_tokens", "tool_timeout_seconds",
        "worker_startup_timeout_seconds", "request_timeout_seconds",
        "aging_seconds", "telemetry_interval_seconds",
    )
    for key in positive_fields:
        value = runtime.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ExperimentError(f"runtime.{key} must be finite and positive")
    starvation_threshold = runtime.get("starvation_threshold")
    if not isinstance(starvation_threshold, int) or starvation_threshold < 1:
        raise ExperimentError("runtime.starvation_threshold must be a positive integer request count")
    warmups = runtime.get("warmup_requests")
    if not isinstance(warmups, int) or warmups < 0:
        raise ExperimentError("runtime.warmup_requests must be a non-negative integer")
    deadlines = config.get("deadlines", {})
    for key in ("server_startup_seconds", "arm_seconds", "server_shutdown_seconds"):
        value = deadlines.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ExperimentError(f"deadlines.{key} must be finite and positive")
    if not isinstance(config.get("seed", {}).get("base"), int):
        raise ExperimentError("seed.base must be an integer")
    gate = config.get("opportunity_gate", {})
    for key in ("minimum_cpu_related_idle_candidate_seconds",
                "minimum_client_cpu_blocked_seconds", "minimum_tool_queue_p95_seconds"):
        value = gate.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ExperimentError(f"opportunity_gate.{key} must be finite and non-negative")
    fraction = gate.get("minimum_joint_observation_coverage_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ExperimentError("opportunity_gate.minimum_joint_observation_coverage_fraction must be in (0, 1]")
    depth = gate.get("minimum_cpu_max_queue_depth")
    if not isinstance(depth, int) or depth < 2:
        raise ExperimentError("opportunity_gate.minimum_cpu_max_queue_depth must be at least 2")


def _resolve_input(explicit: str | None, env_name: str, label: str) -> Path:
    value = explicit or os.environ.get(env_name)
    if not value:
        raise ExperimentError(f"Provide {label} with its CLI option or {env_name}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ExperimentError(f"{label} does not exist: {path}")
    return path


def _load_model_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(__file__).with_name("model_files.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read the pinned model file manifest: {exc}") from exc
    if (manifest.get("repository"), manifest.get("revision")) != (
            config["model"]["repository"], config["model"]["revision"]):
        raise ExperimentError("Pinned model file manifest does not match the configured model revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ExperimentError("Pinned model manifest has no file digests")
    return manifest


def _digest_model_file(path: Path, size: int, algorithm: str) -> str:
    if algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode("ascii"))
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ExperimentError(f"Unsupported model manifest digest algorithm: {algorithm}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_snapshot(model_path: Path, config: dict[str, Any], *,
                          full_hashes: bool = False) -> dict[str, Any]:
    """Check all official files and, for run, every pinned content digest."""
    manifest = _load_model_manifest(config)
    result: dict[str, Any] = {
        "repository": manifest["repository"],
        "requested_revision": manifest["revision"],
        "weights_revision_verified": False,
        "hash_verification_status": "pending_run" if not full_hashes else "running",
        "verification_source": "official_huggingface_file_digests",
        "files_checked": [],
        "missing_files": [],
        "wrong_size_files": [],
        "digest_mismatches": [],
    }
    if not model_path.is_dir():
        result["missing_files"].append("<model directory>")
        result["complete"] = False
        return result
    file_records = manifest["files"]
    names = {item.get("name") for item in file_records if isinstance(item, dict)}
    if "config.json" not in names or "model.safetensors.index.json" not in names:
        raise ExperimentError("Model manifest lacks the official config or safetensors index")
    expected_shards = {item["name"] for item in file_records
                       if re.fullmatch(r"model-\d+-of-\d+\.safetensors", item["name"])}
    if not expected_shards:
        raise ExperimentError("Model manifest contains no pinned safetensors shards")
    for item in file_records:
        name = item["name"]
        expected_size = item["size"]
        file_path = model_path / name
        if not file_path.is_file():
            result["missing_files"].append(name)
            continue
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            result["wrong_size_files"].append({"name": name, "expected": expected_size,
                                                "actual": actual_size})
            continue
        is_large_weight = name in expected_shards
        if full_hashes or (not is_large_weight and actual_size <= 16 * 1024 * 1024):
            actual_digest = _digest_model_file(file_path, actual_size, item["algorithm"])
            if actual_digest != item["digest"]:
                result["digest_mismatches"].append(name)
                continue
            result["files_checked"].append(name)
        elif is_large_weight:
            result.setdefault("large_weight_hashes_pending", []).append(name)
        else:
            result.setdefault("small_file_hashes_pending", []).append(name)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file() and index_path.stat().st_size == next(
            item["size"] for item in file_records if item["name"] == "model.safetensors.index.json"):
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            referenced_shards = set(index_data["weight_map"].values())
            if referenced_shards != expected_shards:
                result["digest_mismatches"].append("model.safetensors.index.json:shard_set")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            result["digest_mismatches"].append(f"model.safetensors.index.json:invalid:{exc}")
    result["complete"] = not (result["missing_files"] or result["wrong_size_files"]
                               or result["digest_mismatches"])
    if full_hashes and result["complete"] and not result.get("large_weight_hashes_pending"):
        result["weights_revision_verified"] = True
        result["hash_verification_status"] = "all_content_digests_verified"
    return result


def resolve_inputs(args: argparse.Namespace, config: dict[str, Any], *,
                   full_model_hashes: bool = False) -> dict[str, Any]:
    model_path = _resolve_input(args.model_path, ENV_MODEL, "model snapshot path")
    workload_path = _resolve_input(args.workload_path, ENV_WORKLOAD, "workload path")
    tool_root = _resolve_input(args.tool_root, ENV_TOOL_ROOT, "ToolMATH functions directory")
    if not workload_path.is_file():
        raise ExperimentError(f"Workload must be a JSONL file: {workload_path}")
    if not tool_root.is_dir():
        raise ExperimentError(f"Tool root must be a directory: {tool_root}")
    model_audit = verify_model_snapshot(model_path, config, full_hashes=full_model_hashes)
    if not model_audit.get("complete"):
        problems = (model_audit.get("missing_files", [])
                    + [item["name"] for item in model_audit.get("wrong_size_files", [])]
                    + model_audit.get("digest_mismatches", []))
        raise ExperimentError(f"Model snapshot failed pinned manifest checks: {', '.join(problems)}")
    problems = load_jsonl(workload_path, tool_root=tool_root)
    expected = config["workload"]["comparison_tasks"]
    if len(problems) != expected:
        raise ExperimentError(f"Expected exactly {expected} selected workload tasks; found {len(problems)}")
    tool_hashes: dict[str, str] = {}
    seen_tools = 0
    source_indices: list[int] = []
    for problem in problems:
        if problem.metadata.get("sample_seed") != config["workload"]["selection_seed"]:
            raise ExperimentError(
                f"Task {problem.id} is not from the configured seeded selection "
                f"({config['workload']['selection_seed']})")
        if problem.metadata.get("source_sha256") != config["workload"]["source_sha256"]:
            raise ExperimentError(f"Task {problem.id} does not match the pinned ToolMATH source file")
        source_index = problem.metadata.get("source_index")
        if not isinstance(source_index, int):
            raise ExperimentError(f"Task {problem.id} is missing its ToolMATH source index")
        source_indices.append(source_index)
        if not problem.tools:
            raise ExperimentError(f"ToolMATH task {problem.id} has no CPU tool; no synthetic work is added")
        for tool in problem.tools:
            implementation = Path(tool.get("implementation", ""))
            if not implementation.is_absolute():
                implementation = (tool_root / implementation).resolve()
            else:
                implementation = implementation.resolve()
            try:
                relative = implementation.relative_to(tool_root)
            except ValueError as exc:
                raise ExperimentError(f"Tool implementation escapes the supplied tool root: {tool['name']}") from exc
            if not implementation.is_file():
                raise ExperimentError(f"Missing implementation for tool {tool['name']}: {relative.as_posix()}")
            key = f"{tool['name']}:{relative.as_posix()}"
            tool_hashes[key] = sha256_file(implementation)
            seen_tools += 1
    if seen_tools == 0:
        raise ExperimentError("Selected workload contains no tool implementations")
    if len(set(source_indices)) != len(source_indices):
        raise ExperimentError("Selected ToolMATH source indices must be unique")
    return {
        "model_path": model_path,
        "workload_path": workload_path,
        "tool_root": tool_root,
        "problems": problems,
        "model_snapshot": model_audit,
        "workload_sha256": sha256_file(workload_path),
        "workload_ids": [problem.id for problem in problems],
        "tool_hashes": tool_hashes,
        "declared_workload_revision_status": "declared_source_revision_not_independently_verified",
    }


def software_versions() -> dict[str, str | None]:
    packages = ("asyncroll", "vllm", "torch", "transformers", "httpx",
                "prometheus-client", "nvidia-ml-py")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def source_digests() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = [*package.glob("*.py"), package / "model_files.json"]
    return {f"asyncroll/{path.name}": sha256_file(path)
            for path in sorted(paths) if path.is_file()}


def inspect_gpu(gpu_index: int, config: dict[str, Any]) -> dict[str, Any]:
    """Read physical GPU, driver, memory, and compute-process state."""
    if sys.platform != "linux":
        raise ExperimentError("Live single-GPU runs require Linux")
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise ExperimentError("nvidia-smi is required for dedicated-GPU preflight")
    gpu_result = subprocess.run(
        [executable, "--query-gpu=index,name,driver_version,memory.total,memory.used,uuid",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    rows = list(csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True))
    gpu_row = next((row for row in rows if len(row) >= 6 and int(row[0]) == gpu_index), None)
    if gpu_row is None:
        raise ExperimentError(f"Physical GPU index {gpu_index} was not reported by nvidia-smi")
    gpu = {
        "physical_index": int(gpu_row[0]),
        "name": gpu_row[1].strip(),
        "driver_version": gpu_row[2].strip(),
        "memory_total_mib": int(gpu_row[3]),
        "memory_used_mib": int(gpu_row[4]),
        "uuid": gpu_row[5].strip(),
    }
    process_result = subprocess.run(
        [executable, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    processes = []
    for row in csv.reader(process_result.stdout.splitlines(), skipinitialspace=True):
        if len(row) >= 4 and row[0].strip() == gpu["uuid"]:
            processes.append({"pid": row[1].strip(), "process_name": row[2].strip(),
                              "used_memory_mib": row[3].strip()})
    gpu["existing_compute_processes"] = processes
    tokens = [str(token).casefold() for token in config["hardware"]["required_name_tokens"]]
    gpu["name_matches_required"] = all(token in gpu["name"].casefold() for token in tokens)
    gpu["busy"] = bool(processes) or gpu["memory_used_mib"] > int(
        config["hardware"]["max_preexisting_memory_mib"])
    if not gpu["name_matches_required"]:
        raise ExperimentError(f"Wrong GPU at physical index {gpu_index}: {gpu['name']}")
    if gpu["busy"]:
        raise ExperimentError(
            f"GPU {gpu_index} is busy: {len(processes)} compute processes, "
            f"{gpu['memory_used_mib']} MiB already used")
    return gpu


def audit_startup_log(text: str) -> dict[str, Any]:
    """Extract explicit backend, compilation, and successful graph-capture evidence."""
    lines = text.splitlines()
    attention_lines: list[str] = []
    attention_backend: str | None = None
    compilation_lines: list[str] = []
    graph_lines: list[str] = []
    graph_negative_lines: list[str] = []
    positive_graph = re.compile(
        r"(?:graph|cudagraph|cuda\s+graph).*(?:capturing\s+finished|capture\s+finished|"
        r"capture\s+complete|captured\b.*(?:graph|shape)|graphs?\s+captured\b|"
        r"capture\s+succeeded|capture\s+completed)", re.IGNORECASE)
    negative_graph = re.compile(
        r"(?:no\s+(?:cuda\s+)?graphs?\s+captured|(?:cuda\s+)?graphs?\s+(?:are\s+)?disabled|"
        r"enforce[_ -]?eager\s*[:=]\s*(?:true|1)\b|enforce[_ -]?eager\s+(?:is\s+)?enabled|"
        r"without\s+(?:cuda\s+)?graph)", re.IGNORECASE)
    backend_patterns = (
        re.compile(r"\b(?:using|selected)\s+([A-Za-z0-9_.+-]+)\s+(?:attention\s+)?backend\b", re.I),
        re.compile(r"\b(?:using|selected)\s+([A-Za-z0-9_.+-]+)\s+as\s+(?:the\s+)?(?:attention\s+)?backend\b", re.I),
        re.compile(r"\b(?:using|selected)\s+(?:attention\s+)?backend\s*[:=]\s*([A-Za-z0-9_.+-]+)", re.I),
    )
    attention_names = {
        "FLASH_ATTN", "FLASHATTENTION", "FLASHINFER", "FLASHINFER_MLA",
        "TRITON_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHMLA", "CUTLASS_MLA",
        "TRITON_MLA", "TRTLLM_GEN", "FLEX_ATTENTION", "TREE_ATTN",
    }
    for line in lines:
        lower = line.casefold()
        if ("attention" in lower and "backend" in lower) or re.search(
                r"\b(?:using|selected)\s+\S+\s+backend\b", line, re.I):
            for pattern in backend_patterns:
                match = pattern.search(line)
                if match:
                    selected = match.group(1)
                    # A grammar, collective or compilation backend is not an
                    # attention implementation. New names require explicit
                    # attention context; unknown generic selections fail closed.
                    explicit_attention = re.search(r"\battention\s+backend\b", line, re.I)
                    if explicit_attention or selected.upper() in attention_names:
                        attention_backend = selected
                        attention_lines.append(line[:1000])
                    break
        if any(word in lower for word in ("compile", "compilation", "inductor")):
            compilation_lines.append(line[:1000])
        if negative_graph.search(line):
            graph_negative_lines.append(line[:1000])
        if ("graph" in lower or "cudagraph" in lower) and any(
                word in lower for word in ("captur", "eager", "disabled", "graph")):
            if positive_graph.search(line):
                graph_lines.append(line[:1000])
    graph_captured = bool(graph_lines) and not graph_negative_lines
    backend_verified = bool(attention_backend and attention_lines)
    verified = backend_verified and graph_captured
    return {
        "status": "verified" if verified else "unverified",
        "attention_backend": attention_backend,
        "attention_backend_verified": backend_verified,
        "attention_lines": attention_lines[:100],
        "cuda_graph_capture_verified": graph_captured,
        "cuda_graph_positive_lines": graph_lines[:100],
        "cuda_graph_negative_lines": graph_negative_lines[:100],
        "compilation_lines": compilation_lines[:100],
        "compilation_line_count": len(compilation_lines),
        "audit_note": ("Requires a logged selected attention backend and successful CUDA graph capture; "
                       "a requested config value alone is not evidence."),
    }


def audit_vllm_startup(text: str) -> dict[str, Any]:
    """Alias that names the source of the startup audit explicitly."""
    return audit_startup_log(text)


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
        except OSError:
            return False
    return True


def _resolve_vllm_executable(*, required: bool = True) -> str:
    name = "vllm.exe" if os.name == "nt" else "vllm"
    candidates = (Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / name,
                  Path(os.path.abspath(sys.executable)).parent / name)
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        from_path = shutil.which("vllm")
        if from_path:
            candidate = Path(from_path).resolve()
            try:
                candidate.relative_to(Path(sys.prefix).resolve())
            except ValueError:
                raise ExperimentError("PATH vllm executable is outside the active Python environment")
            executable = candidate
    if executable is None:
        if required:
            raise ExperimentError("The vllm executable is missing from the active Python environment")
        return str(candidates[0].absolute())
    return str(executable.resolve())


def server_command(config: dict[str, Any], model_path: Path, port: int,
                   vllm_executable: str | None = None) -> list[str]:
    server = config["server"]
    executable = vllm_executable or _resolve_vllm_executable()
    command = [executable, "serve", str(model_path),
               "--revision", config["model"]["revision"],
               "--served-model-name", config["model"]["served_name"],
               "--host", server["host"], "--port", str(port),
               "--dtype", server["dtype"],
               "--tensor-parallel-size", str(server["tensor_parallel_size"]),
               "--gpu-memory-utilization", str(server["gpu_memory_utilization"]),
               "--max-model-len", str(server["max_model_len"]),
               "--max-num-seqs", str(server["max_num_seqs"]),
               "--max-num-batched-tokens", str(server["max_num_batched_tokens"])]
    if server["prefix_caching"]:
        command.append("--enable-prefix-caching")
    if server["chunked_prefill"]:
        command.append("--enable-chunked-prefill")
    command.extend(("--structured-outputs-config", json.dumps({
        "backend": server["structured_outputs_backend"],
        "disable_any_whitespace": server["disable_any_whitespace"],
    }, separators=(",", ":"))))
    return command


def client_command(config: dict[str, Any], inputs: dict[str, Any], *, policy: str,
                   seed: int, output: Path, port: int, limit: int | None = None,
                   nvtx: bool = False, python_executable: str | None = None) -> list[str]:
    if policy not in POLICIES:
        raise ExperimentError(f"Policy must be one of {POLICIES}")
    runtime = config["runtime"]
    # Resolving a venv's Python symlink would silently select the base interpreter.
    command = [python_executable or os.path.abspath(sys.executable), "-m", "asyncroll.cli", "run",
               "--workload", str(inputs["workload_path"]), "--tool-root", str(inputs["tool_root"]),
               "--output", str(output), "--policy", policy, "--backend", "vllm",
               "--model", config["model"]["served_name"],
               "--endpoint", f"http://{config['server']['host']}:{port}",
               "--metrics-url", f"http://{config['server']['host']}:{port}/metrics",
               "--limit", str(limit) if limit is not None else str(config["workload"]["comparison_tasks"]),
               "--cpu-workers", str(runtime["cpu_workers"]),
               "--max-active-trajectories", str(runtime["max_active_trajectories"]),
               "--max-inflight-model-requests", str(runtime["max_inflight_model_requests"]),
               "--max-turns", str(runtime["max_turns"]),
               "--max-tokens", str(runtime["max_tokens"]),
               "--warmup-requests", str(runtime["warmup_requests"]),
               "--tool-timeout", str(runtime["tool_timeout_seconds"]),
               "--worker-startup-timeout", str(runtime["worker_startup_timeout_seconds"]),
               "--request-timeout", str(runtime["request_timeout_seconds"]),
               "--nvml-device", str(config["hardware"]["gpu_index"]),
               "--telemetry-interval", str(runtime["telemetry_interval_seconds"]),
               "--starvation-threshold", str(runtime["starvation_threshold"]),
               "--aging-seconds", str(runtime["aging_seconds"]),
               "--seed", str(seed), "--dedicated-gpu"]
    if nvtx:
        command.append("--nvtx")
    return command


def build_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = int(config["seed"]["base"])
    pairs = []
    for replicate, order in enumerate(config["comparison"]["orders"], start=1):
        pairs.append({
            "pair_id": f"pair-{replicate:02d}",
            "replicate": replicate,
            "seed": base + replicate - 1,
            "order": list(order),
        })
    if len(pairs) != config["comparison"]["replicates"]:
        raise ExperimentError("Pair schedule does not match configured replicate count")
    return pairs


def render_plan(config: dict[str, Any], inputs: dict[str, Any], *, port: int | None = None,
                output_root: Path | None = None, config_path: Path = CONFIG_PATH,
                vllm_executable: str | None = None,
                python_executable: str | None = None) -> dict[str, Any]:
    """Build all commands without launching a process or probing hardware."""
    if port is not None and port != config["server"]["port"]:
        raise ExperimentError("Set server.port in the frozen config; plan cannot override run settings")
    selected_port = int(port or config["server"]["port"])
    vllm_cmd = vllm_executable or _resolve_vllm_executable(required=False)
    python_cmd = python_executable or os.path.abspath(sys.executable)
    output_base = (output_root or (Path("runs") / "single_5090_preflight")).expanduser().resolve()
    common = [python_cmd, "-m", "asyncroll.experiment", "run", "--config", str(config_path.resolve()),
              "--model-path", str(inputs["model_path"]),
              "--workload-path", str(inputs["workload_path"]),
              "--tool-root", str(inputs["tool_root"])]
    stages: list[dict[str, Any]] = []
    protocol_commands: list[dict[str, Any]] = []
    qualification_seed = int(config["seed"]["base"]) + int(config["seed"]["qualification_offset"])
    for phase, policy, seed, limit in (
            ("qualification", "fifo", qualification_seed, config["workload"]["qualification_tasks"]),
            ("opportunity", "fifo", int(config["seed"]["base"]), config["workload"]["comparison_tasks"])):
        stage_root = output_base / phase
        client_out = stage_root / "arms" / f"{phase}-{policy}" / "asyncroll-run"
        runner_command = [*common, "--phase", phase, "--output", str(stage_root)]
        if phase == "opportunity":
            runner_command.extend(("--qualification-receipt", str(output_base / "qualification" / "experiment.json")))
        protocol_commands.append({"phase": phase, "output_dir": str(stage_root),
                                 "command": runner_command})
        stages.append({"phase": phase, "policy": policy, "seed": seed, "task_limit": limit,
                       "server_command": server_command(config, inputs["model_path"], selected_port,
                                                        vllm_cmd),
                       "client_command": client_command(config, inputs, policy=policy, seed=seed,
                                                        output=Path(client_out), port=selected_port,
                                                        limit=limit, python_executable=python_cmd)})
    for pair in build_pairs(config):
        stage_root = output_base / "comparison"
        for policy in pair["order"]:
            client_out = stage_root / "arms" / f"{pair['pair_id']}-{policy}" / "asyncroll-run"
            stages.append({"phase": "comparison", **pair, "policy": policy,
                           "server_command": server_command(config, inputs["model_path"], selected_port,
                                                            vllm_cmd),
                           "client_command": client_command(config, inputs, policy=policy,
                                                            seed=pair["seed"], output=Path(client_out),
                                                            port=selected_port, python_executable=python_cmd)})
    protocol_commands.append({
        "phase": "comparison", "output_dir": str(output_base / "comparison"),
        "command": [*common, "--phase", "comparison", "--output", str(output_base / "comparison"),
                    "--qualification-receipt", str(output_base / "qualification" / "experiment.json"),
                    "--opportunity-receipt", str(output_base / "opportunity" / "experiment.json")],
    })
    return {
        "protocol_id": config["protocol_id"],
        "launch_performed": False,
        "hardware_preflight": "deferred_until_run",
        "model_snapshot": inputs["model_snapshot"],
        "model_revision": config["model"]["revision"],
        "workload_revision": config["workload"]["revision"],
        "workload_sha256": inputs["workload_sha256"],
        "workload_ids": inputs["workload_ids"],
        "tool_implementation_sha256": inputs["tool_hashes"],
        "output_root": str(output_base),
        "cuda_visible_devices": None,
        "cuda_visible_devices_physical_index_pending_uuid_lookup": config["hardware"]["gpu_index"],
        "workload_revision_status": inputs["declared_workload_revision_status"],
        "nvml_physical_index": config["hardware"]["gpu_index"],
        "server_flags_audit": {
            "prefix_caching_enabled": config["server"]["prefix_caching"],
            "chunked_prefill_enabled": config["server"]["chunked_prefill"],
            "tensor_parallel_size": config["server"]["tensor_parallel_size"],
            "attention_backend_forced": False,
            "eager_mode_forced": False,
        },
        "deadlines": config["deadlines"],
        "protocol_commands": protocol_commands,
        "stages": stages,
        "plan_note": "No server or client is launched by plan. The selected GPU is checked only by run.",
    }


def _new_run_directory(path: Path) -> Path:
    output = path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "experiment.json"
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read prerequisite receipt {resolved}: {exc}") from exc
    return payload, hashlib.sha256(raw).hexdigest()


def _provenance(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "protocol_id": config["protocol_id"],
        "config_sha256": _canonical_sha256(config),
        "model_revision": config["model"]["revision"],
        "model_repository": config["model"]["repository"],
        "model_weights_revision_verified": inputs["model_snapshot"]["weights_revision_verified"],
        "workload_revision": config["workload"]["revision"],
        "workload_sha256": inputs["workload_sha256"],
        "workload_ids": inputs["workload_ids"],
        "tool_implementation_sha256": inputs["tool_hashes"],
        "source_sha256": source_digests(),
        "software_versions": software_versions(),
    }
    identity["protocol_fingerprint"] = _canonical_sha256(identity)
    return identity


def validate_prerequisites(phase: str, config: dict[str, Any], inputs: dict[str, Any],
                           qualification_path: Path | None,
                           opportunity_path: Path | None = None) -> dict[str, Any]:
    """Read previous phase receipts and reject mismatched or failed gates."""
    provenance = _provenance(config, inputs)
    result: dict[str, Any] = {"provenance": provenance}
    if phase in {"opportunity", "comparison"}:
        if qualification_path is None:
            raise ExperimentError(f"--qualification-receipt is required for {phase}")
        qual, qual_hash = _read_receipt(qualification_path)
        if qual.get("phase") != "qualification" or qual.get("status") != "qualified":
            raise ExperimentError("Qualification receipt did not pass all qualification gates")
        if qual.get("provenance", {}).get("protocol_fingerprint") != provenance["protocol_fingerprint"]:
            raise ExperimentError("Qualification receipt does not match the current model, workload, tools, and config")
        result["qualification_receipt"] = str((qualification_path / "experiment.json"
                                                if qualification_path.is_dir() else qualification_path).resolve())
        result["qualification_receipt_sha256"] = qual_hash
        result["qualification_summary"] = qual.get("stage_result")
        result["qualification_payload"] = qual
    if phase == "comparison":
        if opportunity_path is None:
            raise ExperimentError("--opportunity-receipt is required for comparison")
        opportunity, opportunity_hash = _read_receipt(opportunity_path)
        if opportunity.get("phase") != "opportunity" or opportunity.get("status") != "opportunity_pass":
            raise ExperimentError("Opportunity receipt did not pass the frozen CPU-opportunity gate")
        if opportunity.get("provenance", {}).get("protocol_fingerprint") != provenance["protocol_fingerprint"]:
            raise ExperimentError("Opportunity receipt does not match the current model, workload, tools, and config")
        if opportunity.get("qualification_receipt_sha256") != result["qualification_receipt_sha256"]:
            raise ExperimentError("Opportunity screen was run under a different qualification receipt")
        result["opportunity_receipt"] = str((opportunity_path / "experiment.json"
                                              if opportunity_path.is_dir() else opportunity_path).resolve())
        result["opportunity_receipt_sha256"] = opportunity_hash
        result["opportunity_summary"] = opportunity.get("stage_result")
        result["opportunity_payload"] = opportunity
    return result


def validate_prerequisite_hardware(pre: dict[str, Any], gpu: dict[str, Any]) -> None:
    """Require qualification/screen runs to share the same host runtime and GPU."""
    for key in ("qualification_payload", "opportunity_payload"):
        prior = pre.get(key)
        if not prior:
            continue
        prior_gpu = prior.get("hardware", {}).get("gpu", {})
        if (prior_gpu.get("uuid") != gpu.get("uuid")
                or prior_gpu.get("driver_version") != gpu.get("driver_version")):
            raise ExperimentError("Prerequisite phase used a different physical GPU or driver")
        if prior.get("software_versions") != software_versions():
            raise ExperimentError("Prerequisite phase used different installed software versions")


def _write_receipt(receipt: dict[str, Any], output: Path) -> None:
    receipt["updated_utc"] = utc_now()
    atomic_json(output / "experiment.json", receipt)


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _http_ready(endpoint: str, served_name: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(endpoint + HEALTH_PATH, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
        return any(item.get("id") == served_name for item in payload.get("data", []))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, TypeError):
        return False


def terminate_owned_group(process: Any, grace_seconds: float = 5.0) -> None:
    """Stop only the session/process group created for this owned subprocess."""
    if process is None:
        return
    process_id = getattr(process, "pid", None)
    if not process_id:
        return
    if os.name == "posix":
        def send(sig: int) -> None:
            try:
                os.killpg(process_id, sig)
            except ProcessLookupError:
                pass
        send(signal.SIGTERM)
        try:
            process.wait(timeout=grace_seconds)
        except (subprocess.TimeoutExpired, TimeoutError):
            send(signal.SIGKILL)
            try:
                process.wait(timeout=max(1.0, grace_seconds))
            except (subprocess.TimeoutExpired, TimeoutError):
                pass
        # The direct process can exit while a descendant remains in its group.
        send(signal.SIGKILL)
        return
    try:
        process.terminate()
        process.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired, TimeoutError):
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=max(1.0, grace_seconds))
        except (subprocess.TimeoutExpired, TimeoutError):
            pass


def _spawn(command: list[str], log_path: Path, env: dict[str, str], cwd: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
    except BaseException:
        stream.close()
        raise
    process._asyncroll_log_stream = stream
    return process


def _close_process_stream(process: Any) -> None:
    stream = getattr(process, "_asyncroll_log_stream", None)
    if stream is not None:
        stream.close()


def run_arm(config: dict[str, Any], inputs: dict[str, Any], *, output: Path,
            policy: str, seed: int, limit: int, gpu: dict[str, Any],
            nvtx: bool = False, profile_label: str | None = None) -> dict[str, Any]:
    """Start a fresh local vLLM engine, run one client, then clean both groups."""
    port = int(config["server"]["port"])
    if not _port_available(config["server"]["host"], port):
        raise ExperimentError(f"Configured server port {port} is already in use")
    # Recheck the physical GPU before each arm. This catches an unrelated
    # process that appeared after the initial preflight.
    current_gpu = inspect_gpu(int(config["hardware"]["gpu_index"]), config)
    if current_gpu["uuid"] != gpu["uuid"] or current_gpu["driver_version"] != gpu["driver_version"]:
        raise ExperimentError("GPU identity or driver changed during the experiment")
    output.mkdir(parents=True, exist_ok=False)
    logs_dir = output / "logs"
    logs_dir.mkdir()
    server_log = logs_dir / "vllm-startup.log"
    client_log = logs_dir / "client.log"
    server_cmd = server_command(config, inputs["model_path"], port)
    client_cmd = client_command(config, inputs, policy=policy, seed=seed,
                                output=output / "asyncroll-run", port=port,
                                limit=limit, nvtx=nvtx)
    env = os.environ.copy()
    # The physical index and UUID were already cross-checked above.  Use the
    # numeric index for the child environment because some vLLM releases parse
    # CUDA_VISIBLE_DEVICES as an integer during import and reject valid GPU UUIDs.
    cuda_visible_device = str(current_gpu["physical_index"])
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    deadline = config["deadlines"]
    arm: dict[str, Any] = {
        "policy": policy,
        "seed": seed,
        "task_limit": limit,
        "nvtx_enabled": nvtx,
        "vllm_worker_multiproc_method": "spawn",
        "dedicated_single_gpu_declared": True,
        "gpu": current_gpu,
        "cuda_visible_devices": cuda_visible_device,
        "nvml_physical_index": current_gpu["physical_index"],
        "server_command": server_cmd,
        "client_command": client_cmd,
        "server_log": str(server_log),
        "client_log": str(client_log),
        "asyncroll_run_dir": str(output / "asyncroll-run"),
        "status": "starting_server",
        "started_utc": utc_now(),
    }
    arm_receipt_path = output / "arm.json"

    def persist_arm() -> None:
        atomic_json(arm_receipt_path, arm)

    server_process = None
    client_process = None
    server_ready = False
    try:
        persist_arm()
        server_process = _spawn(server_cmd, server_log, env, Path(__file__).resolve().parents[2])
        arm["server_pid"] = server_process.pid
        persist_arm()
        startup_deadline = time.monotonic() + float(deadline["server_startup_seconds"])
        endpoint = f"http://{config['server']['host']}:{port}"
        while time.monotonic() < startup_deadline:
            if server_process.poll() is not None:
                raise ExperimentError(f"vLLM server exited with status {server_process.returncode}")
            if _http_ready(endpoint, config["model"]["served_name"]):
                server_ready = True
                break
            time.sleep(1.0)
        if not server_ready:
            raise ExperimentError("vLLM server readiness deadline expired")
        arm["server_ready_utc"] = utc_now()
        arm["startup_audit"] = audit_startup_log(_read_log(server_log))
        arm["status"] = "server_ready"
        persist_arm()
        if arm["startup_audit"]["status"] != "verified":
            arm["status"] = "audit_unverified"
            raise AuditUnverified("vLLM startup log did not establish both actual attention backend and completed CUDA graph capture")
        arm["status"] = "running_client"
        persist_arm()
        client_process = _spawn(client_cmd, client_log, env, Path(__file__).resolve().parents[2])
        arm["client_pid"] = client_process.pid
        persist_arm()
        try:
            client_process.wait(timeout=float(deadline["arm_seconds"]))
        except subprocess.TimeoutExpired as exc:
            raise ExperimentError(f"Client exceeded the {deadline['arm_seconds']} second arm deadline") from exc
        arm["client_exit_code"] = client_process.returncode
        if client_process.returncode != 0:
            raise ExperimentError(f"AsyncRoll client exited with status {client_process.returncode}")
        run_dir = output / "asyncroll-run"
        summary_path = run_dir / "summary.json"
        receipt_path = run_dir / "run.json"
        if not summary_path.is_file() or not receipt_path.is_file():
            raise ExperimentError("AsyncRoll child completed without summary.json and run.json")
        arm["summary_file"] = str(summary_path)
        arm["run_receipt_file"] = str(receipt_path)
        arm["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        arm["run_receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))
        arm["status"] = "complete"
        arm["finished_utc"] = utc_now()
        persist_arm()
        return arm
    except BaseException as exc:
        arm.setdefault("startup_audit", audit_startup_log(_read_log(server_log)))
        arm["error"] = f"{type(exc).__name__}: {exc}"
        arm["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else (
            "audit_unverified" if isinstance(exc, AuditUnverified) else "failed")
        arm["finished_utc"] = utc_now()
        setattr(exc, "asyncroll_arm_receipt", arm)
        if output.exists():
            persist_arm()
        raise
    finally:
        if client_process is not None:
            terminate_owned_group(client_process, float(deadline["server_shutdown_seconds"]))
            _close_process_stream(client_process)
        if server_process is not None:
            terminate_owned_group(server_process, float(deadline["server_shutdown_seconds"]))
            _close_process_stream(server_process)
        arm["cleanup_completed_utc"] = utc_now()
        if output.exists():
            persist_arm()


def qualification_pass(summary: dict[str, Any], task_count: int) -> tuple[bool, dict[str, Any]]:
    results = summary.get("results", [])
    probe = summary.get("telemetry_probe") or {}
    availability = probe.get("availability") or {}
    checks = {
        "exact_task_count": summary.get("count") == task_count and len(results) == task_count,
        "all_completed": summary.get("completed") == task_count and summary.get("failed") == 0,
        "all_action_valid": summary.get("action_valid_rate") == 1.0,
        "all_tasks_used_a_tool": summary.get("trajectory_tool_call_rate") == 1.0,
        "every_task_has_tool_call": len(results) == task_count and all(
            result.get("status") == "completed" and int(result.get("tool_calls", 0)) > 0
            for result in results),
        "vllm_metrics_available": availability.get("vllm") is True,
        "nvml_available": availability.get("nvml") is True,
    }
    return all(checks.values()), checks


def opportunity_pass(summary: dict[str, Any], gate: dict[str, Any],
                     expected_task_count: int | None = None) -> tuple[bool, dict[str, Any]]:
    resources = summary.get("resource_metrics")
    if not isinstance(resources, dict):
        resources = {}
    checks: dict[str, Any] = {}
    for field, threshold_key in (
            ("cpu_related_idle_candidate_seconds", "minimum_cpu_related_idle_candidate_seconds"),
            ("client_cpu_blocked_seconds", "minimum_client_cpu_blocked_seconds"),
            ("cpu_max_queue_depth", "minimum_cpu_max_queue_depth")):
        value = resources.get(field)
        threshold = float(gate[threshold_key])
        valid = isinstance(value, (int, float)) and math.isfinite(value)
        checks[field] = {"value": value, "minimum": threshold,
                         "pass": bool(valid and value > threshold if field != "cpu_max_queue_depth"
                                      else valid and value >= threshold)}
    elapsed = summary.get("elapsed_seconds")
    joint_seconds = resources.get("joint_observation_coverage_seconds")
    joint_fraction = (joint_seconds / elapsed if isinstance(joint_seconds, (int, float))
                      and isinstance(elapsed, (int, float)) and elapsed > 0 else None)
    fraction_minimum = float(gate["minimum_joint_observation_coverage_fraction"])
    checks["joint_observation_coverage_fraction"] = {
        "value": joint_fraction, "minimum": fraction_minimum,
        "pass": bool(isinstance(joint_fraction, (int, float))
                     and math.isfinite(joint_fraction) and joint_fraction >= fraction_minimum),
    }
    queue_p95 = summary.get("phase_seconds", {}).get("tool_queue", {}).get("p95_seconds")
    queue_minimum = float(gate["minimum_tool_queue_p95_seconds"])
    checks["tool_queue_p95_seconds"] = {
        "value": queue_p95, "minimum": queue_minimum,
        "pass": bool(isinstance(queue_p95, (int, float)) and math.isfinite(queue_p95)
                     and queue_p95 >= queue_minimum),
    }
    quality_ok, quality_checks = qualification_pass(
        summary, int(expected_task_count if expected_task_count is not None else summary.get("count", 0)))
    checks["full_workload_quality"] = {"pass": quality_ok, "checks": quality_checks}
    return all(item.get("pass", False) for item in checks.values()), checks


def _arm_brief(arm: dict[str, Any], seed: int) -> dict[str, Any]:
    summary = arm["summary"]
    return {
        "run_dir": arm["asyncroll_run_dir"],
        "summary_file": arm["summary_file"],
        "policy": arm["policy"],
        "seed": seed,
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "completed": summary.get("completed"),
        "failed": summary.get("failed"),
        "action_valid_rate": summary.get("action_valid_rate"),
        "trajectory_tool_call_rate": summary.get("trajectory_tool_call_rate"),
        "resource_metrics": summary.get("resource_metrics"),
        "nvtx_enabled": arm.get("nvtx_enabled", False),
        "dedicated_single_gpu_declared": arm.get("dedicated_single_gpu_declared", True),
        "startup_audit": arm.get("startup_audit"),
    }


def run_experiment(args: argparse.Namespace, config: dict[str, Any], inputs: dict[str, Any]) -> int:
    phase = args.phase
    pre = validate_prerequisites(phase, config, inputs,
                                 getattr(args, "qualification_receipt", None),
                                 getattr(args, "opportunity_receipt", None))
    output = _new_run_directory(Path(args.output))
    provenance = pre["provenance"]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "phase": phase,
        "status": "preparing",
        "started_utc": utc_now(),
        "output_dir": str(output),
        "config": config,
        "config_sha256": provenance["config_sha256"],
        "provenance": provenance,
        "inputs": {
            "model_path": str(inputs["model_path"]),
            "model_snapshot": inputs["model_snapshot"],
            "workload_path": str(inputs["workload_path"]),
            "workload_sha256": inputs["workload_sha256"],
            "workload_ids": inputs["workload_ids"],
            "tool_root": str(inputs["tool_root"]),
            "tool_implementation_sha256": inputs["tool_hashes"],
            "workload_revision_status": inputs["declared_workload_revision_status"],
        },
        **{key: value for key, value in pre.items()
           if key not in {"provenance", "qualification_payload", "opportunity_payload"}},
        "software_versions": software_versions(),
        "arms": [],
    }
    _write_receipt(receipt, output)
    gpu: dict[str, Any] | None = None
    try:
        if not inputs["model_snapshot"].get("weights_revision_verified"):
            raise ExperimentError("Model path does not prove the required Hugging Face snapshot revision")
        if receipt["software_versions"].get("vllm") is None:
            raise ExperimentError("vLLM is not installed in this Python environment")
        if os.environ.get("VLLM_ATTENTION_BACKEND"):
            raise ExperimentError("Unset VLLM_ATTENTION_BACKEND; backend selection must remain automatic")
        gpu = inspect_gpu(int(config["hardware"]["gpu_index"]), config)
        validate_prerequisite_hardware(pre, gpu)
        receipt["hardware"] = {"platform": platform.platform(), "gpu": gpu,
                                "cuda_visible_devices": gpu["uuid"],
                                "nvml_physical_index": gpu["physical_index"]}
        receipt["status"] = "running"
        _write_receipt(receipt, output)

        if phase == "qualification":
            arm_dir = output / "arms" / "qualification-fifo"
            try:
                arm = run_arm(config, inputs, output=arm_dir, policy="fifo",
                              seed=int(config["seed"]["base"]) + int(config["seed"]["qualification_offset"]),
                              limit=int(config["workload"]["qualification_tasks"]), gpu=gpu)
            except BaseException as exc:
                partial = getattr(exc, "asyncroll_arm_receipt", None)
                if partial:
                    receipt["arms"].append(partial)
                elif arm_dir.exists():
                    receipt["arms"].append({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                            "logs_dir": str(arm_dir / "logs")})
                raise
            receipt["arms"].append(arm)
            passed, checks = qualification_pass(arm["summary"], int(config["workload"]["qualification_tasks"]))
            receipt["stage_result"] = {"checks": checks, "summary_file": arm["summary_file"],
                                       "action_valid_rate": arm["summary"].get("action_valid_rate"),
                                       "trajectory_tool_call_rate": arm["summary"].get("trajectory_tool_call_rate"),
                                       "completed": arm["summary"].get("completed"),
                                       "failed": arm["summary"].get("failed")}
            receipt["status"] = "qualified" if passed else "failed_qualification"
            receipt["finished_utc"] = utc_now()
            _write_receipt(receipt, output)
            if not passed:
                print(f"Qualification failed; see {output / 'experiment.json'}")
                return 2
            print(f"Qualification passed; receipt: {output / 'experiment.json'}")
            return 0

        if phase == "opportunity":
            arm_dir = output / "arms" / "opportunity-fifo"
            try:
                arm = run_arm(config, inputs, output=arm_dir, policy="fifo",
                              seed=int(config["seed"]["base"]),
                              limit=int(config["workload"]["comparison_tasks"]), gpu=gpu)
            except BaseException as exc:
                partial = getattr(exc, "asyncroll_arm_receipt", None)
                if partial:
                    receipt["arms"].append(partial)
                elif arm_dir.exists():
                    receipt["arms"].append({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                            "logs_dir": str(arm_dir / "logs")})
                raise
            receipt["arms"].append(arm)
            passed, checks = opportunity_pass(
                arm["summary"], config["opportunity_gate"],
                int(config["workload"]["comparison_tasks"]))
            receipt["stage_result"] = {"checks": checks, "resource_metrics": arm["summary"].get("resource_metrics"),
                                       "summary_file": arm["summary_file"]}
            receipt["status"] = "opportunity_pass" if passed else "stopped_no_cpu_opportunity"
            receipt["finished_utc"] = utc_now()
            _write_receipt(receipt, output)
            if not passed:
                print(f"No qualifying CPU-related idle opportunity; comparison stopped. Receipt: {output / 'experiment.json'}")
                return 0
            print(f"CPU opportunity gate passed; receipt: {output / 'experiment.json'}")
            return 0

        comparison = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "phase": "comparison",
            "status": "running",
            "qualification_receipt_sha256": receipt["qualification_receipt_sha256"],
            "opportunity_receipt_sha256": receipt["opportunity_receipt_sha256"],
            "workload_sha256": inputs["workload_sha256"],
            "tool_implementation_sha256": inputs["tool_hashes"],
            "comparison_tasks": config["workload"]["comparison_tasks"],
            "pairs": [],
            "order_counts": {"fifo_first": 2, "asyncroll_first": 1},
            "report_note": "Three paired raw replicates with alternating order (2 FIFO-first, 1 AsyncRoll-first). "
                           "No pooled confidence interval is defined by this protocol.",
        }
        atomic_json(output / "comparison.json", comparison)
        pairs = build_pairs(config)
        for pair in pairs:
            arms_by_policy: dict[str, dict[str, Any]] = {}
            pair_record: dict[str, Any] = {
                "pair_id": pair["pair_id"], "replicate": pair["replicate"],
                "seed": pair["seed"], "order": pair["order"], "arms": {},
            }
            comparison["pairs"].append(pair_record)
            atomic_json(output / "comparison.json", comparison)
            for policy in pair["order"]:
                arm_dir = output / "arms" / f"{pair['pair_id']}-{policy}"
                try:
                    arm = run_arm(config, inputs, output=arm_dir, policy=policy,
                                  seed=pair["seed"],
                                  limit=int(config["workload"]["comparison_tasks"]), gpu=gpu)
                except BaseException as exc:
                    partial = getattr(exc, "asyncroll_arm_receipt", None)
                    if partial:
                        receipt["arms"].append(partial)
                        pair_record["arms"][policy] = {
                            "policy": policy, "seed": pair["seed"],
                            "status": partial.get("status"),
                            "error": partial.get("error"),
                            "arm_receipt_file": str(arm_dir / "arm.json"),
                            "startup_audit": partial.get("startup_audit"),
                            "nvtx_enabled": False,
                            "dedicated_single_gpu_declared": True,
                        }
                    elif arm_dir.exists():
                        receipt["arms"].append({"pair_id": pair["pair_id"], "status": "failed",
                                                "policy": policy,
                                                "error": f"{type(exc).__name__}: {exc}",
                                                "logs_dir": str(arm_dir / "logs")})
                        pair_record["arms"][policy] = {
                            "policy": policy, "seed": pair["seed"], "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "nvtx_enabled": False, "dedicated_single_gpu_declared": True,
                        }
                    atomic_json(output / "comparison.json", comparison)
                    raise
                receipt["arms"].append(arm)
                arms_by_policy[policy] = arm
                pair_record["arms"][policy] = _arm_brief(arm, pair["seed"])
                receipt["status"] = "running"
                _write_receipt(receipt, output)
                atomic_json(output / "comparison.json", comparison)
            fifo_backend = str(arms_by_policy["fifo"]["startup_audit"].get("attention_backend") or "").casefold()
            asyncroll_backend = str(arms_by_policy["asyncroll"]["startup_audit"].get("attention_backend") or "").casefold()
            if fifo_backend != asyncroll_backend:
                raise ExperimentError(
                    f"Selected attention backend changed within {pair['pair_id']}: "
                    f"{fifo_backend!r} vs {asyncroll_backend!r}")
            if "selected_attention_backend" not in comparison:
                comparison["selected_attention_backend"] = fifo_backend
            elif comparison["selected_attention_backend"] != fifo_backend:
                raise ExperimentError("Selected attention backend changed across paired replicates")
            # The pair was appended before its arms so each completed arm and
            # any interrupted second arm remain visible in comparison.json.
            atomic_json(output / "comparison.json", comparison)
        comparison["status"] = "complete"
        comparison["completed_utc"] = utc_now()
        atomic_json(output / "comparison.json", comparison)
        receipt["status"] = "complete"
        receipt["comparison_file"] = str(output / "comparison.json")
        receipt["finished_utc"] = utc_now()
        _write_receipt(receipt, output)
        print(f"Paired comparison complete; review {output / 'comparison.json'}")
        return 0
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            receipt["status"] = "interrupted"
        else:
            receipt["status"] = "failed_preflight" if receipt.get("status") == "preparing" else "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["finished_utc"] = utc_now()
        _write_receipt(receipt, output)
        comparison_path = output / "comparison.json"
        if comparison_path.is_file():
            try:
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                comparison["status"] = receipt["status"]
                comparison["error"] = receipt["error"]
                comparison["finished_utc"] = receipt["finished_utc"]
                atomic_json(comparison_path, comparison)
            except (OSError, json.JSONDecodeError):
                pass
        raise


def run_profile(args: argparse.Namespace, config: dict[str, Any], inputs: dict[str, Any]) -> int:
    pre = validate_prerequisites("opportunity", config, inputs,
                                 args.qualification_receipt)
    if args.opportunity_receipt is not None:
        opportunity, opportunity_hash = _read_receipt(args.opportunity_receipt)
        if (opportunity.get("phase") != "opportunity"
                or opportunity.get("provenance", {}).get("protocol_fingerprint")
                != pre["provenance"]["protocol_fingerprint"]
                or opportunity.get("qualification_receipt_sha256")
                != pre["qualification_receipt_sha256"]):
            raise ExperimentError("Optional opportunity receipt does not match this qualification and workload")
        pre["diagnostic_opportunity_receipt_sha256"] = opportunity_hash
        pre["diagnostic_opportunity_status"] = opportunity.get("status")
        pre["opportunity_payload"] = opportunity
        pre["diagnostic_opportunity_receipt"] = str(
            (args.opportunity_receipt / "experiment.json"
             if args.opportunity_receipt.is_dir() else args.opportunity_receipt).resolve())
    output = _new_run_directory(Path(args.output))
    receipt = {
        "schema_version": 1, "protocol_id": config["protocol_id"],
        "phase": "profile", "status": "preparing", "started_utc": utc_now(),
        "output_dir": str(output), "profile_policy": args.profile_policy,
        "config": config, "config_sha256": pre["provenance"]["config_sha256"],
        "provenance": pre["provenance"],
        "inputs": {"workload_sha256": inputs["workload_sha256"],
                   "tool_implementation_sha256": inputs["tool_hashes"],
                   "model_snapshot": inputs["model_snapshot"]},
        "qualification_receipt_sha256": pre["qualification_receipt_sha256"],
        "opportunity_receipt_sha256": pre.get("diagnostic_opportunity_receipt_sha256"),
        "opportunity_receipt": pre.get("diagnostic_opportunity_receipt"),
        "opportunity_status": pre.get("diagnostic_opportunity_status"),
        "software_versions": software_versions(), "arms": [],
        "source_sha256": pre["provenance"]["source_sha256"],
        "trace_output": str(args.trace_output) if args.trace_output else None,
        "throughput_included": False,
        "profile_note": "One policy trace. Do not pool profile timing with unprofiled throughput arms.",
    }
    _write_receipt(receipt, output)
    try:
        if not inputs["model_snapshot"].get("weights_revision_verified"):
            raise ExperimentError("Model path does not prove the required Hugging Face snapshot revision")
        if receipt["software_versions"].get("vllm") is None:
            raise ExperimentError("vLLM is not installed in this Python environment")
        if os.environ.get("VLLM_ATTENTION_BACKEND"):
            raise ExperimentError("Unset VLLM_ATTENTION_BACKEND; backend selection must remain automatic")
        gpu = inspect_gpu(int(config["hardware"]["gpu_index"]), config)
        validate_prerequisite_hardware(pre, gpu)
        receipt["hardware"] = {"gpu": gpu, "cuda_visible_devices": gpu["uuid"],
                                "nvml_physical_index": gpu["physical_index"]}
        receipt["status"] = "running"
        _write_receipt(receipt, output)
        arm_dir = output / "arms" / f"profile-{args.profile_policy}"
        try:
            arm = run_arm(config, inputs, output=arm_dir, policy=args.profile_policy,
                          seed=int(config["seed"]["base"]),
                          limit=int(config["workload"]["comparison_tasks"]), gpu=gpu, nvtx=True)
        except BaseException as exc:
            partial = getattr(exc, "asyncroll_arm_receipt", None)
            if partial:
                receipt["arms"].append({key: value for key, value in partial.items() if key != "summary"})
            raise
        # A profile run is intentionally not stored in comparison.json.
        receipt["arms"].append({key: value for key, value in arm.items() if key != "summary"})
        receipt["profile_summary_file"] = arm["summary_file"]
        receipt["status"] = "diagnostic_complete"
        receipt["finished_utc"] = utc_now()
        _write_receipt(receipt, output)
        print(f"Diagnostic profile arm complete; trace it by wrapping this command in nsys profile. Receipt: {output / 'experiment.json'}")
        return 0
    except BaseException as exc:
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["finished_utc"] = utc_now()
        _write_receipt(receipt, output)
        raise


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--model-path", default=None,
                        help=f"Local pinned HF snapshot; or set {ENV_MODEL}")
    parser.add_argument("--workload-path", "--workload", dest="workload_path", default=None,
                        help=f"Fixed 32-task ToolMATH JSONL; or set {ENV_WORKLOAD}")
    parser.add_argument("--tool-root", default=None,
                        help=f"Local ToolMATH functions directory; or set {ENV_TOOL_ROOT}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m asyncroll.experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate local inputs and print commands; launch nothing")
    _add_common_arguments(plan)
    plan.add_argument("--port", type=int, default=None)
    plan.add_argument("--output-root", type=Path, default=Path("runs") / "single_5090_preflight")

    run = commands.add_parser("run", help="run one bounded protocol phase")
    _add_common_arguments(run)
    run.add_argument("--phase", choices=PHASES, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--qualification-receipt", type=Path)
    run.add_argument("--opportunity-receipt", type=Path)

    profile = commands.add_parser("profile", help="run one separate NVTX diagnostic policy arm")
    _add_common_arguments(profile)
    profile.add_argument("--profile-policy", choices=POLICIES, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--qualification-receipt", type=Path, required=True)
    profile.add_argument("--opportunity-receipt", type=Path)
    profile.add_argument("--trace-output", type=Path,
                         help="Expected external nsys output base; this command does not start Nsight")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Convert normal termination into the same orderly cleanup path as Ctrl+C.
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    previous_handlers = {sig: signal.signal(sig, interrupt)
                         for sig in {signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)}}
    try:
        config = load_config(args.config)
        inputs = resolve_inputs(args, config, full_model_hashes=(args.command != "plan"))
        if args.command == "plan":
            plan = render_plan(config, inputs, port=args.port, output_root=args.output_root,
                               config_path=args.config)
            print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
            return
        if args.command == "run":
            code = run_experiment(args, config, inputs)
        else:
            code = run_profile(args, config, inputs)
        raise SystemExit(code)
    except KeyboardInterrupt:
        print("Interrupted; owned process groups were cleaned and the receipt was preserved.", file=sys.stderr)
        raise SystemExit(130)
    except (ExperimentError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"experiment error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
