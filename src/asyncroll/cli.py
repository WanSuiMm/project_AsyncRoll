"""Benchmark CLI. Each run has durable events and an explicit terminal receipt."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .model import ScriptedBackend, VLLMBackend
from .runtime import run
from .telemetry import Telemetry
from .timeline import write_timeline
from .workload import convert_toolmath, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="asyncroll")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-toolmath")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--functions-dir", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--limit", type=int)
    convert.add_argument("--seed", type=int, default=0)

    bench = commands.add_parser("run")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--tool-root", type=Path)
    bench.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    bench.add_argument("--policy", choices=["sync", "fifo", "tool_first", "gpu_first", "asyncroll"], required=True)
    bench.add_argument("--backend", choices=["scripted", "vllm"], required=True)
    bench.add_argument("--model")
    bench.add_argument("--endpoint", default="http://localhost:8000")
    bench.add_argument("--limit", type=int)
    bench.add_argument("--cpu-workers", type=int, default=2)
    bench.add_argument("--max-active-trajectories", type=int, default=16)
    bench.add_argument("--max-inflight-model-requests", "--gpu-slots", type=int, default=4,
                       help="Client request limit; --gpu-slots is a legacy alias")
    bench.add_argument("--max-turns", type=int, default=8)
    bench.add_argument("--max-tokens", type=int, default=512)
    bench.add_argument("--warmup-requests", type=int, default=2)
    bench.add_argument("--tool-timeout", type=float, default=60)
    bench.add_argument("--worker-startup-timeout", type=float, default=120)
    bench.add_argument("--request-timeout", type=float, default=120)
    bench.add_argument("--no-structured-output", action="store_true")
    bench.add_argument("--metrics-url", help="vLLM /metrics endpoint; defaults to endpoint/metrics")
    bench.add_argument("--no-telemetry", action="store_true")
    bench.add_argument("--nvml-device", type=int, help="Physical NVML index on THIS host")
    bench.add_argument("--telemetry-interval", type=float, default=0.2)
    bench.add_argument("--seed", type=int, default=0, help="Model decoding seed; not a determinism guarantee")
    bench.add_argument("--starvation-threshold", type=int, default=1)
    bench.add_argument("--aging-seconds", type=float, default=1.0)
    bench.add_argument("--nvtx", action="store_true", help="Diagnostic Nsight ranges; requires profiling extra")
    bench.add_argument("--dedicated-gpu", action="store_true", help="Declare one dedicated GPU for measured allocation-hour rate")
    args = parser.parse_args()
    if args.command == "convert-toolmath":
        count = convert_toolmath(args.source, args.functions_dir, args.output, args.limit, args.seed)
        print(f"Wrote {count} problems to {args.output}")
        return
    if args.backend == "vllm" and not args.model:
        parser.error("--model is required with --backend vllm")
    if args.dedicated_gpu and args.backend != "vllm":
        parser.error("--dedicated-gpu requires a real vLLM backend")
    if min(args.cpu_workers, args.max_active_trajectories, args.max_inflight_model_requests,
           args.max_turns, args.max_tokens, args.tool_timeout, args.worker_startup_timeout,
           args.request_timeout, args.telemetry_interval, args.starvation_threshold, args.aging_seconds) <= 0 or args.warmup_requests < 0:
        parser.error("Limits/timeouts must be positive; warmup count must be nonnegative")
    if args.no_telemetry and (args.metrics_url or args.nvml_device is not None):
        parser.error("Telemetry sources conflict with --no-telemetry")
    problems = load_jsonl(args.workload, args.limit, args.tool_root)
    if args.backend == "scripted" and any(p.script is None for p in problems):
        parser.error("Every problem needs a script for --backend scripted")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics_url = args.metrics_url or (args.endpoint.rstrip("/") + "/metrics"
                                     if args.backend == "vllm" else None)
    manifest = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    manifest.update(status="preparing", started_utc=datetime.now(timezone.utc).isoformat(),
                    host=socket.gethostname(), pid=os.getpid(),
                    workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                    workload_ids=[p.id for p in problems], count=len(problems),
                    metrics_url=metrics_url, python=platform.python_version(),
                    client_versions={name: version(name) for name in ("httpx", "prometheus-client")},
                    tool_hashes={t["implementation"]: hashlib.sha256(Path(t["implementation"]).read_bytes()).hexdigest()
                                 for p in problems for t in p.tools if "implementation" in t})

    def receipt() -> None:
        temporary = args.output / "run.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(args.output / "run.json")

    receipt()
    events = []
    try:
        backend = (ScriptedBackend() if args.backend == "scripted" else VLLMBackend(
            args.endpoint, args.model, args.request_timeout, args.max_inflight_model_requests,
            args.max_tokens, not args.no_structured_output, seed=args.seed))
        telemetry = (Telemetry(metrics_url, args.model, args.nvml_device, args.telemetry_interval)
                     if not args.no_telemetry and (metrics_url or args.nvml_device is not None) else None)
        with (args.output / "events.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
            def record(event: dict) -> None:
                events.append(event)
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                if event["event"] == "measurement_started":
                    manifest["status"] = "running"
                    manifest["measurement_started_utc"] = datetime.now(timezone.utc).isoformat()
                    receipt()

            summary, _ = asyncio.run(run(
                problems, backend, args.policy, cpu_workers=args.cpu_workers,
                max_inflight_model_requests=args.max_inflight_model_requests,
                max_turns=args.max_turns, max_active_trajectories=args.max_active_trajectories,
                warmup_requests=args.warmup_requests, tool_timeout=args.tool_timeout,
                worker_startup_timeout=args.worker_startup_timeout, telemetry=telemetry,
                event_sink=record, starvation_threshold=args.starvation_threshold,
                aging_seconds=args.aging_seconds, nvtx_enabled=args.nvtx,
                dedicated_gpu=args.dedicated_gpu))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_timeline(events, args.output)
        manifest["status"] = "complete"
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt()


if __name__ == "__main__":
    main()
