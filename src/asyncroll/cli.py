"""Command-line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from .model import ScriptedBackend, VLLMBackend
from .runtime import run
from .workload import convert_toolmath, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="asyncroll")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-toolmath")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--functions-dir", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--limit", type=int)

    bench = commands.add_parser("run")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True,
                       help="New run directory; must not exist")
    bench.add_argument("--policy", choices=["sync", "fifo", "gpu_first"], required=True)
    bench.add_argument("--backend", choices=["scripted", "vllm"], required=True)
    bench.add_argument("--model", help="vLLM model name")
    bench.add_argument("--endpoint", default="http://localhost:8000")
    bench.add_argument("--limit", type=int)
    bench.add_argument("--cpu-workers", type=int, default=2)
    bench.add_argument("--gpu-slots", type=int, default=4)
    bench.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()
    if args.command == "convert-toolmath":
        count = convert_toolmath(args.source, args.functions_dir, args.output,
                                 args.limit)
        print(f"Wrote {count} problems to {args.output}")
        return

    if args.backend == "vllm" and not args.model:
        parser.error("--model is required with --backend vllm")
    problems = load_jsonl(args.workload, args.limit)
    if args.backend == "scripted" and any(p.script is None for p in problems):
        parser.error("Every problem needs a script for --backend scripted")
    backend = (ScriptedBackend() if args.backend == "scripted"
               else VLLMBackend(args.endpoint, args.model))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workload": str(args.workload.resolve()),
        "backend": args.backend, "model": args.model,
        "endpoint": args.endpoint if args.backend == "vllm" else None,
        "policy": args.policy, "cpu_workers": args.cpu_workers,
        "gpu_slots": args.gpu_slots, "max_turns": args.max_turns,
        "count": len(problems),
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    summary, events = asyncio.run(run(problems, backend, args.policy,
                                      args.cpu_workers, args.gpu_slots,
                                      args.max_turns))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    with (args.output / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
