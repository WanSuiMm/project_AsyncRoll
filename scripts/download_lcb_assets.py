"""Download pinned model and LiveCodeBench assets with durable receipts."""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("model", "dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    repository = ("Qwen/Qwen2.5-Coder-7B-Instruct" if args.kind == "model"
                  else "livecodebench/code_generation")
    repository_type = "model" if args.kind == "model" else "dataset"
    api = HfApi()
    info = (api.model_info(repository) if args.kind == "model"
            else api.dataset_info(repository))
    revision = info.sha
    base = dict(kind=args.kind, repository=repository,
                repository_type=repository_type, revision=revision,
                output=str(args.output.resolve()), workers=args.workers,
                host=socket.gethostname(), pid=os.getpid(), started_utc=utc_now())
    write_receipt(args.receipt, **base, status="running")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.kind == "model":
            result = snapshot_download(
                repo_id=repository, repo_type="model", revision=revision,
                local_dir=args.output, max_workers=args.workers)
        else:
            result = hf_hub_download(
                repo_id=repository, repo_type="dataset", revision=revision,
                filename="test.jsonl", local_dir=args.output)
        write_receipt(args.receipt, **base, status="complete",
                      resolved_path=str(Path(result).resolve()), finished_utc=utc_now())
    except BaseException as exc:
        write_receipt(args.receipt, **base, status="failed",
                      error=f"{type(exc).__name__}: {exc}", finished_utc=utc_now())
        raise


if __name__ == "__main__":
    main()
