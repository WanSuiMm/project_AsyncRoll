"""Model-only serving benchmarks used by the optional 5090 ablation.

This module deliberately excludes tool execution and AsyncRoll scheduling.  It
can benchmark an OpenAI-compatible server or a local Transformers fixed batch
with the same prompts and greedy decoding budget.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from .model import initial_messages
from .workload import load_jsonl


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_messages(path: Path, warmup: int, measured: int) -> list[dict[str, Any]]:
    if warmup < 0 or measured < 1:
        raise ValueError("warmup must be non-negative and measured must be positive")
    problems = load_jsonl(path, limit=warmup + measured)
    if len(problems) < warmup + measured:
        raise ValueError(f"need {warmup + measured} prompts, found {len(problems)}")
    return [{"id": problem.id, "messages": initial_messages(problem)}
            for problem in problems]


def prompt_digest(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def summarize(samples: list[dict[str, Any]], elapsed: float,
              engine: str, prompt_sha256: str) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.get("ok")]
    latencies = [float(sample["latency_seconds"]) for sample in successful]
    output_tokens = sum(int(sample.get("completion_tokens", 0)) for sample in successful)
    input_tokens = sum(int(sample.get("prompt_tokens", 0)) for sample in successful)
    output_hash = hashlib.sha256("\n".join(
        str(sample.get("output_sha256", "")) for sample in successful
    ).encode("ascii")).hexdigest()
    return {
        "schema_version": 1,
        "engine": engine,
        "status": "complete" if len(successful) == len(samples) else "failed_requests",
        "prompt_sha256": prompt_sha256,
        "requests": len(samples),
        "successful_requests": len(successful),
        "elapsed_seconds": elapsed,
        "requests_per_second": len(successful) / elapsed if elapsed > 0 else None,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed if elapsed > 0 else None,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "combined_output_sha256": output_hash,
        "samples": samples,
    }


async def benchmark_http(endpoint: str, model: str, records: list[dict[str, Any]],
                         warmup: int, concurrency: int, max_tokens: int,
                         timeout: float, seed: int,
                         transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    if concurrency < 1 or max_tokens < 1:
        raise ValueError("concurrency and max_tokens must be positive")
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    limits = httpx.Limits(max_connections=concurrency,
                          max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits,
                                 transport=transport) as client:
        async def request(record: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(url, json={
                    "model": model,
                    "messages": record["messages"],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "seed": seed,
                })
                latency = time.perf_counter() - started
                try:
                    response.raise_for_status()
                    body = response.json()
                    content = body["choices"][0]["message"]["content"] or ""
                    usage = body.get("usage") or {}
                    return {
                        "id": record["id"], "ok": True,
                        "latency_seconds": latency,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    }
                except Exception as exc:
                    return {"id": record["id"], "ok": False,
                            "latency_seconds": latency,
                            "error": f"{type(exc).__name__}: {exc}"}

        for record in records[:warmup]:
            sample = await request(record)
            if not sample["ok"]:
                raise RuntimeError(f"warmup request failed: {sample['error']}")
        measured = records[warmup:]
        started = time.perf_counter()
        samples = await asyncio.gather(*(request(record) for record in measured))
        elapsed = time.perf_counter() - started
    return summarize(samples, elapsed, "vllm_http", prompt_digest(measured))


def _generated_length(tokens: list[int], eos_ids: set[int], pad_id: int | None) -> int:
    count = 0
    for token in tokens:
        if pad_id is not None and token == pad_id and token not in eos_ids:
            break
        count += 1
        if token in eos_ids:
            break
    return count


def benchmark_transformers(model_path: Path, revision: str, records: list[dict[str, Any]],
                           warmup: int, batch_size: int, max_tokens: int,
                           dtype_name: str, attention: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers benchmark requires torch and transformers") from exc
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), revision=revision,
                                               trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), revision=revision, torch_dtype=dtype,
        attn_implementation=attention, device_map={"": 0}, trust_remote_code=False)
    model.eval()

    def run_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts = [tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                               add_generation_prompt=True)
                 for row in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        input_width = int(encoded["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**encoded, do_sample=False,
                                     max_new_tokens=max_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        eos = tokenizer.eos_token_id
        eos_ids = {int(eos)} if isinstance(eos, int) else set(int(x) for x in (eos or []))
        samples = []
        for row, input_ids, output_ids in zip(batch, encoded["attention_mask"], outputs):
            generated = output_ids[input_width:].tolist()
            length = _generated_length(generated, eos_ids, tokenizer.pad_token_id)
            content = tokenizer.decode(generated[:length], skip_special_tokens=True)
            samples.append({
                "id": row["id"], "ok": True, "latency_seconds": latency,
                "prompt_tokens": int(input_ids.sum().item()),
                "completion_tokens": length,
                "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })
        return samples

    for offset in range(0, warmup, batch_size):
        run_batch(records[offset:min(warmup, offset + batch_size)])
    measured = records[warmup:]
    samples: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(measured), batch_size):
        samples.extend(run_batch(measured[offset:offset + batch_size]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = summarize(samples, elapsed, "transformers_fixed_batch",
                       prompt_digest(measured))
    result["batch_size"] = batch_size
    result["attention_implementation"] = attention
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workload", type=Path, required=True)
    common.add_argument("--warmup", type=int, required=True)
    common.add_argument("--measured", type=int, required=True)
    common.add_argument("--max-tokens", type=int, required=True)
    common.add_argument("--output", type=Path, required=True)

    http_parser = subparsers.add_parser("http", parents=[common])
    http_parser.add_argument("--endpoint", required=True)
    http_parser.add_argument("--model", required=True)
    http_parser.add_argument("--concurrency", type=int, required=True)
    http_parser.add_argument("--timeout", type=float, default=300)
    http_parser.add_argument("--seed", type=int, default=0)

    transformer_parser = subparsers.add_parser("transformers", parents=[common])
    transformer_parser.add_argument("--model-path", type=Path, required=True)
    transformer_parser.add_argument("--revision", required=True)
    transformer_parser.add_argument("--batch-size", type=int, required=True)
    transformer_parser.add_argument("--dtype", choices=["bfloat16", "float16"],
                                    default="bfloat16")
    transformer_parser.add_argument("--attention", default="flash_attention_2")
    args = parser.parse_args()
    records = load_messages(args.workload, args.warmup, args.measured)
    if args.command == "http":
        result = asyncio.run(benchmark_http(
            args.endpoint, args.model, records, args.warmup, args.concurrency,
            args.max_tokens, args.timeout, args.seed))
    else:
        result = benchmark_transformers(
            args.model_path, args.revision, records, args.warmup, args.batch_size,
            args.max_tokens, args.dtype, args.attention)
    write_result(args.output, result)


if __name__ == "__main__":
    main()
