import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from asyncroll.serving_ablation import (
    _generated_length,
    benchmark_http,
    load_messages,
    percentile,
    prompt_digest,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_serving_ablation", ROOT / "scripts/run_serving_ablation.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class ServingAblationTest(unittest.TestCase):
    def test_config_freezes_matched_engine_and_attention_comparisons(self):
        config = json.loads((ROOT / "experiments/serving_ablation_5090.json")
                            .read_text(encoding="utf-8"))
        RUNNER.validate_config(config)
        self.assertEqual(config["benchmark"]["concurrency"], 4)
        self.assertEqual(config["transformers"]["batch_size"], 4)
        self.assertEqual(config["transformers"]["attention_implementation"],
                         "flash_attention_2")
        self.assertEqual(config["attention"]["primary_backend"], "FLASH_ATTN")
        self.assertFalse(config["vllm"]["prefix_caching"])
        self.assertEqual(config["benchmark"]["repeats"], 3)

    def test_prompt_loading_digest_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            rows = [{"id": str(i), "prompt": f"problem {i}",
                     "metadata": {"prompt_metadata": {"protocol": "one_repair"}}}
                    for i in range(3)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                            encoding="utf-8")
            records = load_messages(path, warmup=1, measured=2)
        self.assertEqual([row["id"] for row in records], ["0", "1", "2"])
        self.assertEqual(prompt_digest(records), prompt_digest(records))
        samples = [{"id": "1", "ok": True, "latency_seconds": 1.0,
                    "prompt_tokens": 10, "completion_tokens": 4,
                    "output_sha256": "a"},
                   {"id": "2", "ok": True, "latency_seconds": 3.0,
                    "prompt_tokens": 12, "completion_tokens": 6,
                    "output_sha256": "b"}]
        result = summarize(samples, 2.0, "test", "digest")
        self.assertEqual(result["requests_per_second"], 1.0)
        self.assertEqual(result["output_tokens_per_second"], 5.0)
        self.assertEqual(result["latency_seconds"]["p50"], 2.0)
        self.assertEqual(percentile([1.0, 3.0], 0.95), 2.9)

    def test_http_benchmark_excludes_warmup_and_uses_greedy_budget(self):
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            payloads.append(payload)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            })

        records = [{"id": str(i), "messages": [{"role": "user", "content": str(i)}]}
                   for i in range(3)]
        result = asyncio.run(benchmark_http(
            "http://test", "model", records, warmup=1, concurrency=2,
            max_tokens=17, timeout=2, seed=9,
            transport=httpx.MockTransport(handler)))
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["completion_tokens"], 4)
        self.assertEqual(len(payloads), 3)
        self.assertTrue(all(payload["temperature"] == 0 for payload in payloads))
        self.assertTrue(all(payload["max_tokens"] == 17 for payload in payloads))
        self.assertTrue(all(payload["seed"] == 9 for payload in payloads))

    def test_token_length_stops_at_eos(self):
        self.assertEqual(_generated_length([3, 4, 2, 9], {2}, 0), 3)
        self.assertEqual(_generated_length([3, 0, 0], {2}, 0), 1)

    def test_aggregate_pairs_replicates_and_checks_outputs(self):
        def stage(arm, rps, token_rate, output):
            return {"replicate": 1, "arm": arm, "status": "complete", "metrics": {
                "requests_per_second": rps, "output_tokens_per_second": token_rate,
                "samples": [{"id": "x", "ok": True, "output_sha256": output}]}}
        result = RUNNER.aggregate([
            stage("transformers_flash_attn_2", 1.0, 10.0, "same"),
            stage("vllm_flash_attn", 2.0, 15.0, "same"),
            stage("vllm_reference_attention", 1.6, 12.0, "different"),
        ], repeats=1)
        engine = result["vllm_vs_transformers"]
        attention = result["flash_attention_vs_reference"]
        self.assertEqual(engine["mean_requests_per_second_change"], 1.0)
        self.assertEqual(engine["mean_exact_output_match_fraction"], 1.0)
        self.assertAlmostEqual(attention["mean_requests_per_second_change"], 0.25)
        self.assertEqual(attention["mean_exact_output_match_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
