import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from asyncroll.telemetry import Telemetry, counter_rates, parse_vllm_metrics
from asyncroll.timeline import build_timeline, write_timeline


class TelemetryTest(unittest.TestCase):
    def test_parse_aggregates_and_filters_model_series(self):
        text = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="math",engine="0"} 2
vllm:num_requests_running{model_name="math",engine="1"} 3
vllm:num_requests_running{model_name="other"} 99
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="math"} 4
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="math",engine="0"} 100
vllm:prompt_tokens_total{model_name="math",engine="1"} 20
vllm:prompt_tokens_total{model_name="other"} 900
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="math"} 60
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="math",engine="0"} 0.25
vllm:kv_cache_usage_perc{model_name="math",engine="1"} 0.70
"""
        metrics = parse_vllm_metrics(text, model_name="math")
        self.assertEqual(metrics["running_requests"], 5)
        self.assertEqual(metrics["waiting_requests"], 4)
        self.assertEqual(metrics["prompt_tokens_total"], 120)
        self.assertEqual(metrics["generation_tokens_total"], 60)
        self.assertEqual(metrics["kv_cache_fraction"], 0.70)
        self.assertEqual(metrics["missing_metrics"], [])

    def test_missing_series_are_null_and_reported(self):
        metrics = parse_vllm_metrics(
            '# TYPE vllm:num_requests_running gauge\n'
            'vllm:num_requests_running{model_name="other"} 7\n',
            model_name="math",
        )
        self.assertIsNone(metrics["running_requests"])
        self.assertIsNone(metrics["waiting_requests"])
        self.assertIsNone(metrics["prompt_tokens_total"])
        self.assertIsNone(metrics["generation_tokens_total"])
        self.assertIsNone(metrics["kv_cache_fraction"])
        self.assertEqual(len(metrics["missing_metrics"]), 5)

    def test_legacy_kv_metric_fallback(self):
        metrics = parse_vllm_metrics(
            '# TYPE vllm:gpu_cache_usage_perc gauge\n'
            'vllm:gpu_cache_usage_perc{model_name="math"} 0.4\n',
            model_name="math",
        )
        self.assertEqual(metrics["kv_cache_fraction"], 0.4)

    def test_rates_and_counter_resets(self):
        rates, resets = counter_rates(
            {"prompt_tokens_total": 50, "generation_tokens_total": 20},
            {"prompt_tokens_total": 10, "generation_tokens_total": 10},
            2.0,
        )
        self.assertEqual(rates, {
            "prompt_tokens_total": 20.0,
            "generation_tokens_total": 5.0,
        })
        self.assertEqual(resets, [])

        rates, resets = counter_rates(
            {"prompt_tokens_total": 2, "generation_tokens_total": None},
            {"prompt_tokens_total": 10, "generation_tokens_total": 10},
            1.0,
        )
        self.assertIsNone(rates["prompt_tokens_total"])
        self.assertIsNone(rates["generation_tokens_total"])
        self.assertEqual(resets, ["prompt_tokens_total"])

    def test_prepare_and_collect_reuse_persistent_http_client(self):
        payloads = iter((
            self._metrics_text(prompt=10, generation=4),
            self._metrics_text(prompt=30, generation=14),
        ))

        class Response:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeClient:
            def __init__(self, *, timeout):
                self.timeout = timeout
                self.closed = False
                self.get_calls = 0

            async def get(self, _url):
                self.get_calls += 1
                return Response(next(payloads))

            async def aclose(self):
                self.closed = True

        async def exercise(factory):
            telemetry = Telemetry("http://local/metrics", model_name="math")
            probe = await telemetry.prepare()
            repeated_probe = await telemetry.prepare()
            client = telemetry._client
            sample = await telemetry.collect()
            await telemetry.close()
            return client, sample, probe, repeated_probe

        with patch("asyncroll.telemetry.httpx.AsyncClient", side_effect=FakeClient) as factory:
            client, sample, probe, repeated_probe = asyncio.run(exercise(factory))
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(client.get_calls, 2)
        self.assertTrue(client.closed)
        self.assertEqual(probe, repeated_probe)
        self.assertEqual(probe["requested_sources"], {"vllm": True, "nvml": False})
        self.assertEqual(probe["availability"], {"vllm": True, "nvml": None})
        self.assertEqual(sample["vllm_running_requests"], 1)
        self.assertGreater(sample["prompt_tokens_per_second"], 0)
        self.assertGreater(sample["generation_tokens_per_second"], 0)
        self.assertIsNone(sample["errors"]["metrics"])

    def test_partial_probe_is_unavailable_and_idempotent(self):
        class Response:
            text = ('# TYPE vllm:num_requests_running gauge\n'
                    'vllm:num_requests_running{model_name="math"} 1\n')

            def raise_for_status(self):
                return None

        class FakeClient:
            async def get(self, _url):
                return Response()

            async def aclose(self):
                return None

        async def exercise():
            telemetry = Telemetry("http://local/metrics", model_name="math")
            first = await telemetry.prepare()
            first["availability"]["vllm"] = True
            second = await telemetry.prepare()
            await telemetry.close()
            return second

        with patch("asyncroll.telemetry.httpx.AsyncClient", return_value=FakeClient()):
            probe = asyncio.run(exercise())
        self.assertFalse(probe["availability"]["vllm"])
        self.assertIn("vllm:num_requests_waiting", probe["missing_metrics"])
        self.assertIsNone(probe["probe_errors"]["vllm"])

    def test_non_finite_series_become_null_and_are_explicit(self):
        text = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="math"} NaN
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="math"} +Inf
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="math"} 2
"""
        metrics = parse_vllm_metrics(text, model_name="math")
        self.assertIsNone(metrics["running_requests"])
        self.assertIsNone(metrics["waiting_requests"])
        self.assertEqual(metrics["prompt_tokens_total"], 2)
        self.assertEqual(len(metrics["invalid_metrics"]), 2)
        json.dumps(metrics, allow_nan=False)

    def test_total_deadline_bounds_a_stalled_metrics_client(self):
        class FakeClient:
            async def get(self, _url):
                await asyncio.sleep(1)

            async def aclose(self):
                return None

        async def exercise():
            telemetry = Telemetry("http://local/metrics", timeout=0.02)
            started = time.perf_counter()
            probe = await telemetry.prepare()
            elapsed = time.perf_counter() - started
            await telemetry.close()
            return probe, elapsed

        with patch("asyncroll.telemetry.httpx.AsyncClient", return_value=FakeClient()):
            probe, elapsed = asyncio.run(exercise())
        self.assertFalse(probe["availability"]["vllm"])
        self.assertIsNotNone(probe["probe_errors"]["vllm"])
        self.assertLess(elapsed, 0.5)

    def test_sampler_subtracts_collection_time_and_skips_stop_race_sample(self):
        class SlowTelemetry(Telemetry):
            def __init__(self):
                super().__init__(None, interval=0.2)
                self._prepared = True
                self._probe_receipt = {
                    "requested_sources": {"vllm": False, "nvml": False},
                    "probe_errors": {"vllm": None, "nvml": None},
                    "missing_metrics": [], "invalid_metrics": [],
                    "availability": {"vllm": None, "nvml": None},
                }

            async def collect(self):
                await asyncio.sleep(0.05)
                return {"requested_sources": {"vllm": False, "nvml": False},
                        "availability": {"vllm": None, "nvml": None},
                        "errors": {"metrics": None, "nvml": None},
                        "missing_metrics": [], "invalid_metrics": []}

        class Recorder:
            def __init__(self):
                self.start = time.perf_counter()
                self.events = []

            def emit(self, problem_id, event, **values):
                self.events.append({"t": time.perf_counter() - self.start,
                                    "id": problem_id, "event": event, **values})

        async def exercise():
            telemetry = SlowTelemetry()
            recorder = Recorder()
            stop = asyncio.Event()
            sampler = asyncio.create_task(telemetry.sample_loop(recorder, stop))
            await asyncio.sleep(0.41)
            stop.set()
            await sampler
            return recorder.events

        events = asyncio.run(exercise())
        samples = [event for event in events if event["event"] == "resource_sample"]
        self.assertGreaterEqual(len(samples), 1)
        self.assertLessEqual(len(samples), 2)
        self.assertTrue(any(event["event"] == "resource_probe" for event in events))
        for sample in samples:
            self.assertLess(sample["collection_started_offset_seconds"],
                            sample["collection_finished_offset_seconds"])
            self.assertGreaterEqual(sample["collection_seconds"], 0.04)
            self.assertLessEqual(sample["collection_finished_offset_seconds"], sample["t"])
        if len(samples) > 1:
            start_gap = (samples[1]["collection_started_offset_seconds"]
                         - samples[0]["collection_started_offset_seconds"])
            self.assertLess(start_gap, 0.24)

    @staticmethod
    def _metrics_text(prompt, generation):
        return f"""# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="math"}} 1
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model_name="math"}} 2
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{{model_name="math"}} {prompt}
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model_name="math"}} {generation}
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{model_name="math"}} 0.5
"""


class TimelineTest(unittest.TestCase):
    def test_pairs_failures_and_unmatched_spans(self):
        events = [
            {"t": 0.0, "id": "ok", "event": "trajectory_started"},
            {"t": 0.1, "id": "ok", "event": "model_enqueued", "request_id": "r1"},
            {"t": 0.2, "id": "ok", "event": "model_started", "request_id": "r1"},
            {"t": 0.5, "id": "ok", "event": "model_failed", "request_id": "r1",
             "error": "offline"},
            {"t": 0.6, "id": "orphan", "event": "model_finished", "request_id": "r2"},
            {"t": 0.7, "id": "queued", "event": "cpu_enqueued", "job_id": "j1",
             "kind": "tool"},
            {"t": 0.8, "id": "queued", "event": "cpu_started", "job_id": "j1",
             "kind": "tool", "worker_id": 2},
            {"t": 1.1, "id": "queued", "event": "cpu_failed", "job_id": "j1",
             "kind": "tool", "worker_id": 2, "dispatch_seconds": 0.3,
             "worker_started": 0.82, "worker_finished": 1.02,
             "execute_seconds": 0.15, "serialization_seconds": 0.03},
            {"t": 1.2, "id": "pending", "event": "cpu_enqueued", "job_id": "j2",
             "kind": "grade"},
            {"t": 1.3, "id": "open", "event": "trajectory_started"},
            {"t": 1.5, "id": "ok", "event": "trajectory_finished", "status": "failed"},
            {"t": 1.6, "id": "__resources__", "event": "resource_sample",
             "vllm_running_requests": 1, "nvml_gpu_utilization_percent": 60},
        ]
        timeline = build_timeline(events)
        outcomes = [span["args"]["outcome"] for span in timeline["spans"]]
        self.assertIn("failed", outcomes)
        self.assertIn("incomplete", outcomes)
        self.assertIn("unmatched completion", outcomes)
        self.assertTrue(any(span["track"] == "CPU dispatch / worker 2"
                            for span in timeline["spans"]))
        child = next(span for span in timeline["spans"]
                     if span["track"] == "CPU child execution / worker 2")
        self.assertAlmostEqual(child["start"], 0.82)
        self.assertAlmostEqual(child["end"] - child["start"], 0.15)
        self.assertEqual(len(timeline["samples"]), 1)
        model_span = next(span for span in timeline["spans"]
                          if span["track"] == "Model HTTP client request / ok")
        self.assertEqual(model_span["end"] - model_span["start"], 0.3)

    def test_overlapping_trajectories_have_separate_phase_rows(self):
        events = [
            {"t": 0.0, "id": "A", "event": "trajectory_started"},
            {"t": 0.0, "id": "B", "event": "trajectory_started"},
            {"t": 0.1, "id": "A", "event": "model_enqueued", "request_id": "a1"},
            {"t": 0.1, "id": "B", "event": "model_enqueued", "request_id": "b1"},
            {"t": 0.2, "id": "A", "event": "model_started", "request_id": "a1"},
            {"t": 0.2, "id": "B", "event": "model_started", "request_id": "b1"},
            {"t": 0.5, "id": "A", "event": "model_finished", "request_id": "a1"},
            {"t": 0.6, "id": "B", "event": "model_finished", "request_id": "b1"},
            {"t": 0.7, "id": "A", "event": "trajectory_finished", "status": "completed"},
            {"t": 0.8, "id": "B", "event": "trajectory_finished", "status": "completed"},
        ]
        tracks = {span["track"] for span in build_timeline(events)["spans"]}
        self.assertIn("Trajectory / A", tracks)
        self.assertIn("Trajectory / B", tracks)
        self.assertIn("Model queue / A", tracks)
        self.assertIn("Model queue / B", tracks)
        self.assertIn("Model HTTP client request / A", tracks)
        self.assertIn("Model HTTP client request / B", tracks)

    def test_writer_creates_chrome_trace_and_self_contained_escaped_html(self):
        injected = '<img src=x onerror="alert(1)"></script>&'
        events = [
            {"t": 0.0, "id": injected, "event": "trajectory_started"},
            {"t": 0.1, "id": "__resources__", "event": "resource_probe",
             "requested_sources": {"vllm": True, "nvml": False},
             "probe_errors": {"vllm": None, "nvml": None},
             "missing_metrics": ["vllm:num_requests_waiting"],
             "availability": {"vllm": False, "nvml": None}},
            {"t": 0.5, "id": injected, "event": "trajectory_finished", "status": "completed"},
            {"t": 0.25, "id": "__resources__", "event": "resource_sample",
             "prompt_tokens_per_second": 12.5, "nvml_source": "local_host",
             "requested_sources": {"vllm": True, "nvml": True},
             "availability": {"vllm": True, "nvml": True},
             "errors": {"metrics": None, "nvml": None}},
            {"t": 0.35, "id": "__resources__", "event": "resource_sample",
             "prompt_tokens_per_second": None,
             "requested_sources": {"vllm": True, "nvml": True},
             "availability": {"vllm": False, "nvml": True},
             "errors": {"metrics": "TimeoutError", "nvml": None}},
            {"t": 0.45, "id": "__resources__", "event": "resource_sample",
             "prompt_tokens_per_second": 15.0,
             "requested_sources": {"vllm": True, "nvml": True},
             "availability": {"vllm": True, "nvml": True},
             "errors": {"metrics": None, "nvml": None}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace_path, html_path = write_timeline(events, Path(directory))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            page = html_path.read_text(encoding="utf-8")
        self.assertTrue(trace_path.name == "trace.json")
        self.assertTrue(html_path.name == "timeline.html")
        self.assertTrue(any(item.get("ph") == "X" for item in trace["traceEvents"]))
        self.assertEqual(trace["metadata"]["model_span_semantics"],
                         "HTTP/client time; not GPU execution")
        self.assertIn("const DATA = ", page)
        self.assertIn("(function(){", page)
        self.assertIn("const H = plotTop", page)
        self.assertNotIn("const left = 270, right = 26, top =", page)
        self.assertIn("toPrecision(3)+\" ms\"", page)
        self.assertIn('id="source-status"', page)
        self.assertIn("active=false", page)
        self.assertIn("missing_metrics", page)
        self.assertIn("vllm:num_requests_waiting", page)
        self.assertNotIn(injected, page)
        self.assertNotIn("https://", page)
        self.assertIn("prompt_tokens_per_second", page)


if __name__ == "__main__":
    unittest.main()
