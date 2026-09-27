import asyncio
import unittest
from unittest.mock import patch

from asyncroll.profiling import Profiler
from asyncroll.workers import Worker, _serve


class FakeNVTX:
    def __init__(self):
        self.next_id = 0
        self.started = []
        self.ended = []

    def start_range(self, **kwargs):
        self.next_id += 1
        range_id = (self.next_id, 0)
        self.started.append((kwargs, range_id))
        return range_id

    def end_range(self, range_id):
        self.ended.append(range_id)


class ProfilingTest(unittest.TestCase):
    def test_disabled_profiler_does_not_import_nvtx(self):
        with patch("asyncroll.profiling.import_module") as import_module:
            profiler = Profiler(enabled=False)
            handle = profiler.start("MODEL_REQUEST", object())
            profiler.end(handle)
            profiler.close()
        self.assertIsNone(handle)
        import_module.assert_not_called()

    def test_enabled_profiler_uses_explicit_overlapping_ranges_and_payloads(self):
        fake = FakeNVTX()
        with patch("asyncroll.profiling.import_module", return_value=fake):
            profiler = Profiler(enabled=True)
            first = profiler.start("MODEL_WAIT", 17)
            second = profiler.start("MODEL_REQUEST", 18)
            profiler.end(first)
            profiler.close()

        self.assertIsNot(first, second)
        self.assertEqual([call[0] for call in fake.started], [
            {"message": "MODEL_WAIT", "payload": 17},
            {"message": "MODEL_REQUEST", "payload": 18},
        ])
        self.assertEqual(fake.ended, [(1, 0), (2, 0)])

    def test_end_is_idempotent_and_identity_is_numeric(self):
        fake = FakeNVTX()
        with patch("asyncroll.profiling.import_module", return_value=fake):
            profiler = Profiler(enabled=True)
            handle = profiler.start("TOOL_QUEUE")
            profiler.end(handle)
            profiler.end(handle)
            with self.assertRaises(TypeError):
                profiler.start("TOOL_QUEUE", "job-17")
        self.assertEqual(fake.ended, [(1, 0)])

    def test_missing_optional_dependency_has_clear_error(self):
        missing = ModuleNotFoundError("No module named 'nvtx'")
        with patch("asyncroll.profiling.import_module", side_effect=missing):
            with self.assertRaisesRegex(ImportError, "optional 'nvtx' package"):
                Profiler(enabled=True)

    def test_worker_brackets_tool_and_grade_calls_including_errors(self):
        events = []

        class RecordingProfiler:
            def __init__(self, enabled=False):
                self.enabled = enabled

            def start(self, label, identity=None):
                events.append(("start", label, identity))
                return label

            def end(self, handle):
                events.append(("end", handle))

            def close(self):
                events.append(("close",))

        class Connection:
            def __init__(self):
                self.jobs = [
                    {"kind": "tool", "tool": {"name": "bad"},
                     "arguments": {}, "nvtx_identity": 27},
                    {"kind": "grade", "answer": "1", "reference": "1",
                     "nvtx_identity": 28},
                    None,
                ]
                self.sent = []

            def recv(self):
                return self.jobs.pop(0)

            def send(self, value):
                self.sent.append(value)

            def close(self):
                pass

        connection = Connection()
        def fake_call_tool(tool, arguments):
            if tool["name"] == "add":
                return 0
            raise ValueError("bad tool")

        with patch("asyncroll.workers.Profiler", RecordingProfiler), \
                patch("asyncroll.workers.call_tool", side_effect=fake_call_tool), \
                patch("asyncroll.workers.grade", return_value=True):
            _serve(connection, [], nvtx_enabled=True)

        self.assertEqual(events, [
            ("start", "TOOL_EXEC", 27), ("end", "TOOL_EXEC"),
            ("start", "GRADE_EXEC", 28), ("end", "GRADE_EXEC"),
            ("close",),
        ])
        self.assertTrue(connection.sent[0]["ready"])
        self.assertIn("ValueError: bad tool", connection.sent[1]["error"])
        self.assertTrue(connection.sent[2]["result"])

    def test_cpu_time_control_message_is_separate_from_tool_jobs(self):
        async def exercise():
            worker = Worker(0, [], startup_timeout=10)
            try:
                await worker.start()
                before = await worker.cpu_time()
                result = await worker.execute({
                    "kind": "tool", "tool": {"name": "add"},
                    "arguments": {"a": 2, "b": 3}, "nvtx_identity": 1,
                }, timeout=10)
                after = await worker.cpu_time()
                return result, before, after
            finally:
                await worker.dispose()

        result, before, after = asyncio.run(exercise())
        self.assertEqual(result["result"], '{"result": 5}')
        self.assertGreaterEqual(before, 0)
        self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main()
