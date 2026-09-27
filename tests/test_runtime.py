import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from asyncroll.model import Generation, ScriptedBackend, parse_action
from asyncroll.runtime import CPUQueue, Recorder, run
from asyncroll.workload import Problem, load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    def test_smoke_all_policies(self):
        problems = load_jsonl(ROOT / "examples" / "smoke.jsonl")
        for policy in ("sync", "fifo", "tool_first"):
            with self.subTest(policy=policy):
                summary, events = asyncio.run(run(
                    problems, ScriptedBackend(), policy,
                    cpu_workers=1, max_inflight_model_requests=1))
                self.assertEqual(summary["completed"], 3)
                self.assertEqual(summary["total_tool_calls"], 2)
                self.assertEqual(summary["exact_accuracy"], 1)
                self.assertTrue(any(e["event"] == "cpu_started" for e in events))

    def test_bad_action_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('{"type":"tool","name":"add"}')

    def test_unknown_tool_is_not_a_valid_action(self):
        problem = Problem("unknown", "call tool", script=[
            {"type": "tool", "name": "missing", "arguments": {}}])
        summary, _ = asyncio.run(run([problem], ScriptedBackend(), "fifo",
                                     cpu_workers=1, warmup_requests=0))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["action_valid_rate"], 0)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup.jsonl"
            row = json.dumps({"id": "same", "prompt": "hello"})
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path)

    def test_tool_first_queues_tool_before_grade(self):
        async def exercise():
            recorder = Recorder()
            async with CPUQueue(1, "tool_first", recorder) as cpu:
                grading = asyncio.create_task(cpu.submit(
                    "finished", "grade", answer="1", reference="1"))
                tool = asyncio.create_task(cpu.submit(
                    "continues", "tool", tool={"name": "add"},
                    arguments={"a": 1, "b": 2}))
                await asyncio.gather(grading, tool)
            return [e["id"] for e in recorder.events
                    if e["event"] == "cpu_started"]

        self.assertEqual(asyncio.run(exercise()), ["continues", "finished"])

    def test_independent_limits_and_no_unlabeled_grading(self):
        class DelayedBackend:
            active = 0
            peak = 0

            async def generate(self, problem, messages, turn):
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0.02)
                self.active -= 1
                return Generation({"type": "final", "answer": "1"}, request_seconds=0.02)

        for trajectories, requests in ((3, 1), (2, 5)):
            backend = DelayedBackend()
            problems = [Problem(str(i), "answer 1") for i in range(6)]
            summary, events = asyncio.run(run(
                problems, backend, "fifo", cpu_workers=1,
                max_active_trajectories=trajectories,
                max_inflight_model_requests=requests, warmup_requests=0))
            self.assertEqual(backend.peak, min(trajectories, requests))
            active, peak = 0, 0
            for event in events:
                active += (event["event"] == "trajectory_started") - (event["event"] == "trajectory_finished")
                peak = max(peak, active)
            self.assertEqual(peak, trajectories)
            self.assertFalse(any(e["event"] == "cpu_enqueued" for e in events))
            self.assertEqual(summary["graded"], 0)
            self.assertEqual(summary["completed"], 6)

    def test_timeout_kills_worker_and_next_job_recovers(self):
        async def exercise():
            recorder = Recorder()
            async with CPUQueue(1, "fifo", recorder, timeout=0.05) as cpu:
                old_process = cpu.workers[0].process
                with self.assertRaises(TimeoutError):
                    await cpu.submit("slow", "tool", tool={"name": "sleep_ms"},
                                     arguments={"milliseconds": 1000})
                self.assertFalse(old_process.is_alive())
                value = await cpu.submit("next", "tool", tool={"name": "add"},
                                         arguments={"a": 2, "b": 3})
                self.assertEqual(json.loads(value), {"result": 5})
                self.assertEqual(cpu.workers[0].generation, 2)
            self.assertTrue(any(e["event"] == "worker_restarting" for e in recorder.events))
        asyncio.run(exercise())

    def test_warm_workers_cache_modules_and_exclude_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.py"
            path.write_text("import time\ntime.sleep(0.1)\ncount = 0\n"
                            "def counter():\n    global count\n    count += 1\n    return count\n")
            tool = {"name": "counter", "implementation": str(path)}
            problem = Problem("cache", "counter twice", [tool], script=[
                {"type": "tool", "name": "counter", "arguments": {}},
                {"type": "tool", "name": "counter", "arguments": {}},
                {"type": "final", "answer": "done"}])

            class InspectBackend(ScriptedBackend):
                async def generate(self, problem, messages, turn):
                    if turn == 2:
                        self.observations = [m["content"] for m in messages if "returned:" in m["content"]]
                    return await super().generate(problem, messages, turn)

            backend = InspectBackend()
            summary, events = asyncio.run(run([problem], backend, "fifo", cpu_workers=2))
            # Both workers imported before measurement; each has its own cache/state.
            self.assertGreaterEqual(summary["warmup"]["worker_seconds"], 0.1)
            self.assertEqual(summary["model_attempts"], 3)
            self.assertEqual(summary["warmup"]["requests"], 2)
            executed = [e["execute_seconds"] for e in events if e["event"] == "cpu_finished"]
            self.assertTrue(all(value < 0.1 for value in executed))
            self.assertTrue(all(e["t"] >= 0 for e in events))

            async def cache_check():
                async with CPUQueue(1, "fifo", Recorder(), [tool]) as cpu:
                    one = await cpu.submit("x", "tool", tool=tool, arguments={})
                    two = await cpu.submit("x", "tool", tool=tool, arguments={})
                    self.assertEqual((one, two), ("1", "2"))
            asyncio.run(cache_check())


if __name__ == "__main__":
    unittest.main()
