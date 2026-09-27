import asyncio
import json
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from asyncroll.model import ScriptedBackend, parse_action
from asyncroll.runtime import CPUQueue, Recorder, run
from asyncroll.workload import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    def test_smoke_all_policies(self):
        problems = load_jsonl(ROOT / "examples" / "smoke.jsonl")
        for policy in ("sync", "fifo", "gpu_first"):
            with self.subTest(policy=policy):
                summary, events = asyncio.run(run(
                    problems, ScriptedBackend(), policy,
                    cpu_workers=1, gpu_slots=1))
                self.assertEqual(summary["completed"], 3)
                self.assertEqual(summary["total_tool_calls"], 2)
                self.assertEqual(summary["exact_accuracy"], 1)
                self.assertTrue(any(e["event"] == "cpu_started" for e in events))

    def test_bad_action_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('{"type":"tool","name":"add"}')

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup.jsonl"
            row = json.dumps({"id": "same", "prompt": "hello"})
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path)

    def test_gpu_first_queues_gpu_returning_tool_before_grade(self):
        async def exercise():
            recorder = Recorder()
            with ProcessPoolExecutor(max_workers=1) as executor:
                async with CPUQueue(1, "gpu_first", recorder, executor) as cpu:
                    grading = asyncio.create_task(cpu.submit(
                        "finished", "grade", answer="1", reference="1"))
                    tool = asyncio.create_task(cpu.submit(
                        "continues", "tool", tool={"name": "add"},
                        arguments={"a": 1, "b": 2}))
                    await asyncio.gather(grading, tool)
            return [e["id"] for e in recorder.events
                    if e["event"] == "cpu_started"]

        self.assertEqual(asyncio.run(exercise()), ["continues", "finished"])


if __name__ == "__main__":
    unittest.main()
