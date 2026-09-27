import asyncio
import unittest

from asyncroll.code_executor import evaluate_python
from asyncroll.model import ScriptedBackend, action_schema, initial_messages
from asyncroll.runtime import run
from asyncroll.workload import Problem


def problem(script):
    tests = {"public": [{"input": "2\n", "output": "4\n"}], "private": []}
    tool = {"name": "lcb_evaluate", "inputs": {"code": "str"},
            "bound_arguments": {"reference_tests": tests,
                                "execution_metadata": {}, "timeout_seconds": 2}}
    metadata = {"reference_tests": tests, "prompt_metadata": {
        "protocol": "one_repair", "max_generations": 2,
        "repair_instruction": "Repair once."}}
    return Problem("code", "double input", [tool], script=script, metadata=metadata)


class CodeExecutorTest(unittest.TestCase):
    def test_stdin_pass_and_failure_feedback(self):
        tests = {"public": [{"input": "3\n", "output": "6\n"}], "private": []}
        passed = evaluate_python("print(int(input())*2)", tests, timeout_seconds=2)
        failed = evaluate_python("print(0)", tests, timeout_seconds=2)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertIn('"expected": "6', failed["feedback"])

    def test_schema_and_prompt_are_code_specific(self):
        item = problem([])
        schema = action_schema(item, 0)
        self.assertEqual(schema["properties"]["type"], {"const": "submit"})
        self.assertEqual(schema["properties"]["code"]["maxLength"], 20000)
        self.assertIn("complete Python 3 solution", initial_messages(item)[0]["content"])

    def test_one_repair_executes_twice_and_records_quality(self):
        item = problem([
            {"type": "submit", "code": "print(0)"},
            {"type": "submit", "code": "print(int(input())*2)"},
        ])
        streamed = []
        summary, events = asyncio.run(run(
            [item], ScriptedBackend(), "fifo", cpu_workers=1,
            warmup_requests=0, max_turns=2, result_sink=streamed.append))
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["exact_accuracy"], 1.0)
        self.assertEqual(summary["total_tool_calls"], 2)
        self.assertTrue(summary["results"][0]["repair_attempted"])
        self.assertNotIn("reference_tests", summary["results"][0]["metadata"])
        self.assertNotIn("prompt_metadata", summary["results"][0]["metadata"])
        self.assertEqual(streamed, summary["results"])
        self.assertEqual(sum(e["event"] == "cpu_finished" for e in events), 2)

    def test_failed_repair_is_completed_but_not_correct(self):
        item = problem([
            {"type": "submit", "code": "print(0)"},
            {"type": "submit", "code": "print(1)"},
        ])
        summary, _ = asyncio.run(run(
            [item], ScriptedBackend(), "fifo", cpu_workers=1,
            warmup_requests=0, max_turns=2))
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["exact_accuracy"], 0.0)
        self.assertFalse(summary["results"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
