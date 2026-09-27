import json
import base64
import pickle
import zlib
import tempfile
import unittest
from pathlib import Path

from asyncroll.code_workload import convert_livecodebench
from asyncroll.workload import load_jsonl


def _record(question_id: str, test_marker: str | None = None) -> dict:
    tests = [{"input": "1\n", "output": "1\n"}]
    if test_marker is not None:
        tests = [{"input": test_marker, "output": "private expected output"}]
    return {
        "question_id": question_id,
        "question_title": f"Title {question_id}",
        "question_content": f"Read input and return the value for {question_id}.",
        "starter_code": "def solve():\n    pass\n",
        "public_test_cases": json.dumps(tests),
        "private_test_cases": [{"input": "2\n", "output": "2\n"}],
        # Source solutions are data only and must never be executed or prompted.
        "canonical_solution": "raise RuntimeError('must not execute')",
        "platform": "synthetic",
        "difficulty": "easy",
    }


def _output_ids(path: Path) -> set[str]:
    return {row["id"] for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


class CodeWorkloadTest(unittest.TestCase):
    def test_seeded_selection_is_repeatable_and_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_record(f"q-{index}") for index in range(40)]
            source = root / "release.json"
            output_a, output_b = root / "a.jsonl", root / "b.jsonl"
            source.write_text(json.dumps(rows), encoding="utf-8")

            self.assertEqual(convert_livecodebench(source, output_a, limit=9, seed=1729), 9)
            self.assertEqual(convert_livecodebench(source, output_b, limit=9, seed=1729), 9)
            first_bytes = output_a.read_bytes()
            self.assertEqual(first_bytes, output_b.read_bytes())

            reordered_source, reordered_output = root / "reordered.jsonl", root / "reordered-out.jsonl"
            reordered_source.write_text(
                "\n".join(json.dumps(row) for row in reversed(rows)) + "\n",
                encoding="utf-8",
            )
            convert_livecodebench(reordered_source, reordered_output, limit=9, seed=1729)
            self.assertEqual(_output_ids(output_a), _output_ids(reordered_output))

    def test_output_loads_as_problem_and_reference_tests_stay_out_of_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "one.jsonl", root / "workload.jsonl"
            row = _record("lcb/question-1", test_marker="TEST_ONLY_SENTINEL")
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            self.assertEqual(convert_livecodebench(source, output, seed=7), 1)
            problems = load_jsonl(output)
            self.assertEqual(len(problems), 1)
            problem = problems[0]
            self.assertEqual(problem.id, "lcb/question-1")
            self.assertIn("Read input and return", problem.prompt)
            self.assertIn("def solve()", problem.prompt)
            self.assertNotIn("TEST_ONLY_SENTINEL", problem.prompt)
            self.assertNotIn("private expected output", problem.prompt)
            self.assertNotIn("must not execute", problem.prompt)
            self.assertEqual(problem.tools[0]["name"], "lcb_evaluate")
            self.assertEqual(problem.tools[0]["inputs"], {"code": "str"})
            self.assertIn("reference_tests", problem.tools[0]["bound_arguments"])
            self.assertIsNone(problem.reference_answer)
            self.assertEqual(problem.metadata["sample_seed"], 7)
            self.assertEqual(problem.metadata["reference_tests"]["public"][0]["input"],
                             "TEST_ONLY_SENTINEL")
            self.assertEqual(problem.metadata["reference_tests"]["private"][0]["output"],
                             "2\n")
            prompt_metadata = problem.metadata["prompt_metadata"]
            self.assertEqual(prompt_metadata["protocol"], "one_repair")
            self.assertEqual(prompt_metadata["initial_attempts"], 1)
            self.assertEqual(prompt_metadata["repair_attempts"], 1)
            self.assertFalse(prompt_metadata["reference_tests_in_prompt"])
            self.assertIn("one corrected complete program", prompt_metadata["repair_instruction"])

    def test_required_fields_and_reference_tests_fail_explicitly(self):
        invalid_records = [
            ({"question_content": "statement", "public_test_cases": [{"in": "1"}]},
             "missing required non-empty question_id"),
            ({"question_id": "q", "public_test_cases": [{"in": "1"}]},
             "missing required question_content"),
            ({"question_id": "q", "question_content": "statement"},
             "missing non-empty reference test cases"),
            ({**_record("q"), "private_test_cases": "not JSON"},
             "private_test_cases must contain JSON or the pinned"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (row, message) in enumerate(invalid_records):
                with self.subTest(message=message):
                    source, output = root / f"invalid-{index}.json", root / f"out-{index}.jsonl"
                    source.write_text(json.dumps([row]), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        convert_livecodebench(source, output)
                    self.assertFalse(output.exists())

    def test_invalid_jsonl_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "invalid.jsonl", root / "out.jsonl"
            source.write_text('{"question_id":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"invalid\.jsonl:1: invalid JSONL record"):
                convert_livecodebench(source, output)

            source.write_text(
                "\n".join(json.dumps(_record("same")) for _ in range(2)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate problem ID"):
                convert_livecodebench(source, output, limit=1)
            self.assertFalse(output.exists())

    def test_rejects_nonpositive_limit_and_source_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "release.json"
            source.write_text(json.dumps([_record("q")]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "limit must be a positive integer"):
                convert_livecodebench(source, Path(directory) / "out.jsonl", limit=0)
            with self.assertRaisesRegex(ValueError, "source and destination must be different"):
                convert_livecodebench(source, source)

    def test_decodes_livecodebench_compressed_private_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _record("packed")
            private = json.dumps([{"input": "9\n", "output": "18\n"}])
            row["private_test_cases"] = base64.b64encode(
                zlib.compress(pickle.dumps(private))).decode()
            source, output = root / "packed.jsonl", root / "out.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            convert_livecodebench(source, output)
            problem = load_jsonl(output)[0]
            tests = problem.metadata["reference_tests"]["private"]
            self.assertEqual(tests[0]["output"], "18\n")


if __name__ == "__main__":
    unittest.main()
