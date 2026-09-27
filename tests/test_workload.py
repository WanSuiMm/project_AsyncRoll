import json
import tempfile
import unittest
from pathlib import Path

from asyncroll.workload import convert_toolmath, load_jsonl


class WorkloadTest(unittest.TestCase):
    def test_sample_seed_metadata_and_portable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            functions = root / "functions"
            functions.mkdir()
            (functions / "f.py").write_text("def f(): return 1\n")
            source = root / "source.json"
            source.write_text(json.dumps([dict(function="f.py", name="f", description="f", inputs={},
                source_problem=str(i), difficulty="easy", category="algebra") for i in range(20)]))
            output = root / "workload.jsonl"
            convert_toolmath(source, functions, output, limit=5, seed=7)
            first = output.read_text()
            convert_toolmath(source, functions, output, limit=5, seed=7)
            self.assertEqual(first, output.read_text())
            self.assertNotIn(str(functions), first)
            problems = load_jsonl(output, tool_root=functions)
            self.assertEqual(problems[0].metadata["category"], "algebra")
            self.assertEqual(Path(problems[0].tools[0]["implementation"]), functions / "f.py")
            self.assertNotEqual([p.metadata["source_index"] for p in problems], list(range(5)))
