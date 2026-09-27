import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from asyncroll.report import aggregate, write_report


class ReportTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional figure dependency not installed")
    def test_offline_figure_export_from_synthetic_fixture(self):
        arms = {p: {"seed": 0, "completed": 3, "failed": 0,
                    "resource_metrics": {"dedicated_single_gpu_declared": True,
                        "completed_per_gpu_hour": 100, "model_starvation_seconds": 0.1,
                        "cpu_core_seconds_per_completed_trajectory": 0.01}}
                for p in ("fifo", "asyncroll")}
        receipt = {"status": "complete", "phase": "comparison", "pairs": [{"pair_id": "fixture", "arms": arms}]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.json"
            source.write_text(json.dumps(receipt), encoding="utf-8")
            output = Path(directory) / "report"
            write_report(source, output, figure=True)
            self.assertTrue((output / "comparison.pdf").read_bytes().startswith(b"%PDF"))
            self.assertTrue((output / "comparison.png").read_bytes().startswith(b"\x89PNG"))
            self.assertIn("not mathematical correctness", (output / "REPORT.md").read_text(encoding="utf-8"))

    def test_preserves_paired_units_missing_cost_and_failure_denominators(self):
        arms = {p: {"seed": 0, "completed": n, "failed": 3-n,
                    "action_valid_rate": 1, "trajectory_tool_call_rate": 1,
                    "resource_metrics": {"dedicated_single_gpu_declared": True,
                        "completed_per_gpu_hour": n*100, "model_starvation_seconds": 0,
                        "cpu_core_seconds_per_completed_trajectory": None}}
                for p, n in (("fifo", 3), ("asyncroll", 2))}
        receipt = {"status": "complete", "phase": "comparison", "pairs": [{"pair_id": 0, "arms": arms}]}
        report = aggregate(receipt)
        self.assertAlmostEqual(report["metrics"]["completed_per_gpu_hour"]["paired_asyncroll_over_fifo"][0], 2/3)
        self.assertFalse(report["metrics"]["cpu_core_seconds_per_completed_trajectory"]["complete"])
        self.assertEqual(report["quality"][1]["failed"], 1)
        bad = copy.deepcopy(receipt)
        bad["pairs"][0]["arms"]["fifo"]["nvtx_enabled"] = True
        with self.assertRaises(ValueError):
            aggregate(bad)
        receipt["pairs"].append(receipt["pairs"][0])
        with self.assertRaises(ValueError):
            aggregate(receipt)


if __name__ == "__main__":
    unittest.main()
