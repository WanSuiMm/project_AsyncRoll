import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BoundedProtocolTest(unittest.TestCase):
    def test_single_operating_point_and_activation_gate_are_frozen(self):
        config = json.loads((ROOT / "experiments/livecodebench_bounded_8x4.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(config["protocol_id"], "asyncroll-bounded-8x4-qwen-coder-lcb-v3")
        self.assertEqual(config["runtime"]["max_active_trajectories"], 8)
        self.assertEqual(config["runtime"]["max_inflight_model_requests"], 4)
        self.assertEqual(config["runtime"]["cpu_workers"], 2)
        self.assertEqual(config["runtime"]["aging_seconds"], 30)
        self.assertNotIn("sweep", config)
        self.assertGreaterEqual(config["activation_gate"]["minimum_reorders"], 1)
        self.assertEqual(config["comparison"]["replicates"], 3)

    def test_runner_requires_v3_protocol_and_activation_stage(self):
        runner = (ROOT / "scripts/run_lcb_bounded_single_gpu.py").read_text(encoding="utf-8")
        self.assertIn("asyncroll-bounded-8x4-qwen-coder-lcb-v3", runner)
        self.assertIn('run_arm("activation-asyncroll", "asyncroll"', runner)
        self.assertIn("stopped_scheduler_not_activated", runner)

    def test_v6_binds_trigger_to_capacity_and_uses_simple_activation_gate(self):
        config = json.loads((ROOT / "experiments/livecodebench_bounded_8x4_v6.json")
                            .read_text(encoding="utf-8"))
        runtime = config["runtime"]
        self.assertEqual(config["protocol_id"], "asyncroll-bounded-8x4-qwen-coder-lcb-v6")
        self.assertEqual(runtime["starvation_threshold"], 4)
        self.assertEqual(runtime["starvation_threshold"], runtime["max_inflight_model_requests"])
        self.assertEqual(runtime["max_active_trajectories"], 8)
        self.assertEqual(runtime["cpu_workers"], 2)
        self.assertEqual(runtime["aging_seconds"], 30)
        gate = config["activation_gate"]
        self.assertEqual(set(gate), {"minimum_eligible_decisions", "minimum_reorders",
                                     "minimum_repair_prediction_log_pearson"})
        self.assertEqual(gate["minimum_eligible_decisions"], 20)
        self.assertEqual(gate["minimum_reorders"], 10)
        runner = (ROOT / "scripts/run_lcb_bounded_v6_single_gpu.py").read_text(encoding="utf-8")
        self.assertIn("starvation_threshold == max_inflight_model_requests", runner)
        self.assertIn("repair_prediction_log_pearson", runner)


if __name__ == "__main__":
    unittest.main()
