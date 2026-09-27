import unittest
from types import SimpleNamespace

from asyncroll.scheduling import Scheduler


def job(number, test_count=1, code_bytes=10, enqueued=0, turn=0, kind="tool"):
    tests = [{"input": "x" * (10 * test_count), "output": "ignored"}
             for _ in range(test_count)]
    tool = {"name": "lcb_evaluate", "bound_arguments": {
        "reference_tests": {"public": tests, "private": []}}}
    payload = {"kind": kind, "turn": turn, "tool": tool,
               "arguments": {"code": "x" * code_bytes}}
    return SimpleNamespace(job_id=number, enqueued=enqueued, payload=payload)


class SchedulingTest(unittest.TestCase):
    def test_per_job_predictor_reorders_only_under_low_model_supply(self):
        slow, fast = job(0, test_count=20, code_bytes=1000), job(1, test_count=1)
        scheduler = Scheduler("asyncroll", aging_seconds=30, aging_weight=0)
        self.assertIs(scheduler.select([slow, fast], 0.1, 0)[0], slow)
        scheduler.observe(slow.payload, 5.0)
        scheduler.observe(fast.payload, 0.05)
        chosen, evidence = scheduler.select([slow, fast], 0.2, 0)
        self.assertIs(chosen, fast)
        self.assertTrue(evidence["reordered"])
        self.assertTrue(evidence["eligible_for_reorder"])
        self.assertEqual(evidence["reason"], "low_model_supply_predicted_unlock")
        self.assertIs(scheduler.select([slow, fast], 0.2, 1)[0], slow)
        self.assertIs(Scheduler("fifo").select([slow, fast], 0.2, 0)[0], slow)

    def test_soft_aging_changes_score_without_immediate_fifo_override(self):
        old_slow = job(0, test_count=20, enqueued=0)
        fresh_fast = job(1, test_count=1, enqueued=9)
        scheduler = Scheduler("asyncroll", aging_seconds=30, aging_weight=1.0)
        scheduler.observe(old_slow.payload, 5.0)
        scheduler.observe(fresh_fast.payload, 0.05)
        chosen, evidence = scheduler.select([old_slow, fresh_fast], 10, 0)
        self.assertIs(chosen, old_slow)
        self.assertEqual(evidence["reason"], "low_model_supply_predicted_unlock")

    def test_hard_deadline_prevents_starvation(self):
        expired = job(0, test_count=20, enqueued=0)
        fresh = job(1, test_count=1, enqueued=29)
        scheduler = Scheduler("asyncroll", aging_seconds=30, aging_weight=0)
        scheduler.observe(expired.payload, 5.0)
        scheduler.observe(fresh.payload, 0.05)
        chosen, evidence = scheduler.select([expired, fresh], 31, 0)
        self.assertIs(chosen, expired)
        self.assertEqual(evidence["reason"], "hard_starvation_fifo")

    def test_predictor_resets_each_run_and_features_are_job_specific(self):
        scheduler = Scheduler("asyncroll")
        small, large = job(0, test_count=1), job(1, test_count=10)
        scheduler.observe(small.payload, 0.1)
        first = scheduler.select([small], 0, 0)[1]
        second = scheduler.select([large], 0, 0)[1]
        self.assertNotEqual(first["feature_key"], second["feature_key"])
        self.assertEqual(scheduler.predictor.observations, 1)
        self.assertEqual(Scheduler("asyncroll").predictor.observations, 0)


if __name__ == "__main__":
    unittest.main()
