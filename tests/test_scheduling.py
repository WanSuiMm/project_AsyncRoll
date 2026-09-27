import unittest
from types import SimpleNamespace

from asyncroll.scheduling import Scheduler


def job(number, name, enqueued=0, kind="tool"):
    return SimpleNamespace(job_id=number, enqueued=enqueued,
                           payload={"kind": kind, "tool": {"name": name}})


class SchedulingTest(unittest.TestCase):
    def test_only_low_model_supply_reorders_observed_tools(self):
        slow, fast = job(0, "slow"), job(1, "fast")
        scheduler = Scheduler("asyncroll", aging_seconds=10)
        # Unknown tools stay FIFO; predictions cannot inspect future outcomes.
        self.assertIs(scheduler.select([slow, fast], 0.1, 0)[0], slow)
        scheduler.observe(slow.payload, 0.5)
        scheduler.observe(fast.payload, 0.01)
        chosen, evidence = scheduler.select([slow, fast], 0.2, 0)
        self.assertIs(chosen, fast)
        self.assertTrue(evidence["reordered"])
        self.assertTrue(evidence["estimate_known"])
        self.assertIs(scheduler.select([slow, fast], 0.2, 1)[0], slow)
        self.assertIs(Scheduler("fifo").select([slow, fast], 0.2, 0)[0], slow)

    def test_aging_overrides_shortest_job_without_preemption(self):
        scheduler = Scheduler("asyncroll", aging_seconds=1)
        slow, fresh = job(0, "slow", 0), job(1, "fast", 1.05)
        scheduler.observe(slow.payload, 10)
        scheduler.observe(fresh.payload, 0.001)
        chosen, evidence = scheduler.select([slow, fresh], 1.1, 0)
        self.assertIs(chosen, slow)
        self.assertEqual(evidence["reason"], "aging_fifo")

    def test_observation_is_per_callable_and_reset_each_run(self):
        scheduler = Scheduler("asyncroll")
        first = job(0, "same")
        second = job(1, "same")
        second.payload["tool"]["implementation"] = "different.py"
        scheduler.observe(first.payload, 0.1)
        scheduler.observe(first.payload, 0.3)
        self.assertAlmostEqual(scheduler.select([first], 0, 0)[1]["estimated_execute_seconds"], 0.2)
        self.assertFalse(scheduler.select([second], 0, 0)[1]["estimate_known"])
        self.assertFalse(Scheduler("asyncroll").estimates)


if __name__ == "__main__":
    unittest.main()
