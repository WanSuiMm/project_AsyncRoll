import unittest

from asyncroll.metrics import resource_metrics


def event(t, name, **kw):
    return {"t": t, "id": "a", "event": name, **kw}


def sample(t, running=0, gpu=0):
    return event(t, "resource_sample", vllm_running_requests=running,
                 vllm_waiting_requests=0, nvml_gpu_utilization_percent=gpu,
                 collection_started_offset_seconds=t - 0.01)


class ResourceMetricsTest(unittest.TestCase):
    def test_intersects_exact_client_state_and_sampled_server_state(self):
        events = [event(0, "trajectory_started"),
                  event(0.1, "cpu_enqueued", job_id=0, kind="tool"), sample(0.2),
                  event(0.3, "model_enqueued", request_id="other"),
                  event(0.4, "model_finished", request_id="other"), sample(0.5),
                  event(0.6, "cpu_finished", job_id=0),
                  event(0.8, "trajectory_finished")]
        result = resource_metrics(events, 0.8, 1, True)
        self.assertAlmostEqual(result["model_starvation_seconds"], 0.19)
        self.assertAlmostEqual(result["cpu_related_idle_candidate_seconds"], 0.19)
        self.assertAlmostEqual(result["client_cpu_blocked_seconds"], 0.4)
        self.assertEqual(result["completed_per_gpu_hour"], 4500)

    def test_missing_telemetry_is_unknown_and_gaps_are_not_filled(self):
        events = [event(0, "trajectory_started"), event(0, "cpu_enqueued", job_id=0, kind="tool"),
                  sample(0.1, running=None), sample(0.3), sample(2),
                  event(2.1, "cpu_finished", job_id=0)]
        result = resource_metrics(events, 2.1, 1)
        self.assertIsNone(result["model_starvation_seconds"])
        self.assertIsNone(result["cpu_related_idle_candidate_seconds"])
        self.assertIsNone(result["completed_per_gpu_hour"])

    def test_empty_server_is_not_low_gpu_or_pending_cpu_proof(self):
        events = [event(0, "trajectory_started"), event(0, "cpu_enqueued", job_id=0, kind="tool"),
                  sample(0.1, gpu=50), sample(0.3, gpu=60),
                  event(0.4, "cpu_finished", job_id=0)]
        result = resource_metrics(events, 0.4, 1)
        self.assertGreater(result["model_starvation_seconds"], 0)
        self.assertEqual(result["cpu_related_idle_candidate_seconds"], 0)
        events = [sample(0.1), sample(0.3)]
        self.assertEqual(resource_metrics(events, 0.4, 1)["model_starvation_seconds"], 0)

    def test_scheduler_activation_and_prediction_diagnostics(self):
        events = [
            event(0.1, "cpu_selected", job_id=0, queue_depth=2, reordered=True,
                  eligible_for_reorder=True, reason="low_model_supply_predicted_unlock",
                  feature_key="a", estimate_known=True, estimated_execute_seconds=0.1,
                  predictor_source="same_problem_first_evaluation"),
            event(0.2, "cpu_finished", job_id=0, execute_seconds=0.2),
            event(0.3, "cpu_selected", job_id=1, queue_depth=2, reordered=False,
                  eligible_for_reorder=True, reason="low_model_supply_predicted_unlock",
                  feature_key="b", estimate_known=True, estimated_execute_seconds=1.0,
                  predictor_source="same_problem_first_evaluation"),
            event(0.4, "cpu_finished", job_id=1, execute_seconds=1.2),
            event(0.5, "cpu_selected", job_id=2, queue_depth=1, reordered=False,
                  eligible_for_reorder=False, reason="fifo", feature_key="c",
                  estimate_known=True, estimated_execute_seconds=2.0,
                  predictor_source="same_problem_first_evaluation"),
            event(0.6, "cpu_finished", job_id=2, execute_seconds=2.1),
        ]
        result = resource_metrics(events, 1.0, 3)
        self.assertEqual(result["scheduler_eligible_decisions"], 2)
        self.assertEqual(result["scheduler_reorders"], 1)
        self.assertEqual(result["scheduler_activation_rate"], 0.5)
        self.assertEqual(result["scheduler_distinct_feature_keys"], 3)
        self.assertGreater(result["prediction_log_pearson"], 0.9)
        self.assertGreater(result["repair_prediction_log_pearson"], 0.9)
        self.assertEqual(result["repair_prediction_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
