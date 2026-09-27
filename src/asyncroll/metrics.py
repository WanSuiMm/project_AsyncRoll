"""Conservative sampled attribution; no counterfactual GPU recovery claim."""

from __future__ import annotations

import math
from collections import Counter


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs)
                            * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def resource_metrics(events: list[dict], elapsed: float, completed: int,
                     dedicated_gpu: bool = False, max_sample_gap: float = 0.6,
                     low_gpu_percent: float = 10.0) -> dict:
    # Exact parent-state spans; intersect them with sampled server evidence.
    active, model, tools = set(), set(), set()
    blocked = []
    last = 0.0
    for event in events:
        end = min(elapsed, max(last, event["t"]))
        if active and not model and tools and end > last:
            blocked.append((last, end))
        name = event["event"]
        if name == "trajectory_started":
            active.add(event["id"])
        elif name == "trajectory_finished":
            active.discard(event["id"])
        elif name == "model_enqueued":
            model.add(event["request_id"])
        elif name in {"model_finished", "model_failed"}:
            model.discard(event["request_id"])
        elif name == "cpu_enqueued" and event.get("kind") == "tool":
            tools.add(event["job_id"])
        elif name in {"cpu_finished", "cpu_failed"}:
            tools.discard(event["job_id"])
        last = end
    samples = [e for e in events if e["event"] == "resource_sample"]
    coverage = gpu_coverage = starvation = candidate = 0.0
    for first, second in zip(samples, samples[1:]):
        # Keep away from both scrape windows; don't extrapolate at run edges.
        left = first["t"]
        right = second.get("collection_started_offset_seconds", second["t"])
        width = right - left
        if width <= 0 or second["t"] - first["t"] > max_sample_gap:
            continue
        values = [s for s in (first, second)]
        fields = ("vllm_running_requests", "vllm_waiting_requests")
        if not all(_finite(v.get(k)) for v in values for k in fields):
            continue
        coverage += width
        gpu = [s.get("nvml_gpu_utilization_percent") for s in (first, second)]
        gpu_valid = all(_finite(v) and v <= 100 for v in gpu)
        if gpu_valid:
            gpu_coverage += width
        if any(v[k] != 0 for v in values for k in fields):
            continue
        overlap = sum(max(0.0, min(right, b) - max(left, a)) for a, b in blocked)
        starvation += overlap
        if gpu_valid and max(gpu) <= low_gpu_percent:
            candidate += overlap
    decisions = [e for e in events if e["event"] == "cpu_selected"]
    actual_by_job = {e["job_id"]: e.get("execute_seconds") for e in events
                     if e["event"] in {"cpu_finished", "cpu_failed"}}
    prediction_pairs = [(math.log(max(0.001, e["estimated_execute_seconds"])),
                         math.log(max(0.001, actual_by_job[e["job_id"]])))
                        for e in decisions if _finite(e.get("estimated_execute_seconds"))
                        and _finite(actual_by_job.get(e["job_id"]))]
    eligible = [e for e in decisions if e.get("eligible_for_reorder")]
    reason_counts = Counter(e.get("reason", "unknown") for e in decisions)
    return {
        "completed_per_gpu_hour": completed / elapsed * 3600 if dedicated_gpu and elapsed else None,
        "dedicated_single_gpu_declared": dedicated_gpu,
        "client_cpu_blocked_seconds": sum(b - a for a, b in blocked),
        "model_starvation_seconds": starvation if coverage else None,
        "cpu_related_idle_candidate_seconds": candidate if gpu_coverage else None,
        "server_observation_coverage_seconds": coverage,
        "joint_observation_coverage_seconds": gpu_coverage,
        "max_sample_gap_seconds": max_sample_gap, "low_gpu_percent": low_gpu_percent,
        "scheduler_decisions": len(decisions),
        "cpu_max_queue_depth": max((e["queue_depth"] for e in decisions), default=0),
        "scheduler_reorders": sum(e["reordered"] for e in decisions),
        "scheduler_eligible_decisions": len(eligible),
        "scheduler_activation_rate": (sum(e["reordered"] for e in eligible) / len(eligible)
                                      if eligible else None),
        "scheduler_reason_counts": dict(sorted(reason_counts.items())),
        "scheduler_distinct_feature_keys": len({e.get("feature_key") for e in decisions
                                                  if e.get("feature_key") not in {None, "grade"}}),
        "known_cost_decisions": sum(e["estimate_known"] for e in decisions),
        "prediction_log_pearson": _pearson(prediction_pairs),
        "prediction_pairs": len(prediction_pairs),
        "interpretation": "Sampled empty vLLM queues overlapping pending CPU continuations and no "
                          "client model work. Endpoints do not prove continuous server idleness. "
                          "Low NVML activity yields a candidate, not recoverable GPU time. "
                          "GPU-hour counts one dedicated allocated GPU during measurement only.",
    }
