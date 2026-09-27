"""Non-preemptive pressure-aware CPU selection, using past observations only."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


def tool_key(payload: dict) -> tuple[str, str]:
    tool = payload.get("tool") or {}
    return (tool.get("implementation", "builtin"), tool.get("name", payload["kind"]))


@dataclass
class Scheduler:
    policy: str
    starvation_threshold: int = 1
    aging_seconds: float = 1.0
    estimates: dict[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.starvation_threshold < 1 or not math.isfinite(self.aging_seconds) or self.aging_seconds <= 0:
            raise ValueError("Scheduler threshold and aging deadline must be positive")

    def observe(self, payload: dict, seconds: float) -> None:
        if payload["kind"] == "tool" and math.isfinite(seconds) and seconds >= 0:
            key = tool_key(payload)
            self.estimates[key] = 0.5 * seconds + 0.5 * self.estimates.get(key, seconds)

    def select(self, jobs: list[Any], now: float, model_outstanding: int) -> tuple[Any, dict]:
        """Re-evaluate priority at dispatch; a running callable is never preempted.

        Unknown tools use the median of past observed tool durations (zero before
        any observation). No arguments, answers, future trace or evaluation labels
        enter this estimator. New runs start with an empty estimator.
        """
        oldest = min(jobs, key=lambda j: j.job_id)
        reason = "fifo"
        selected = oldest
        fallback = statistics.median(self.estimates.values()) if self.estimates else 0.0
        estimate = lambda j: self.estimates.get(tool_key(j.payload), fallback)
        if self.policy == "tool_first":
            selected = min(jobs, key=lambda j: (j.payload["kind"] != "tool", j.job_id))
            reason = "tool_before_grade"
        elif self.policy == "asyncroll":
            aged = [j for j in jobs if now - j.enqueued >= self.aging_seconds]
            if aged:
                selected = min(aged, key=lambda j: j.job_id)
                reason = "aging_fifo"
            elif model_outstanding < self.starvation_threshold:
                selected = min(jobs, key=lambda j: (j.payload["kind"] != "tool", estimate(j), j.job_id))
                reason = "low_model_supply_short_observed_tool"
        return selected, {
            "reason": reason, "queue_depth": len(jobs),
            "model_outstanding": model_outstanding,
            "estimated_execute_seconds": estimate(selected),
            "estimate_known": tool_key(selected.payload) in self.estimates,
            "oldest_job_id": oldest.job_id, "reordered": selected is not oldest,
        }
