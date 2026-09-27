"""Non-preemptive CPU scheduling from observable per-job cost features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


def _payload_features(payload: dict[str, Any]) -> tuple[float, ...]:
    """Return pre-execution features without reading expected test outputs."""
    cached = payload.get("_scheduling_features")
    if isinstance(cached, tuple) and len(cached) == 5:
        return cached
    tool = payload.get("tool") or {}
    bound = tool.get("bound_arguments") or {}
    tests = bound.get("reference_tests") or {}
    cases = [case for split in ("public", "private")
             for case in (tests.get(split) or []) if isinstance(case, dict)]
    input_bytes = sum(len(json.dumps(case.get("input", ""), ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8"))
                      for case in cases)
    arguments = payload.get("arguments") or {}
    code = arguments.get("code", "")
    code_bytes = len(str(code).encode("utf-8"))
    repair = int(payload.get("turn", 0) > 0)
    features = (1.0, math.log1p(len(cases)), math.log1p(input_bytes),
                math.log1p(code_bytes), float(repair))
    payload["_scheduling_features"] = features
    return features


def _feature_key(features: tuple[float, ...]) -> str:
    # Coarse, non-sensitive diagnostics; no hidden test contents are logged.
    return ":".join(str(round(value, 2)) for value in features[1:])


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Small partial-pivot Gaussian solve for the five-feature ridge model."""
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        if abs(scale) < 1e-12:
            return [0.0] * size
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [left - factor * right
                              for left, right in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]


@dataclass
class OnlineRidge:
    regularization: float = 2.0
    observations: int = 0
    gram: list[list[float]] = field(default_factory=list)
    target: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.gram:
            self.gram = [[self.regularization if i == j else 0.0 for j in range(5)]
                         for i in range(5)]
            # A 100 ms prior avoids an unrealistic one-second cold estimate.
            self.target = [self.regularization * math.log(0.1), 0.0, 0.0, 0.0, 0.0]

    def predict(self, features: tuple[float, ...]) -> float:
        weights = _solve(self.gram, self.target)
        log_seconds = sum(weight * value for weight, value in zip(weights, features))
        return math.exp(max(math.log(0.001), min(math.log(60.0), log_seconds)))

    def observe(self, features: tuple[float, ...], seconds: float) -> None:
        target = math.log(max(0.001, seconds))
        for row in range(5):
            self.target[row] += features[row] * target
            for column in range(5):
                self.gram[row][column] += features[row] * features[column]
        self.observations += 1


@dataclass
class Scheduler:
    policy: str
    starvation_threshold: int = 1
    aging_seconds: float = 30.0
    aging_weight: float = 0.02
    predictor: OnlineRidge = field(default_factory=OnlineRidge)
    first_evaluation_seconds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.starvation_threshold < 1 or not math.isfinite(self.aging_seconds)
                or self.aging_seconds <= 0 or not math.isfinite(self.aging_weight)
                or self.aging_weight < 0):
            raise ValueError("Scheduler threshold, aging deadline and weight must be valid")

    def observe(self, payload: dict, seconds: float) -> None:
        if payload["kind"] == "tool" and math.isfinite(seconds) and seconds >= 0:
            self.predictor.observe(_payload_features(payload), seconds)
            problem_id = payload.get("problem_id")
            if problem_id is not None and int(payload.get("turn", 0)) == 0:
                self.first_evaluation_seconds[str(problem_id)] = seconds

    def select(self, jobs: list[Any], now: float, model_outstanding: int) -> tuple[Any, dict]:
        """Choose a queued job using only information available before execution."""
        oldest = min(jobs, key=lambda job: job.job_id)
        selected, reason = oldest, "fifo"

        def prediction(job: Any) -> tuple[float, str]:
            if job.payload["kind"] != "tool":
                return 0.0, "grade"
            problem_id = str(job.payload.get("problem_id", ""))
            if int(job.payload.get("turn", 0)) > 0 and problem_id in self.first_evaluation_seconds:
                return self.first_evaluation_seconds[problem_id], "same_problem_first_evaluation"
            return self.predictor.predict(_payload_features(job.payload)), "ridge_fallback"

        eligible = (self.policy == "asyncroll" and len(jobs) > 1
                    and model_outstanding < self.starvation_threshold)
        if self.policy == "tool_first":
            selected = min(jobs, key=lambda job: (job.payload["kind"] != "tool", job.job_id))
            reason = "tool_before_grade"
        elif self.policy == "asyncroll":
            expired = [job for job in jobs if now - job.enqueued >= self.aging_seconds]
            if expired:
                selected = min(expired, key=lambda job: job.job_id)
                reason = "hard_starvation_fifo"
            elif eligible:
                selected = min(jobs, key=lambda job: (
                    job.payload["kind"] != "tool",
                    prediction(job)[0] - self.aging_weight * max(0.0, now - job.enqueued),
                    job.job_id))
                reason = "low_model_supply_predicted_unlock"

        features = _payload_features(selected.payload) if selected.payload["kind"] == "tool" else None
        predicted_seconds, predictor_source = prediction(selected)
        return selected, {
            "reason": reason, "queue_depth": len(jobs),
            "model_outstanding": model_outstanding,
            "estimated_execute_seconds": predicted_seconds,
            "estimate_known": self.predictor.observations > 0,
            "predictor_observations": self.predictor.observations,
            "predictor_source": predictor_source,
            "feature_key": _feature_key(features) if features else "grade",
            "eligible_for_reorder": eligible,
            "oldest_job_id": oldest.job_id, "reordered": selected is not oldest,
        }
