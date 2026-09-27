"""Bounded trajectory admission, measured CPU queues and model requests."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from .model import ModelBackend, initial_messages
from .metrics import resource_metrics
from .profiling import Profiler
from .scheduling import Scheduler
from .workers import Worker
from .workload import Problem


@dataclass
class Recorder:
    start: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    sink: Callable[[dict[str, Any]], None] | None = None
    recording_seconds: float = 0.0
    profiler: Any = None
    model_outstanding: set[str] = field(default_factory=set)
    spans: dict[tuple[str, Any], Any] = field(default_factory=dict)
    nvtx_requests: dict[str, int] = field(default_factory=dict)

    def emit(self, problem_id: str, event: str, **values: Any) -> None:
        started = time.perf_counter()
        request_id = values.get("request_id")
        if event == "model_enqueued":
            self.model_outstanding.add(request_id)
        elif event in {"model_finished", "model_failed"}:
            self.model_outstanding.discard(request_id)
        if self.profiler and self.profiler.enabled:
            if event == "model_enqueued":
                self.nvtx_requests[request_id] = len(self.events)
            identity = self.nvtx_requests.get(request_id) if request_id is not None else values.get("job_id")
            if identity is not None:
                values["nvtx_identity"] = identity
            pairs = {
                "measurement_started": (None, "RUN_MEASUREMENT"),
                "measurement_finished": ("RUN_MEASUREMENT", None),
                "model_enqueued": (None, "MODEL_WAIT"),
                "model_started": ("MODEL_WAIT", "MODEL_REQUEST"),
                "model_finished": ("MODEL_REQUEST", None),
                "model_failed": ("MODEL_REQUEST", None),
                "cpu_enqueued": (None, "TOOL_QUEUE" if values.get("kind") == "tool" else "GRADE_QUEUE"),
                "cpu_started": ("TOOL_QUEUE" if values.get("kind") == "tool" else "GRADE_QUEUE", None),
            }
            if event in pairs:
                before, after = pairs[event]
                if before:
                    self.profiler.end(self.spans.pop((before, identity), None))
                if after:
                    self.spans[(after, identity)] = self.profiler.start(after, identity)
            if event in {"model_finished", "model_failed"}:
                self.nvtx_requests.pop(request_id, None)
        record = {"t": started - self.start,
                  "id": problem_id, "event": event, **values}
        self.events.append(record)
        if self.sink:
            self.sink(record)
        self.recording_seconds += time.perf_counter() - started


@dataclass
class CPUJob:
    problem_id: str
    job_id: int
    turn: int
    payload: dict[str, Any]
    future: asyncio.Future[Any]
    enqueued: float


class CPUQueue:
    def __init__(self, workers: int, policy: str, recorder: Recorder,
                 tools: list[dict[str, Any]] | None = None,
                 timeout: float = 60, startup_timeout: float = 120,
                 starvation_threshold: int = 1, aging_seconds: float = 1,
                 nvtx_enabled: bool = False):
        self.pending: list[CPUJob] = []
        self.available = asyncio.Event()
        self.closing = False
        self.policy, self.recorder, self.timeout = policy, recorder, timeout
        self.scheduler = Scheduler(policy, starvation_threshold, aging_seconds)
        self.workers = [Worker(i, tools or [], startup_timeout, nvtx_enabled=nvtx_enabled) for i in range(workers)]
        self.sequence = 0
        self.tasks: list[asyncio.Task] = []

    async def __aenter__(self) -> "CPUQueue":
        results = await asyncio.gather(*(w.start() for w in self.workers),
                                       return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            await asyncio.gather(*(w.dispose() for w in self.workers))
            raise failures[0]
        self.tasks = [asyncio.create_task(self._worker(w)) for w in self.workers]
        return self

    async def __aexit__(self, *_: object) -> None:
        self.closing = True
        self.available.set()
        try:
            await asyncio.gather(*self.tasks)
        finally:
            await asyncio.gather(*(w.dispose() for w in self.workers))

    async def submit(self, problem_id: str, kind: str, turn: int = 0,
                     tool: dict[str, Any] | None = None,
                     arguments: dict[str, Any] | None = None,
                     answer: str | None = None,
                     reference: str | None = None) -> Any:
        if kind not in {"tool", "grade"}:
            raise ValueError(f"Unknown CPU job kind {kind}")
        if kind == "grade" and reference is None:
            return {"graded": False, "correct": None}
        future = asyncio.get_running_loop().create_future()
        job = CPUJob(problem_id, self.sequence, turn,
                     dict(kind=kind, tool=tool, arguments=arguments,
                          answer=answer, reference=reference), future, time.perf_counter())
        self.sequence += 1
        self.recorder.emit(problem_id, "cpu_enqueued", **self._fields(job))
        job.enqueued = time.perf_counter()
        self.pending.append(job)
        self.available.set()
        return await future

    @staticmethod
    def _fields(job: CPUJob) -> dict[str, Any]:
        return {"kind": job.payload["kind"], "job_id": job.job_id, "turn": job.turn,
                "tool_name": (job.payload["tool"] or {}).get("name")}

    async def _worker(self, worker: Worker) -> None:
        while True:
            while not self.pending:
                if self.closing:
                    return
                self.available.clear()
                await self.available.wait()
            job, decision = self.scheduler.select(self.pending, time.perf_counter(),
                                                  len(self.recorder.model_outstanding))
            self.pending.remove(job)
            self.recorder.emit(job.problem_id, "cpu_selected", **self._fields(job), **decision)
            fields = {**self._fields(job), "worker_id": worker.worker_id}
            dispatched = None
            details: dict[str, Any] = {}
            try:
                if not worker.process.is_alive():
                    self.recorder.emit(job.problem_id, "worker_restarting", **fields)
                    await worker.start()
                self.recorder.emit(job.problem_id, "cpu_started", **fields,
                                   worker_generation=worker.generation,
                                   queue_seconds=time.perf_counter() - job.enqueued)
                dispatched = time.perf_counter()
                reply = await worker.execute({**job.payload, "nvtx_identity": job.job_id}, self.timeout)
                dispatch_seconds = time.perf_counter() - dispatched
                details = {k: v for k, v in reply.items() if k not in {"result", "error"}}
                for name in ("worker_started", "worker_finished"):
                    if name in details:
                        details[name] -= self.recorder.start
                details["dispatch_seconds"] = dispatch_seconds
                details["harness_seconds"] = max(0.0, dispatch_seconds
                    - details.get("execute_seconds", 0) - details.get("serialization_seconds", 0))
                if "execute_seconds" in details:
                    self.scheduler.observe(job.payload, details["execute_seconds"])
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                self.recorder.emit(job.problem_id, "cpu_finished", **fields, **details)
                if not job.future.done():
                    job.future.set_result(reply["result"])
            except Exception as exc:
                if "dispatch_seconds" not in details:
                    details["dispatch_seconds"] = (time.perf_counter() - dispatched
                                                    if dispatched is not None else None)
                self.recorder.emit(job.problem_id, "cpu_failed", **fields, **details,
                                   error=str(exc), timeout=isinstance(exc, TimeoutError))
                if not job.future.done():
                    # Never transfer a traceback containing this long-lived
                    # coroutine: consumers may clear its suspended frames.
                    error_type = TimeoutError if isinstance(exc, TimeoutError) else RuntimeError
                    job.future.set_exception(error_type(str(exc)))


async def _trajectory(problem: Problem, backend: ModelBackend,
                      model_slots: asyncio.Semaphore, cpu: CPUQueue,
                      recorder: Recorder, max_turns: int) -> dict[str, Any]:
    started = time.perf_counter()
    recorder.emit(problem.id, "trajectory_started", admission_seconds=started - recorder.start)
    messages = initial_messages(problem)
    tool_calls, tool_attempts = 0, 0
    try:
        for turn in range(max_turns):
            request_id = f"{problem.id}:{turn}"
            fields = dict(turn=turn, request_id=request_id)
            recorder.emit(problem.id, "model_enqueued", **fields)
            enqueued = time.perf_counter()
            async with model_slots:
                recorder.emit(problem.id, "model_started", **fields,
                              queue_seconds=time.perf_counter() - enqueued)
                try:
                    generated = await backend.generate(problem, messages, turn)
                except Exception as exc:
                    recorder.emit(problem.id, "model_failed", **fields,
                                  error=str(exc), **getattr(exc, "timings", {}))
                    raise
                action = generated.action
                known_tool = (action["type"] != "tool" or any(
                    t["name"] == action["name"] for t in problem.tools))
                recorder.emit(problem.id, "model_finished", **fields,
                              action=action["type"], action_valid=known_tool,
                              schema_valid=True,
                              tool_name_valid=known_tool,
                              request_seconds=generated.request_seconds,
                              parsing_seconds=generated.parsing_seconds, usage=generated.usage)
            messages.append({"role": "assistant", "content": json.dumps(action)})
            if action["type"] == "final":
                answer = action["answer"]
                verdict = {"graded": False, "correct": None}
                if problem.reference_answer is not None:
                    verdict = await cpu.submit(problem.id, "grade", turn,
                                               answer=answer, reference=problem.reference_answer)
                result = {"id": problem.id, "status": "completed", "answer": answer, **verdict}
                break
            tool_attempts += 1
            tool = next((item for item in problem.tools if item["name"] == action["name"]), None)
            if tool is None:
                raise ValueError(f"Unknown tool {action['name']}")
            observation = await cpu.submit(problem.id, "tool", turn, tool=tool,
                                           arguments=action["arguments"])
            tool_calls += 1
            messages.append({"role": "user", "content":
                             f"Tool {action['name']} returned: {observation}"})
        else:
            result = {"id": problem.id, "status": "max_turns"}
    except Exception as exc:
        result = {"id": problem.id, "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}"}
    result.update(tool_calls=tool_calls, tool_attempts=tool_attempts,
                  latency_seconds=time.perf_counter() - started,
                  metadata=problem.metadata)
    recorder.emit(problem.id, "trajectory_finished", status=result["status"])
    return result


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(values), "total_seconds": sum(values),
            "p50_seconds": statistics.median(values) if values else None,
            "p95_seconds": ordered[min(len(ordered)-1, int((len(ordered)-1)*0.95+0.5))]
            if values else None}


def summarize(results: list[dict], recorder: Recorder, elapsed: float) -> dict[str, Any]:
    by_id: dict[str, list[dict]] = {}
    for event in recorder.events:
        by_id.setdefault(event["id"], []).append(event)
    for result in results:
        stages = dict(model_queue=0.0, model_request=0.0, parsing=0.0,
                      cpu_queue=0.0, cpu_dispatch=0.0)
        for event in by_id.get(result["id"], []):
            if event["event"] == "model_started":
                stages["model_queue"] += event["queue_seconds"]
            if event["event"] in {"model_finished", "model_failed"}:
                stages["model_request"] += event.get("request_seconds", 0)
                stages["parsing"] += event.get("parsing_seconds", 0)
            if event["event"] == "cpu_started":
                stages["cpu_queue"] += event["queue_seconds"]
            if event["event"] in {"cpu_finished", "cpu_failed"}:
                stages["cpu_dispatch"] += event.get("dispatch_seconds") or 0
        stages["runtime_and_unattributed"] = max(0.0, result["latency_seconds"] - sum(stages.values()))
        result["latency_breakdown_seconds"] = stages
    completed = [r for r in results if r["status"] == "completed"]
    graded = [r for r in completed if r["graded"]]
    model_results = [e for e in recorder.events if e["event"] in {"model_finished", "model_failed"}]
    received = [e for e in model_results if e.get("response_received") or e["event"] == "model_finished"]
    phases = {}
    for label, event_names, key, kind in [
        ("model_queue", {"model_started"}, "queue_seconds", None),
        ("model_http", {"model_finished", "model_failed"}, "request_seconds", None),
        ("parsing", {"model_finished", "model_failed"}, "parsing_seconds", None),
        *[(f"{kind}_{phase}", names, key, kind) for kind in ("tool", "grade")
          for phase, names, key in [
              ("queue", {"cpu_started"}, "queue_seconds"),
              ("execute", {"cpu_finished", "cpu_failed"}, "execute_seconds"),
              ("process_cpu", {"cpu_finished", "cpu_failed"}, "process_cpu_seconds"),
              ("serialization", {"cpu_finished", "cpu_failed"}, "serialization_seconds"),
              ("dispatch", {"cpu_finished", "cpu_failed"}, "dispatch_seconds"),
              ("harness", {"cpu_finished", "cpu_failed"}, "harness_seconds")]]]:
        phases[label] = distribution([e[key] for e in recorder.events
            if e["event"] in event_names and e.get(key) is not None
            and (kind is None or e.get("kind") == kind)])
    return {
        "count": len(results), "completed": len(completed),
        "failed": len(results)-len(completed), "elapsed_seconds": elapsed,
        "completed_per_wall_hour": len(completed)/elapsed*3600 if elapsed else None,
        "total_tool_calls": sum(r["tool_calls"] for r in results),
        "total_tool_attempts": sum(r["tool_attempts"] for r in results),
        "trajectory_tool_call_rate": sum(r["tool_calls"] > 0 for r in results)/len(results),
        "model_attempts": len(model_results), "model_responses": len(received),
        "model_success_rate": sum(e["event"] == "model_finished" for e in model_results)/len(model_results)
            if model_results else None,
        "action_valid_rate": sum(bool(e.get("action_valid")) for e in received)/len(received)
            if received else None,
        "graded": len(graded),
        "exact_accuracy": sum(bool(r["correct"]) for r in graded)/len(graded) if graded else None,
        "trajectory_latency": distribution([r["latency_seconds"] for r in results]),
        "phase_seconds": phases,
        "event_recording_seconds": recorder.recording_seconds,
        "phase_note": "Overlapping accumulated durations; not additive wall-clock shares. "
                      "HTTP includes transport and server queue/inference. Timed-out execution is unknown.",
        "results": results,
    }


async def run(problems: list[Problem], backend: ModelBackend, policy: str,
              cpu_workers: int = 2, max_inflight_model_requests: int = 4,
              max_turns: int = 8, max_active_trajectories: int = 16,
              warmup_requests: int = 2, tool_timeout: float = 60,
              worker_startup_timeout: float = 120, telemetry: Any = None,
              event_sink: Callable[[dict], None] | None = None,
              gpu_slots: int | None = None, starvation_threshold: int = 1,
              aging_seconds: float = 1.0, nvtx_enabled: bool = False,
              dedicated_gpu: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Owns backend/telemetry lifecycle; excludes warmup and shutdown from timing."""
    requested_policy = policy
    if policy == "gpu_first":
        warnings.warn("gpu_first is a legacy alias for tool_first", FutureWarning)
        policy = "tool_first"
    if gpu_slots is not None:
        warnings.warn("gpu_slots is now max_inflight_model_requests", FutureWarning)
        max_inflight_model_requests = gpu_slots
    if policy not in {"sync", "fifo", "tool_first", "asyncroll"}:
        raise ValueError(f"Unknown policy {policy}")
    if not problems or len({p.id for p in problems}) != len(problems):
        raise ValueError("A nonempty workload with unique IDs is required")
    if min(cpu_workers, max_inflight_model_requests, max_turns,
           max_active_trajectories, tool_timeout, worker_startup_timeout) <= 0 or warmup_requests < 0:
        raise ValueError("Limits/timeouts must be positive; warmup count must be nonnegative")
    profiler = None
    recorder = Recorder(sink=event_sink)
    setup_started = time.perf_counter()
    tools = list({(t.get("implementation"), t["name"]): t for p in problems for t in p.tools}.values())
    slots = asyncio.Semaphore(max_inflight_model_requests)
    stop = asyncio.Event()
    sampler = None
    try:
        profiler = Profiler(nvtx_enabled)
        recorder.profiler = profiler
        async with CPUQueue(cpu_workers, policy, recorder, tools, tool_timeout,
                            worker_startup_timeout, starvation_threshold, aging_seconds,
                            nvtx_enabled) as cpu:
            worker_warmup_seconds = time.perf_counter() - setup_started
            warm_started = time.perf_counter()
            # Bounded batches; never create a task for every trajectory at once.
            for offset in range(0, warmup_requests, max_inflight_model_requests):
                await asyncio.gather(*(backend.generate(
                    problems[i % len(problems)], initial_messages(problems[i % len(problems)]), 0)
                    for i in range(offset, min(warmup_requests, offset+max_inflight_model_requests))))
            model_warmup_seconds = time.perf_counter() - warm_started
            telemetry_probe = await telemetry.prepare() if telemetry else None
            worker_before = await asyncio.gather(*(w.cpu_time() for w in cpu.workers))
            generations = [w.generation for w in cpu.workers]
            parent_cpu_before = time.process_time()
            recorder.start = time.perf_counter()
            recorder.emit("__run__", "measurement_started")
            if telemetry:
                sampler = asyncio.create_task(telemetry.sample_loop(recorder, stop))
            results: list[Any] = [None] * len(problems)
            pending = iter(enumerate(problems))

            async def consume() -> None:
                for index, problem in pending:
                    results[index] = await _trajectory(problem, backend, slots, cpu, recorder, max_turns)

            active_limit = 1 if policy == "sync" else max_active_trajectories
            consumers = [asyncio.create_task(consume()) for _ in range(min(active_limit, len(problems)))]
            try:
                await asyncio.gather(*consumers)
            finally:
                for task in consumers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*consumers, return_exceptions=True)
                elapsed = time.perf_counter() - recorder.start
                parent_cpu_seconds = time.process_time() - parent_cpu_before
                recorder.emit("__run__", "measurement_finished", elapsed_seconds=elapsed)
                stop.set()
                if sampler:
                    await sampler
            worker_after = await asyncio.gather(*(w.cpu_time() for w in cpu.workers), return_exceptions=True)
            cpu_cost_complete = all(w.generation == g and isinstance(end, (float, int))
                                    for w, g, end in zip(cpu.workers, generations, worker_after))
            worker_cpu_seconds = sum(end - start for start, end in zip(worker_before, worker_after)) if cpu_cost_complete else None
        summary = summarize(results, recorder, elapsed)
        resources = resource_metrics(recorder.events, elapsed, summary["completed"], dedicated_gpu,
                                     max_sample_gap=3 * getattr(telemetry, "interval", 0.2))
        cpu_total = parent_cpu_seconds + worker_cpu_seconds if cpu_cost_complete else None
        resources.update(
            cpu_cost_complete=cpu_cost_complete,
            parent_cpu_core_seconds=parent_cpu_seconds,
            worker_cpu_core_seconds=worker_cpu_seconds,
            cpu_core_seconds=cpu_total,
            cpu_core_seconds_per_completed_trajectory=cpu_total / summary["completed"] if cpu_cost_complete and summary["completed"] else None,
            cpu_cost_scope="Client process and persistent tool workers; excludes vLLM/server, "
                           "other processes and startup. Worker IPC boundary skew is included; "
                           "cost is unknown if a worker was killed/replaced.")
        summary["resource_metrics"] = resources
        summary["scheduler"] = {"starvation_threshold": starvation_threshold, "aging_seconds": aging_seconds,
                                "estimator": "Per-tool observed execution EWMA alpha=0.5; unknown=median history; reset each run"}
        summary["nvtx_enabled"] = nvtx_enabled
        samples = [e for e in recorder.events if e["event"] == "resource_sample"]
        summary.update(policy=policy, requested_policy=requested_policy,
                       cpu_workers=cpu_workers, max_inflight_model_requests=max_inflight_model_requests,
                       max_active_trajectories=max_active_trajectories,
                       effective_max_active_trajectories=active_limit,
                       warmup={"worker_seconds": worker_warmup_seconds,
                               "model_seconds": model_warmup_seconds, "requests": warmup_requests,
                               "note": "First-turn workload requests; may warm prefix/grammar caches. "
                                       "Does not certify all CUDA graph/kernel shapes are warm."},
                       telemetry_probe=telemetry_probe,
                       telemetry={"enabled": telemetry is not None, "sample_count": len(samples),
                                  "error_samples": sum(any(e.get("errors", {}).values()) for e in samples),
                                  "missing_metric_samples": sum(bool(e.get("missing_metrics")) for e in samples),
                                  "invalid_metric_samples": sum(bool(e.get("invalid_metrics")) for e in samples)})
        return summary, recorder.events
    finally:
        stop.set()
        if profiler:
            profiler.close()
        if telemetry:
            await telemetry.close()
        close = getattr(backend, "aclose", None)
        if close:
            await close()
