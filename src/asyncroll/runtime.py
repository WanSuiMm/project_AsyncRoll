"""Trajectory runtime and CPU queue policies."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .model import ModelBackend, initial_messages
from .tools import execute_tool, grade
from .workload import Problem


@dataclass
class Recorder:
    start: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, problem_id: str, event: str, **values: Any) -> None:
        self.events.append({"t": time.perf_counter() - self.start,
                            "id": problem_id, "event": event, **values})


@dataclass
class CPUJob:
    problem_id: str
    kind: str
    tool: dict[str, Any] | None
    arguments: dict[str, Any] | None
    answer: str | None
    reference: str | None
    future: asyncio.Future[Any]
    enqueued: float


class CPUQueue:
    def __init__(self, workers: int, policy: str, recorder: Recorder,
                 executor: ProcessPoolExecutor):
        self.queue: asyncio.PriorityQueue[tuple[int, int, CPUJob | None]] = asyncio.PriorityQueue()
        self.workers = workers
        self.policy = policy
        self.recorder = recorder
        self.executor = executor
        self.sequence = 0
        self.tasks: list[asyncio.Task[None]] = []

    async def __aenter__(self) -> "CPUQueue":
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
        return self

    async def __aexit__(self, *_: object) -> None:
        for _ in self.tasks:
            await self.queue.put((2, self.sequence, None))
            self.sequence += 1
        await asyncio.gather(*self.tasks)

    async def submit(self, problem_id: str, kind: str,
                     tool: dict[str, Any] | None = None,
                     arguments: dict[str, Any] | None = None,
                     answer: str | None = None,
                     reference: str | None = None) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        job = CPUJob(problem_id, kind, tool, arguments, answer, reference,
                     future, time.perf_counter())
        priority = 0 if self.policy == "gpu_first" and kind == "tool" else 1
        await self.queue.put((priority, self.sequence, job))
        self.sequence += 1
        self.recorder.emit(problem_id, "cpu_enqueued", kind=kind)
        return await future

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            _, _, job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            self.recorder.emit(job.problem_id, "cpu_started", kind=job.kind,
                               queue_seconds=time.perf_counter() - job.enqueued)
            try:
                if job.kind == "tool":
                    result = await loop.run_in_executor(
                        self.executor, execute_tool, job.tool, job.arguments)
                else:
                    result = await loop.run_in_executor(
                        self.executor, grade, job.answer, job.reference)
                self.recorder.emit(job.problem_id, "cpu_finished", kind=job.kind)
                job.future.set_result(result)
            except Exception as exc:
                self.recorder.emit(job.problem_id, "cpu_failed", kind=job.kind,
                                   error=str(exc))
                job.future.set_exception(exc)
            finally:
                self.queue.task_done()


async def _trajectory(problem: Problem, backend: ModelBackend,
                      gpu: asyncio.Semaphore, cpu: CPUQueue,
                      recorder: Recorder, max_turns: int) -> dict[str, Any]:
    started = time.perf_counter()
    messages = initial_messages(problem)
    tool_count = 0
    recorder.emit(problem.id, "trajectory_started")
    try:
        for turn in range(max_turns):
            recorder.emit(problem.id, "gpu_enqueued", turn=turn)
            async with gpu:
                recorder.emit(problem.id, "gpu_started", turn=turn)
                action = await backend.generate(problem, messages, turn)
                recorder.emit(problem.id, "gpu_finished", turn=turn,
                              action=action["type"])
            messages.append({"role": "assistant", "content": json.dumps(action)})
            if action["type"] == "final":
                answer = str(action["answer"])
                verdict = await cpu.submit(problem.id, "grade", answer=answer,
                                           reference=problem.reference_answer)
                result = {"id": problem.id, "status": "completed", "answer": answer,
                          "tool_calls": tool_count, **verdict}
                break
            tool = next((item for item in problem.tools
                         if item["name"] == action["name"]), None)
            if tool is None:
                raise ValueError(f"Unknown tool {action['name']}")
            observation = await cpu.submit(problem.id, "tool", tool=tool,
                                           arguments=action["arguments"])
            tool_count += 1
            messages.append({"role": "user", "content":
                             f"Tool {action['name']} returned: {observation}"})
        else:
            result = {"id": problem.id, "status": "max_turns", "tool_calls": tool_count}
    except Exception as exc:
        result = {"id": problem.id, "status": "failed", "error": str(exc),
                  "tool_calls": tool_count}
    result["latency_seconds"] = time.perf_counter() - started
    recorder.emit(problem.id, "trajectory_finished", status=result["status"])
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))]


async def run(problems: list[Problem], backend: ModelBackend, policy: str,
              cpu_workers: int = 2, gpu_slots: int = 4,
              max_turns: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if policy not in {"sync", "fifo", "gpu_first"}:
        raise ValueError(f"Unknown policy {policy}")
    if cpu_workers < 1 or gpu_slots < 1 or max_turns < 1:
        raise ValueError("Workers, GPU slots, and max turns must be positive")
    recorder = Recorder()
    gpu = asyncio.Semaphore(gpu_slots)
    with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        async with CPUQueue(cpu_workers, policy, recorder, executor) as cpu:
            if policy == "sync":
                results = []
                for problem in problems:
                    results.append(await _trajectory(problem, backend, gpu, cpu,
                                                     recorder, max_turns))
            else:
                results = await asyncio.gather(*[
                    _trajectory(problem, backend, gpu, cpu, recorder, max_turns)
                    for problem in problems])
    elapsed = time.perf_counter() - recorder.start
    completed = [item for item in results if item["status"] == "completed"]
    latencies = [item["latency_seconds"] for item in completed]
    graded = [item for item in completed if item["graded"]]
    queues = [item["queue_seconds"] for item in recorder.events
              if item["event"] == "cpu_started"]
    summary = {
        "policy": policy, "count": len(problems), "completed": len(completed),
        "failed": len(problems) - len(completed), "elapsed_seconds": elapsed,
        "completed_per_gpu_hour": len(completed) / elapsed * 3600 if elapsed else None,
        "total_tool_calls": sum(item["tool_calls"] for item in results),
        "graded": len(graded),
        "exact_accuracy": (sum(bool(item["correct"]) for item in graded) / len(graded)
                           if graded else None),
        "trajectory_p50_seconds": statistics.median(latencies) if latencies else None,
        "trajectory_p95_seconds": _percentile(latencies, 0.95),
        "cpu_queue_p50_seconds": statistics.median(queues) if queues else None,
        "cpu_queue_p95_seconds": _percentile(queues, 0.95),
        "results": results,
    }
    return summary, recorder.events
