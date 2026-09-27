"""Warm persistent processes with kill-and-replace deadlines for trusted tools.

This is process isolation, not a security sandbox. Tools must not create child
processes: terminating a worker does not promise to kill its descendants.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .profiling import Profiler
from .tools import call_tool, grade, load_tool


def _serve(connection: Any, tools: list[dict[str, Any]],
           nvtx_enabled: bool = False) -> None:
    profiler: Profiler | None = None
    try:
        profiler = Profiler(enabled=nvtx_enabled)
        for tool in tools:
            if "implementation" in tool:
                load_tool(tool)
        call_tool({"name": "add"}, {"a": 0, "b": 0})
        grade("0", "0")
        connection.send({"ready": True})
        while True:
            job = connection.recv()
            if job is None:
                return
            if job.get("_worker_command") == "cpu_time":
                connection.send({"worker_cpu_time_seconds": time.process_time()})
                continue
            started, cpu_started = time.perf_counter(), time.process_time()
            label = "TOOL_EXEC" if job["kind"] == "tool" else "GRADE_EXEC"
            try:
                range_handle = profiler.start(label, job.get("nvtx_identity"))
                try:
                    value = (call_tool(job["tool"], job["arguments"])
                             if job["kind"] == "tool"
                             else grade(job["answer"], job["reference"]))
                finally:
                    profiler.end(range_handle)
                executed = time.perf_counter()
                cpu_seconds = time.process_time() - cpu_started
                result = (json.dumps(value, ensure_ascii=False, default=str)
                          if job["kind"] == "tool" else value)
                reply = {"result": result, "execute_seconds": executed - started,
                         "process_cpu_seconds": cpu_seconds,
                         "serialization_seconds": time.perf_counter() - executed}
            except BaseException as exc:
                reply = {"error": f"{type(exc).__name__}: {exc}",
                         "execute_seconds": time.perf_counter() - started,
                         "process_cpu_seconds": time.process_time() - cpu_started}
            connection.send({**reply, "worker_started": started,
                             "worker_finished": time.perf_counter()})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        connection.send({"error": f"Worker startup: {type(exc).__name__}: {exc}"})
    finally:
        try:
            if profiler is not None:
                profiler.close()
        finally:
            connection.close()


class Worker:
    def __init__(self, worker_id: int, tools: list[dict[str, Any]],
                 startup_timeout: float = 120, nvtx_enabled: bool = False):
        self.worker_id, self.tools = worker_id, tools
        self.startup_timeout = startup_timeout
        self.nvtx_enabled = nvtx_enabled
        self.process: Any = None
        self.connection: Any = None
        self.generation = 0
        # One waiting IPC thread per worker avoids the default asyncio pool's
        # 32-thread cap becoming an unintended CPU concurrency limit.
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cpu-ipc-{worker_id}")

    def _start(self) -> None:
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_serve, args=(child, self.tools, self.nvtx_enabled), daemon=True)
        try:
            self.process.start()
            child.close()
            if not self.connection.poll(self.startup_timeout):
                raise TimeoutError("CPU worker warmup timed out")
            reply = self.connection.recv()
            if not reply.get("ready"):
                raise RuntimeError(reply.get("error", "CPU worker failed to warm up"))
            self.generation += 1
        except BaseException:
            child.close()
            self._stop()
            raise

    def _stop(self) -> None:
        if self.process is not None and self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if self.process.is_alive():
                raise RuntimeError("Unable to terminate CPU worker")
        if self.connection is not None:
            self.connection.close()

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self.io, self._start)

    async def close(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self.io, self._stop)

    async def dispose(self) -> None:
        await self.close()
        self.io.shutdown(wait=True)

    def _exchange(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.connection.send(payload)
        if not self.connection.poll(timeout):
            raise TimeoutError(f"CPU job exceeded {timeout:g}s; worker terminated")
        return self.connection.recv()

    async def execute(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            return await asyncio.get_running_loop().run_in_executor(self.io, self._exchange, payload, timeout)
        except (TimeoutError, EOFError, BrokenPipeError, OSError):
            await self.close()
            raise

    async def cpu_time(self) -> float:
        """Read worker process CPU seconds through a control-only IPC message.

        Take a baseline after worker warmup and subtract it from a later sample
        to measure CPU consumed by the run without charging startup warmup.
        """
        if self.process is None or self.connection is None:
            raise RuntimeError("CPU worker has not been started")
        if not self.process.is_alive():
            raise RuntimeError("CPU worker is not alive")
        try:
            reply = await asyncio.get_running_loop().run_in_executor(
                self.io, self._exchange,
                {"_worker_command": "cpu_time"}, self.startup_timeout)
        except (TimeoutError, EOFError, BrokenPipeError, OSError):
            await self.close()
            raise
        value = reply.get("worker_cpu_time_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(reply.get("error", "CPU worker returned an invalid CPU time"))
        return float(value)
