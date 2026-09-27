"""Resource-bounded local evaluator for an ephemeral benchmark host.

This reduces accidental damage and runaway processes. It is not a hardened
security sandbox and must not be used on a host containing credentials or data.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _limits(memory_mb: int, cpu_seconds: int,
            run_uid: int | None, run_gid: int | None) -> None:
    import resource

    os.setsid()
    memory = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    if run_gid is not None and run_uid is not None:
        os.setgroups([])
        os.setgid(run_gid)
        os.setuid(run_uid)


def _run(command: list[str], cwd: Path, stdin: str, timeout: float,
         memory_mb: int, run_uid: int | None,
         run_gid: int | None) -> tuple[int | None, str, str, bool]:
    process = subprocess.Popen(
        command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=False,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
             "PYTHONDONTWRITEBYTECODE": "1"},
        preexec_fn=(lambda: _limits(memory_mb, max(1, int(timeout)),
                                    run_uid, run_gid))
        if os.name == "posix" else None)
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        return None, stdout, stderr, True


def _normalized(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def evaluate_python(code: str, reference_tests: dict[str, list[Any]],
                    execution_metadata: dict[str, Any] | None = None,
                    timeout_seconds: float = 6.0, memory_mb: int = 768) -> dict[str, Any]:
    """Execute all supplied stdin tests and return bounded first-failure feedback."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("submission must be non-empty Python source")
    if timeout_seconds <= 0 or memory_mb < 64:
        raise ValueError("invalid evaluator limits")
    tests = list(reference_tests.get("public") or []) + list(reference_tests.get("private") or [])
    if not tests:
        raise ValueError("reference test set is empty")
    # Functional-call cases require the official LiveCodeBench checker. Refuse
    # rather than silently grading them with stdin semantics.
    metadata = execution_metadata or {}
    if metadata.get("func_name"):
        return {"passed": False, "tests_run": 0, "tests_total": len(tests),
                "feedback": "unsupported functional-call case", "unsupported": True}

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="asyncroll-lcb-") as directory:
        root = Path(directory)
        run_uid = run_gid = None
        if os.name == "posix" and os.geteuid() == 0:
            import pwd

            account = pwd.getpwnam("nobody")
            run_uid, run_gid = account.pw_uid, account.pw_gid
            os.chown(root, run_uid, run_gid)
            root.chmod(0o700)
        program = root / "solution.py"
        program.write_text(code, encoding="utf-8")
        if run_uid is not None and run_gid is not None:
            os.chown(program, run_uid, run_gid)
            program.chmod(0o600)
        for index, case in enumerate(tests):
            if not isinstance(case, dict) or "input" not in case or "output" not in case:
                return {"passed": False, "tests_run": index, "tests_total": len(tests),
                        "feedback": "invalid reference test format", "harness_error": True}
            remaining = timeout_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                return {"passed": False, "tests_run": index, "tests_total": len(tests),
                        "feedback": "evaluation timed out", "timed_out": True}
            code_value, stdout, stderr, timed_out = _run(
                [sys.executable, "-I", str(program)], root, str(case["input"]),
                remaining, memory_mb, run_uid, run_gid)
            if timed_out:
                return {"passed": False, "tests_run": index + 1,
                        "tests_total": len(tests), "feedback": "execution timed out",
                        "timed_out": True}
            expected = str(case["output"])
            if code_value != 0 or _normalized(stdout) != _normalized(expected):
                feedback = {
                    "case": index,
                    "exit_code": code_value,
                    "input": str(case["input"])[:1000],
                    "expected": expected[:1000],
                    "actual": stdout[:1000],
                    "stderr": stderr[:1000],
                }
                return {"passed": False, "tests_run": index + 1,
                        "tests_total": len(tests),
                        "feedback": json.dumps(feedback, ensure_ascii=False)}
    return {"passed": True, "tests_run": len(tests), "tests_total": len(tests),
            "feedback": "all tests passed"}
