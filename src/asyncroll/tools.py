"""CPU tool execution. Only execute trusted, locally inspected tool files."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any


def execute_tool(tool: dict[str, Any], arguments: dict[str, Any]) -> str:
    name = tool["name"]
    if name == "sleep_ms" and "implementation" not in tool:
        time.sleep(float(arguments["milliseconds"]) / 1000)
        return json.dumps({"slept_ms": arguments["milliseconds"]})
    if name == "add" and "implementation" not in tool:
        return json.dumps({"result": arguments["a"] + arguments["b"]})
    path = Path(tool["implementation"]).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"Invalid tool implementation: {path}")
    spec = importlib.util.spec_from_file_location("asyncroll_external_tool", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"{path} does not define callable {name}")
    return json.dumps(function(**arguments), ensure_ascii=False, default=str)


def grade(answer: str, reference: str | None) -> dict[str, Any]:
    if reference is None:
        return {"graded": False, "correct": None}
    # Exact comparison is deliberately narrow. No mathematical equivalence claim.
    normalized = lambda value: " ".join(value.strip().split())
    return {"graded": True, "correct": normalized(answer) == normalized(reference)}
