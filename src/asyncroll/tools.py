"""CPU tool execution. Only execute trusted, locally inspected tool files."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


_TOOL_CACHE: dict[tuple[str, str], Any] = {}


def _coerce_argument(value: Any, type_name: str) -> Any:
    if type_name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("expected int")
        return value
    if type_name == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected float")
        return float(value)
    if type_name == "str":
        if not isinstance(value, str):
            raise TypeError("expected str")
        return value
    if type_name == "dict[int, int]":
        if not isinstance(value, dict):
            raise TypeError("expected dict[int, int]")
        try:
            return {int(key): _coerce_argument(item, "int")
                    for key, item in value.items()}
        except (TypeError, ValueError) as exc:
            raise TypeError("expected decimal integer keys and integer values") from exc
    if type_name in {"List[int]", "List[float]"}:
        if not isinstance(value, list):
            raise TypeError(f"expected {type_name}")
        item_type = "int" if type_name == "List[int]" else "float"
        return [_coerce_argument(item, item_type) for item in value]
    raise TypeError(f"unsupported tool input type {type_name}")


def load_tool(tool: dict[str, Any]) -> Any:
    """Cache trusted callables per worker; module state persists within a run."""
    name = tool["name"]
    path = Path(tool["implementation"]).resolve()
    key = (str(path), name)
    if key in _TOOL_CACHE:
        return _TOOL_CACHE[key]
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"Invalid tool implementation: {path}")
    module_name = "asyncroll_tool_" + hashlib.sha256(str(path).encode()).hexdigest()
    if module_name in sys.modules:
        function = getattr(sys.modules[module_name], name, None)
        if not callable(function):
            raise ValueError(f"{path} does not define callable {name}")
        _TOOL_CACHE[key] = function
        return function
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"{path} does not define callable {name}")
    _TOOL_CACHE[key] = function
    return function


def call_tool(tool: dict[str, Any], arguments: dict[str, Any]) -> Any:
    name = tool["name"]
    if name == "sleep_ms" and "implementation" not in tool:
        time.sleep(float(arguments["milliseconds"]) / 1000)
        return {"slept_ms": arguments["milliseconds"]}
    if name == "add" and "implementation" not in tool:
        return {"result": arguments["a"] + arguments["b"]}
    if name == "lcb_evaluate" and "implementation" not in tool:
        from .code_executor import evaluate_python

        if set(arguments) != {"code"}:
            raise TypeError("lcb_evaluate requires exactly one code argument")
        bound = tool.get("bound_arguments") or {}
        return evaluate_python(arguments["code"],
                               bound.get("reference_tests") or {},
                               bound.get("execution_metadata") or {},
                               float(bound.get("timeout_seconds", 6.0)),
                               int(bound.get("memory_mb", 768)))
    inputs = tool.get("inputs") or {}
    if set(arguments) != set(inputs):
        raise TypeError("tool arguments do not match the declared inputs")
    coerced = {name: _coerce_argument(arguments[name], type_name)
               for name, type_name in inputs.items()}
    return load_tool(tool)(**coerced)


def execute_tool(tool: dict[str, Any], arguments: dict[str, Any]) -> str:
    return json.dumps(call_tool(tool, arguments), ensure_ascii=False, default=str)


def grade(answer: str, reference: str | None) -> dict[str, Any]:
    if reference is None:
        return {"graded": False, "correct": None}
    # Exact comparison is deliberately narrow. No mathematical equivalence claim.
    normalized = lambda value: " ".join(value.strip().split())
    return {"graded": True, "correct": normalized(answer) == normalized(reference)}
