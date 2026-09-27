"""Persistent asynchronous HTTP; request latency is not GPU execution time."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .workload import Problem


@dataclass
class Generation:
    action: dict[str, Any]
    request_seconds: float = 0.0
    parsing_seconds: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


class GenerationError(ValueError):
    def __init__(self, message: str, **timings: Any):
        super().__init__(message)
        self.timings = timings


class ModelBackend(Protocol):
    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> Generation: ...


def parse_action(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    action = json.loads(cleaned)
    if not isinstance(action, dict) or action.get("type") not in {"tool", "final", "submit"}:
        raise ValueError("Model must return a JSON object with type tool, final, or submit")
    if action["type"] == "tool" and (
        not isinstance(action.get("name"), str)
        or not isinstance(action.get("arguments"), dict)
    ):
        raise ValueError("Tool action requires name and arguments object")
    if action["type"] == "final" and not isinstance(action.get("answer"), str):
        raise ValueError("Final action requires a string answer")
    if action["type"] == "submit" and not isinstance(action.get("code"), str):
        raise ValueError("Submit action requires Python source in a code string")
    return action


def parse_code_submission(content: str) -> dict[str, str]:
    """Accept ordinary source output and remove one optional Markdown fence."""
    code = content.strip()
    fenced = re.fullmatch(r"```(?:python|py)?\s*\n?(.*?)\n?```", code,
                          flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        code = fenced.group(1).strip()
    if not code:
        raise ValueError("Model returned an empty code submission")
    return {"type": "submit", "code": code}


def _input_schema(type_name: str) -> dict[str, Any]:
    schemas = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "dict[int, int]": {
            "type": "object", "additionalProperties": {"type": "integer"}},
        "List[int]": {"type": "array", "items": {"type": "integer"}},
        "List[float]": {"type": "array", "items": {"type": "number"}},
    }
    try:
        return schemas[type_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported ToolMATH input type: {type_name}") from exc


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    inputs = tool.get("inputs") or {}
    arguments = {
        "type": "object",
        "properties": {name: _input_schema(type_name)
                       for name, type_name in inputs.items()},
        "required": list(inputs),
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {
        "type": {"const": "tool"}, "name": {"const": tool["name"]},
        "arguments": arguments},
        "required": ["type", "name", "arguments"], "additionalProperties": False}


def action_schema(problem: Problem, turn: int = 0) -> dict[str, Any]:
    """Require one typed tool call first, then one bounded final response."""
    if problem.metadata.get("prompt_metadata", {}).get("protocol") == "one_repair":
        return {"type": "object", "properties": {
            "type": {"const": "submit"},
            "code": {"type": "string", "minLength": 1, "maxLength": 20000}},
            "required": ["type", "code"], "additionalProperties": False}
    if problem.tools and turn == 0:
        variants = [_tool_schema(tool) for tool in problem.tools]
        return variants[0] if len(variants) == 1 else {"anyOf": variants}
    return {"type": "object", "properties": {
        "type": {"const": "final"},
        "answer": {"type": "string", "maxLength": 256}},
        "required": ["type", "answer"], "additionalProperties": False}


class ScriptedBackend:
    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> Generation:
        if problem.script is None or turn >= len(problem.script):
            raise ValueError(f"No scripted action for {problem.id} turn {turn}")
        started = time.perf_counter()
        try:
            action = parse_action(json.dumps(problem.script[turn]))
        except ValueError as exc:
            raise GenerationError(str(exc), request_seconds=0.0,
                                  parsing_seconds=time.perf_counter() - started,
                                  response_received=True, action_valid=False) from exc
        return Generation(action, parsing_seconds=time.perf_counter() - started)


class VLLMBackend:
    def __init__(self, endpoint: str, model: str, timeout: float = 120,
                 max_connections: int = 4, max_tokens: int = 512,
                 structured_output: bool = True,
                 transport: httpx.AsyncBaseTransport | None = None,
                 seed: int = 0):
        if timeout <= 0 or max_connections < 1 or max_tokens < 1:
            raise ValueError("HTTP timeout, connections and max tokens must be positive")
        self.endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model, self.timeout = model, timeout
        self.max_tokens, self.structured_output = max_tokens, structured_output
        self.seed = seed
        self.client = httpx.AsyncClient(
            timeout=timeout, transport=transport,
            limits=httpx.Limits(max_connections=max_connections,
                               max_keepalive_connections=max_connections))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> Generation:
        payload = {"model": self.model, "messages": messages,
                   "temperature": 0, "max_tokens": self.max_tokens, "seed": self.seed}
        if self.structured_output:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "agent_action", "strict": True,
                "schema": action_schema(problem, turn)}}
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.client.post(self.endpoint, json=payload), self.timeout)
        except Exception as exc:
            raise GenerationError(str(exc) or type(exc).__name__,
                                  request_seconds=time.perf_counter() - started,
                                  parsing_seconds=0.0, response_received=False) from exc
        received = time.perf_counter()
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GenerationError(str(exc), request_seconds=received - started,
                                  parsing_seconds=0.0, response_received=True,
                                  http_status=response.status_code, action_valid=False) from exc
        try:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            if (not self.structured_output
                    and problem.metadata.get("prompt_metadata", {}).get("protocol") == "one_repair"):
                action = parse_code_submission(content)
            else:
                action = parse_action(content)
        except Exception as exc:
            raise GenerationError(str(exc), request_seconds=received - started,
                                  parsing_seconds=time.perf_counter() - received,
                                  response_received=True, action_valid=False) from exc
        return Generation(action, received - started, time.perf_counter() - received,
                          result.get("usage") or {})


def initial_messages(problem: Problem) -> list[dict[str, str]]:
    if problem.metadata.get("prompt_metadata", {}).get("protocol") == "one_repair":
        return [
            {"role": "system", "content": (
                "Write a complete Python 3 solution. Return source code only, "
                "without Markdown fences or explanation.")},
            {"role": "user", "content": problem.prompt},
        ]
    tools = [{key: tool[key] for key in ("name", "description", "inputs")
              if key in tool} for tool in problem.tools]
    return [
        {"role": "system", "content": (
            "Solve the math problem using the listed tool protocol. "
            "On the first response, call exactly one listed tool with concrete "
            "JSON values that satisfy its input types. After the tool result, "
            "return the final answer. For dict[int, int], encode integer keys "
            "as decimal JSON strings, for example {\"2\":3}. "
            'Reply with exactly one JSON object, either '
            '{"type":"tool","name":"...","arguments":{...}} '
            'or {"type":"final","answer":"..."}. '
            "No markdown or prose outside JSON. "
            f"Tools: {json.dumps(tools, ensure_ascii=False)}")},
        {"role": "user", "content": problem.prompt},
    ]
