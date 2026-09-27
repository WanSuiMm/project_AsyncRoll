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
    if not isinstance(action, dict) or action.get("type") not in {"tool", "final"}:
        raise ValueError("Model must return a JSON object with type tool or final")
    if action["type"] == "tool" and (
        not isinstance(action.get("name"), str)
        or not isinstance(action.get("arguments"), dict)
    ):
        raise ValueError("Tool action requires name and arguments object")
    if action["type"] == "final" and not isinstance(action.get("answer"), str):
        raise ValueError("Final action requires a string answer")
    return action


def action_schema(problem: Problem) -> dict[str, Any]:
    variants = [{"type": "object", "properties": {
        "type": {"const": "final"}, "answer": {"type": "string"}},
        "required": ["type", "answer"], "additionalProperties": False}]
    if problem.tools:
        variants.append({"type": "object", "properties": {
            "type": {"const": "tool"},
            "name": {"enum": sorted({tool["name"] for tool in problem.tools})},
            "arguments": {"type": "object"}},
            "required": ["type", "name", "arguments"], "additionalProperties": False})
    return {"anyOf": variants}


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
                 transport: httpx.AsyncBaseTransport | None = None):
        if timeout <= 0 or max_connections < 1 or max_tokens < 1:
            raise ValueError("HTTP timeout, connections and max tokens must be positive")
        self.endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model, self.timeout = model, timeout
        self.max_tokens, self.structured_output = max_tokens, structured_output
        self.client = httpx.AsyncClient(
            timeout=timeout, transport=transport,
            limits=httpx.Limits(max_connections=max_connections,
                               max_keepalive_connections=max_connections))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> Generation:
        payload = {"model": self.model, "messages": messages,
                   "temperature": 0, "max_tokens": self.max_tokens}
        if self.structured_output:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "agent_action", "schema": action_schema(problem)}}
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
            action = parse_action(content)
        except Exception as exc:
            raise GenerationError(str(exc), request_seconds=received - started,
                                  parsing_seconds=time.perf_counter() - received,
                                  response_received=True, action_valid=False) from exc
        return Generation(action, received - started, time.perf_counter() - received,
                          result.get("usage") or {})


def initial_messages(problem: Problem) -> list[dict[str, str]]:
    tools = [{key: tool[key] for key in ("name", "description", "inputs")
              if key in tool} for tool in problem.tools]
    return [
        {"role": "system", "content": (
            "Solve the math problem. You may call only the listed tools. "
            'Reply with exactly one JSON object, either '
            '{"type":"tool","name":"...","arguments":{...}} '
            'or {"type":"final","answer":"..."}. '
            "No markdown or prose outside JSON. "
            f"Tools: {json.dumps(tools, ensure_ascii=False)}")},
        {"role": "user", "content": problem.prompt},
    ]
