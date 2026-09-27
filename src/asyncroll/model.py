"""Scripted smoke backend and OpenAI-compatible vLLM backend."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from typing import Any, Protocol

from .workload import Problem


class ModelBackend(Protocol):
    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> dict[str, Any]: ...


def parse_action(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    action = json.loads(cleaned)
    if not isinstance(action, dict) or action.get("type") not in {"tool", "final"}:
        raise ValueError("Model must return a JSON object with type tool or final")
    if action["type"] == "tool" and (
        not isinstance(action.get("name"), str)
        or not isinstance(action.get("arguments"), dict)
    ):
        raise ValueError("Tool action requires name and arguments object")
    if action["type"] == "final" and "answer" not in action:
        raise ValueError("Final action requires answer")
    return action


class ScriptedBackend:
    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> dict[str, Any]:
        if problem.script is None or turn >= len(problem.script):
            raise ValueError(f"No scripted action for {problem.id} turn {turn}")
        return parse_action(json.dumps(problem.script[turn]))


class VLLMBackend:
    def __init__(self, endpoint: str, model: str, timeout: float = 120):
        self.endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": 0, "max_tokens": 512,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        return parse_action(content)

    async def generate(self, problem: Problem, messages: list[dict[str, str]],
                       turn: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._request, messages)


def initial_messages(problem: Problem) -> list[dict[str, str]]:
    tools = [{key: tool[key] for key in ("name", "description", "inputs")
              if key in tool} for tool in problem.tools]
    return [
        {"role": "system", "content": (
            "Solve the math problem. You may call only the listed tools. "
            "Reply with exactly one JSON object, either "
            "{\"type\":\"tool\",\"name\":\"...\",\"arguments\":{...}} "
            "or {\"type\":\"final\",\"answer\":\"...\"}. "
            "No markdown or prose outside JSON. "
            f"Tools: {json.dumps(tools, ensure_ascii=False)}")},
        {"role": "user", "content": problem.prompt},
    ]
