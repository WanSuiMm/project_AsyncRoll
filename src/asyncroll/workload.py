"""Load a small, explicit agent workload or adapt ToolMATH records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    reference_answer: str | None = None
    script: list[dict[str, Any]] | None = None


def load_jsonl(path: Path, limit: int | None = None) -> list[Problem]:
    problems: list[Problem] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"{path}:{line_number}: missing prompt")
            problems.append(Problem(
                id=str(row.get("id", line_number)),
                prompt=str(row["prompt"]),
                tools=list(row.get("tools", [])),
                reference_answer=row.get("reference_answer"),
                script=row.get("script"),
            ))
            if limit is not None and len(problems) >= limit:
                break
    if not problems:
        raise ValueError(f"No problems in {path}")
    if len({p.id for p in problems}) != len(problems):
        raise ValueError("Problem IDs must be unique")
    return problems


def convert_toolmath(source: Path, functions_dir: Path, destination: Path,
                     limit: int | None = None) -> int:
    """Map ToolMATH metadata to this runtime's JSONL format.

    A ToolMATH row specifies one tool and its source problem. It does not supply
    a verified multi-turn trajectory or an unambiguous final-answer label.
    """
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a ToolMATH JSON array")
    selected = rows[:limit] if limit is not None else rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(selected):
            implementation = (functions_dir / row["function"]).resolve()
            if not implementation.is_file():
                raise FileNotFoundError(implementation)
            record = {
                "id": f"toolmath-{index}",
                "prompt": row["source_problem"],
                "tools": [{
                    "name": row["name"],
                    "description": row["description"],
                    "inputs": row["inputs"],
                    "implementation": str(implementation),
                }],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(selected)
