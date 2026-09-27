"""Load a small, explicit agent workload or adapt ToolMATH records."""

from __future__ import annotations

import json
import hashlib
import random
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
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: Path, limit: int | None = None,
               tool_root: Path | None = None) -> list[Problem]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    problems: list[Problem] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"{path}:{line_number}: missing prompt")
            tools = list(row.get("tools", []))
            if len({t["name"] for t in tools}) != len(tools):
                raise ValueError("Tool names must be unique within a problem")
            for tool in tools:
                if "implementation" in tool:
                    implementation = Path(tool["implementation"])
                    if not implementation.is_absolute():
                        root = (tool_root or path.parent).resolve()
                        implementation = (root / implementation).resolve()
                        if not implementation.is_relative_to(root):
                            raise ValueError("Relative tool path escapes tool root")
                    tool["implementation"] = str(implementation)
            problems.append(Problem(
                id=str(row.get("id", line_number)),
                prompt=str(row["prompt"]),
                tools=tools,
                reference_answer=(str(row["reference_answer"])
                                  if row.get("reference_answer") is not None else None),
                script=row.get("script"),
                metadata=dict(row.get("metadata", {})),
            ))
            if limit is not None and len(problems) >= limit:
                break
    if not problems:
        raise ValueError(f"No problems in {path}")
    if len({p.id for p in problems}) != len(problems):
        raise ValueError("Problem IDs must be unique")
    return problems


def convert_toolmath(source: Path, functions_dir: Path, destination: Path,
                     limit: int | None = None, seed: int = 0) -> int:
    """Map ToolMATH metadata to this runtime's JSONL format.

    A ToolMATH row specifies one tool and its source problem. It does not supply
    a verified multi-turn trajectory or an unambiguous final-answer label.
    """
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    rows = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a ToolMATH JSON array")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = (random.Random(seed).sample(range(len(rows)), min(limit, len(rows)))
                if limit is not None else list(range(len(rows))))
    functions_dir = functions_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for index in selected:
            row = rows[index]
            implementation = (functions_dir / row["function"]).resolve()
            if not implementation.is_relative_to(functions_dir):
                raise ValueError("Tool implementation escapes functions directory")
            if not implementation.is_file():
                raise FileNotFoundError(implementation)
            record = {
                "id": f"toolmath-{index}",
                "prompt": row["source_problem"],
                "tools": [{
                    "name": row["name"],
                    "description": row["description"],
                    "inputs": row["inputs"],
                    "implementation": implementation.relative_to(functions_dir).as_posix(),
                }],
                "metadata": {**{k: row[k] for k in ("difficulty", "category", "level", "type")
                                 if k in row}, "source_index": index, "sample_seed": seed,
                             "source_sha256": source_sha256},
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(selected)
