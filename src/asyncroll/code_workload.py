"""Convert LiveCodeBench release_v6 records into AsyncRoll JSONL problems.

This module only parses and packages benchmark data. It never evaluates a
submission or runs code from an input record.
"""

from __future__ import annotations

import hashlib
import json
import base64
import io
import pickle
import zlib
from pathlib import Path
from typing import Any


_REPAIR_INSTRUCTION = (
    "Your previous complete Python 3 program failed the supplied test feedback. "
    "Use that feedback to produce one corrected complete program. Return source "
    "code only, without Markdown fences or explanation."
)


def _read_records(source: Path, source_bytes: bytes) -> list[dict[str, Any]]:
    """Read a JSON array/single object or JSONL file without evaluating input."""
    try:
        text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: input must be UTF-8 JSON or JSONL") from exc
    if not text.strip():
        raise ValueError(f"{source}: input contains no records")

    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSONL record: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: each record must be an object")
            records.append(record)
        if not records:
            raise ValueError(f"{source}: input contains no records")
        return records

    if isinstance(document, list):
        if any(not isinstance(record, dict) for record in document):
            raise ValueError(f"{source}: every JSON array record must be an object")
        return document
    if isinstance(document, dict):
        return [document]
    raise ValueError(f"{source}: expected a JSON object, JSON array, or JSONL records")


def _record_id(record: dict[str, Any], source: Path, index: int) -> str:
    for key in ("question_id", "task_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    raise ValueError(
        f"{source}: record {index}: missing required non-empty question_id, task_id, or id"
    )


def _test_cases(record: dict[str, Any], key: str, source: Path,
                index: int) -> list[Any]:
    value = record.get(key)
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            if key != "private_test_cases":
                raise ValueError(
                    f"{source}: record {index}: {key} must contain a JSON array"
                ) from exc
            try:
                class StringsOnlyUnpickler(pickle.Unpickler):
                    def find_class(self, module: str, name: str) -> Any:
                        raise pickle.UnpicklingError("global objects are forbidden")

                packed = zlib.decompress(base64.b64decode(value, validate=True))
                decoded = StringsOnlyUnpickler(io.BytesIO(packed)).load()
                value = json.loads(decoded) if isinstance(decoded, str) else decoded
            except Exception as packed_exc:
                raise ValueError(
                    f"{source}: record {index}: {key} must contain JSON or the "
                    "pinned LiveCodeBench compressed test format"
                ) from packed_exc
    if not isinstance(value, list):
        raise ValueError(f"{source}: record {index}: {key} must be a JSON array")
    return value


def _execution_metadata(record: dict[str, Any], source: Path,
                        index: int) -> dict[str, Any]:
    value = record.get("metadata") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: record {index}: metadata must be JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source}: record {index}: metadata must be an object")
    return value


def _prompt(record: dict[str, Any], source: Path, index: int) -> str:
    statement = record.get("question_content", record.get("question"))
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError(
            f"{source}: record {index}: missing required question_content string"
        )
    title = record.get("question_title")
    if title is not None and not isinstance(title, str):
        raise ValueError(f"{source}: record {index}: question_title must be a string")
    starter = record.get("starter_code")
    if starter is not None and not isinstance(starter, str):
        raise ValueError(f"{source}: record {index}: starter_code must be a string or null")

    parts = [
        "Solve the following programming problem in Python 3.",
        "Return a complete program as source code only, without Markdown fences or explanation.",
    ]
    if title and title.strip():
        parts.extend((f"Problem: {title.strip()}", ""))
    parts.extend(("Problem statement:", statement.strip()))
    if starter and starter.strip():
        parts.extend(("", "Starter code:", starter.rstrip()))
    return "\n".join(parts)


def _sampling_key(seed: int, record_id: str) -> bytes:
    payload = json.dumps([seed, record_id], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _prepare_record(record: dict[str, Any], source: Path, index: int,
                    source_sha256: str, seed: int,
                    stdin_only: bool) -> tuple[str, dict[str, Any]] | None:
    record_id = _record_id(record, source, index)
    execution_metadata = _execution_metadata(record, source, index)
    if stdin_only and execution_metadata.get("func_name"):
        return None
    public_tests = _test_cases(record, "public_test_cases", source, index)
    private_tests = _test_cases(record, "private_test_cases", source, index)
    if not public_tests and not private_tests:
        public_tests = _test_cases(record, "test_cases", source, index)
    if not public_tests and not private_tests:
        raise ValueError(f"{source}: record {index}: missing non-empty reference test cases")
    metadata: dict[str, Any] = {
        "dataset": "livecodebench", "release": "release_v6",
        "source_record_id": record_id, "source_index": index,
        "source_sha256": source_sha256, "sample_seed": seed,
        "reference_tests": {"public": public_tests, "private": private_tests},
        "execution_metadata": execution_metadata,
        "prompt_metadata": {
            "protocol": "one_repair", "language": "python3",
            "initial_attempts": 1, "repair_attempts": 1, "max_generations": 2,
            "repair_instruction": _REPAIR_INSTRUCTION,
            "reference_tests_in_prompt": False,
        },
    }
    for key in ("platform", "contest", "difficulty"):
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return record_id, {
        "id": record_id, "prompt": _prompt(record, source, index),
        "tools": [{"name": "lcb_evaluate",
                   "description": "Submit a complete Python 3 program for hidden evaluation.",
                   "inputs": {"code": "str"}}],
        "reference_answer": None, "metadata": metadata,
    }


def convert_livecodebench(source: Path, destination: Path,
                          limit: int | None = None, seed: int = 0,
                          stdin_only: bool = False) -> int:
    """Write a deterministic AsyncRoll JSONL sample from release_v6-style data.

    Accepted source formats are a JSON array, one JSON record, or JSONL. Each
    record needs an ID, a problem statement, and at least one public/private
    reference test case. Test cases are kept in ``metadata.reference_tests``;
    only the statement, optional title, and optional starter code enter prompt.

    Sampling ranks stable record IDs by a seed-keyed SHA-256 digest. This makes
    the selected set independent of input row order and Python's RNG version.
    If ``limit`` is omitted or covers the complete source, source order is kept.
    """
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")

    if source.suffix.lower() == ".jsonl":
        digest = hashlib.sha256()
        candidates: list[tuple[bytes, str, int]] = []
        source_ids: set[str] = set()
        with source.open("rb") as stream:
            for index, line in enumerate(stream):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{source}:{index + 1}: invalid JSONL record: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{source}:{index + 1}: each record must be an object")
                record_id = _record_id(record, source, index)
                if record_id in source_ids:
                    raise ValueError(f"{source}: duplicate problem ID {record_id!r}")
                source_ids.add(record_id)
                execution_metadata = _execution_metadata(record, source, index)
                if stdin_only and execution_metadata.get("func_name"):
                    continue
                candidates.append((_sampling_key(seed, record_id), record_id, index))
        if not candidates:
            raise ValueError(f"{source}: input contains no problems")
        chosen = (sorted(candidates)[:limit] if limit is not None and limit < len(candidates)
                  else candidates)
        selected_indices = {item[2] for item in chosen}
        source_sha256 = digest.hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with source.open(encoding="utf-8-sig") as input_stream, \
                destination.open("w", encoding="utf-8", newline="\n") as output_stream:
            for index, line in enumerate(input_stream):
                if index not in selected_indices:
                    continue
                prepared = _prepare_record(json.loads(line), source, index,
                                           source_sha256, seed, stdin_only)
                if prepared is not None:
                    output_stream.write(json.dumps(prepared[1], ensure_ascii=False) + "\n")
                    written += 1
        return written

    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    records = _read_records(source, source_bytes)
    source_ids: set[str] = set()
    prepared: list[tuple[str, dict[str, Any]]] = []

    for index, record in enumerate(records):
        record_id = _record_id(record, source, index)
        if record_id in source_ids:
            raise ValueError(f"{source}: duplicate problem ID {record_id!r}")
        source_ids.add(record_id)

        item = _prepare_record(record, source, index, source_sha256, seed, stdin_only)
        if item is not None:
            prepared.append(item)

    if not prepared:
        raise ValueError(f"{source}: input contains no problems")
    if limit is not None and limit < len(prepared):
        prepared = sorted(
            prepared,
            key=lambda item: (_sampling_key(seed, item[0]), item[0]),
        )[:limit]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for _, output_record in prepared:
            stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    return len(prepared)
