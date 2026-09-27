# Incremental review handoff

- Review base: `00b6d31c496413efb4dc73bb4422cfc1325e857c`
- Code/evidence head: `e16d3313a30305969ded386ad4dc6e6b1635832a`
- This file is a later documentation-only commit; the evidence head stays fixed.
- Validation date: 2026-09-27.

## What changed

1. HTTP requests use one persistent async client with JSON-schema output,
   separate request/parsing timing and a total deadline.
2. Persistent CPU workers preload cached callables before measurement. Tool
   deadlines terminate the process; later jobs trigger a preloaded replacement.
3. Trajectory admission, model inflight calls and CPU workers have independent
   controls. Warmup and teardown are excluded from throughput timing.
4. Unlabelled tasks create no grading work. `tool_first` is the canonical name
   for the existing tool-over-grade priority; `gpu_first` remains an alias.
5. vLLM metric parsing, optional local NVML, phase summaries, job identities,
   durable event logs and standalone/Chrome resource timelines are implemented.
6. Seeded ToolMATH conversion preserves metadata and uses portable tool paths.

The primary rate is now `completed_per_wall_hour`. Per-trajectory latency
components and accumulated phase totals have different meanings; read their
definitions before making any attribution.

## What remains unchanged

There is no empirical scheduling advantage, live Qwen/ToolMATH qualification,
math-equivalence grader, RL training, or recoverable GPU-bubble result. No
CUDA/vLLM internals, CPU affinity or NUMA tuning was added. Unlabelled ToolMATH
does not differentiate FIFO and tool-first's CPU priority order.

## New verification evidence

| Check | Observed result | Boundary |
| --- | --- | --- |
| `python -m unittest discover -s tests -v` | 24 tests passed | Local behavior; HTTP/metrics mocked |
| CLI scripted smoke, 3 active / 2 model slots / 2 CPU workers | 3/3 completed, 2 tool calls, 3/3 exact labels | Synthetic functional check, no GPU |
| Run outputs | Complete receipt, summary, events, trace, HTML | Local artifacts excluded from Git |
| Browser rendering | Separate trajectory and CPU lanes render; absent telemetry is visible | No real resource samples |
| Package build | `asyncroll-0.2.0` wheel built | Client package only |
| Repository delivery checks | Markdown targets valid; staged whitespace/size/identifier scans passed | No model weights or private receipts |

Tests exercised actual worker process death and replacement after timeout.
They also cover cached imports, concurrency limits, absent no-op grading,
unknown tool validity, HTTP error versus timeout, telemetry gaps/deadlines,
counter resets and HTML escaping. Live vLLM structured-output compatibility,
Linux worker behavior and real NVML collection remain unverified.

## Minimal reading order and questions

1. [MEASUREMENT.md](MEASUREMENT.md): can any field still be mistaken for GPU
   execution or additive wall-clock share?
2. [runtime.py](src/asyncroll/runtime.py), [workers.py](src/asyncroll/workers.py):
   do admission, warmup, replacement and failure denominators preserve fairness?
3. [model.py](src/asyncroll/model.py), [telemetry.py](src/asyncroll/telemetry.py):
   which live protocol/version assumptions must Gate 0 qualify?
4. [PROJECT.md](PROJECT.md): is there actual CPU-related idle opportunity to
   justify any policy beyond the current coarse baseline?

Reproduce local verification using [README.md](README.md). Review summaries and
tests first; do not request raw per-event logs before the measurement semantics
are clear. Do not compare scripted throughput to a GPU serving benchmark.
