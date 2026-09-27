# AsyncRoll context for reviewers

## Current answer and boundaries

**No empirical answer exists yet** to whether CPU scheduling improves model
feeding on real ToolMATH trajectories. This inference-only repository has
local behavior tests and measurement infrastructure. It has no live vLLM
trace, math-equivalence grading, RL loop, or demonstrated systems speedup.
Model download receipts and machine operations are intentionally outside Git.

| Variant | Definition | Source |
| --- | --- | --- |
| `sync` | One active trajectory | `runtime.run` |
| `fifo` | Bounded active trajectories; FIFO CPU queue | `CPUQueue.submit` |
| `tool_first` | Same admission; tools before terminal grading | `CPUQueue.submit` |

`gpu_first` is a deprecated alias of `tool_first`, not a pressure-aware
scheduler. Without labels, there are no grade jobs and these asynchronous
policies have the same priority order. Client inflight requests are not vLLM
internal batch size. HTTP duration includes network, server queue and inference.

## Implementation map

| Concern | Exact entry |
| --- | --- |
| Persistent asynchronous HTTP, schema request, parsing | `model.VLLMBackend.generate`, `action_schema` |
| Bounded admission, warmup exclusion, phase summaries | `runtime.run`, `_trajectory`, `summarize` |
| Non-preemptive CPU policy and identified jobs | `runtime.CPUQueue` |
| Persistent spawn workers, deadlines and replacement | `workers.Worker`, `_serve` |
| Per-process callable/module cache | `tools.load_tool` |
| Portable seeded workload conversion | `workload.convert_toolmath`, `load_jsonl` |
| Model-filtered vLLM metrics, optional local NVML | `telemetry.Telemetry`, `parse_vllm_metrics` |
| Request/queue/worker/trajectory timeline | `timeline.write_timeline`, `build_timeline` |
| New run directory, event stream and terminal receipt | `cli.main` |

## Evidence routing

- `tests/test_runtime.py`: policies, independent limits, absent no-op grading,
  warmup/cache behavior and actual process death/replacement after timeout.
- `tests/test_model.py`: asynchronous overlapping requests, reused client,
  schema payload, parse errors and total request deadline via mock transport.
- `tests/test_telemetry.py`: metric parsing/rates/errors and timeline export
  with synthetic samples. This is not a server/NVML integration result.
- `tests/test_workload.py`: seeded portable ToolMATH conversion and metadata.
- Local `runs/` contains smoke receipts; it is excluded from Git. Read the
  concise summary before raw events. No smoke throughput is a GPU result.

## Review route

Read [PROJECT.md](PROJECT.md), [MEASUREMENT.md](MEASUREMENT.md), then
`runtime.py`, `workers.py`, `model.py` and their tests. Review telemetry and
timeline code for attribution limits. `completed_per_wall_hour` replaces
the misleading `completed_per_gpu_hour` field. Phase totals overlap across
trajectories and must not be presented as wall-clock percentages.
