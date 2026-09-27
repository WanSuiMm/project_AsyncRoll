# AsyncRoll context for reviewers

## Current answer and boundaries

The live single-GPU stack qualified, but the frozen FIFO opportunity screen
found no useful CPU scheduling leverage: CPU-blocked time and CPU-related idle
candidate time were both zero, maximum CPU queue depth was one, and tool queue
p95 was 1.707 ms. The pre-registered stop rule fired before comparison. There
is no FIFO versus AsyncRoll result, demonstrated speedup, math-equivalence
grading, or RL loop. See [RESULTS.md](RESULTS.md).

| Formal field | Status |
| --- | --- |
| Qualification | `qualified` (8/8 completed and used a tool) |
| Opportunity screen | `stopped_no_cpu_opportunity` (31/32 completed) |
| Formal comparison | Not run |
| Speedup claim | Unsupported |

| Variant | Definition | Source |
| --- | --- | --- |
| `sync` | One active trajectory | `runtime.run` |
| `fifo` | Bounded active trajectories; FIFO CPU queue | `CPUQueue.submit` |
| `asyncroll` | Low client model supply triggers shortest observed tool first; aging restores FIFO | `scheduling.Scheduler.select` |
| `tool_first` | Same admission; tools before terminal grading | `CPUQueue.submit` |

`gpu_first` is a deprecated alias of `tool_first`, not a pressure-aware
scheduler. Without labels, there are no grade jobs and these asynchronous
legacy policies have the same priority order. The formal comparison is FIFO
versus `asyncroll`. Client inflight requests are not vLLM
internal batch size. HTTP duration includes network, server queue and inference.

## Implementation map

| Concern | Exact entry |
| --- | --- |
| Persistent asynchronous HTTP, schema request, parsing | `model.VLLMBackend.generate`, `action_schema` |
| Bounded admission, warmup exclusion, phase summaries | `runtime.run`, `_trajectory`, `summarize` |
| Non-preemptive CPU policy and identified jobs | `runtime.CPUQueue` |
| Pressure signal, past-cost estimate, aging, dispatch decisions | `scheduling.Scheduler.select`, `observe` |
| Sampled CPU-related starvation and coverage | `metrics.resource_metrics` |
| Whole client/worker process CPU accounting | `runtime.run`, `workers.Worker.cpu_time` |
| Async-safe NVTX and real worker execution ranges | `profiling.Profiler`, `runtime.Recorder.emit`, `workers._serve` |
| Single-GPU deployment, gates, paired execution, serving audit | `experiment.py`, `experiments/single_5090.json` |
| Offline paired report and one three-panel figure | `report.aggregate`, `write_report` |
| Persistent spawn workers, deadlines and replacement | `workers.Worker`, `_serve` |
| Per-process callable/module cache | `tools.load_tool` |
| Portable seeded workload conversion | `workload.convert_toolmath`, `load_jsonl` |
| Model-filtered vLLM metrics, optional local NVML | `telemetry.Telemetry`, `parse_vllm_metrics` |
| Request/queue/worker/trajectory timeline | `timeline.write_timeline`, `build_timeline` |
| New run directory, event stream and terminal receipt | `cli.main` |

## Evidence routing

- `RESULTS.md`: canonical human-readable gates, verdict and claim boundary.
- `results/single_5090_20260927.json`: sanitized machine-readable aggregate.

- `tests/test_runtime.py`: policies, independent limits, absent no-op grading,
  warmup/cache behavior and actual process death/replacement after timeout.
- `tests/test_model.py`: asynchronous overlapping requests, reused client,
  schema payload, parse errors and total request deadline via mock transport.
- `tests/test_telemetry.py`: metric parsing/rates/errors and timeline export
  with synthetic samples. This is not a server/NVML integration result.
- `tests/test_workload.py`: seeded portable ToolMATH conversion and metadata.
- `tests/test_scheduling.py`: low-supply dispatch, past-only estimates, aging.
- `tests/test_metrics.py`: client/server interval intersection, missing data/gaps.
- `tests/test_profiling.py`: explicit overlapping ranges, failed tool spans and
  CPU-time control messages. NVTX itself is mocked; real Nsight remains untested.
- `tests/test_experiment.py`: configuration/planning, evidence gates and process
  lifecycle checks using mocks. A successful test does not qualify live vLLM.
- `tests/test_report.py`: independent paired units and missing-cost semantics.
- Local `runs/` contains smoke receipts; it is excluded from Git. Read the
  concise summary before raw events. No smoke throughput is a GPU result.

## Review route

Read [PROJECT.md](PROJECT.md), [MEASUREMENT.md](MEASUREMENT.md), then
`runtime.py`, `workers.py`, `model.py` and their tests. Review telemetry and
timeline code for attribution limits. `completed_per_wall_hour` replaces
the unconditional legacy GPU-hour field. `resource_metrics.completed_per_gpu_hour`
is now emitted only with an explicit dedicated-one-GPU declaration; the launcher
checks hardware/process exclusivity. It measures allocation during the timed
window, excluding warmup. Phase totals overlap across trajectories and must not
be presented as wall-clock percentages. CPU cost explicitly excludes serving CPU.
