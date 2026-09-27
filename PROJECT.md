# AsyncRoll

## Research question

When agents alternate between GPU model execution and CPU tools, how much
model idle time coincides with pending CPU continuations, and can a CPU queue
policy improve completed trajectories per elapsed hour at equal resources?

## Current status (2026-09-27)

The runtime implements pooled asynchronous HTTP, warm cached tool workers,
independent concurrency limits, vLLM/NVML telemetry, optional NVTX and portable
experiment launchers. The completed 32-active LiveCodeBench v2 comparison found
-0.37% mean throughput change and zero AsyncRoll reorders. V3 keeps that result
and fixes a single bounded-concurrency operating point at 8 active trajectories,
4 model requests and 2 CPU workers. It added per-job online ridge prediction,
soft aging and an activation gate. V3 stopped at that gate: only 2 decisions
were eligible and 1 reordered, so the formal comparison did not run. No positive
gain is claimed. This project is inference only, without RL. See [RESULTS.md](RESULTS.md).

V6 keeps the 8/4/2 operating point and changes only the mechanism exposed by
v5: it triggers below the four-request serving capacity and uses each problem's
first evaluation as the repair-time prior. Its simplified activation gate
requires 20 eligible decisions, 10 reorders and positive repair prediction
correlation. V6 completed all formal arms: paired throughput changes were
+0.92%, +1.03% and +2.37%, with +1.44% mean and a descriptive interval spanning
zero. The mechanism activated; the result does not support a 20% gain.

An optional model-only ablation harness is implemented but has not been run.
It compares vLLM online serving with Transformers fixed batches while holding
FlashAttention fixed in both stacks. It does not include AsyncRoll scheduling
and cannot explain or strengthen the completed +1.44% scheduler result. See
[SERVING_ABLATION.md](SERVING_ABLATION.md).

The completed scheduler comparison used
`scripts/run_lcb_bounded_v6_single_gpu.py` with
`experiments/livecodebench_bounded_8x4_v6.json`. See
[PROTOCOL_LIVECODEBENCH.md](PROTOCOL_LIVECODEBENCH.md).

## One experiment on a dedicated rented RTX 5090

1. Freeze a seeded ToolMATH subset, model revision, server version/settings,
   decoding, warmup/cache policy, and hardware allocation. Qualify action
   validity and actual tool use first. Retain failures in the denominator.
2. Collect aligned request/CPU traces, vLLM gauges/counters and server-local
   NVML samples. Inspect tool durations, CPU queues, model admission queues,
   and client overhead before choosing a scheduling intervention.
3. Freeze one configuration before comparison. The sole intervention is the
   CPU policy: FIFO versus `asyncroll`. Restart the serving process before
   each arm, use identical warmup, counterbalance arm order, and repeat full
   runs. Individual requests are not independent throughput replicates.
4. Primary metric: completed trajectories per dedicated allocated GPU-hour,
   excluding startup/warmup. Also report the wall-hour rate, with failure and
   tool-call rates. Exact string accuracy applies only to labelled problems.
   Stop the scheduling claim if tools rarely run/queue or FIFO leaves no
   measured CPU-related idle opportunity. Synthetic sleep is a stress test.

The `asyncroll` policy reorders pending tools when client model supply is low,
using an online per-job predictor over information available before execution.
Soft aging balances predicted unlock time; a 30-second deadline prevents
starvation. If the activation gate fails, stop before comparison. The legacy
`tool_first` baseline only prioritizes tools over grading and is not the formal
comparison. See [PROTOCOL.md](PROTOCOL.md) for the hypothesis and stop rules.

## Scope and entry

Priorities: correct measurement, then a clean scheduler and real workload.
The serving stack is fixed: vLLM batching, prefix caching, chunked prefill,
compilation/CUDA graphs and its automatically selected attention backend.
Retain actual startup evidence. Nsight/NVTX diagnoses the same experiment;
profiled runs stay separate from throughput runs. CPU affinity, oversubscription
and NUMA require later trace evidence. No other engine, custom kernel or
optimization ladder is part of this experiment.

Run from this repository root:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m asyncroll.cli run --workload examples/smoke.jsonl --backend scripted --policy fifo --output runs/smoke-fifo
```

See [docs/SINGLE_GPU.md](docs/SINGLE_GPU.md) for rental setup and execution,
[MEASUREMENT.md](MEASUREMENT.md) for metric semantics, and
[RESULTS.md](RESULTS.md) for the completed screen and stop decision.
