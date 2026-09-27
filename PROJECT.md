# AsyncRoll

## Research question

When agents alternate between GPU model execution and CPU tools, how much
model idle time coincides with pending CPU continuations, and can a CPU queue
policy improve completed trajectories per elapsed hour at equal resources?

## Current status (2026-09-27)

The measurement refactor implements asynchronous pooled HTTP, warm cached
tool workers, process termination/replacement on timeout, independent
trajectory/model/CPU limits, phase timing, vLLM/NVML telemetry collection,
and timeline export. Local tests and scripted smoke validate harness behavior.
No live model/ToolMATH run, speedup, accuracy qualification, or causal bubble
recovery result is established. This project is inference only, without RL.

## First screen (Gate 0)

1. Freeze a seeded ToolMATH subset, model revision, server version/settings,
   decoding, warmup/cache policy, and hardware allocation. Qualify action
   validity and actual tool use first. Retain failures in the denominator.
2. Collect aligned request/CPU traces, vLLM gauges/counters and server-local
   NVML samples. Inspect tool durations, CPU queues, model admission queues,
   and client overhead before choosing a scheduling intervention.
3. Independently vary active trajectories, model inflight limit and CPU
   worker count in a bounded, preregistered grid. Hold each configuration
   fixed for policy comparisons. Repeat independent full runs; individual
   requests are not independent throughput replicates.
4. Primary metric: completed trajectories per wall hour, with failure and
   tool-call rates. Exact string accuracy applies only to labelled problems.
   Stop the scheduling claim if tools rarely run/queue or FIFO leaves no
   measured CPU-related idle opportunity. Synthetic sleep is a stress test.

The current `tool_first` baseline only prioritizes tools over terminal grading.
Unlabelled ToolMATH provides no such competing grading class; FIFO and
tool-first are expected to behave alike. Do not invent no-op grading to create
headroom. A pressure-aware policy requires a later evidence-based decision.

## Scope and entry

Priorities: correct measurement, then a clean scheduler and real workload.
CPU affinity, oversubscription and NUMA are deferred until the first trace
justifies them. CUDA kernels, vLLM internals, tokenizer optimization, custom
RPC/shared memory and a C++ runtime are outside this version.

Run from this repository root:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m asyncroll.cli run --workload examples/smoke.jsonl --backend scripted --policy fifo --output runs/smoke-fifo
```

See [README.md](README.md) for live configuration and [MEASUREMENT.md](MEASUREMENT.md)
for exact metric semantics. This update authorizes code verification only;
no live experiment is launched as part of it.
