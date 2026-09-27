# Incremental review handoff

- Review base: `5cb6a72d33711085e93e152ddfe5c81abc89d31a`
- Code/evidence head: `797b6fe94817aa65c607c48def40f20b356346bb`
- This file is a later documentation-only commit; the evidence head stays fixed.
- Validation date: 2026-09-27. Release: 0.3.0.

## What changed

1. The project now has one fixed-resource experiment for a dedicated rented
   RTX 5090: FIFO versus `asyncroll`. The configuration has 8 qualification
   tasks, a 32-task opportunity screen, and three alternating-order full-run
   pairs. A failed/negative screen stops the comparison.
2. `scheduling.Scheduler` implements a non-preemptive policy using client model
   supply, per-tool past observed duration and aging. It has no future-work or
   answer oracle. Unknown tools initially preserve FIFO; ineffective reordering
   is an admissible negative result.
3. CPU cost uses whole client/worker process CPU deltas; worker death makes the
   total unknown. Sampled starvation intersects client state with empty server
   queues, with explicit coverage and additional NVML idle-candidate criteria.
4. Optional async-safe NVTX ranges cover model waiting/HTTP, CPU queueing, actual
   worker execution and the measured window. Nsight wraps server and client
   descendants. Diagnostic profile timings never enter the formal comparison.
5. The launcher checks pinned model content digests, workload/tool/code/dependency
   fingerprints, GPU ownership, selected attention backend and graph-capture
   evidence. It restarts the server for each arm and preserves owned PID/terminal
   receipts, partial comparisons and bounded cleanup.
6. The offline report exports raw paired points, quality/failure counts and one
   three-panel PDF/PNG. No example speedup numbers have been added.

## Claim boundaries retained

There is no live rented-GPU result, Qwen/ToolMATH protocol qualification,
demonstrated speedup, mathematical-equivalence accuracy, causal recoverable
bubble estimate, or RL training. vLLM/NVML/NVTX/Nsight integration still needs the
actual rental environment. Startup audit rejects unfamiliar/unproven evidence;
requested flags alone are insufficient. Generic grammar backend log entries do
not count as attention evidence.

Completed trajectories are not necessarily correct answers. CPU cost excludes
vLLM serving CPU. Dedicated GPU-hours cover the measured allocation window,
excluding model loading and warmup. Sampled empty queues do not prove continuous
CUDA idleness. A short/cheap/unique ToolMATH tool mix may leave no useful schedule
intervention. Three pairs are a pilot comparison with a 2:1 first-arm imbalance.

## New verification evidence

| Check | Observed result | Boundary |
| --- | --- | --- |
| `python -m unittest discover -s tests -q` | 49 tests passed | Local behavior; GPU/HTTP/NVTX/vendor startup evidence mocked |
| Scripted AsyncRoll CLI smoke | 3/3 completed, two tool calls, zero failures | No GPU; throughput is not a scientific result |
| Offline report export | PDF/PNG generated from temporary synthetic fixtures | Export behavior only; fixtures are not experimental data |
| Packaging | 0.3.0 wheel built; new modules and model manifest included | Rental dependencies/driver remain unqualified |
| Documentation | Seven Markdown files checked; no missing local link targets | Use editable install from repository root as documented |
| Delivery | Staged size/whitespace/secret/machine-identifier checks passed | Caches, models, local runs and private receipts excluded |

Targeted tests cover pressure/aging, estimator reset, killed-worker accounting,
missing telemetry and sample gaps, overlapping NVTX correlation, fresh arm plans,
UUID/NVML mapping, actual backend versus grammar logs, hash mismatches, failed-arm
PID persistence, cleanup, and retaining the active Python environment path.

## Minimal reading order and review questions

1. [PROTOCOL.md](PROTOCOL.md), [single_5090.json](experiments/single_5090.json):
   are the frozen screen, quality constraints and negative-result interpretation
   sufficient for a useful pilot?
2. [scheduling.py](src/asyncroll/scheduling.py), [metrics.py](src/asyncroll/metrics.py):
   can this past-cost heuristic distinguish real tools, and are sampled
   associations/cost scope labelled honestly?
3. [experiment.py](src/asyncroll/experiment.py), [test_experiment.py](tests/test_experiment.py):
   which rental-specific vLLM/log/driver assumptions still need qualification?
4. [SINGLE_GPU.md](docs/SINGLE_GPU.md), [MEASUREMENT.md](MEASUREMENT.md):
   can an independent reader execute the same comparison and explain its trace?

Read concise summaries before per-event logs. Previously reviewed HTTP pooling,
worker import caching, independent concurrency and baseline timeline code remain
the foundation; do not infer new empirical evidence from those unchanged pieces.
