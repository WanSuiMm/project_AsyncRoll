# Single dedicated GPU protocol

## Question and intervention

Can scheduling CPU continuations improve completed trajectories per allocated
GPU-hour when tool-using agents share one dedicated RTX 5090 and one vLLM
server? Compare **FIFO versus AsyncRoll**. Inference only; no RL.

Hold model revision, ToolMATH subset/order, tools, decoding, CPU workers,
trajectory admission, model request concurrency, vLLM settings, warmup, telemetry
cadence and hardware fixed. The executable starting configuration is
[single_5090.json](experiments/single_5090.json). Record its hash before running.
Request seeds match within each pair. Temperature zero plus a fixed seed does
not guarantee identical output under changed concurrent batching.

## Policy

`scheduling.Scheduler.select` is non-preemptive. When client queued + inflight
model requests fall below the configured threshold, prefer CPU tools that can
return agents to model execution, ordered by estimated execution time. Use an
EWMA (alpha 0.5) of only earlier completed callable executions from this run.
Unknown callables use median observed duration; no observations means FIFO.
At/above the supply threshold use FIFO. Aged pending jobs take FIFO precedence
over cost. The estimator resets for every arm. Both policies collect the same
timings and decision fields.

This is a concrete candidate heuristic, not an optimal scheduling claim.
It neither observes remaining runtime of executing tools nor reads future
actions, answer labels or a replay oracle. The pressure signal is client supply,
not instantaneous CUDA occupancy. Many ToolMATH functions are cheap and unique;
there may be neither an accurate cost signal nor an opportunity to reorder.

## Bounded sequence and stop rules

1. **Qualification:** eight fixed tasks. Check real schema/argument validity,
   completion and actual tool calls, actual attention/graph startup evidence,
   and server-local telemetry. Failed qualification ends this configuration.
2. **Opportunity screen:** FIFO on the fixed 32-task subset. Require at least
   two queued jobs, tool queue p95 >= 5 ms, CPU-related idle candidate > 1 ms,
   and joint observation coverage >= 80 percent. The exact thresholds live in
   the configuration. A zero/unknown result does not justify a scheduling
   claim. Preserve artifacts; inspect the timeline without adding artificial
   sleeps, no-op grading or selecting convenient slow tasks after seeing results.
3. **Comparison:** three independent full-run pairs, alternating arm order,
   identical resources. Start a fresh serving process before each arm to remove
   cross-arm KV/prefix state, then apply the same warmup. Process startup and
   model warmup are reported separately. Bound startup/run duration and retain
   failed arms; never silently retry until a favorable result appears.
4. **Mechanism trace:** one diagnostic Nsight capture per policy, same workload
   and settings. Capture a process tree including the vLLM server, client and
   workers. Keep profiler-enabled timing out of the main comparison.

The machine is not yet rented, so runtime/driver qualification remains pending.
Freeze the working environment/configuration before comparison. Startup capture
can be expensive; record phase deadlines and separate setup times rather than
hiding them in throughput. Three pairs alternate order but are not perfectly
balanced (two FIFO-first, one AsyncRoll-first); retain that order information.

## Endpoints and interpretation

- Primary: completions per **dedicated allocated GPU-hour during measurement**.
  Report count, elapsed time, failures, action validity and tool-use rate together.
  This is not a rental-bill efficiency metric; startup/warmup are excluded.
- Mechanism: sampled empty vLLM queues intersected with client CPU-blocked
  intervals; low NVML activity additionally defines a CPU-related idle candidate.
  Report sample coverage. This cannot prove continuous kernel idleness or
  recoverability; inspect Nsight and the controlled policy difference.
- Cost: process CPU seconds for client plus all persistent tool workers divided
  by completions. Includes IPC, serialization and client telemetry overhead;
  excludes vLLM serving CPU. A killed worker makes total cost unknown.
- Quality: completion is a protocol endpoint, not mathematical correctness.
  Unlabelled ToolMATH cannot establish useful/correct-answer throughput. Report
  labelled exact-match accuracy only when genuine references exist; do not
  derive a correctness oracle by copying the source solution into the prompt.

Independent unit: a full run, paired by configuration and request seed. Retain
raw paired points/ratios. Three pairs give a pilot systems comparison, not a
broad generalization across models, GPUs or workloads. No positive result is
assumed. If throughput rises while failure/quality/workload differs, investigate
before attributing a benefit to recovered model starvation.
