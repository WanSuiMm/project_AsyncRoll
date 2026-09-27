# Incremental review handoff

- Review base: `56a36ee7e3c81a364fec9e59d9af82ed9a1e371a`
- Evidence head: `2800b9d202355459320ff7693b847f2966d319cf`
- This handoff is a later metadata-only commit; review the evidence head above.
- Validation date: 2026-09-27.

## Decision-relevant change

The LiveCodeBench v2 formal experiment completed three paired comparisons. All
six arms completed 219/219 trajectories. AsyncRoll's paired throughput changes
relative to FIFO were +1.38%, -2.90%, and +0.40%; paired mean -0.37%, with a
descriptive 95% t interval of [-5.94%, +5.20%]. The result does not support an
AsyncRoll throughput improvement under this configuration.

Mean exact pass rate was 28.46% for FIFO and 29.07% for AsyncRoll. This +0.61
percentage-point descriptive difference is not a quality-improvement claim.

## Protocol boundary

The v1 screen stopped because joint vLLM/NVML coverage was 76.07% against an
80% gate, although the four direct CPU-opportunity checks passed. At the user's
explicit direction, v2 recorded telemetry coverage but removed it as a stop
condition. The v1 receipt remains unchanged. Interpret the comparison only
under the documented v2 amendment.

## Minimal reading order

1. `RESULTS.md`: result, uncertainty, and claim boundary.
2. `results/livecodebench_20260927_v4/README.md`: compact comparison table.
3. `results/livecodebench_20260927_v4/experiment.json`: canonical aggregate.
4. `PROTOCOL_LIVECODEBENCH.md`: workload and protocol amendment.
5. `src/asyncroll/runtime.py` and `scripts/run_lcb_single_gpu.py`: runtime and gated runner.

Large hidden tests, model files, per-request logs, and machine-specific launch
receipts are intentionally excluded. The earlier ToolMATH negative result is a
separate frozen workload and is unchanged.

## Reviewer questions

1. Does the paired result justify the no-speedup conclusion without implying
   equivalence outside the interval supported by three pairs?
2. Is the post-screen v2 amendment disclosed clearly enough?
3. Do the published aggregates preserve the distinction between throughput and
   pass-rate observations?
