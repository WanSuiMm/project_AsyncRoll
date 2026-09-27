# Incremental review handoff

- Review base: `2c5d7beaef4cc0a5fd438f9286510d3382854089`
- Evidence head: `1834f2f12a93eb1161bce1b4132f2855cd86e30d`
- This handoff is a later metadata-only commit; review the evidence head above.
- Validation date: 2026-09-27.

## V6 change

V5 stopped because `model_outstanding < 1` yielded only two eligible decisions.
V6 keeps the frozen 8-active/4-inflight/2-worker point, 219-task workload, vLLM
settings, soft aging and 30-second hard deadline. It changes two mechanisms:

1. Reordering is eligible when model outstanding is below the four-request
   client capacity, so CPU dispatch replenishes supply before the queue empties.
2. A repair evaluation uses the same problem's observed first-evaluation wall
   time. Initial evaluations retain the online ridge fallback.

The activation arm passed with 38 eligible decisions, 19 reorders and +0.559
same-problem repair correlation. All six formal arms completed. AsyncRoll's
paired throughput changes were +0.92%, +1.03% and +2.37%; paired mean +1.44%
with a descriptive 95% t interval of [-0.56%, +3.44%]. The mechanism activated
and all point estimates were positive, while the interval includes zero and the
result does not support a 20% claim.

## Verification

- 69 tests and 8 subtests passed before deployment.
- Python compilation passed for the scheduler, runtime, metrics and v6 runner.
- The sanitized complete v6 receipt is published under `results/`.

## Minimal reading order

1. `PROTOCOL_LIVECODEBENCH.md`: v6 mechanism and boundary.
2. `experiments/livecodebench_bounded_8x4_v6.json`: frozen configuration.
3. `src/asyncroll/scheduling.py`: proactive trigger and same-problem prior.
4. `src/asyncroll/metrics.py`: repair-specific prediction correlation.
5. `scripts/run_lcb_bounded_v6_single_gpu.py`: activation and comparison gates.
6. `RESULTS.md`: preserved v2 and v5 negative evidence.

## Reviewer questions

1. Is same-problem first-evaluation latency a valid online repair predictor?
2. Does binding the trigger to client capacity match the intended supply signal?
3. Does the simplified activation gate test the mechanism before comparison?
