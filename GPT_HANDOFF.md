# Incremental review handoff

- Review base: `2800b9d202355459320ff7693b847f2966d319cf`
- Evidence head: `bb4913035f2cbcecac63172b0f6bbff7cb04469a`
- This handoff is a later metadata-only commit; review the evidence head above.
- Validation date: 2026-09-27.

## What changed

V3 is a code-only bounded-concurrency optimization; it has not been deployed.
The operating point is fixed at 8 active trajectories, 4 client model requests,
2 CPU workers and vLLM `max_num_seqs=8`. There is no concurrency sweep.

AsyncRoll now predicts per-job evaluator wall time with an online ridge model
using pre-execution test count, test-input bytes, generated-code bytes and
initial/repair status. It applies soft aging and a 30-second hard starvation
deadline. Metrics now report eligible decisions, activation rate, reorder count,
decision reasons, feature diversity and predicted/actual log-time correlation.

An activation arm must demonstrate real reordering and positive prediction
correlation before the unchanged three-pair FIFO/AsyncRoll comparison runs.
FIFO and AsyncRoll use identical tasks and resource limits. A 20% gain is a
target, not a result. V1/v2 receipts and their negative conclusions are intact.

## Verification

- 67 tests and 8 subtests passed locally.
- Python compilation passed for the runtime, scheduler, metrics, CLI and runner.
- Staged content passed whitespace and sensitive-identifier scans.
- No server experiment was launched.

## Minimal reading order

1. `PROTOCOL_LIVECODEBENCH.md`: v3 boundary and activation gate.
2. `experiments/livecodebench_bounded_8x4.json`: frozen operating point.
3. `src/asyncroll/scheduling.py`: predictor and selection rule.
4. `src/asyncroll/metrics.py`: activation and prediction diagnostics.
5. `scripts/run_lcb_bounded_single_gpu.py`: qualification-to-comparison runner.
6. `RESULTS.md`: unchanged completed v1/v2 evidence.

## Reviewer questions

1. Are all predictor features available before execution and free of expected
   test outputs?
2. Does the activation gate prevent another FIFO-equivalent comparison?
3. Does soft aging preserve useful reorder decisions while bounding starvation?
