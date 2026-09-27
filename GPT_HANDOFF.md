# Incremental review handoff

- Review base: `bb4913035f2cbcecac63172b0f6bbff7cb04469a`
- Evidence head: `6e1418ccc7302bb4430171f19e18c592edc8325f`
- This handoff is a later metadata-only commit; review the evidence head above.
- Validation date: 2026-09-27.

## What changed

V3 completed qualification, FIFO opportunity and AsyncRoll activation, then
stopped before comparison because the scheduler did not activate often enough.
The operating point is fixed at 8 active trajectories, 4 client model requests,
2 CPU workers and vLLM `max_num_seqs=8`. There is no concurrency sweep.

AsyncRoll now predicts per-job evaluator wall time with an online ridge model
using pre-execution test count, test-input bytes, generated-code bytes and
initial/repair status. It applies soft aging and a 30-second hard starvation
deadline. Metrics now report eligible decisions, activation rate, reorder count,
decision reasons, feature diversity and predicted/actual log-time correlation.

The activation arm completed 128/128 trajectories, exposed 224 distinct feature
keys and measured +0.105 predicted/actual log-duration correlation. Only two
decisions were eligible and one reordered, below the required counts of 10 and
5. The three-pair comparison did not run; no gain is established. V1/v2
receipts and their negative conclusions are intact.

## Verification

- 67 tests and 8 subtests passed locally.
- Python compilation passed for the runtime, scheduler, metrics, CLI and runner.
- Staged content passed whitespace and sensitive-identifier scans.
- The sanitized v3 activation receipt is published under `results/`.

## Minimal reading order

1. `PROTOCOL_LIVECODEBENCH.md`: v3 boundary and activation gate.
2. `experiments/livecodebench_bounded_8x4.json`: frozen operating point.
3. `src/asyncroll/scheduling.py`: predictor and selection rule.
4. `src/asyncroll/metrics.py`: activation and prediction diagnostics.
5. `scripts/run_lcb_bounded_single_gpu.py`: qualification-to-comparison runner.
6. `RESULTS.md`: completed v1/v2 evidence and the v3 activation stop.

## Reviewer questions

1. Are all predictor features available before execution and free of expected
   test outputs?
2. Does the activation gate prevent another FIFO-equivalent comparison?
3. Does soft aging preserve useful reorder decisions while bounding starvation?
