# Incremental review handoff

- Review base: `53c998a3fa795f89b40d8c8967590cd0d805e957`
- Scheduler evidence head: `1834f2f12a93eb1161bce1b4132f2855cd86e30d`
- Validation date: 2026-09-28.

## Code-only serving ablation

This update adds an optional model-only benchmark. No new GPU experiment or
result is included, because the rental server was already shut down.

The harness answers two separate engineering questions:

1. vLLM online continuous serving versus Transformers fixed batches, with both
   sides requesting FlashAttention, the same prompts, greedy decoding, a
   four-request/batch load and the same output-token limit.
2. vLLM `FLASH_ATTN` versus a configured supported vLLM reference attention
   backend, with all other server settings fixed.

It runs three rotated replicates, starts a fresh process for every arm, excludes
startup and eight warmup prompts, disables prefix caching, and records request
rate, output-token rate, latency and exact output-match fraction. If the
reference backend cannot start, the receipt marks it unsupported and emits no
attention gain.

The completed AsyncRoll v6 evidence is unchanged: +1.44% paired mean scheduler
throughput change with a descriptive interval spanning zero. The new harness
cannot be used to attribute that result to vLLM or FlashAttention.

## Verification

- 74 unit tests passed, including five new serving-ablation tests.
- Python compilation passed for the benchmark module, runner and tests.
- No GPU or server measurement was attempted.

## Minimal reading order

1. `SERVING_ABLATION.md`: estimands, boundaries and run command.
2. `experiments/serving_ablation_5090.json`: fixed prompts/load/order/settings.
3. `src/asyncroll/serving_ablation.py`: HTTP and Transformers measurements.
4. `scripts/run_serving_ablation.py`: fresh-process orchestration and receipt.
5. `tests/test_serving_ablation.py`: local behavior checks.

## Reviewer questions

1. Is the fixed-batch Transformers comparator labelled narrowly enough?
2. Does exact-output matching adequately expose decoding drift between arms?
3. Should a future host replace `TRITON_ATTN` only if startup proves it
   unsupported, while preserving the replacement in a new configuration?
