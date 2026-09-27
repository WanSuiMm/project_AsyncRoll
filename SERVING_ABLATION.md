# Optional vLLM serving ablation

This is a model-only side measurement. It does not include CPU tools, repair
execution or AsyncRoll scheduling, and its numbers must not be presented as the
cause of the completed AsyncRoll scheduler result.

## Question

At four concurrent requests, how much faster is vLLM online continuous serving
than fixed batches of four from Transformers when both request FlashAttention
and use the same greedy prompts and token limit?

FlashAttention is held fixed as part of both stacks. This protocol does not
measure or claim an independent FlashAttention gain.

## Frozen measurement

- Model: pinned Qwen2.5-Coder-7B-Instruct revision.
- Prompts: first 8 warmups and next 64 measured prompts from the pinned selected
  LiveCodeBench workload.
- Decoding: greedy, 256 maximum new tokens, fixed seed.
- Load: concurrency 4 for vLLM and fixed batch size 4 for Transformers.
- Repetitions: three paired runs, with alternating arm order.
- Prefix caching: disabled. Server startup and warmup are excluded.
- Primary metrics: completed requests/s and output tokens/s. p50/p95 request
  latency and exact output-match fraction are diagnostic checks.

The Transformers arm uses `attn_implementation=flash_attention_2`, while the
vLLM engine arm requests `FLASH_ATTN`. This makes the engine comparison closer
to a serving-stack comparison rather than an attention-kernel comparison.

## Run on a fresh RTX 5090 instance

Install the project in the existing vLLM environment and ensure Transformers
and its FlashAttention 2 dependency are available. Validate the immutable plan
without loading a model:

```bash
python scripts/run_serving_ablation.py \
  --config experiments/serving_ablation_5090.json \
  --model-path /path/to/Qwen2.5-Coder-7B-Instruct \
  --workload /path/to/selected-219.jsonl \
  --output /tmp/not-created-in-plan-mode \
  --plan
```

Then launch into a new result directory:

```bash
python scripts/run_serving_ablation.py \
  --config experiments/serving_ablation_5090.json \
  --model-path /path/to/Qwen2.5-Coder-7B-Instruct \
  --workload /path/to/selected-219.jsonl \
  --output runs/serving-ablation-5090-v1 \
  --gpu-index 0
```

`experiment.json` is updated after every arm. A valid completed comparison
requires three paired replicates and reports vLLM's change in requests/s and
output tokens/s relative to the matching Transformers fixed-batch arm.
