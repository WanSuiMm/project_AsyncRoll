# Results

## Bounded-concurrency v3 activation screen

The fixed 8-active/4-inflight/2-worker run qualified and passed the FIFO CPU
opportunity screen, then stopped at the AsyncRoll activation gate. The 128-task
activation arm completed 128/128 trajectories, exposed 224 distinct feature
keys and achieved +0.105 predicted/actual log-duration correlation. However,
only two dispatches were eligible under the low-model-supply trigger and only
one reordered, below the frozen minima of 10 and 5. The formal comparison was
not run and no throughput gain is established.

This localizes the remaining mechanism failure: per-job costs are now
distinguishable, while `model_outstanding < 1` makes ordering eligible too
rarely. The canonical receipt is
[`results/livecodebench_bounded_20260927_v5/experiment.json`](results/livecodebench_bounded_20260927_v5/experiment.json).

## LiveCodeBench v2 formal comparison

The amended v2 protocol completed all three paired FIFO/AsyncRoll comparisons;
every arm completed 219/219 trajectories. AsyncRoll's paired throughput changes
were **+1.38%, -2.90%, and +0.40%**. The paired mean was **-0.37%** with a
descriptive 95% t interval of **[-5.94%, +5.20%]** across three pairs. Pooling
elapsed time across arms gives the same conclusion: AsyncRoll throughput was
0.37% lower than FIFO. This experiment provides no evidence that AsyncRoll
improves throughput under the tested configuration.

Mean exact pass rate was 28.46% for FIFO and 29.07% for AsyncRoll, a descriptive
+0.61 percentage-point difference. Pass rate was a comparability check rather
than evidence for a quality improvement, and three pairs are insufficient for a
quality claim. The canonical receipt is
[`results/livecodebench_20260927_v4/experiment.json`](results/livecodebench_20260927_v4/experiment.json).

## LiveCodeBench v1 opportunity screen

The Qwen2.5-Coder-7B-Instruct one-repair workload qualified and completed its
128-task FIFO opportunity screen. Four of five v1 gates passed: client CPU
blocking was 2.708 s, maximum CPU queue depth was 29, CPU-related idle candidate
time was 1.751 s, and tool-queue p95 was 10.169 s. Joint vLLM/NVML observation
coverage was 76.07%, below the 80% telemetry gate, so v1 stopped before the
comparison. The 128-task pass rate was 26.56%.

At the user's explicit direction, v2 removes joint telemetry coverage as a hard
stop while continuing to report it. The other four opportunity gates and the
three paired FIFO/AsyncRoll comparisons remain unchanged. This is a documented
post-screen protocol amendment; the v1 receipt is preserved at
[`results/livecodebench_20260927_v3/experiment.json`](results/livecodebench_20260927_v3/experiment.json).

## Current result

The frozen single-RTX-5090 experiment qualified the live Qwen/ToolMATH stack,
then stopped after the FIFO opportunity screen. Real ToolMATH calls were too
short and did not queue deeply enough to create a measurable CPU scheduling
opportunity. The planned FIFO versus AsyncRoll comparison was therefore not
run. This result supports no speedup or causal GPU-bubble recovery claim.

Canonical machine-readable values are in
[`results/single_5090_20260927.json`](results/single_5090_20260927.json).

## Qualification

| Check | Result |
| --- | ---: |
| Completed trajectories | 8 / 8 |
| Valid structured actions | 100% |
| Trajectories using a tool | 100% |
| vLLM and NVML telemetry | Available |
| Telemetry errors | 0 / 241 samples |
| Verdict | **Qualified** |

The live stack used Qwen2.5-Math-7B-Instruct at the pinned model revision,
vLLM 0.11.2, FLASH_ATTN, verified CUDA graph capture, xgrammar structured
outputs, and a dedicated RTX 5090.

## FIFO opportunity screen

| Frozen gate | Observed | Required | Pass |
| --- | ---: | ---: | :---: |
| Full-workload quality | 31 / 32 completed; 31 / 32 used a tool | All | No |
| Client CPU-blocked time | 0 s | >= 0.001 s | No |
| Maximum CPU queue depth | 1 | >= 2 | No |
| CPU-related idle candidate | 0 s | >= 0.001 s | No |
| Joint observation coverage | 76.81% | >= 80% | No |
| Tool queue p95 | 1.707 ms | >= 5 ms | No |

Tool execution itself had a p95 of 0.355 ms. The screen recorded 591 telemetry
samples without collection errors. It completed 31 trajectories in 118.58 s;
the descriptive rate was 941.13 completions per allocated GPU-hour. That rate
is a single FIFO measurement whose quality gate failed, so it is not a baseline
for a speedup claim.

One retained trajectory, `toolmath-10884`, produced an invalid division-by-zero
tool call. It remains in the denominator as required by the protocol.

## Decision and claim boundary

The pre-registered stop rule fired: this workload and configuration showed no
qualifying CPU-related GPU-idle opportunity. Adding artificial sleeps or changing
the workload after seeing the result would answer a different question.

- Established: the live inference stack, structured tool loop, telemetry, and
  evidence gates ran end to end on the dedicated GPU.
- Established: the sampled real-tool workload offered no useful CPU scheduling
  leverage under this fixed configuration.
- Not established: FIFO versus AsyncRoll performance, answer accuracy,
  recoverable GPU time, or any AsyncRoll speedup.

Raw traces and launch receipts remain local because they include machine-specific
paths, process identifiers, and device identifiers. This repository publishes
the decision-relevant aggregate without those private deployment details.
