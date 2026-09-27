# Results

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
