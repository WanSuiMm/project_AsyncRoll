# AsyncRoll context for reviewers

## Question and current answer

Can a CPU worker queue that favors tools returning to model inference improve
completed tool-using trajectories per GPU-hour over an asynchronous FIFO queue
at the same CPU allocation? **No empirical answer exists yet.** The repository
contains an inference-only prototype and a deterministic scripted smoke test.
It does not contain a live model run, a GPU utilization trace, an RL loop, or a
systems performance claim.

## Variants and implementation

| Variant | What it does | Source |
| --- | --- | --- |
| `sync` | Runs trajectories sequentially | `runtime.run` |
| `fifo` | Runs trajectories concurrently and serves CPU jobs in arrival order | `CPUQueue.submit` |
| `gpu_first` | Gives tool jobs returning to inference priority over terminal grading | `CPUQueue.submit` |

`_trajectory` alternates model actions with CPU tools. `VLLMBackend` sends
OpenAI-compatible chat requests; `ScriptedBackend` supplies fixed actions for
tests. `execute_tool` imports trusted Python tool files into worker processes.
`convert_toolmath` maps a ToolMATH problem and named tool to this runtime's
JSONL schema; it does not supply a verified trajectory or answer label.

The current priority rule is deliberately coarse. It does not use tool progress,
predict remaining tool time, preempt running work, or modify vLLM internals.
`gpu_slots` is a client request semaphore, not an observed GPU batch size.

## Evidence and claim boundary

- `tests/test_runtime.py`: four local unit/smoke tests passed on 2026-09-27.
  The scripted workload completed three of three trajectories. This checks
  execution flow, not throughput on a GPU.
- There are no live ToolMATH trajectories or verified performance results.
- `completed_per_gpu_hour` is wall-clock completion rate under a one-GPU
  assumption. It is meaningless as a GPU benchmark in scripted mode.
- `exact_accuracy` compares normalized answer strings only when the workload
  includes an explicit `reference_answer`. Mathematical equivalence is not
  established.
- The requested Qwen instruction model was not successfully acquired as of
  2026-09-27. This repository does not include model weights or checkpoints.

## First decisive experiment

Use one frozen ToolMATH subset and decoding configuration across all three
policies. Measure real tool-call rate, CPU run and queue times, model request
times, completed trajectories per elapsed hour, and failures. Compare FIFO to
GPU-first at identical worker count and trajectory concurrency. Stop the
scheduling claim if real tools seldom queue or FIFO already hides their delay.
Server-side GPU telemetry is required before attributing a difference to GPU
idle recovery. Artificial delay experiments, if added, must be labeled as
controlled stress tests rather than real-workload results.

## Minimal reading order

Read this file, then `PROJECT.md`, `src/asyncroll/runtime.py`,
`src/asyncroll/model.py`, and `tests/test_runtime.py`. The sample run outputs
under `runs/` and downloaded data under `data/` are local and excluded from Git.
