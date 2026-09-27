# LiveCodeBench one-repair protocol

This is a new experiment. It does not replace the frozen ToolMATH result in
[RESULTS.md](RESULTS.md).

## Question

On one dedicated RTX 5090, can a CPU queue policy improve completed code-agent
trajectories per allocated GPU-hour when each trajectory alternates between
Qwen2.5-Coder generation and LiveCodeBench full test execution?

The online trajectory is: generate one Python program, run its hidden tests,
provide the first bounded failure feedback when needed, generate at most one
repair, run the same tests again, and terminate. This adapts LiveCodeBench data
and test semantics to an online AsyncRoll trajectory. It is not a claim that the
official offline Self Repair CLI implements this runtime loop.

## Frozen stages

1. Qualify 8 fixed stdin-style tasks, structured submissions, local execution,
   vLLM metrics and NVML telemetry.
2. Run a 128-task FIFO opportunity screen. Stop if execution does not create a
   CPU queue and a sampled CPU-related idle opportunity. Do not add sleeps or
   select convenient slow tasks after observing the result.
3. If the screen passes, compare FIFO with AsyncRoll on all 219 stdin-style
   tasks available in the pinned full release_v6 export
   in three alternating-order pairs. Restart vLLM before each arm and use the
   same warmup, model, tests, decoding, resource limits and concurrency.

The source contains 454 release_v6 tasks; 219 use stdin semantics and 235 use
function-call semantics. V1 freezes the 219 supported tasks instead of mixing
checker implementations or claiming the requested approximate 512 count.

The fixed starting configuration is
[`experiments/livecodebench_5090.json`](experiments/livecodebench_5090.json).
The independent unit is a full arm, not a trajectory or test case.

## Execution and security boundary

Each evaluator job gets a new temporary directory, wall and CPU deadlines,
memory/file/process limits and a low-privilege account when the launcher is
root. These controls are a benchmark harness, not a hardened security sandbox.
Run it only on a disposable host without credentials or valuable data. The v1
experiment uses stdin-style problems; functional-call cases are excluded rather
than silently reinterpreted.

## Endpoints

- Primary: completed trajectories per dedicated allocated GPU-hour.
- Mechanism: sampled model starvation, CPU queue p95 and maximum queue depth.
- Quality: final test pass rate, failures and repair rate.

A positive throughput difference is uninterpretable if task quality, resource
allocation, telemetry coverage or test sets differ. Preserve timeouts and failed
arms. No positive result is assumed.
