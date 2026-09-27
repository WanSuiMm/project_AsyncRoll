# LiveCodeBench one-repair protocol

## v3 fixed bounded-concurrency optimization (not yet run)

V2 showed that 32 active trajectories hid nearly all CPU evaluator latency and
that AsyncRoll-v0 made zero reorder decisions. V3 fixes one operating point
before execution: 8 active trajectories, 4 client model requests, 2 CPU workers,
vLLM `max_num_seqs=8`, and a 30-second hard starvation deadline. There is no
concurrency sweep.

The scheduler predicts each queued evaluation from pre-execution test count,
test-input bytes, generated-code bytes, and initial/repair status using an
online ridge model. It ranks predicted unlock time with soft aging. A 128-task
activation arm must record at least 10 eligible decisions, 5 reorders, 5%
activation, 8 distinct feature keys, and positive predicted/actual log-duration
correlation before the unchanged three-pair FIFO/AsyncRoll comparison may run.
FIFO and AsyncRoll receive identical resources and tasks. A 20% improvement is
an engineering target, not a result or gate.

The frozen configuration and runner are
[`experiments/livecodebench_bounded_8x4.json`](experiments/livecodebench_bounded_8x4.json)
and `scripts/run_lcb_bounded_single_gpu.py`. V1 and v2 evidence remains intact.

## v2 protocol amendment (2026-09-27)

The v1 opportunity screen completed 128/128 tasks and passed four of five
opportunity checks. Joint vLLM/NVML observation coverage was 0.7607 versus the
0.80 stop threshold. Because coverage measures telemetry alignment rather than
the presence of CPU scheduling opportunity, v2 records this value but removes
it as a hard stop gate. The CPU queue, client blocking, CPU-related idle
candidate, and tool-queue latency gates remain required. The v1 receipt is
preserved unchanged; v2 comparisons are interpreted under this amended protocol.

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

Code generations use ordinary source output with optional Markdown-fence
removal. JSON-schema constrained decoding is disabled because live qualification
showed that grammar-constraining an escaped multi-kilobyte program reduced
generation throughput to about 5.6 tokens/s and would dominate the measurement.

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
