# What the measurements mean

## Lifecycle

Each run starts all configured worker processes using `spawn`, imports each
selected tool once per worker, and executes a dummy tool/grading call. It then
sends `--warmup-requests` first-turn requests in bounded batches and probes
telemetry. Only after these steps does the measurement clock start. Timing
ends when all trajectories finish, before worker/client teardown and timeline
export. Setup durations and effective limits are recorded separately.

Warmup uses initial prompts from the workload, cycling when necessary. It may
populate prefix and grammar caches; hold that policy constant across runs.
Client warmup does not establish that every vLLM/CUDA graph, sequence length
or kernel is warm. Record server startup/warmup configuration independently.

## Timing and concurrency

| Field | Meaning |
| --- | --- |
| Trajectory admission | Time since measurement start before a trajectory begins |
| `model_queue` | Waiting for the client model semaphore |
| `model_http` / per-trajectory `model_request` | Async HTTP round trip, including server queue and inference |
| `parsing` | Response envelope/action JSON decoding and validation |
| `tool_queue`, `grade_queue` | Parent CPU queue wait, including any worker replacement before dispatch |
| `*_execute` | Child callable wall time, excluding result serialization |
| `*_process_cpu` | Child process CPU time during that callable |
| `*_serialization` | Child result encoding |
| `*_dispatch` | Parent dispatch until reply/error, including IPC, execution and serialization |
| `*_harness` | Dispatch minus measured execution and serialization, including IPC/event-loop delay |

Each job has trajectory/turn/job IDs, kind, tool name, worker slot and worker
generation. Successful replies include child start/finish times on the same
host monotonic clock. Timeout execution duration is **unknown**, not zero;
dispatch time includes termination handling. A replacement worker is preloaded
again, with recovery inside the measured run.

Each trajectory has an additive latency breakdown with runtime/unattributed
residual; CPU dispatch already includes tool execution. It excludes time waiting
for trajectory admission. Across trajectories, phase totals overlap, so they
are accumulated work/latency, not additive percentages of elapsed wall time.
Completed-per-wall-hour includes failed trajectories' elapsed cost. Exact
accuracy uses only labelled completed trajectories; report its denominator
alongside total failures.

Line-buffered event writing and sampling also consume CPU. Keep their settings
fixed across configurations. `event_recording_seconds` reports time in the
recorder/synchronous event sink; queue/dispatch timers start after their entry
event is written. Other tasks' event writing can still delay the event loop
and is part of runtime contention. Small tools may be dominated by Python/IPC; the
recorded harness duration makes this visible. Do not subtract it silently.

## Resource sources

The collector uses [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/):
running/waiting request gauges, prompt/generation token counters and KV cache
fraction. It filters by the configured served model name, sums request/counter
series and takes maximum KV usage across matching engines. Token rates use
counter deltas; missing samples/resets are null, never zero. Model-name mismatch
is a visible missing-metric condition. Older `gpu_cache_usage_perc` is accepted.

NVML is optional and local to the process running AsyncRoll. Select the physical
GPU hosting vLLM; remote `/metrics` plus unrelated local NVML is not a valid
combined trace. NVML utilization is sampled device activity; VRAM bytes/fraction
are allocation, not memory-bandwidth utilization. Other processes may contribute.

Scrape times, errors and missing values are recorded. Sampling and vLLM metric
refresh cadences limit temporal resolution: a 200 ms collector cannot establish
microsecond kernel gaps, and faster scraping need not mean fresher server data.
Use an otherwise isolated server and retain its version and configuration.

## Reading the timeline

Open `timeline.html` locally, or load `trace.json` in a Chrome/Perfetto trace
viewer. Align CPU queues/execution and trajectory phases with vLLM running/
waiting requests, token rates, KV usage and local GPU utilization. Model-client
bars are HTTP intervals, not GPU kernel execution spans. Missing resource
series do not imply idleness.

The first question is whether low observed GPU activity coincides with queued
or executing CPU continuations while no model request is ready. Such intervals
are **candidate CPU-related idle time**, not established recoverable bubbles.
Recoverability needs a controlled intervention at equal resources and preserved
workload/quality. This version deliberately emits no causal `recoverable_bubble`
scalar and does not subtract HTTP time to guess server inference time.

HTTP pooling follows [HTTPX async client guidance](https://www.python-httpx.org/async/).
Schema output follows [vLLM structured output guidance](https://docs.vllm.ai/en/latest/features/structured_outputs/);
argument correctness and Qwen native TIR compatibility still need real qualification.
