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

## Single-GPU endpoints

`resource_metrics` reports the following:

| Field | Definition |
| --- | --- |
| `completed_per_gpu_hour` | Completions / timed wall hours, only when one dedicated allocated GPU is explicitly declared |
| `client_cpu_blocked_seconds` | Exact parent-event intervals with active trajectories, pending tool continuations and zero queued/inflight model requests |
| `model_starvation_seconds` | Those client intervals intersected with consecutive observations of zero vLLM running/waiting requests |
| `cpu_related_idle_candidate_seconds` | Same intersection, also requiring both endpoint NVML utilizations <= 10 percent |
| `server_observation_coverage_seconds` | Consecutive valid server-sample intervals admitted to the estimate |
| `joint_observation_coverage_seconds` | Admitted intervals with valid server and NVML samples |
| `cpu_core_seconds` | Client process CPU delta + persistent worker process CPU deltas |
| `cpu_core_seconds_per_completed_trajectory` | That measured CPU cost / completions; failed work stays in the numerator |

Interpolation admits intervals only when adjacent samples are no farther apart
than three configured collection periods. It excludes scrape windows and does
not fill leading/trailing gaps. Missing series yield null, not zero. Valid zero
candidate time can still miss short gaps; a positive estimate is not proof that
the GPU was continuously idle between samples. Telemetry refresh cadence and
collection cadence differ. Record coverage alongside every reported estimate.

CPU accounting uses `time.process_time()` around the measured window for the
client and IPC snapshots of each worker's process clock. This counts actual
scheduled core time (including worker IPC/serialization), not worker wall-time
multiplied by worker count. Snapshot boundary skew is included. Startup/preload
is excluded. A killed or replaced worker invalidates the full CPU total; callable
CPU subtotals remain diagnostic. **vLLM/server CPU and other processes are not
included.** This field must be labelled client + tool CPU cost, not machine total.

`--dedicated-gpu` is a declaration on the low-level run CLI. The experiment runner
checks the selected physical GPU and foreign compute processes; no software can
guarantee a future tenant/job will not intrude. GPU allocation time excludes
model loading, warmup, artifact export and shutdown, so it is not rental cost.

## Policy evidence and profiling

Every `cpu_selected` event records model supply, queue depth, predicted cost,
whether the estimate is known, the FIFO-oldest job, decision reason and whether
selection reordered work. Timing both policies includes the same observer and
event overhead. The estimator does not learn from future work or a different
policy arm. If most tools are unseen or no decision changes, report it.

NVTX is optional and off for the throughput series. `Profiler` uses explicit
start/end handles so overlapping asyncio requests do not corrupt thread-local
range stacks. Stable messages are `MODEL_WAIT`, `MODEL_REQUEST`, `TOOL_QUEUE`
and `TOOL_EXEC` (plus grading where labels exist). Numeric payloads map to
`nvtx_identity` in parent events; worker payloads equal CPU job IDs. Worker
`TOOL_EXEC` spans bracket actual callables. Requests still mark HTTP, not kernels.
`RUN_MEASUREMENT` marks the timed client window, excluding warmup and teardown.

Follow [the rental guide](docs/SINGLE_GPU.md) to launch Nsight around both vLLM
and AsyncRoll descendants. Profiling only the HTTP client cannot reveal server
CUDA execution. Preserve actual selected attention backend, compilation and
completed graph-capture log evidence. Requested flags alone are insufficient.
Relevant references: [NVTX process ranges](https://nvidia.github.io/NVTX/python/reference.html)
and [Nsight Systems CLI](https://docs.nvidia.com/nsight-systems/UserGuide/).
