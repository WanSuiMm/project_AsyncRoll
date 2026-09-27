# AsyncRoll: measured scheduling for tool-using agents

AsyncRoll investigates whether CPU scheduling can keep an inference engine fed
while agent trajectories alternate between model requests and Python tools.
The current version provides an inference-only runtime, bounded concurrency,
warm workers, a pressure-aware CPU policy, NVTX instrumentation and a resource
timeline. The completed 32-active LiveCodeBench comparison found -0.37% mean
AsyncRoll throughput change; its AsyncRoll arms made zero reorder decisions and
therefore tested a FIFO-equivalent policy. That result remains frozen.

The v3 engineering optimization fixed a latency-sensitive operating point at 8
active trajectories, 4 inflight model requests and 2 CPU workers. Its per-job
predictor distinguished 224 feature keys with positive log-time correlation,
but only 2 dispatches became eligible and 1 reordered. The activation gate
stopped the run before comparison, so no positive gain is claimed. See
[PROTOCOL_LIVECODEBENCH.md](PROTOCOL_LIVECODEBENCH.md).

V6 kept the same operating point and made the supply trigger proactive:
reordering becomes eligible below the four-request client capacity. Repair jobs
use their own first-evaluation wall time as an online prior, with ridge fallback
for initial evaluations. It completed all six formal arms with paired throughput
changes of +0.92%, +1.03% and +2.37% (+1.44% mean; descriptive 95% interval
[-0.56%, +3.44%]). This is a small consistent point estimate, not a 20% claim.

## Start here

1. [RESULTS.md](RESULTS.md): canonical result, gates and claim boundary.
2. [PROJECT.md](PROJECT.md): question, scope and current decision.
3. [PROTOCOL.md](PROTOCOL.md): single experiment, estimand and stop rules.
4. [docs/SINGLE_GPU.md](docs/SINGLE_GPU.md): rental setup, staged execution and Nsight.
5. [MEASUREMENT.md](MEASUREMENT.md): metric definitions and confounders.
6. [GPT_CONTEXT.md](GPT_CONTEXT.md): implementation map and evidence routing.
7. [PROTOCOL_LIVECODEBENCH.md](PROTOCOL_LIVECODEBENCH.md): new code-agent experiment.

## Local check

From this repository root, with Python 3.10 or newer:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m asyncroll.cli run --workload examples/smoke.jsonl --backend scripted --policy fifo --cpu-workers 2 --max-active-trajectories 8 --max-inflight-model-requests 4 --output runs/smoke-fifo
```

Use a new output directory each time. Outputs are `run.json`, `summary.json`,
line-buffered `events.jsonl`, Chrome/Perfetto `trace.json`, and standalone
`timeline.html`. Scripted timings check execution behavior only. A complete
run receipt means the harness finished; inspect failure counts and telemetry
coverage before interpreting it.

## Independent concurrency controls

| Option | Controls |
| --- | --- |
| `--max-active-trajectories` | Admitted trajectories across model/CPU stages |
| `--max-inflight-model-requests` | Client model calls; not vLLM batch size |
| `--cpu-workers` | Persistent Python tool/grading processes |

The three limits remain independent knobs, held fixed for the main comparison.
`asyncroll` checks queued + inflight client model requests at each CPU dispatch.
Below `--starvation-threshold` (default 1), it predicts each tool continuation
from test count, test-input bytes, generated-code bytes and repair status using
an online ridge model. The score subtracts `--aging-weight` per queued second;
after `--aging-seconds` (default 30), the oldest expired job wins. This is
non-preemptive. Every prediction, reason, eligible decision and reorder is logged.

`sync` admits one trajectory at a time. `fifo` uses CPU arrival order.
`tool_first` gives queued tool calls priority over queued terminal grading.
It is non-preemptive and does not predict remaining tool duration or inspect GPU
pressure. **On an unlabelled workload, FIFO and tool-first have the same CPU
priority ordering**, since there is no grading work. The old `gpu_first` name
and `--gpu-slots` flag remain compatibility aliases.

## Live model and ToolMATH

The [single-GPU runner](docs/SINGLE_GPU.md) owns a vLLM server per arm and records
its actual backend and graph evidence. Manual existing-server runs below remain
available for diagnostics. The backend uses a persistent HTTPX AsyncClient
and requests JSON-schema output by default. Schema compliance is not proof
that [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
can use this action protocol correctly; native TIR qualification remains open.
An unsupported schema response fails visibly; `--no-structured-output` is an
explicit separate configuration, never an automatic fallback.

Inspect the Python functions in [ToolMATH](https://huggingface.co/datasets/CHJ0417/ToolMATH)
before execution, then convert a reproducible random subset:

```powershell
python -m asyncroll.cli convert-toolmath --source data/ToolMATH.json --functions-dir data/function_ToolMATH --output data/toolmath-small.jsonl --limit 100 --seed 0
python -m pip install -e ".[telemetry]"
python -m asyncroll.cli run --workload data/toolmath-small.jsonl --tool-root data/function_ToolMATH --backend vllm --model Qwen/Qwen2.5-Math-7B-Instruct --policy fifo --cpu-workers 2 --max-active-trajectories 16 --max-inflight-model-requests 4 --warmup-requests 8 --nvml-device 0 --output runs/toolmath-fifo-001
```

Run this client **on the GPU server** when collecting NVML: `--nvml-device`
selects a physical NVML device on the client's host, not a remote or
`CUDA_VISIBLE_DEVICES`-remapped index. Check that it matches the vLLM device.
The default metrics URL is `http://localhost:8000/metrics`; override it using
`--metrics-url` and the server address with `--endpoint`. Missing metrics and
NVML errors stay visible. `--no-telemetry` is for explicit diagnostic runs.

The converter preserves source indices, sample seed and available difficulty/
category metadata. Function paths are relative to `--tool-root`; no answer
labels are fabricated from source solutions. Only explicit `reference_answer`
values produce grading jobs, using normalized string equality.

All workers preload the selected trusted modules and execute a dummy call
before timed work. Module state persists within each worker; use stateless
tools for policy comparisons. `--tool-timeout` terminates a stuck worker;
its replacement is preloaded and recovery overhead remains in the measured
run. This is not a security sandbox or a descendant-process cleanup mechanism.

See [MEASUREMENT.md](MEASUREMENT.md) before using any throughput number.

## LiveCodeBench one-repair workload

Convert a pinned full-data export to a deterministic stdin-only subset:

```powershell
python -m asyncroll.cli convert-livecodebench --source data/livecodebench/test.jsonl --output data/livecodebench/selected-219.jsonl --limit 219 --seed 20260927 --stdin-only
```

Each trajectory submits one complete Python program to the CPU evaluator and,
after bounded failure feedback, may submit one repair. Test execution uses a
fresh temporary directory, resource limits and a low-privilege account when
launched as root. It remains a local benchmark harness rather than a hardened
security sandbox and belongs only on a disposable credential-free host.

The fixed bounded-concurrency runner is intentionally a single operating point:

```bash
python scripts/run_lcb_bounded_single_gpu.py \
  --config experiments/livecodebench_bounded_8x4.json \
  --model-path /path/to/Qwen2.5-Coder-7B-Instruct \
  --workload /path/to/selected-219.jsonl \
  --output runs/lcb-bounded-8x4
```

## One result figure

After an unprofiled comparison completes:

```bash
python -m pip install -e '.[figures]'
python -m asyncroll.report --comparison runs/comparison/comparison.json --output runs/report --figure
```

This produces a compact report and one three-panel PDF/PNG: completions per
allocated GPU-hour, sampled model starvation, and client + tool-worker CPU cost
per completion. Lines connect independent paired full runs. Missing values stay
unknown; correctness and a causal recovery claim need additional evidence.
