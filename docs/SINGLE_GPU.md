# Run on one rented RTX 5090

Status: **prepared and locally tested; real GPU qualification is pending**.
Use Linux on the dedicated rented machine. Run commands from this repository
root. The launcher does not install drivers, modify a shared environment,
download a model automatically, or connect to a previous server.

## 1. Prepare the existing serving environment

Prefer the rental image's working PyTorch/vLLM environment. Confirm that its
Python, vLLM executable and installed CUDA runtime belong to the same environment
before changing packages. A specific untested vLLM/driver combination is not
certified by this repository. If the image is incompatible, resolve that once
before qualification and then freeze it.

```bash
python -m pip install -e '.[telemetry,profiling,figures]'
python -m unittest discover -s tests -v
nvidia-smi
```

Only one selected GPU is visible to the serving child. The launcher verifies
its physical index and UUID, then passes the verified numeric index to serving
for compatibility with vLLM releases that parse `CUDA_VISIBLE_DEVICES` as an
integer. NVML uses the same physical index. It refuses busy or
non-5090 devices. Do not start unrelated GPU jobs during the experiment.

Default fixed allocation: 2 CPU workers, 16 active trajectories, 4 inflight
model requests; vLLM tensor parallelism 1, BF16, context 4096, maximum 16
sequences, prefix caching and chunked prefill enabled. These are starting
protocol settings, not claimed optimal settings. Inspect
[the configuration](../experiments/single_5090.json) before qualification.

## 2. Prepare pinned inputs

The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
can fetch the frozen revisions. Install `huggingface_hub` only if `hf` is absent.
An already downloaded, verified copy may be transferred instead.

```bash
hf download Qwen/Qwen2.5-Math-7B-Instruct \
  --revision ef9926d75ab1d54532f6a30dd5e760355eb9aa4d \
  --local-dir data/model
hf download CHJ0417/ToolMATH ToolMATH.json function_ToolMATH.tar.gz \
  --repo-type dataset --revision f439a4af8dddbf061246ea9d68f6e977b18cbece \
  --local-dir data/toolmath
tar -tzf data/toolmath/function_ToolMATH.tar.gz
tar -xzf data/toolmath/function_ToolMATH.tar.gz -C data/toolmath
python -m asyncroll.cli convert-toolmath \
  --source data/toolmath/ToolMATH.json \
  --functions-dir data/toolmath/function_ToolMATH \
  --output data/toolmath/selected-32.jsonl --limit 32 --seed 20260927
```

Inspect the selected Python tools before execution. The archive is trusted
research code; workers are not a security sandbox. The converter records the
source SHA256, selection seed and indices. No solution text becomes a fabricated
answer label. The first eight selected tasks are used for qualification.

```bash
export ASYNCROLL_MODEL_PATH="$PWD/data/model"
export ASYNCROLL_WORKLOAD_PATH="$PWD/data/toolmath/selected-32.jsonl"
export ASYNCROLL_TOOL_ROOT="$PWD/data/toolmath/function_ToolMATH"
python -m asyncroll.experiment plan
```

`plan` validates inputs and prints commands without launching processes or
probing a GPU. Large weight hashes remain pending at this step. Before each
live phase, all 14 files, including four weight shards, are checked against
[the pinned digest manifest](../src/asyncroll/model_files.json). Directory
names alone never prove the model revision. Full hashing reads about 15.2 GB;
this happens before GPU startup and outside the benchmark clock.

## 3. Run the bounded experiment

Every output directory must be new. The three phases belong to one experiment;
there is no hyperparameter sweep or optimization ladder.

```bash
python -m asyncroll.experiment run --phase qualification --output runs/qualification
python -m asyncroll.experiment run --phase opportunity \
  --qualification-receipt runs/qualification --output runs/opportunity
python -m asyncroll.experiment run --phase comparison \
  --qualification-receipt runs/qualification \
  --opportunity-receipt runs/opportunity --output runs/comparison
```

Proceed only when the preceding receipt passes. Qualification requires 8/8
valid completions with actual tool use, available server/NVML telemetry and
verified serving startup. The opportunity screen uses the same fixed 32 tasks:
it requires meaningful CPU queueing and observed CPU-related idle opportunity
with adequate coverage. `stopped_no_cpu_opportunity` is a preserved negative
screen, not permission to add sleeps or search for favorable tasks.

Comparison consists of three paired full runs, with orders FIFO/AsyncRoll,
AsyncRoll/FIFO, FIFO/AsyncRoll. Each pair uses the same decoding seed. Order is
alternating, with a 2:1 first-arm imbalance; it is not perfectly balanced.
Each arm starts a fresh server and identical client warmup. Sampling, trace
writing and concurrency settings are identical; NVTX is off for these arms.

The launcher verifies actual selected attention backend and completed graph
capture in startup logs. It retains compilation evidence and fails visibly
when log formats cannot establish the audit. Automatic kernel choice remains
with [vLLM](https://docs.vllm.ai/en/stable/cli/serve/); no FlashAttention version
is forced. Source/dependency/input fingerprints and GPU identity must match
the prerequisite receipts. New code or environment changes require a new
qualification, not reuse of an old pass.

Deadlines are explicit in the config: server startup up to 30 minutes, client
arm up to 60 minutes, shutdown grace 30 seconds. These are timeout ceilings,
not estimated runtimes. Start with qualification and inspect its measured
startup/warmup durations before committing rental time to the remaining arms.
Ctrl+C triggers cleanup of owned child groups; receipts preserve interruption.
Hard host termination cannot guarantee orderly cleanup.

## 4. Capture one diagnostic trace per policy

Install a compatible Nsight Systems CLI on the rental image and check
`nsys --version`. Wrap the **experiment launcher**, which creates both the
vLLM server and AsyncRoll children. Wrapping only an HTTP client would omit
server CUDA work. The launcher uses spawned vLLM workers in all phases.

```bash
mkdir -p runs/traces
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=runs/traces/fifo \
  python -m asyncroll.experiment profile --profile-policy fifo \
  --qualification-receipt runs/qualification --output runs/profile-fifo
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=runs/traces/asyncroll \
  python -m asyncroll.experiment profile --profile-policy asyncroll \
  --qualification-receipt runs/qualification --output runs/profile-asyncroll
```

These settings collect CUDA/NVTX/OS runtime activity without CPU instruction
sampling or context-switch collection. They do not change system perf
permissions. Inspect `RUN_MEASUREMENT`, model request/wait, tool queue and actual
worker execution ranges against CUDA activity. Keep `.nsys-rep` files beside
the diagnostic receipts; a completed Python arm alone does not verify that
Nsight captured usable CUDA events. See the
[Nsight guide](https://docs.nvidia.com/nsight-systems/UserGuide/).

Diagnostic profiling remains available after a qualified but negative
opportunity screen. It never unlocks formal comparison by itself. Do not pool
profile timings with the unprofiled comparison.

## 5. Read the result

```bash
python -m asyncroll.report --comparison runs/comparison/comparison.json \
  --output runs/report --figure
```

Start with `REPORT.md` and `aggregate.json`, then the three-panel `comparison.pdf`.
Use per-arm `summary.json` and `timeline.html` to explain the result; raw events,
startup logs and Nsight traces come afterwards. `experiment.json` is the phase
receipt and each `arm.json` records commands, owned PIDs and terminal status.

Read [metric semantics](../MEASUREMENT.md) before interpretation. Completed
trajectories are not necessarily correct answers. CPU cost covers client and
tool workers, excluding serving CPU. GPU-hour throughput covers the measured
allocation window, excluding setup. Sampled starvation is not established
recoverable GPU time. Runs, caches, private machine identifiers and large traces
stay local under ignored `runs/` and `data/`.
