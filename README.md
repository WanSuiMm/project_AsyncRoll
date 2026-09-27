# AsyncRoll: GPU-first runtime for tool-using agents

AsyncRoll asks whether CPU tool priority can reduce the time a model GPU
waits for the next agent turn. It compares sequential trajectories, an
asynchronous FIFO CPU queue, and an asynchronous queue that prioritizes tools
which return to the GPU over final grading. The current code is a prototype;
there are no live benchmark results yet. The proposed Qwen model is not yet
available in the development environment.

## Start here

1. [PROJECT.md](PROJECT.md) states the question, first screen, and claim boundary.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md) maps the claims, components, and evidence.
3. [runtime.py](src/asyncroll/runtime.py) contains the trajectory loop and CPU policies.
4. [smoke.jsonl](examples/smoke.jsonl) shows the workload schema.

## Local smoke

From the repository root, with Python 3.10 or newer:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
asyncroll run --workload examples/smoke.jsonl --backend scripted --policy sync --output runs/smoke-sync
asyncroll run --workload examples/smoke.jsonl --backend scripted --policy fifo --output runs/smoke-fifo
asyncroll run --workload examples/smoke.jsonl --backend scripted --policy gpu_first --output runs/smoke-gpu-first
```

Each run writes `run.json`, `summary.json`, and `events.jsonl` into a new
directory. Scripted mode checks behavior only; its throughput is not a GPU
measurement. `gpu_slots` bounds in-flight model HTTP requests, not vLLM's
internal batch size.

## Live model and ToolMATH

Install and serve a compatible model separately, for example
[Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
with vLLM on a suitable GPU. For a live run, install vLLM in that environment
and start `vllm serve Qwen/Qwen2.5-Math-7B-Instruct`. The backend calls the
OpenAI-compatible chat completions endpoint and requests strict JSON actions.
This prompting protocol has **not** yet been qualified for Qwen's native TIR
format. Parse failures are recorded as failed trajectories.

Download [ToolMATH](https://huggingface.co/datasets/CHJ0417/ToolMATH)
separately, inspect the Python tool archive, and extract it.
The converter expects the main JSON array and the extracted function directory:

```powershell
asyncroll convert-toolmath --source data/ToolMATH.json --functions-dir data/function_ToolMATH --output data/toolmath-small.jsonl --limit 100
asyncroll run --workload data/toolmath-small.jsonl --backend vllm --model Qwen/Qwen2.5-Math-7B-Instruct --policy fifo --cpu-workers 2 --gpu-slots 4 --output runs/toolmath-fifo-001
```

The converter supplies each problem's named tool and implementation. It does
not manufacture a trajectory or a verified answer. ToolMATH's source solution
is not automatically treated as a final-answer label. External tool files are
imported as Python code in worker processes, so use only trusted files you
have inspected. The runtime requires the file to define a callable matching
the declared tool name; incompatible records fail visibly.

`completed_per_gpu_hour` assumes a one-GPU live run and uses wall time; it
does not prove actual GPU utilization. Exact accuracy is reported only for
workloads with explicit `reference_answer` values, using a narrow string
comparison. A future math-equivalence grader needs independent validation.
