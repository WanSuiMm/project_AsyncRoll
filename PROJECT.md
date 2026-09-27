# AsyncRoll

## Research question

When a tool-using agent shares limited CPU workers with one model GPU, does
prioritizing tools whose completion produces a GPU-ready request improve
completed trajectories per GPU-hour over ordinary asynchronous FIFO?

## Current status (2026-09-27)

Initial runtime implementation and deterministic smoke workload exist. No live
vLLM or ToolMATH trajectory has been run. No performance or accuracy result is
claimed. This first version is inference only; it does not train with RL.
The attempted acquisition of the proposed instruction-tuned model did not
complete; no model-availability claim is made by this repository.

## Frozen first screen

1. Convert a small ToolMATH subset and run the same workload, model, decoding
   settings, GPU request limit, and CPU worker count under sync, FIFO, and
   GPU-first policies. Retain failed trajectories in the denominator.
2. Inspect actual tool-call rate, tool duration, queue delay, model request
   duration, and output validity. Compare FIFO to GPU-first at the same CPU
   allocation. Use completed trajectories per elapsed hour as the primary
   metric, with exact accuracy only where an explicit answer label exists.
3. Stop the scheduling claim if tools rarely run, CPU work rarely queues, or
   FIFO leaves no measurable headroom. Artificial latency belongs in a separate
   stress-test protocol and cannot establish a real-workload gain.

The runtime currently logs client-side request events, not true GPU occupancy.
A live screen must add server-side vLLM / GPU telemetry before attributing a
gain to GPU bubble recovery.

## Entry commands

Run from this repository's root:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
asyncroll run --workload examples/smoke.jsonl --backend scripted --policy fifo --output runs/smoke-fifo
```

See `README.md` for ToolMATH conversion and live vLLM usage.
