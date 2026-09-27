# Incremental review handoff

- Prior evidence head: `797b6fe94817aa65c607c48def40f20b356346bb`
- New code/evidence head: `496f4c2e1edf66ab8781aa88140bf0a2f0e13e79`
- This file is a later documentation-only commit; review the evidence head above.
- Validation date: 2026-09-27. Release: 0.3.0.

## What changed

1. The action protocol now uses typed tool parameters, requires a tool action on
   the first turn, and uses bounded second-turn final output. The live server
   used xgrammar with arbitrary whitespace disabled.
2. The dedicated RTX 5090 stack qualified: 8/8 trajectories completed, all
   structured actions were valid, all trajectories used a tool, and vLLM/NVML
   telemetry was complete.
3. The frozen 32-task FIFO opportunity screen completed 31 tasks and retained
   one division-by-zero tool failure. CPU-blocked time and CPU-related idle
   candidate time were zero, maximum CPU queue depth was one, and tool queue
   p95 was 1.707 ms.
4. The pre-registered stop rule fired. The FIFO versus AsyncRoll comparison was
   not run; the repository makes no speedup or causal bubble-recovery claim.
5. `RESULTS.md` and `results/single_5090_20260927.json` publish the sanitized
   aggregate. Raw logs and receipts remain local because they contain deployment
   paths, process identifiers and device identifiers.

## Claim boundaries retained

The result establishes that the live inference/tool/telemetry stack ran end to
end and that this fixed real-tool workload offered no qualifying CPU scheduling
opportunity. It does not establish answer accuracy, comparative performance,
recoverable GPU time, or an AsyncRoll benefit. The reported 941.13 completions
per allocated GPU-hour is a descriptive single-arm measurement whose full
workload quality gate failed.

## New verification evidence

| Check | Observed result | Boundary |
| --- | --- | --- |
| Unit/integration suite | 52 tests passed | Local behavior checks |
| Qualification | 8/8 complete, 100% valid actions and tool use | Live dedicated GPU |
| Opportunity screen | 31/32 complete; no CPU opportunity gate passed | One FIFO arm only |
| Telemetry | 591 screen samples, zero collection errors | Sampled observation |
| Documentation | Local Markdown targets present; result JSON parses | Raw receipts excluded |
| Delivery audit | Whitespace and sensitive-identifier scans passed | Sanitized aggregate only |

## Minimal reading order and review questions

1. [RESULTS.md](RESULTS.md) and
   [single_5090_20260927.json](results/single_5090_20260927.json): do the gate
   values justify stopping before comparison and keep the negative claim bounded?
2. [PROTOCOL.md](PROTOCOL.md) and
   [single_5090.json](experiments/single_5090.json): was the stop decision applied
   consistently with the frozen protocol?
3. [model.py](src/asyncroll/model.py) and [test_model.py](tests/test_model.py):
   is the typed two-turn action protocol sufficiently constrained without hiding
   model/tool failures?
4. [MEASUREMENT.md](MEASUREMENT.md): are telemetry coverage and sampled-idle
   limitations stated clearly enough for a systems reader?
