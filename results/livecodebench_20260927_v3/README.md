# LiveCodeBench v1 screen

`experiment.json` is the compact, sanitized control-plane receipt from the
completed v1 qualification and opportunity screen. The run stopped because
joint telemetry coverage was 0.7607 against the original 0.80 gate. Four other
CPU-opportunity checks passed. Large per-task summaries were excluded because
they duplicated hidden tests; the decision-relevant aggregates are retained in
this receipt.

This result contains no FIFO versus AsyncRoll comparison and supports no
speedup claim. Protocol v2 preserves this receipt and removes telemetry coverage
as a hard stop before launching the paired comparison.
