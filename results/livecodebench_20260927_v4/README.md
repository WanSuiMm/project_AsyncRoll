# LiveCodeBench v2 formal comparison

Status: **complete**. All six comparison arms completed 219/219 trajectories.

| Pair | FIFO trajectories/GPU-hour | AsyncRoll trajectories/GPU-hour | Relative change |
| --- | ---: | ---: | ---: |
| 1 | 5681.92 | 5760.43 | +1.38% |
| 2 | 6050.14 | 5874.71 | -2.90% |
| 3 | 5968.88 | 5992.84 | +0.40% |
| Paired mean | | | **-0.37%** |

The descriptive paired 95% t interval is [-5.94%, +5.20%]. Pooled elapsed time
was 401.15 s for FIFO and 402.62 s for AsyncRoll, equivalent to a -0.37%
AsyncRoll throughput change. Mean exact pass rates were 28.46% and 29.07%,
respectively.

The result does not support a throughput improvement claim. `experiment.json`
is the sanitized canonical control-plane receipt containing qualification,
opportunity-screen, comparison, telemetry, and configuration aggregates.
