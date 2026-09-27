# Bounded-concurrency v6 formal comparison

Status: **complete**. The activation gate and all six formal arms completed.

| Pair | FIFO trajectories/GPU-hour | AsyncRoll trajectories/GPU-hour | Relative change | AsyncRoll reorders |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2978.32 | 3005.77 | +0.92% | 26 |
| 2 | 3008.43 | 3039.48 | +1.03% | 34 |
| 3 | 2950.36 | 3020.20 | +2.37% | 28 |
| Paired mean | | | **+1.44%** | |

The descriptive paired 95% t interval is [-0.56%, +3.44%]. Pooled totals were
654 FIFO completions in 790.38 s and 655 AsyncRoll completions in 780.34 s,
equivalent to +1.44% throughput. Mean exact pass rate differed by +0.26
percentage points.

The activation arm recorded 38 eligible decisions, 19 reorders and +0.559
same-problem repair prediction correlation. The formal AsyncRoll arms continued
to reorder and had positive repair correlations. The result establishes that
the scheduler mechanism activates and reports a small consistent point estimate;
three pairs do not establish a nonzero population effect or a 20% gain.

`experiment.json` is the sanitized canonical receipt.
