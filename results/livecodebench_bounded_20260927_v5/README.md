# Bounded-concurrency v3 activation screen

Status: **stopped_scheduler_not_activated**.

| Stage | Completed | Wall time | Pass rate |
| --- | ---: | ---: | ---: |
| Qualification FIFO | 8/8 | 10.23 s | 12.50% |
| Opportunity FIFO | 127/128 | 158.21 s | 25.98% |
| Activation AsyncRoll | 128/128 | 167.10 s | 25.00% |

The activation arm recorded 224 distinct feature keys and +0.105 log-duration
prediction correlation, but only two eligible scheduling decisions and one
reorder. Its nominal 50% activation rate is therefore a small-denominator
statistic and did not pass the minimum counts. The three-pair comparison did not
run. `experiment.json` is the sanitized canonical receipt.
