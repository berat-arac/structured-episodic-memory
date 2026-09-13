# SEM Motor Retention / Forgetting Summary

## Protocol

A pretrained SEM is exposed to a hidden actuator remap: A1 normal, B1 inverted, A2 normal, B2 inverted. SEM is never told when the mapping changes and memory is not reset between phases. Because mapping identity is not observable, this test measures **re-learning savings / retention**, not simultaneous frozen recall of two explicit policies.

## 1. A -> B -> A -> B retention, 6 paired seeds

| Metric | B1 first inverted | B2 second inverted |
|---|---:|---:|
| Early 20 hit rate | 17.5% | 30.0% |
| First 50 hit rate | 24.3% | 40.7% |
| First 100 hit rate | 37.8% | 51.7% |
| Late 20 hit rate | 74.2% | 77.5% |
| Mean latency to 75% rolling-20 | 158.8 | 91.7 |

B2 reached the 75% criterion faster than B1 in **6/6 seeds**. Normal mapping recovery A2 was fast: mean latency **31.7** outcomes, late-20 hit rate **98.3%**.

## 2. Matched-experience control

The control SEM receives the same amount of pre-target experience but has never experienced inversion before its late B phase.

| Metric | Retention B2 | Matched late first-B control |
|---|---:|---:|
| First 50 hit rate | 40.7% | 23.0% |
| First 100 hit rate | 51.7% | 36.5% |
| Mean 75% latency | 91.7 | 124.6 among adapted controls |

B2 was higher in the first 50 and first 100 outcomes in **6/6 seeds**. Of five runs where both reached the latency threshold, B2 was faster in **4** and tied in **1**; one extra control never reached threshold within the phase.

## 3. Interference / forgetting

| Normal outcomes between B1 and B2 | B2 early 20 | B2 first 50 | B2 first 100 | 75% latency |
|---|---:|---:|---:|---:|
| 40 | 37.5% | 49.5% | 60.8% | 73.0 |
| 220 | 22.5% | 34.5% | 46.5% | 104.2 |
| 440 | 18.8% | 33.0% | 45.5% | 101.2 |

The immediate savings is strongest after a short gap and weakens after longer normal-mapping interference. The 220 -> 440 range does not show another large collapse in these seeds, so a residual savings signal remains, but this is a behavioral description rather than a proven internal mechanism.

## 4. Same-history timescale branch probe

All branches share the **exact same A1 -> B1 -> A2 history**. At the B2 boundary the agent is cloned; only the survival-motor value readout changes.

| B2 readout | First 50 | First 100 | Late 20 | Adapted | Mean 75% latency |
|---|---:|---:|---:|---:|---:|
| Full fast+slow | 45.3% | 56.5% | 80.0% | 6/6 | 77.0 |
| Fast-only | 48.0% | 62.2% | 89.2% | 6/6 | 67.7 |
| Slow-only | 33.3% | 38.2% | 63.3% | 2/6 | 170.0 |

This probe **does not support slow memory as the main retention carrier**. Fast-only was at least as effective as the full readout on B2, while slow-only adapted in only 2/6 runs. The observed savings therefore appears to live largely in distributed fast-value state / incompletely overwritten overlapping state-action abstractions rather than in the slow estimate alone.

## Core validation

After adding these benchmark scripts, `run_validation.py` remains **PASS**. No SEM core decision rule was changed for these retention experiments.
