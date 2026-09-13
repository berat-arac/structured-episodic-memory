# SEM Generalization + Continual Learning Summary
## Protocol
- Physics generalization uses the existing 40-run checkpoint. Frozen tests disable exploration and do not update learned tables. Adaptive tests use the same checkpoint and sparse outcome updates. The physics shift is not explicitly signaled to SEM.
- Abstraction transfer ablation disables aim residuals (`aim_weight=0`) and compares full overlapping survival keys, only the most-specific key, and only the coarser keys. Learning is frozen.
- Motor remap is A1 normal actuator -> B externally inverted command sign -> A2 normal actuator. Same serve RNG and same coach are reused across phases; SEM is not told about the switch and memory is never reset.
## Paired physics generalization, 6 seeds
| Condition | Frozen hit | Adaptive hit | Adaptive early | Adaptive late | Exact specific key seen | Exact unseen + coarse seen |
|---|---:|---:|---:|---:|---:|---:|
| in_distribution | 0.971 | 0.981 | 0.975 | 0.958 | 0.813 | 0.187 |
| tiny_paddle | 0.995 | 1.000 | 1.000 | 1.000 | 0.658 | 0.342 |
| very_fast_ball | 0.938 | 0.933 | 0.950 | 0.942 | 0.771 | 0.229 |
| very_steep_bounce | 0.974 | 0.979 | 0.983 | 0.975 | 0.726 | 0.274 |
| very_tall_field | 0.981 | 0.971 | 0.992 | 0.958 | 0.767 | 0.233 |
| slow_actuator | 0.960 | 0.976 | 0.983 | 0.958 | 0.809 | 0.191 |
| hard_compound | 0.921 | 0.952 | 0.967 | 0.967 | 0.572 | 0.428 |

Hard compound changes field height, paddle height, SEM max speed, base/max ball speed, acceleration and bounce-angle scale simultaneously. Frozen hit rate remains 0.921.
## Abstraction transfer ablation, 6 seeds
| Physics | Memory view | Hit rate | Hit rate on episodes containing unseen specific keys |
|---|---|---:|---:|
| id | full_overlap | 0.983 | 0.981 |
| id | specific_only | 0.681 | 0.641 |
| id | coarse_only | 0.979 | 0.975 |
| hard_compound | full_overlap | 0.945 | 0.944 |
| hard_compound | specific_only | 0.564 | 0.536 |
| hard_compound | coarse_only | 0.929 | 0.926 |

The coarse-only memory retains almost all full performance, while specific-only collapses. This supports transfer through overlapping abstractions rather than exact-state lookup alone.
## Hidden motor-remap reversal, 6 seeds
- A1 normal: early 0.983, late 0.975.
- B inverted actuator: early 0.175, late 0.733; recovery +0.558; late > early in 6/6 seeds.
- A2 normal restored: early 0.492, late 0.950; recovery +0.458; late > early in 6/6 seeds.

This supports continual re-learning under an unannounced action-consequence reversal. It does not show instant remapping: inverted performance remains below the original mapping after 220 outcomes.
## Core validation
- status: **PASS**
- fresh motor miss reduction: 0.722
- pretrained held-out hit rate: 0.979
- contact intent/actual correlation: 0.400
- analytical future-path solver audit: PASS
## Claim boundary
- SEM still receives engineered current-frame state variables and feature bins. This is not pixel-level blank-slate learning.
- These tests support learned sparse-feedback motor control, overlapping abstraction transfer, context-dependent strategy, fast contingency adaptation, goal-conditioned aiming, and partial continual re-learning after motor remapping.
- Slow memory currently has only weak stabilisation evidence and should not be presented as a major validated advantage.
