# SEM causal ablation summary

All ablations keep the Pong agent oracle-free. No analytical future-path solver is added.

## 1. Fast-memory ablation — hidden A->B->A opponent switch

Paired 6-seed test, 100 strategy outcomes per phase.

- Full fast+slow SEM, hidden A->B switch: 6/6 runs adapted, mean latency 23.17 outcomes, final-20 correct-intent fraction 0.875.
- Slow-only strategy memory: 4/6 runs adapted within the 100-outcome B phase. Among adapted runs, mean latency was 63.25 outcomes; final-20 correct-intent fraction was 0.558.
- In the initial stationary A1 phase there was no meaningful advantage for fast memory (full latency 20.83 vs slow-only 19.83 outcomes).

Interpretation: the fast estimate is specifically useful when the opponent-response contingency changes, rather than merely improving stationary learning.

Note: a planned 12-seed expansion exceeded the execution tool's 120 s single-command limit and is not reported as completed evidence.

## 2. Motor eligibility-trace ablation

Paired 8-seed fresh-learning test, 140 incoming outcomes per seed. Contact-aim residual was disabled in both conditions to isolate survival motor learning.

- Full bounded decaying trace: mean misses 13.75 -> 5.25 (61.8% reduction), late hit rate 0.869, improved in 8/8 seeds.
- Last-decision-only credit: mean misses 25.38 -> 12.13 (52.2% reduction), late hit rate 0.697, improved in 8/8 seeds.
- Full trace had fewer late misses in 7/8 paired seeds, with 6.875 fewer late misses on average.

Interpretation: temporal credit assignment materially improves motor learning, although the last-decision baseline can still learn something.

## 3. Contact-aim ablation

Paired 8-seed evaluation, 100 outcomes per seed, random contact intents, pretrained motor checkpoint.

- Full aim residual: mean intent/actual-contact correlation 0.395, mean absolute contact error 0.421, hit rate 0.964.
- Aim disabled (`aim_weight=0`): correlation -0.020, error 0.596, hit rate 0.986.
- Full aim produced higher intent/contact correlation in 8/8 seeds.

Interpretation: survival/interception does not depend on the aim residual, but the aim memory is causally responsible for converting a chosen contact intent into a changed physical contact distribution.

## 4. Context-memory ablation

Paired 6-seed hidden A->B->A switch test.

- Full global+context memory B-phase adaptation latency: 23.17 outcomes; final-20 correct-intent fraction 0.875.
- Global-only fast+slow memory B-phase adaptation latency: 28.83 outcomes; final-20 correct-intent fraction 0.950.
- A1 and A2 were also very similar between conditions.

Interpretation: this benchmark does **not** establish that context-specific strategy memory is necessary. Context reduced mean B-switch latency by about 5.7 outcomes, but global-only memory matched or exceeded final-phase preference quality. A richer benchmark with genuinely context-dependent opponent weaknesses is required before making a strong claim for the context layer.
