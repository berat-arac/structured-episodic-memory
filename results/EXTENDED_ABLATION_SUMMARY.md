# SEM Extended Ablation Summary

This file records only tests that were actually run in this package. Negative or mixed results are retained rather than hidden.

## 1. Context-dependent conflicting contingency benchmark

Protocol: paired full vs global-only strategy memory. Each trial exposes only the opponent paddle's vertical context cue. The hidden response contingency is opposite across the two balanced contexts: high cue rewards positive/downward contact; low cue rewards negative/upward contact. The mapping is never exposed to SEM.

8 seeds, 320 trials per condition.

- Full context memory, last 100 target-intent fraction: **0.9125**
- Global-only, last 100 target-intent fraction: **0.4675**
- Full context memory, preferred target fraction: **0.9550**
- Global-only, preferred target fraction: **0.52625**
- Full outperformed global-only on target-intent in **8/8 seeds**.
- Target-family actual shots produced opponent misses at about **0.4801**; non-target family in this controlled benchmark: **0.0**.

Interpretation: context-specific memory is causally useful when the same action has opposite value in different observable contexts. The earlier global hidden-switch benchmark did not require context strongly enough; this benchmark does.

## 2. Slow-memory ablation on long A->B->A switches

8 paired seeds, 80 outcomes per phase. Full fast+slow vs fast-only.

- A1 adaptation latency: full **19.6**, fast-only **16.4**
- B adaptation latency: full **31.8**, fast-only **25.3**
- A2 adaptation latency: full **22.9**, fast-only **26.5**

Result is mixed. Fast-only adapts faster to the long B switch. Full is somewhat better on mean A2 reacquisition, but the paired seed result is not consistent enough for a strong claim.

Interpretation: the current fixed slow-memory blend is **not validated as a major performance component** on long switches.

## 3. Slow-memory temporary-shock stability test

8 paired seeds. 100-outcome stable A, 10-outcome temporary B shock, 50-outcome A recovery.

- Recovery first-20 target fraction: full **0.7313**, fast-only **0.7438**
- Recovery last-20 target fraction: full **0.9000**, fast-only **0.8438**
- Full vs fast-only first-recovery comparison split **4 vs 4 seeds**.

Interpretation: no strong early stability benefit. A small late recovery advantage exists, but this is not strong enough to claim that slow memory is necessary.

## 4. Isolated noisy stationary strategy-memory test

Synthetic component isolation only: fixed context, same SEM strategy value updates/exploration, 64 seeds x 4000 steps. Target arm success p=0.58, all others p=0.52.

- Target preferred fraction in final 1000: full **0.2406**, fast-only **0.2199**
- Preferred-action switches / 1000: full **124.17**, fast-only **134.71**
- Mean reward difference: only **+0.00072** for full.

Interpretation: slow memory shows a **small smoothing/stability bias**, not a large reward improvement. Keep its claim modest unless redesigned and revalidated.

## 5. Vertical-velocity ablation on ordinary fresh Pong learning

8 paired seeds x 140 outcomes. Survival motor only; contact aiming disabled.

- Full state late hit rate: **0.89375**
- Vertical-velocity removed late hit rate: **0.921875**
- Both reduced misses by about **72.5%**.

Interpretation: ordinary Pong does **not** prove that current vertical velocity is required. Position/relative-position/distance features can be sufficient, and velocity bins can add state fragmentation.

## 6. Direction-ambiguity benchmark

Controlled test designed specifically to remove that confound. Every trial begins from essentially the same ball x/y and paddle geometry, very close to SEM. Only vertical velocity sign differs: `ball_vy = +/-700`, `ball_vx = 650`. Sparse hit/miss reward only. Full state vs vertical-velocity-neutralized state.

8 paired seeds x 320 episodes.

- Full late hit rate: **0.9625**
- No-vy late hit rate: **0.59375**
- Full first-action direction accuracy: **0.9531**
- No-vy first-action direction accuracy: **0.3391**
- Full had higher late hit rate in **8/8 seeds**.

Interpretation: vertical velocity is not a hard-coded action rule. When current geometry is deliberately ambiguous, the learned policy uses the velocity sensor to disambiguate which motor response works. In ordinary Pong, other state features often make the same information recoverable without it.

## 7. Core regression validation after all benchmark additions

Core code was not changed by these new benchmark scripts. `run_validation.py` was rerun and returned **PASS**.

- Fresh motor learning miss reduction: **0.7222**
- Pretrained held-out hit rate: **0.9792**
- Contact-intent correlation: **0.3996**
- Checkpoint roundtrip: PASS
- Future-path solver audit: PASS

## Current evidence boundary

Supported:
- sparse-feedback motor learning without an analytical future trajectory solver;
- bounded temporal credit improves motor learning;
- fast memory materially improves adaptation to changed response contingencies;
- context-specific strategy memory is necessary when action value reverses across observable contexts;
- aim memory causally links contact intent to actual contact;
- instantaneous vertical-velocity information is learned and useful when geometry alone is ambiguous.

Not strongly supported yet:
- the present slow-memory blend as a necessary high-impact component;
- claims of blank-slate pixel learning;
- human identity recognition;
- biological realism.
