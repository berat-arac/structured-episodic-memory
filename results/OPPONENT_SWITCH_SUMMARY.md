# SEM hidden opponent-behaviour switch test

Protocol: A1 (weak_up) -> B (weak_down) -> A2 (weak_up), 100 strategy outcomes per phase, 6 independent seeds.

SEM receives no phase label, no switch signal, and no opponent-mode input. Pretrained motor/aim memory is kept. Opponent/strategy memory is reset once at test start and is **not** reset between phases.

## Aggregate result

| Phase | First 20 target intents | Last 20 target intents | Mean adaptation latency |
|---|---:|---:|---:|
| A1 weak_up | 65.8% | 94.2% | 20.8 episodes |
| B weak_down | 45.8% | 87.5% | 23.2 episodes |
| A2 weak_up | 53.3% | 93.3% | 23.5 episodes |

All 6/6 seeds crossed the predefined rolling adaptation threshold in every phase.

The controlled reward landscape was strong enough to be observable but was not exposed to SEM. Across phases, shots that actually landed in the currently vulnerable trajectory family caused opponent misses about 42-48% of the time, versus about 4-7% for other trajectory families.

Interpretation: after a hidden contingency switch, SEM initially carries over the previous strategy, then online fast/slow outcome memory shifts the contact-intent distribution toward the new vulnerability. When the old contingency returns, preference shifts back without resetting memory.

Limitations: this is a deliberately controlled non-stationary benchmark opponent, not a human model. It demonstrates online adaptation to changed response statistics, not opponent identity recognition or general game-theoretic intelligence.
