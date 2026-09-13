# Structured Episodic Memory (SEM)

**Structured Episodic Memory (SEM)** is an experimental, non-neural Pong agent built to test how far adaptive control can be obtained from structured associative memory, overlapping state abstractions, sparse outcome credit, and online opponent-response memory.

SEM does **not** use a neural network, backpropagation, autodiff, or an analytical future-ball interception solver. It does receive engineered current-frame state variables and other explicit inductive biases; this is not pixel-level blank-slate learning.

## What is learned

The current system separates three learned functions:

- **Survival motor memory:** current state abstractions -> paddle velocity action, learned from sparse hit/miss outcomes through a bounded decision-level eligibility trace.
- **Goal-conditioned aim memory:** a residual motor preference conditioned on a chosen contact intent, learned from sparse intended-vs-actual contact error.
- **Opponent-response strategy memory:** global and context-specific values over contact zones, updated from returns and opponent misses and used to choose the next contact intent.

## Main results

The archived `v1.0` experiment set contains the following measured results:

| Test | Result |
|---|---:|
| 40-run training | 2,200 incoming outcomes, 0.958 lifetime hit rate |
| Frozen held-out evaluation | 0.974 mean hit rate across 6 seeds |
| Fresh-learning validation | 72.2% reduction in early-to-late misses |
| Eligibility trace ablation | 0.869 late hit vs 0.697 last-decision-only |
| Aim ablation | intent/contact correlation 0.395 vs -0.020 without aim memory |
| Hidden opponent switch | mean B-phase adaptation latency 23.2 outcomes |
| Context-conflict benchmark | 0.913 correct intent vs 0.468 global-only |
| Direction ambiguity | 0.963 late hit vs 0.594 without vertical velocity |
| Frozen hard compound physics shift | 0.921 hit rate |
| Hard compound abstraction ablation | 0.945 full, 0.929 coarse-only, 0.564 specific-only |
| Hidden actuator inversion | 0.175 early -> 0.733 late hit rate |
| Repeated inversion | 158.8 -> 91.7 outcomes to 75% rolling-hit criterion |

The current slow-value blend has only weak stabilization evidence and is **not** presented as a major validated advantage.

## Architecture boundary

Engineered inputs and priors include:

- `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, paddle positions and dimensions;
- hand-designed state bins and overlapping key layouts;
- the `ball_vx > 0` incoming-direction gate;
- discrete motor and contact-intent vocabularies;
- sparse reward definitions;
- actuator smoothing and switching cost;
- the current-frame contact goal-error representation;
- training coaches and curriculum.

Explicitly absent from SEM:

- future paddle-line intersection computation;
- wall-reflected future-y prediction;
- time-to-impact control;
- hand-coded `if ball is above, move up` action rules;
- neural networks or gradient learning.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Run the fixed validation suite:

```bash
python run_validation.py
```

Train a fresh 40-run checkpoint:

```bash
python train_sem.py
```

Play against the included pretrained checkpoint:

```bash
python play_human_vs_trained_sem.py
```

## Reproduce the main research experiments

```bash
python run_motor_trace_ablation.py
python run_aim_ablation.py
python run_opponent_switch_test.py
python run_fast_memory_ablation.py
python run_context_conditional_benchmark.py
python run_direction_ambiguity_benchmark.py
python run_hard_physics_generalization.py
python run_abstraction_transfer_ablation.py
python run_motor_remap_reversal.py
python run_motor_remap_retention.py
python run_motor_remap_retention_control.py
python run_motor_retention_branch_probe.py
```

Some larger suites are intentionally split into separate scripts because full multi-seed runs can take longer on CPU-only machines.

## Paper

The LaTeX source, vector figures, and compiled preview are in [`paper/`](paper/).

Paper title: **Structured Episodic Memory (SEM)**

The paper deliberately reports negative and mixed findings as well as successful ablations. In particular, ordinary Pong does not require the vertical-velocity sensor, and the current slow-memory blend is only weakly supported.

## Repository layout

```text
sem_agent.py                      main SEM agent
sem_core.py                       associative values + eligibility traces
pong_core.py                      Pong environment
coach_opponents.py                engineered training/evaluation opponents
train_sem.py                      40-run training curriculum
play_human_vs_trained_sem.py      interactive human-vs-SEM demo
run_validation.py                 core regression / claim-boundary validation
run_*.py                          research benchmark and ablation scripts
checkpoints/sem_40runs.json.gz    archived pretrained checkpoint
results/                          raw final JSON reports and summaries
paper/                            LaTeX paper source, figures, compiled preview
```

## Scientific claim boundary

A defensible summary is:

> SEM is a non-neural associative-memory Pong agent that learns paddle interception from sparse hit/miss outcomes, learns goal-conditioned contact control from sparse contact error, and adapts opponent-response values online without an analytical future-trajectory controller.

This repository does **not** claim a new reinforcement-learning foundation, biological realism, pixel-level learning, opponent identity recognition, or superiority to modern neural RL.

## Citation

`CITATION.cff` is included. Replace the placeholder GitHub URL and add the final Zenodo DOI after release.

## License

A license has intentionally not been selected on the author's behalf. Choose and add one before public release if you want others to have explicit reuse permissions.
