from __future__ import annotations

import argparse
import json
from pathlib import Path

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'sem_40runs.json.gz'


def reset_episode_without_learning(sem: AdaptiveSEMAgent, events):
    """Advance episode boundaries while keeping all learned tables frozen.

    This mirrors only the runtime reset that normally happens after a right-side
    hit or miss. It deliberately does not call apply_sparse or strategy updates.
    """
    terminal = False
    for ev in events:
        if ev['type'] == 'hit' and ev['side'] == 'right':
            terminal = True
        elif ev['type'] == 'score' and ev['side'] == 'left':
            terminal = True
    if terminal:
        sem.motor_trace.clear()
        sem.aim_trace.clear()
        sem.incoming_decision_audit.clear()
        sem.incoming_active = False
        sem.decision_frames_left = 0
        sem.pending_shot = None


def condition_configs(seed: int):
    common = dict(seed=seed, win_score=999)
    return {
        'in_distribution': PongConfig(**common),
        # 27% smaller target. Observation exposes the new paddle_h, but no
        # controller rule is changed and the checkpoint has never trained here.
        'small_paddle': PongConfig(**common, paddle_h=80),
        # Higher serve speed, faster acceleration, and a speed ceiling outside
        # the training distribution.
        'faster_ball': PongConfig(**common, ball_base_speed=415.0, hit_accel=1.065, max_ball_speed=1280.0),
        # Same basic game but contact generates substantially steeper angles.
        'steeper_bounce': PongConfig(**common, bounce_angle_scale=1.34),
        # Vertical geometry shift. Agent state uses normalized y features, so
        # this probes whether those abstractions transfer.
        'taller_field': PongConfig(**common, height=790),
        # Compound shift: smaller paddle + taller field + faster ball + steeper
        # bounce. No SEM parameter is changed to announce the shift.
        'compound_shift': PongConfig(
            **common,
            height=760,
            paddle_h=88,
            ball_base_speed=390.0,
            hit_accel=1.065,
            max_ball_speed=1240.0,
            bounce_angle_scale=1.28,
        ),
    }


def play_condition(checkpoint: Path, cfg: PongConfig, seed: int, run_index: int,
                   outcomes_target: int, adaptive: bool):
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed ^ 0x6A11, reset_strategy=True)
    sem.set_intent_mode('neutral')
    # Frozen evaluation has no exploration. Adaptive evaluation also has no
    # exploratory action noise: any recovery is from updating learned values,
    # not from adding extra random movement during measurement.
    sem.set_eval_mode(True)
    env = PongEnv(cfg)
    coach = build_coach(seed ^ 0x49B7, 500 + run_index)

    hits = misses = frames = 0
    outcomes = []
    while hits + misses < outcomes_target and frames < 220_000:
        obs = env.observe()
        sem_v = sem.act(obs)
        coach_v = coach.act(obs)
        obs, events = env.step(coach_v, sem_v, 1 / cfg.fps)

        for ev in events:
            if ev['type'] == 'hit' and ev['side'] == 'right':
                hits += 1
                outcomes.append(1)
            elif ev['type'] == 'score' and ev['side'] == 'left':
                misses += 1
                outcomes.append(0)

        if adaptive:
            sem.on_events(obs, events)
        else:
            reset_episode_without_learning(sem, events)
        frames += 1

    k = min(20, max(1, len(outcomes) // 3))
    first = outcomes[:k]
    last = outcomes[-k:]
    return {
        'seed': seed,
        'coach_style': coach.style,
        'adaptive': adaptive,
        'outcomes': len(outcomes),
        'hits': hits,
        'misses': misses,
        'hit_rate': hits / max(1, hits + misses),
        'early_hit_rate': sum(first) / max(1, len(first)),
        'late_hit_rate': sum(last) / max(1, len(last)),
        'early_misses': len(first) - sum(first),
        'late_misses': len(last) - sum(last),
        'frames': frames,
    }


def mean(rows, key):
    return sum(r[key] for r in rows) / max(1, len(rows))


def run(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    results = {}
    for ci, name in enumerate(condition_configs(args.base_seed)):
        frozen_rows = []
        adaptive_rows = []
        for i in range(args.seeds):
            seed = args.base_seed + ci * 10000 + i * 211
            cfg = condition_configs(seed)[name]
            frozen_rows.append(play_condition(checkpoint, cfg, seed, ci * 20 + i,
                                              args.outcomes, adaptive=False))
            # New agent from the same checkpoint; no contamination from frozen run.
            adaptive_rows.append(play_condition(checkpoint, cfg, seed, ci * 20 + i,
                                                args.outcomes, adaptive=True))
        results[name] = {
            'frozen': frozen_rows,
            'adaptive': adaptive_rows,
            'mean_frozen_hit_rate': mean(frozen_rows, 'hit_rate'),
            'mean_adaptive_hit_rate': mean(adaptive_rows, 'hit_rate'),
            'mean_adaptive_early_hit_rate': mean(adaptive_rows, 'early_hit_rate'),
            'mean_adaptive_late_hit_rate': mean(adaptive_rows, 'late_hit_rate'),
            'mean_adaptive_early_misses': mean(adaptive_rows, 'early_misses'),
            'mean_adaptive_late_misses': mean(adaptive_rows, 'late_misses'),
            'seeds_late_better_than_early': sum(
                r['late_hit_rate'] > r['early_hit_rate'] for r in adaptive_rows
            ),
        }
        x = results[name]
        print(
            f"{name:18s} frozen={x['mean_frozen_hit_rate']:.3f}  "
            f"adaptive={x['mean_adaptive_hit_rate']:.3f}  "
            f"early={x['mean_adaptive_early_hit_rate']:.3f}  "
            f"late={x['mean_adaptive_late_hit_rate']:.3f}  "
            f"late>early={x['seeds_late_better_than_early']}/{args.seeds}"
        )

    baseline = results['in_distribution']['mean_frozen_hit_rate']
    summary = {}
    for name, x in results.items():
        summary[name] = {
            'zero_shot_hit_rate': x['mean_frozen_hit_rate'],
            'zero_shot_drop_vs_in_distribution': baseline - x['mean_frozen_hit_rate'],
            'online_mean_hit_rate': x['mean_adaptive_hit_rate'],
            'online_early_hit_rate': x['mean_adaptive_early_hit_rate'],
            'online_late_hit_rate': x['mean_adaptive_late_hit_rate'],
            'online_recovery_delta': x['mean_adaptive_late_hit_rate'] - x['mean_adaptive_early_hit_rate'],
            'late_better_seeds': x['seeds_late_better_than_early'],
        }

    report = {
        'status': 'PHYSICS_GENERALIZATION_COMPLETE',
        'checkpoint': str(checkpoint),
        'seeds': args.seeds,
        'outcomes_per_condition': args.outcomes,
        'protocol': {
            'frozen': 'checkpoint policy, exploration off, no memory updates',
            'adaptive': 'same checkpoint, exploration off, sparse outcome updates enabled',
            'strategy': 'neutral contact intent to isolate survival motor transfer',
            'shift_is_not_signaled_to_sem': True,
        },
        'summary': summary,
        'conditions': results,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nreport: {out}')
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/sem_40runs.json.gz')
    p.add_argument('--base-seed', type=int, default=20270113)
    p.add_argument('--seeds', type=int, default=6)
    p.add_argument('--outcomes', type=int, default=80)
    p.add_argument('--output', default='results/physics_generalization.json')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
