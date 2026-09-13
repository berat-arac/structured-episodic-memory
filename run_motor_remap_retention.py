from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


def rolling_latency(seq, window=20, threshold=0.75):
    if len(seq) < window:
        return None
    for end in range(window, len(seq) + 1):
        chunk = seq[end-window:end]
        if sum(chunk) / window >= threshold:
            return end
    return None


def run_phase(sem, seed, run_index, n, invert, label):
    # Re-create the same environment/coach stream at every phase so the only
    # controlled phase change is the external actuator sign.
    env = PongEnv(PongConfig(seed=seed, win_score=999))
    coach = build_coach(seed ^ 0x2A77, 900 + run_index)
    seq = []
    frames = 0
    start_outcomes = sem.motor_outcomes
    start_hits = sem.motor_hits
    start_misses = sem.motor_misses

    while len(seq) < n and frames < 400000:
        obs = env.observe()
        command = sem.act(obs)
        applied = -command if invert else command
        obs, events = env.step(coach.act(obs), applied, 1 / env.cfg.fps)
        sem.on_events(obs, events)
        for e in events:
            if e['type'] == 'hit' and e['side'] == 'right':
                seq.append(1)
            elif e['type'] == 'score' and e['side'] == 'left':
                seq.append(0)
        frames += 1

    k = min(20, max(1, len(seq)//4))
    early = sum(seq[:k]) / max(1, k)
    late = sum(seq[-k:]) / max(1, k)
    lat75 = rolling_latency(seq, 20, 0.75)
    lat85 = rolling_latency(seq, 20, 0.85)
    return {
        'label': label,
        'invert': invert,
        'outcomes': len(seq),
        'frames': frames,
        'hit_rate': sum(seq) / max(1, len(seq)),
        'early_hit_rate': early,
        'late_hit_rate': late,
        'recovery_delta': late - early,
        'latency_75pct_window20': lat75,
        'latency_85pct_window20': lat85,
        'sequence': seq,
        'sem_outcomes_delta': sem.motor_outcomes - start_outcomes,
        'sem_hits_delta': sem.motor_hits - start_hits,
        'sem_misses_delta': sem.motor_misses - start_misses,
        'motor_explore_end': sem.motor_explore,
    }


def run(args):
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt

    rows = []
    for i in range(args.seeds):
        base = args.base_seed + i * 317
        sem = AdaptiveSEMAgent.load_checkpoint(ckpt, seed=base ^ 0x7731, reset_strategy=True)
        sem.set_intent_mode('neutral')
        sem.set_eval_mode(False)

        phases = [
            ('A1_normal', False, args.a1_outcomes),
            ('B1_inverted', True, args.b_outcomes),
            ('A2_normal_return', False, args.a2_outcomes),
            ('B2_inverted_return', True, args.b_outcomes),
        ]
        result = {'seed': base}
        for label, invert, n in phases:
            result[label] = run_phase(sem, base, i, n, invert, label)
        rows.append(result)

        a1 = result['A1_normal']; b1 = result['B1_inverted']; a2 = result['A2_normal_return']; b2 = result['B2_inverted_return']
        print(
            f"seed {base}: "
            f"A1 {a1['early_hit_rate']:.2f}->{a1['late_hit_rate']:.2f} | "
            f"B1 {b1['early_hit_rate']:.2f}->{b1['late_hit_rate']:.2f} lat75={b1['latency_75pct_window20']} | "
            f"A2 {a2['early_hit_rate']:.2f}->{a2['late_hit_rate']:.2f} lat75={a2['latency_75pct_window20']} | "
            f"B2 {b2['early_hit_rate']:.2f}->{b2['late_hit_rate']:.2f} lat75={b2['latency_75pct_window20']}"
        )

    def vals(phase, key):
        return [r[phase][key] for r in rows]

    def avg(phase, key):
        return mean(vals(phase, key))

    def latency_summary(phase, key):
        raw = vals(phase, key)
        ok = [x for x in raw if x is not None]
        return {
            'values': raw,
            'adapted_runs': len(ok),
            'mean_among_adapted': mean(ok) if ok else None,
        }

    summary = {}
    for phase in ('A1_normal','B1_inverted','A2_normal_return','B2_inverted_return'):
        summary[phase] = {
            'mean_early_hit_rate': avg(phase,'early_hit_rate'),
            'mean_late_hit_rate': avg(phase,'late_hit_rate'),
            'mean_phase_hit_rate': avg(phase,'hit_rate'),
            'mean_recovery_delta': avg(phase,'recovery_delta'),
            'latency75': latency_summary(phase,'latency_75pct_window20'),
            'latency85': latency_summary(phase,'latency_85pct_window20'),
        }

    b1_lats = vals('B1_inverted','latency_75pct_window20')
    b2_lats = vals('B2_inverted_return','latency_75pct_window20')
    paired = []
    for r, l1, l2 in zip(rows, b1_lats, b2_lats):
        paired.append({
            'seed': r['seed'],
            'B1_latency75': l1,
            'B2_latency75': l2,
            'B2_faster': (l1 is not None and l2 is not None and l2 < l1),
            'B2_not_slower': (l1 is not None and l2 is not None and l2 <= l1),
            'B1_early': r['B1_inverted']['early_hit_rate'],
            'B2_early': r['B2_inverted_return']['early_hit_rate'],
            'early_savings': r['B2_inverted_return']['early_hit_rate'] - r['B1_inverted']['early_hit_rate'],
        })
    comparable = [p for p in paired if p['B1_latency75'] is not None and p['B2_latency75'] is not None]
    summary['retention_savings'] = {
        'paired_comparable_runs': len(comparable),
        'B2_faster_runs': sum(p['B2_faster'] for p in comparable),
        'B2_not_slower_runs': sum(p['B2_not_slower'] for p in comparable),
        'mean_B1_latency75_comparable': mean(p['B1_latency75'] for p in comparable) if comparable else None,
        'mean_B2_latency75_comparable': mean(p['B2_latency75'] for p in comparable) if comparable else None,
        'mean_early_savings_B2_minus_B1': mean(p['early_savings'] for p in paired),
        'paired': paired,
    }

    payload = {
        'status': 'MOTOR_REMAP_RETENTION_COMPLETE',
        'seeds': args.seeds,
        'protocol': {
            'phases': ['A1 normal','B1 inverted','A2 normal','B2 inverted'],
            'A1_outcomes': args.a1_outcomes,
            'B_outcomes': args.b_outcomes,
            'A2_outcomes': args.a2_outcomes,
            'same_env_seed_and_coach_each_phase': True,
            'switch_signaled_to_sem': False,
            'memory_reset_between_phases': False,
            'meaning_of_retention': 'savings on re-learning the hidden inverted actuator mapping; simultaneous frozen competence on both mappings is not expected because mapping identity is unobserved',
        },
        'summary': summary,
        'rows': rows,
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('\nRetention savings:')
    print(json.dumps(summary['retention_savings'], indent=2))
    print('report:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/sem_40runs.json.gz')
    p.add_argument('--base-seed', type=int, default=20270601)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--a1-outcomes', type=int, default=100)
    p.add_argument('--b-outcomes', type=int, default=220)
    p.add_argument('--a2-outcomes', type=int, default=220)
    p.add_argument('--output', default='results/motor_remap_retention.json')
    run(p.parse_args())
