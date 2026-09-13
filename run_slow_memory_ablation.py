from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from statistics import mean, median

import run_opponent_switch_test as sw
from sem_agent import AdaptiveSEMAgent
from pong_core import PongConfig, PongEnv

ROOT = Path(__file__).resolve().parent


def install_mode(sem, mode):
    if mode == 'full':
        return
    if mode != 'fast_only':
        raise ValueError(mode)

    def fast_only_value(self, ctx, action):
        g = self.global_stats[action]
        c = self.ctx_stats[ctx][action]
        cw = min(0.72, c.n / 7.0)
        base = (1 - cw) * g.fast + cw * c.fast
        total_n = sum(s.n for s in self.global_stats.values()) + 1
        bonus = 0.15 * math.sqrt(math.log(total_n + 1) / (g.n + 1))
        return base + bonus

    sem._strategy_value = types.MethodType(fast_only_value, sem)


def run_one(checkpoint, seed, episodes=100, max_frames=900000, mode='full'):
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
    sem.set_intent_mode('strategy')
    sem.set_eval_mode(False)
    install_mode(sem, mode)
    env = PongEnv(PongConfig(seed=seed ^ 0xA55A, win_score=999))
    coach = sw.NonStationaryCoach(seed ^ 0xC0A7, mode='weak_up')
    reps = []
    fc = 0
    for pn, om in [('A1', 'weak_up'), ('B', 'weak_down'), ('A2', 'weak_up')]:
        coach.set_mode(om)
        start = sem.shots_learned
        rows = []
        while sem.shots_learned - start < episodes and fc < max_frames:
            before = sem.shots_learned
            obs = env.observe(); sv = sem.act(obs); cv = coach.act(obs)
            obs, evs = env.step(cv, sv, 1/env.cfg.fps); sem.on_events(obs, evs); fc += 1
            if sem.shots_learned > before:
                ep = dict(sem.strategy_episodes[-1]); ctx = tuple(ep['ctx'])
                vals = {a: sem._strategy_value(ctx, a) for a in sem.CONTACT_ACTIONS}; pref = max(vals, key=vals.get)
                rows.append({
                    'intended_contact': float(ep['intended_contact']),
                    'actual_contact': float(ep['actual_contact']),
                    'actual_contact_bin': float(ep['actual_contact_bin']),
                    'reward': float(ep['reward']),
                    'outcome': ep['outcome'],
                    'preferred_action': float(pref),
                    'fast_values': {str(a): float(sem.global_stats[a].fast) for a in sem.CONTACT_ACTIONS},
                    'slow_values': {str(a): float(sem.global_stats[a].slow) for a in sem.CONTACT_ACTIONS},
                })
        reps.append({'phase': pn, **sw.phase_summary(rows, om)})
    return {'seed': seed, 'mode': mode, 'phase_reports': reps}


def agg(runs):
    out = {}
    for ph in ['A1','B','A2']:
        rows = [{p['phase']:p for p in r['phase_reports']}[ph] for r in runs]
        lat = [p['adaptation_latency_episodes'] for p in rows if p['adaptation_latency_episodes'] is not None]
        out[ph] = {
            'mean_first20_target_intent_fraction': mean(p['first20_target_intent_fraction'] for p in rows),
            'mean_last20_target_intent_fraction': mean(p['last20_target_intent_fraction'] for p in rows),
            'mean_intended_target_fraction': mean(p['intended_target_fraction'] for p in rows),
            'mean_reward': mean(p['mean_reward'] for p in rows),
            'adapted_runs': len(lat),
            'mean_adaptation_latency_episodes': mean(lat) if lat else None,
            'median_adaptation_latency_episodes': median(lat) if lat else None,
            'latencies': [p['adaptation_latency_episodes'] for p in rows],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--episodes-per-phase', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--report', default='results/slow_memory_ablation.json')
    args = p.parse_args()
    cp = ROOT/'checkpoints'/'sem_40runs.json.gz'
    modes = {'full': [], 'fast_only': []}
    for i in range(args.seeds):
        seed = args.seed + i*1009
        for m in modes:
            modes[m].append(run_one(cp, seed, args.episodes_per_phase, mode=m))
        f = {p['phase']:p for p in modes['full'][-1]['phase_reports']}
        g = {p['phase']:p for p in modes['fast_only'][-1]['phase_reports']}
        print(seed, 'B lat', f['B']['adaptation_latency_episodes'], g['B']['adaptation_latency_episodes'], 'A2', f['A2']['adaptation_latency_episodes'], g['A2']['adaptation_latency_episodes'])
    A = {m:agg(r) for m,r in modes.items()}
    paired = {}
    for ph in ['A1','B','A2']:
        latdiff=[]; last=[]
        for fr,gr in zip(modes['full'],modes['fast_only']):
            f={p['phase']:p for p in fr['phase_reports']}[ph]; g={p['phase']:p for p in gr['phase_reports']}[ph]
            if f['adaptation_latency_episodes'] is not None and g['adaptation_latency_episodes'] is not None:
                latdiff.append(g['adaptation_latency_episodes']-f['adaptation_latency_episodes'])
            last.append(f['last20_target_intent_fraction']-g['last20_target_intent_fraction'])
        paired[ph] = {
            'mean_latency_penalty_fast_only_minus_full': mean(latdiff) if latdiff else None,
            'mean_last20_advantage_full_minus_fast_only': mean(last),
            'paired_adapted': len(latdiff),
        }
    report = {
        'status':'SEM_SLOW_MEMORY_ABLATION_COMPLETE',
        'protocol':{
            'paired_seeds':True,
            'seeds':args.seeds,
            'episodes_per_phase':args.episodes_per_phase,
            'full':'original fast+slow strategy value',
            'fast_only':'same context weighting/exploration/bonus; slow estimates ignored in action choice',
            'phases':['A1 weak_up','B weak_down','A2 weak_up'],
            'memory_reset_between_phases':False,
            'switch_signal_exposed_to_sem':False,
        },
        'aggregate':A,
        'paired':paired,
        'runs':modes,
    }
    out=ROOT/args.report; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':A,'paired':paired},indent=2)); print('report',out)

if __name__ == '__main__':
    main()
