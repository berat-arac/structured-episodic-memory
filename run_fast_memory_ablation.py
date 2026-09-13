from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import types

import run_opponent_switch_test as sw
from sem_agent import AdaptiveSEMAgent
from pong_core import PongConfig, PongEnv

ROOT = Path(__file__).resolve().parent


def install_strategy_ablation(sem: AdaptiveSEMAgent, mode: str):
    if mode == "full":
        return
    if mode != "slow_only":
        raise ValueError(mode)

    def slow_only_value(self, ctx, action):
        g = self.global_stats[action]
        c = self.ctx_stats[ctx][action]
        cw = min(0.72, c.n / 7.0)
        # Exact same contextual weighting and uncertainty bonus as the full
        # strategy memory. The only ablation is removal of the fast estimate.
        base = (1 - cw) * g.slow + cw * c.slow
        total_n = sum(s.n for s in self.global_stats.values()) + 1
        bonus = 0.15 * math.sqrt(math.log(total_n + 1) / (g.n + 1))
        return base + bonus

    sem._strategy_value = types.MethodType(slow_only_value, sem)


def run_one(checkpoint: Path, seed: int, episodes_per_phase: int, max_frames: int, mode: str):
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
    sem.set_intent_mode("strategy")
    sem.set_eval_mode(False)
    install_strategy_ablation(sem, mode)

    env = PongEnv(PongConfig(seed=seed ^ 0xA55A, win_score=999))
    coach = sw.NonStationaryCoach(seed ^ 0xC0A7, mode="weak_up")
    phases = [("A1", "weak_up"), ("B", "weak_down"), ("A2", "weak_up")]
    phase_reports = []
    frame_count = 0

    for phase_name, opponent_mode in phases:
        coach.set_mode(opponent_mode)
        start_shots = sem.shots_learned
        phase_rows = []
        while (sem.shots_learned - start_shots) < episodes_per_phase and frame_count < max_frames:
            before = sem.shots_learned
            obs = env.observe()
            sem_v = sem.act(obs)
            coach_v = coach.act(obs)
            obs, events = env.step(coach_v, sem_v, 1 / env.cfg.fps)
            sem.on_events(obs, events)
            frame_count += 1

            if sem.shots_learned > before:
                ep = dict(sem.strategy_episodes[-1])
                ctx = tuple(ep["ctx"])
                vals = {a: sem._strategy_value(ctx, a) for a in sem.CONTACT_ACTIONS}
                preferred = max(vals, key=vals.get)
                phase_rows.append({
                    "phase": phase_name,
                    "mode": opponent_mode,
                    "phase_episode": sem.shots_learned - start_shots,
                    "global_episode": sem.shots_learned,
                    "intended_contact": float(ep["intended_contact"]),
                    "actual_contact": float(ep["actual_contact"]),
                    "actual_contact_bin": float(ep["actual_contact_bin"]),
                    "reward": float(ep["reward"]),
                    "outcome": ep["outcome"],
                    "preferred_action": float(preferred),
                    "fast_values": {str(a): float(sem.global_stats[a].fast) for a in sem.CONTACT_ACTIONS},
                    "slow_values": {str(a): float(sem.global_stats[a].slow) for a in sem.CONTACT_ACTIONS},
                })

        phase_reports.append({"phase": phase_name, **sw.phase_summary(phase_rows, opponent_mode)})

    return {
        "seed": seed,
        "mode": mode,
        "frames": frame_count,
        "motor_hit_rate": sem.motor_hits / max(1, sem.motor_hits + sem.motor_misses),
        "phase_reports": phase_reports,
    }


def aggregate_mode(runs):
    out = {}
    for phase in ("A1", "B", "A2"):
        rows = [[p for p in r["phase_reports"] if p["phase"] == phase][0] for r in runs]
        lat = [p["adaptation_latency_episodes"] for p in rows if p["adaptation_latency_episodes"] is not None]
        out[phase] = {
            "mean_first20_target_intent_fraction": mean(p["first20_target_intent_fraction"] for p in rows),
            "mean_last20_target_intent_fraction": mean(p["last20_target_intent_fraction"] for p in rows),
            "mean_intended_target_fraction": mean(p["intended_target_fraction"] for p in rows),
            "mean_opponent_miss_rate": mean(p["opponent_miss_rate"] for p in rows),
            "mean_reward": mean(p["mean_reward"] for p in rows),
            "adapted_runs": len(lat),
            "mean_adaptation_latency_episodes": mean(lat) if lat else None,
            "median_adaptation_latency_episodes": median(lat) if lat else None,
            "adaptation_latencies": [p["adaptation_latency_episodes"] for p in rows],
        }
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Fast-memory ablation on hidden A->B->A opponent switch benchmark")
    p.add_argument("--checkpoint", default="checkpoints/sem_40runs.json.gz")
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--episodes-per-phase", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--max-frames", type=int, default=900000)
    p.add_argument("--report", default="results/fast_memory_ablation.json")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    mode_runs = {"full": [], "slow_only": []}
    seeds = [args.seed + i * 1009 for i in range(args.seeds)]
    for seed in seeds:
        for mode in ("full", "slow_only"):
            r = run_one(checkpoint, seed, args.episodes_per_phase, args.max_frames, mode)
            mode_runs[mode].append(r)
        f = {p["phase"]: p for p in mode_runs["full"][-1]["phase_reports"]}
        s = {p["phase"]: p for p in mode_runs["slow_only"][-1]["phase_reports"]}
        print(
            f"seed {seed}: B latency full={f['B']['adaptation_latency_episodes']} slow={s['B']['adaptation_latency_episodes']} | "
            f"B last20 full={f['B']['last20_target_intent_fraction']:.2f} slow={s['B']['last20_target_intent_fraction']:.2f}"
        )

    agg = {mode: aggregate_mode(runs) for mode, runs in mode_runs.items()}

    paired = {}
    for phase in ("A1", "B", "A2"):
        full_last = []
        slow_last = []
        full_lat = []
        slow_lat = []
        both_lat = []
        for fr, sr in zip(mode_runs["full"], mode_runs["slow_only"]):
            fp = [p for p in fr["phase_reports"] if p["phase"] == phase][0]
            sp = [p for p in sr["phase_reports"] if p["phase"] == phase][0]
            full_last.append(fp["last20_target_intent_fraction"])
            slow_last.append(sp["last20_target_intent_fraction"])
            if fp["adaptation_latency_episodes"] is not None:
                full_lat.append(fp["adaptation_latency_episodes"])
            if sp["adaptation_latency_episodes"] is not None:
                slow_lat.append(sp["adaptation_latency_episodes"])
            if fp["adaptation_latency_episodes"] is not None and sp["adaptation_latency_episodes"] is not None:
                both_lat.append(sp["adaptation_latency_episodes"] - fp["adaptation_latency_episodes"])
        paired[phase] = {
            "mean_last20_advantage_full_minus_slow": mean(a-b for a,b in zip(full_last, slow_last)),
            "both_adapted_seed_count": len(both_lat),
            "mean_latency_penalty_slow_minus_full_on_both_adapted": mean(both_lat) if both_lat else None,
            "median_latency_penalty_slow_minus_full_on_both_adapted": median(both_lat) if both_lat else None,
        }

    report = {
        "status": "SEM_FAST_MEMORY_ABLATION_COMPLETE",
        "checkpoint": str(checkpoint),
        "protocol": {
            "modes": {
                "full": "original fast+slow strategy value",
                "slow_only": "same contextual weights, exploration and uncertainty bonus, but strategy choice ignores all fast estimates",
            },
            "phases": ["A1: weak_up", "B: weak_down", "A2: weak_up"],
            "seeds": args.seeds,
            "episodes_per_phase": args.episodes_per_phase,
            "paired_seeds": True,
            "memory_reset_between_phases": False,
            "switch_signal_exposed_to_sem": False,
        },
        "aggregate": agg,
        "paired": paired,
        "runs": mode_runs,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAggregate")
    for mode in ("full", "slow_only"):
        print(mode)
        for phase in ("A1", "B", "A2"):
            a = agg[mode][phase]
            print(f"  {phase}: first20={a['mean_first20_target_intent_fraction']:.3f} last20={a['mean_last20_target_intent_fraction']:.3f} adapted={a['adapted_runs']}/{args.seeds} lat={a['mean_adaptation_latency_episodes']}")
    print("paired:", json.dumps(paired, indent=2))
    print("report:", report_path)


if __name__ == "__main__":
    main()
