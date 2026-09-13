from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


def configure_trace_mode(sem: AdaptiveSEMAgent, mode: str):
    # Isolate survival motor learning: contact-aim residual is disabled in BOTH
    # conditions. The only difference is temporal eligibility vs last decision.
    sem.aim_weight = 0.0
    if mode == "full_trace":
        return
    if mode != "last_only":
        raise ValueError(mode)
    original_add = sem.motor_trace.add
    def last_only_add(keys, action):
        sem.motor_trace.clear()
        return original_add(keys, action)
    sem.motor_trace.add = last_only_add


def run_one(seed: int, mode: str, outcomes_target: int = 140):
    sem = AdaptiveSEMAgent(seed=seed + 10_000)
    sem.set_intent_mode("neutral")
    configure_trace_mode(sem, mode)
    env = PongEnv(PongConfig(seed=seed, win_score=999))
    coach = build_coach(seed ^ 0x4455, seed % 23)
    outcomes = []
    frames = 0
    while len(outcomes) < outcomes_target and frames < 260_000:
        obs = env.observe()
        sem_v = sem.act(obs)
        obs, events = env.step(coach.act(obs), sem_v, 1 / env.cfg.fps)
        sem.on_events(obs, events)
        for ev in events:
            if ev["type"] == "hit" and ev["side"] == "right":
                outcomes.append(1)
            elif ev["type"] == "score" and ev["side"] == "left":
                outcomes.append(0)
        frames += 1
    first = outcomes[:40]
    last = outcomes[-40:]
    return {
        "seed": seed,
        "mode": mode,
        "coach_style": coach.style,
        "outcomes": len(outcomes),
        "frames": frames,
        "early_misses": len(first) - sum(first),
        "late_misses": len(last) - sum(last),
        "early_hit_rate": sum(first) / max(1, len(first)),
        "late_hit_rate": sum(last) / max(1, len(last)),
        "motor_updates": sem.motor.total_updates,
    }


def summarize(rows):
    early = mean(r["early_misses"] for r in rows)
    late = mean(r["late_misses"] for r in rows)
    return {
        "mean_early_misses": early,
        "mean_late_misses": late,
        "miss_reduction_fraction": (early - late) / max(early, 1e-9),
        "mean_early_hit_rate": mean(r["early_hit_rate"] for r in rows),
        "mean_late_hit_rate": mean(r["late_hit_rate"] for r in rows),
        "improved_seeds": sum(r["late_misses"] < r["early_misses"] for r in rows),
        "equal_seeds": sum(r["late_misses"] == r["early_misses"] for r in rows),
        "worse_seeds": sum(r["late_misses"] > r["early_misses"] for r in rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--outcomes", type=int, default=140)
    p.add_argument("--seed", type=int, default=31000)
    p.add_argument("--report", default="results/motor_trace_ablation.json")
    args = p.parse_args()

    modes = {"full_trace": [], "last_only": []}
    for i in range(args.seeds):
        seed = args.seed + i * 97
        for mode in modes:
            modes[mode].append(run_one(seed, mode, args.outcomes))
        f = modes["full_trace"][-1]
        l = modes["last_only"][-1]
        print(f"seed {seed}: full {f['early_misses']}->{f['late_misses']} | last-only {l['early_misses']}->{l['late_misses']}")

    agg = {m: summarize(rows) for m, rows in modes.items()}
    paired = {
        "mean_late_miss_advantage_last_only_minus_full": mean(
            l["late_misses"] - f["late_misses"]
            for f, l in zip(modes["full_trace"], modes["last_only"])
        ),
        "full_better_late_seeds": sum(
            f["late_misses"] < l["late_misses"]
            for f, l in zip(modes["full_trace"], modes["last_only"])
        ),
        "ties_late_seeds": sum(
            f["late_misses"] == l["late_misses"]
            for f, l in zip(modes["full_trace"], modes["last_only"])
        ),
        "last_only_better_late_seeds": sum(
            f["late_misses"] > l["late_misses"]
            for f, l in zip(modes["full_trace"], modes["last_only"])
        ),
    }
    report = {
        "status": "SEM_MOTOR_TRACE_ABLATION_COMPLETE",
        "protocol": {
            "paired_seeds": True,
            "outcomes_per_seed": args.outcomes,
            "seeds": args.seeds,
            "intent_mode": "neutral",
            "aim_weight": 0.0,
            "full_trace": "original bounded decaying motor-decision eligibility trace",
            "last_only": "trace cleared before every motor decision; only the final decision before sparse hit/miss receives credit",
            "dense_frame_reward": False,
        },
        "aggregate": agg,
        "paired": paired,
        "runs": modes,
    }
    out = Path(args.report)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": agg, "paired": paired}, indent=2))
    print("report:", out)

if __name__ == "__main__":
    main()
