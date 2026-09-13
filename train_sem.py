from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


def _pearson(xs, ys):
    if not xs or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return num / ((dx * dy) ** 0.5)


def evaluate_contact_control(checkpoint, args):
    pairs = []
    total_hits = total_misses = 0
    seeds = max(3, min(5, args.eval_seeds))
    for i in range(seeds):
        seed = args.seed + 220_000 + i * 131
        sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
        sem.set_eval_mode(True)
        sem.set_intent_mode("random")
        env = PongEnv(PongConfig(seed=seed, win_score=999))
        coach = build_coach(seed ^ 0x25A1, 300 + i)
        outcomes = frames = 0
        while outcomes < max(50, args.eval_outcomes) and frames < args.max_frames_per_run:
            obs = env.observe()
            sem_v = sem.act(obs)
            obs, events = env.step(coach.act(obs), sem_v, 1 / env.cfg.fps)
            sem.on_events(obs, events)
            for ev in events:
                if ev["type"] == "hit" and ev["side"] == "right":
                    ep = sem.motor_episodes[-1]
                    pairs.append((float(ep["intent"]), float(ep["actual_contact"])))
                    total_hits += 1
                    outcomes += 1
                elif ev["type"] == "score" and ev["side"] == "left":
                    total_misses += 1
                    outcomes += 1
            frames += 1
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mae = sum(abs(x - y) for x, y in pairs) / max(1, len(pairs))
    return {
        "pairs": len(pairs),
        "hit_rate": total_hits / max(1, total_hits + total_misses),
        "mean_absolute_contact_error": mae,
        "intent_actual_correlation": _pearson(xs, ys),
    }


def curriculum_mode(run_index: int, runs: int) -> str:
    frac = run_index / max(1, runs)
    if frac < 0.35:
        return "neutral"      # first learn to survive / reach the ball
    if frac < 0.90:
        return "random"       # then learn reusable contact-control primitives
    return "strategy"         # finally exercise strategy -> motor intent loop


def run_training(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    if args.resume and checkpoint.exists():
        sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=args.seed)
        print(f"resumed: {checkpoint}")
    else:
        sem = AdaptiveSEMAgent(seed=args.seed)

    rng = random.Random(args.seed)
    rows = []
    start = time.time()

    for run in range(args.runs):
        env_seed = rng.randrange(1, 2**31 - 1)
        coach_seed = rng.randrange(1, 2**31 - 1)
        cfg = PongConfig(seed=env_seed, win_score=999)
        env = PongEnv(cfg)
        coach = build_coach(coach_seed, run)
        mode = curriculum_mode(run, args.runs)
        sem.set_intent_mode(mode)
        sem.set_eval_mode(False)

        before_outcomes = sem.motor_outcomes
        before_hits = sem.motor_hits
        before_misses = sem.motor_misses
        before_shots = sem.shots_learned
        frames = 0
        incoming_hits = 0
        incoming_misses = 0

        while (sem.motor_outcomes - before_outcomes) < args.outcomes_per_run and frames < args.max_frames_per_run:
            obs = env.observe()
            sem_v = sem.act(obs)
            coach_v = coach.act(obs)
            obs, events = env.step(coach_v, sem_v, 1 / cfg.fps)
            sem.on_events(obs, events)
            for ev in events:
                if ev["type"] == "hit" and ev["side"] == "right":
                    incoming_hits += 1
                elif ev["type"] == "score" and ev["side"] == "left":
                    incoming_misses += 1
            frames += 1

        n = incoming_hits + incoming_misses
        row = {
            "run": run + 1,
            "env_seed": env_seed,
            "coach_seed": coach_seed,
            "coach_style": coach.style,
            "coach_reaction_frames": coach.reaction_frames,
            "coach_gain": coach.gain,
            "coach_max_speed": coach.max_speed,
            "intent_mode": mode,
            "frames": frames,
            "outcomes": sem.motor_outcomes - before_outcomes,
            "hits": sem.motor_hits - before_hits,
            "misses": sem.motor_misses - before_misses,
            "hit_rate": incoming_hits / max(1, n),
            "strategy_updates": sem.shots_learned - before_shots,
            "motor_explore": sem._motor_epsilon(),
            "mean_recent_intent_error": sum(sem.intent_errors) / max(1, len(sem.intent_errors)),
        }
        rows.append(row)
        print(
            f"run {run+1:02d}/{args.runs}  seed={env_seed:<10d}  "
            f"coach={coach.style:<10s} mode={mode:<8s}  "
            f"hit={row['hit_rate']:.3f}  eps={row['motor_explore']:.3f}  "
            f"outcomes={sem.motor_outcomes}"
        )

        # A partial checkpoint makes interruption/restart less painful.
        if args.save_every and (run + 1) % args.save_every == 0:
            sem.save_checkpoint(
                checkpoint,
                include_strategy=True,
                metadata={"completed_runs": run + 1, "base_seed": args.seed},
            )

    elapsed = time.time() - start
    sem.set_intent_mode("strategy")
    sem.save_checkpoint(
        checkpoint,
        include_strategy=True,
        metadata={
            "completed_runs": args.runs,
            "outcomes_per_run": args.outcomes_per_run,
            "base_seed": args.seed,
            "elapsed_seconds": elapsed,
            "curriculum": "35% neutral, 55% random contact intents, 10% learned strategy intents",
        },
    )

    # Evaluate interception skill with exploration off and fresh opponent models.
    eval_rows = []
    for i in range(args.eval_seeds):
        eval_seed = args.seed + 100_000 + i * 97
        test_sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=eval_seed, reset_strategy=True)
        test_sem.set_eval_mode(True)
        test_sem.set_intent_mode("neutral")
        env = PongEnv(PongConfig(seed=eval_seed, win_score=999))
        coach = build_coach(eval_seed ^ 0x7711, 100 + i)
        hits = misses = frames = 0
        velocities = []
        prev_sign = 0
        sign_flips = 0
        while hits + misses < args.eval_outcomes and frames < args.max_frames_per_run:
            obs = env.observe()
            sem_v = test_sem.act(obs)
            if obs["ball_vx"] > 0:
                velocities.append(sem_v)
                sign = 1 if sem_v > 30 else (-1 if sem_v < -30 else 0)
                if sign and prev_sign and sign != prev_sign:
                    sign_flips += 1
                if sign:
                    prev_sign = sign
            obs, events = env.step(coach.act(obs), sem_v, 1 / env.cfg.fps)
            test_sem.on_events(obs, events)
            for ev in events:
                if ev["type"] == "hit" and ev["side"] == "right":
                    hits += 1
                elif ev["type"] == "score" and ev["side"] == "left":
                    misses += 1
            frames += 1
        eval_rows.append({
            "seed": eval_seed,
            "coach_style": coach.style,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / max(1, hits + misses),
            "incoming_frames": len(velocities),
            "velocity_sign_flips": sign_flips,
            "sign_flips_per_1000_incoming_frames": 1000.0 * sign_flips / max(1, len(velocities)),
        })

    contact_control = evaluate_contact_control(checkpoint, args)

    report = {
        "status": "SEM_40_RUN_TRAINING_COMPLETE",
        "base_seed": args.seed,
        "runs": rows,
        "training": {
            "elapsed_seconds": elapsed,
            "total_motor_outcomes": sem.motor_outcomes,
            "total_motor_hits": sem.motor_hits,
            "total_motor_misses": sem.motor_misses,
            "lifetime_hit_rate": sem.motor_hits / max(1, sem.motor_hits + sem.motor_misses),
            "motor_decisions": sem.motor_decisions,
            "motor_exploratory_decisions": sem.motor_exploratory_decisions,
            "strategy_updates": sem.shots_learned,
        },
        "evaluation": eval_rows,
        "mean_eval_hit_rate": sum(x["hit_rate"] for x in eval_rows) / max(1, len(eval_rows)),
        "mean_eval_sign_flips_per_1000_frames": sum(x["sign_flips_per_1000_incoming_frames"] for x in eval_rows) / max(1, len(eval_rows)),
        "contact_control": contact_control,
        "checkpoint": str(checkpoint),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\ntraining complete")
    print(f"checkpoint: {checkpoint}")
    print(f"report:     {report_path}")
    print(f"eval hit:   {report['mean_eval_hit_rate']:.3f}")
    print(f"smoothness: {report['mean_eval_sign_flips_per_1000_frames']:.2f} sign flips / 1000 incoming frames")
    print(f"aim corr:   {contact_control['intent_actual_correlation']:.3f}  mae={contact_control['mean_absolute_contact_error']:.3f}")
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Train oracle-free SEM across varied Pong opponents.")
    p.add_argument("--runs", type=int, default=40)
    p.add_argument("--outcomes-per-run", type=int, default=55)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--checkpoint", default="checkpoints/sem_40runs.json.gz")
    p.add_argument("--report", default="results/training_40runs.json")
    p.add_argument("--eval-seeds", type=int, default=6)
    p.add_argument("--eval-outcomes", type=int, default=70)
    p.add_argument("--max-frames-per-run", type=int, default=160000)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
