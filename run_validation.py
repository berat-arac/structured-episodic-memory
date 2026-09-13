from __future__ import annotations

import ast
import json
from pathlib import Path
import random
import tempfile

from coach_opponents import build_coach
from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sem_40runs.json.gz"


def _pearson(xs, ys):
    if not xs or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs
    )
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return num / ((dx * dy) ** 0.5)


def source_audit():
    path = ROOT / "sem_agent.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad_function_fragments = ("inter" + "cept", "trajec" + "tory", "projec" + "tion")
    bad_functions = []
    target_names = []
    divisions_by_horizontal_velocity = 0

    def contains_ball_vx(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and sub.value == "ball_vx":
                return True
            if isinstance(sub, ast.Name) and sub.id == "ball_vx":
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            low = node.name.lower()
            if any(fragment in low for fragment in bad_function_fragments):
                bad_functions.append(node.name)
        if isinstance(node, ast.Name) and node.id in {"target_x", "future_y", "impact_y", "time_to_impact"}:
            target_names.append(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and contains_ball_vx(node.right):
            divisions_by_horizontal_velocity += 1

    return {
        "bad_solver_functions": sorted(set(bad_functions)),
        "future_target_names": sorted(set(target_names)),
        "divisions_by_ball_vx": divisions_by_horizontal_velocity,
        "neural_framework_imports": sum(name in source for name in ("torch", "tensorflow", "jax", "keras")),
        "pass": not bad_functions and not target_names and divisions_by_horizontal_velocity == 0,
        "scope": "AST audit for analytical future-path solver signatures in sem_agent.py",
    }


def fresh_motor_learning(seed, outcomes_target=120):
    sem = AdaptiveSEMAgent(seed=seed + 10_000)
    sem.set_intent_mode("neutral")
    env = PongEnv(PongConfig(seed=seed, win_score=999))
    coach = build_coach(seed ^ 0x4455, seed % 23)
    outcomes = []
    frames = 0
    while len(outcomes) < outcomes_target and frames < 220_000:
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
    first = outcomes[:30]
    last = outcomes[-30:]
    return {
        "seed": seed,
        "coach_style": coach.style,
        "outcomes": len(outcomes),
        "early_misses": len(first) - sum(first),
        "late_misses": len(last) - sum(last),
        "early_hit_rate": sum(first) / max(1, len(first)),
        "late_hit_rate": sum(last) / max(1, len(last)),
    }


def evaluate_pretrained(checkpoint, seeds=4, outcomes_target=60):
    rows = []
    for i in range(seeds):
        seed = 51000 + i * 101
        sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
        sem.set_eval_mode(True)
        sem.set_intent_mode("neutral")
        env = PongEnv(PongConfig(seed=seed, win_score=999))
        coach = build_coach(seed ^ 0x1881, 120 + i)
        hits = misses = frames = 0
        prev_sign = 0
        flips = 0
        incoming_frames = 0
        while hits + misses < outcomes_target and frames < 180_000:
            obs = env.observe()
            v = sem.act(obs)
            if obs["ball_vx"] > 0:
                incoming_frames += 1
                sign = 1 if v > 30 else (-1 if v < -30 else 0)
                if sign and prev_sign and sign != prev_sign:
                    flips += 1
                if sign:
                    prev_sign = sign
            obs, events = env.step(coach.act(obs), v, 1 / env.cfg.fps)
            sem.on_events(obs, events)
            for ev in events:
                if ev["type"] == "hit" and ev["side"] == "right":
                    hits += 1
                elif ev["type"] == "score" and ev["side"] == "left":
                    misses += 1
            frames += 1
        rows.append({
            "seed": seed,
            "coach_style": coach.style,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / max(1, hits + misses),
            "sign_flips_per_1000_incoming_frames": 1000.0 * flips / max(1, incoming_frames),
        })
    return {
        "rows": rows,
        "mean_hit_rate": sum(r["hit_rate"] for r in rows) / max(1, len(rows)),
        "mean_sign_flips_per_1000_incoming_frames": sum(r["sign_flips_per_1000_incoming_frames"] for r in rows) / max(1, len(rows)),
    }


def evaluate_contact_control(checkpoint, seeds=3, outcomes_target=60):
    pairs = []
    hits = misses = 0
    for i in range(seeds):
        seed = 62000 + i * 137
        sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
        sem.set_eval_mode(True)
        sem.set_intent_mode("random")
        env = PongEnv(PongConfig(seed=seed, win_score=999))
        coach = build_coach(seed ^ 0x7788, 220 + i)
        outcomes = frames = 0
        while outcomes < outcomes_target and frames < 180_000:
            obs = env.observe()
            v = sem.act(obs)
            obs, events = env.step(coach.act(obs), v, 1 / env.cfg.fps)
            sem.on_events(obs, events)
            for ev in events:
                if ev["type"] == "hit" and ev["side"] == "right":
                    ep = sem.motor_episodes[-1]
                    pairs.append((float(ep["intent"]), float(ep["actual_contact"])))
                    hits += 1
                    outcomes += 1
                elif ev["type"] == "score" and ev["side"] == "left":
                    misses += 1
                    outcomes += 1
            frames += 1
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    return {
        "pairs": len(pairs),
        "hit_rate": hits / max(1, hits + misses),
        "mean_absolute_contact_error": sum(abs(a - b) for a, b in pairs) / max(1, len(pairs)),
        "intent_actual_correlation": _pearson(xs, ys),
    }


def strategy_change_test():
    sem = AdaptiveSEMAgent(seed=777)
    rng = random.Random(778)
    ctx = (2, 2, 0)
    actions = sem.CONTACT_ACTIONS

    def phase(target, n=170):
        for _ in range(n):
            vals = {a: sem._strategy_value(ctx, a) for a in actions}
            if rng.random() < sem.strategy_explore:
                a = rng.choice(actions)
            else:
                m = max(vals.values())
                a = rng.choice([x for x, v in vals.items() if abs(v - m) < 1e-12])
            closeness = max(0.0, 1.0 - abs(a - target) / 1.1)
            reward = 1.0 if rng.random() < 0.06 + 0.86 * closeness else 0.08 * closeness
            sem.pending_shot = {"ctx": ctx, "actual_contact": a, "actual_contact_bin": a, "intended_contact": a}
            sem._settle_strategy(reward, closeness, "validation")

    hi = max(actions)
    lo = min(actions)
    phase(hi)
    vals_a = {a: sem._strategy_value(ctx, a) for a in actions}
    best_a = max(actions, key=lambda a: vals_a[a])
    phase(lo)
    vals_b = {a: sem._strategy_value(ctx, a) for a in actions}
    best_b = max(actions, key=lambda a: vals_b[a])
    tolerance = 0.24
    return {
        "phase_a_target": hi,
        "phase_a_best": best_a,
        "phase_b_target": lo,
        "phase_b_best": best_b,
        "phase_a_correct": abs(best_a - hi) <= tolerance,
        "phase_b_correct": abs(best_b - lo) <= tolerance,
    }


def checkpoint_roundtrip(checkpoint):
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=9090, reset_strategy=False)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "roundtrip.json.gz"
        sem.save_checkpoint(p, include_strategy=True, metadata={"test": True})
        loaded = AdaptiveSEMAgent.load_checkpoint(p, seed=9090, reset_strategy=False)
        a = sem.motor.to_dict()
        b = loaded.motor.to_dict()
        aa = sem.aim_motor.to_dict()
        bb = loaded.aim_motor.to_dict()
        passed = (
            sem.motor_outcomes == loaded.motor_outcomes
            and sem.motor_hits == loaded.motor_hits
            and sem.motor_misses == loaded.motor_misses
            and a["total_updates"] == b["total_updates"]
            and aa["total_updates"] == bb["total_updates"]
            and len(a["rows"]) == len(b["rows"])
            and len(aa["rows"]) == len(bb["rows"])
            and sem.shots_learned == loaded.shots_learned
        )
    return {"pass": passed}


def main():
    audit = source_audit()
    scratch = [fresh_motor_learning(s) for s in (130, 131, 132, 133)]
    early = sum(x["early_misses"] for x in scratch) / len(scratch)
    late = sum(x["late_misses"] for x in scratch) / len(scratch)
    scratch_reduction = (early - late) / max(early, 1e-9)
    strategy = strategy_change_test()

    if DEFAULT_CHECKPOINT.exists():
        pretrained = evaluate_pretrained(DEFAULT_CHECKPOINT)
        contact = evaluate_contact_control(DEFAULT_CHECKPOINT)
        roundtrip = checkpoint_roundtrip(DEFAULT_CHECKPOINT)
    else:
        pretrained = {"missing_checkpoint": True, "mean_hit_rate": 0.0, "mean_sign_flips_per_1000_incoming_frames": 999.0}
        contact = {"missing_checkpoint": True, "intent_actual_correlation": 0.0, "mean_absolute_contact_error": 999.0}
        roundtrip = {"pass": False, "missing_checkpoint": True}

    gates = {
        "source_audit": audit["pass"],
        "fresh_learning": scratch_reduction > 0.25,
        "pretrained_interception": pretrained["mean_hit_rate"] >= 0.90,
        "smooth_motor": pretrained["mean_sign_flips_per_1000_incoming_frames"] < 70.0,
        "goal_conditioned_contact": contact["intent_actual_correlation"] > 0.20,
        "strategy_change": strategy["phase_a_correct"] and strategy["phase_b_correct"],
        "checkpoint_roundtrip": roundtrip["pass"],
    }
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "source_audit": audit,
        "fresh_motor_learning": {
            "seeds": scratch,
            "mean_early_misses": early,
            "mean_late_misses": late,
            "miss_reduction_fraction": scratch_reduction,
        },
        "pretrained_evaluation": pretrained,
        "contact_control": contact,
        "strategy_change_test": strategy,
        "checkpoint_roundtrip": roundtrip,
        "claim_boundary": [
            "No analytical future ball-path solver is used by SEM.",
            "Current-frame state variables, feature bins, sparse rewards, action vocabulary, actuator smoothing and incoming-direction gating are engineered.",
            "The training opponents are engineered curriculum generators and are not part of SEM.",
            "Motor interception, goal-conditioned contact control and opponent-response values are learned from experience.",
        ],
    }
    out = RES / "validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
