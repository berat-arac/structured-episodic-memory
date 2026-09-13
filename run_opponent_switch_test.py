from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean

from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


class NonStationaryCoach:
    """Controlled opponent whose *behaviour* changes without informing SEM.

    Modes:
      weak_up   : slower/hesitant against upward-travelling incoming balls
      weak_down : slower/hesitant against downward-travelling incoming balls

    The coach uses only current geometry. It does not expose its mode to SEM.
    This is a benchmark opponent, not part of SEM and not a claim about humans.
    """

    def __init__(self, seed: int, mode: str = "weak_up"):
        self.rng = random.Random(seed)
        self.mode = mode
        self.hold = 0
        self.cached_velocity = 0.0
        self.last_vx_sign = None
        self.desired_contact = 0.0
        self.return_index = 0
        self.incoming_vulnerable = False

    def set_mode(self, mode: str):
        if mode not in {"weak_up", "weak_down"}:
            raise ValueError(mode)
        self.mode = mode
        # No SEM state is touched. Clear only the opponent actuator cache so the
        # behaviour switch takes effect immediately and cleanly.
        self.hold = 0
        self.cached_velocity = 0.0

    def _vulnerable_at_departure(self, obs) -> bool:
        # Latch the trajectory family when the ball leaves SEM. A later wall
        # bounce may flip vy, but the opponent keeps the same response mode for
        # that incoming shot. This makes the hidden contingency stable enough to
        # measure without exposing it to SEM.
        if self.mode == "weak_up":
            return obs["ball_vy"] < -35.0
        return obs["ball_vy"] > 35.0

    def _pick_contact(self):
        # Same return repertoire in both modes so the hidden change is primarily
        # a response-contingency change, not a completely different game.
        options = (-0.62, -0.34, 0.0, 0.34, 0.62)
        c = options[self.return_index % len(options)]
        self.return_index += 1
        return float(c)

    def act(self, obs):
        vx_sign = -1 if obs["ball_vx"] < 0 else 1
        if vx_sign != self.last_vx_sign:
            self.last_vx_sign = vx_sign
            if vx_sign < 0:
                self.desired_contact = self._pick_contact()
                self.incoming_vulnerable = self._vulnerable_at_departure(obs)
            self.hold = 0

        if self.hold > 0:
            self.hold -= 1
            return self.cached_velocity

        if obs["ball_vx"] < 0:
            vulnerable = self.incoming_vulnerable
            # Strong normally; noticeably delayed and under-powered only for the
            # currently vulnerable trajectory family.
            if vulnerable:
                reaction_frames = 9
                gain = 4.0
                max_speed = 500.0
                # Occasional hesitation makes the changed response distribution
                # measurable without directly forcing a miss.
                if self.rng.random() < 0.28:
                    v = 0.0
                else:
                    target_y = obs["ball_y"] - self.desired_contact * (obs["paddle_h"] / 2.0)
                    v = (target_y - obs["left_y"]) * gain
            else:
                reaction_frames = 2
                gain = 10.0
                max_speed = 1100.0
                target_y = obs["ball_y"] - self.desired_contact * (obs["paddle_h"] / 2.0)
                v = (target_y - obs["left_y"]) * gain
            v = max(-max_speed, min(max_speed, v))
            self.cached_velocity = v
            self.hold = reaction_frames - 1
            return v

        # Neutral re-centering while waiting. Same in both modes.
        v = (obs["height"] * 0.5 - obs["left_y"]) * 3.8
        self.cached_velocity = max(-850.0, min(850.0, v))
        self.hold = 1
        return self.cached_velocity


def sign_of(x: float, dead: float = 0.08) -> int:
    if x > dead:
        return 1
    if x < -dead:
        return -1
    return 0


def correct_sign(mode: str) -> int:
    # Negative SEM contact produces an upward return toward the left opponent;
    # positive contact produces a downward return.
    return -1 if mode == "weak_up" else 1


def rolling_adaptation_latency(rows, target_sign, window=16, threshold=0.62):
    """First episode index where a rolling intent window favors new weakness."""
    if len(rows) < window:
        return None
    for end in range(window, len(rows) + 1):
        chunk = rows[end-window:end]
        signs = [sign_of(r["intended_contact"]) for r in chunk]
        # Neutral intents count against adaptation; this is deliberately strict.
        frac = sum(1 for s in signs if s == target_sign) / len(signs)
        if frac >= threshold:
            return end
    return None


def phase_summary(rows, mode):
    target = correct_sign(mode)
    if not rows:
        return {}
    intended = [sign_of(r["intended_contact"]) for r in rows]
    actual = [sign_of(r["actual_contact"]) for r in rows]
    misses = sum(1 for r in rows if r["outcome"] == "opponent_miss")
    rewards = [r["reward"] for r in rows]
    target_actual_rows = [r for r in rows if sign_of(r["actual_contact"]) == target]
    other_actual_rows = [r for r in rows if sign_of(r["actual_contact"]) != target]
    target_intent_rows = [r for r in rows if sign_of(r["intended_contact"]) == target]
    other_intent_rows = [r for r in rows if sign_of(r["intended_contact"]) != target]
    n = len(rows)
    first = rows[: min(20, n)]
    last = rows[max(0, n-20):]
    return {
        "mode": mode,
        "episodes": n,
        "target_contact_sign": target,
        "intended_target_fraction": sum(s == target for s in intended) / n,
        "actual_target_fraction": sum(s == target for s in actual) / n,
        "opponent_miss_rate": misses / n,
        "mean_reward": mean(rewards),
        "target_actual_opponent_miss_rate": sum(r["outcome"] == "opponent_miss" for r in target_actual_rows) / max(1, len(target_actual_rows)),
        "other_actual_opponent_miss_rate": sum(r["outcome"] == "opponent_miss" for r in other_actual_rows) / max(1, len(other_actual_rows)),
        "target_intent_opponent_miss_rate": sum(r["outcome"] == "opponent_miss" for r in target_intent_rows) / max(1, len(target_intent_rows)),
        "other_intent_opponent_miss_rate": sum(r["outcome"] == "opponent_miss" for r in other_intent_rows) / max(1, len(other_intent_rows)),
        "first20_target_intent_fraction": sum(sign_of(r["intended_contact"]) == target for r in first) / max(1, len(first)),
        "last20_target_intent_fraction": sum(sign_of(r["intended_contact"]) == target for r in last) / max(1, len(last)),
        "adaptation_latency_episodes": rolling_adaptation_latency(rows, target),
        "end_fast_values": rows[-1]["fast_values"],
        "end_slow_values": rows[-1]["slow_values"],
        "end_preferred_action": rows[-1]["preferred_action"],
    }


def run_one(checkpoint: Path, seed: int, episodes_per_phase: int, max_frames: int):
    # Keep pretrained motor/aim skill, but give strategy a clean slate.
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
    sem.set_intent_mode("strategy")
    sem.set_eval_mode(False)

    env = PongEnv(PongConfig(seed=seed ^ 0xA55A, win_score=999))
    coach = NonStationaryCoach(seed ^ 0xC0A7, mode="weak_up")
    phases = [("A1", "weak_up"), ("B", "weak_down"), ("A2", "weak_up")]
    all_rows = []
    phase_reports = []
    frame_count = 0

    for phase_name, mode in phases:
        coach.set_mode(mode)
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
                row = {
                    "phase": phase_name,
                    "mode": mode,
                    "phase_episode": sem.shots_learned - start_shots,
                    "global_episode": sem.shots_learned,
                    "intended_contact": float(ep["intended_contact"]),
                    "actual_contact": float(ep["actual_contact"]),
                    "actual_contact_bin": float(ep["actual_contact_bin"]),
                    "reward": float(ep["reward"]),
                    "outcome": ep["outcome"],
                    "preferred_action": float(preferred),
                    "preferred_values": {str(a): float(vals[a]) for a in sem.CONTACT_ACTIONS},
                    "fast_values": {str(a): float(sem.global_stats[a].fast) for a in sem.CONTACT_ACTIONS},
                    "slow_values": {str(a): float(sem.global_stats[a].slow) for a in sem.CONTACT_ACTIONS},
                    "counts": {str(a): int(sem.global_stats[a].n) for a in sem.CONTACT_ACTIONS},
                    "strategy_explore": float(sem.strategy_explore),
                }
                phase_rows.append(row)
                all_rows.append(row)

        phase_reports.append({"phase": phase_name, **phase_summary(phase_rows, mode)})

    return {
        "seed": seed,
        "episodes_per_phase": episodes_per_phase,
        "frames": frame_count,
        "motor_hit_rate": sem.motor_hits / max(1, sem.motor_hits + sem.motor_misses),
        "phase_reports": phase_reports,
        "episodes": all_rows,
    }


def aggregate(runs):
    phases = ["A1", "B", "A2"]
    out = {}
    for phase in phases:
        rows = []
        for run in runs:
            rows.extend([p for p in run["phase_reports"] if p["phase"] == phase])
        out[phase] = {
            "mean_intended_target_fraction": mean(p["intended_target_fraction"] for p in rows),
            "mean_actual_target_fraction": mean(p["actual_target_fraction"] for p in rows),
            "mean_opponent_miss_rate": mean(p["opponent_miss_rate"] for p in rows),
            "mean_reward": mean(p["mean_reward"] for p in rows),
            "mean_target_actual_opponent_miss_rate": mean(p["target_actual_opponent_miss_rate"] for p in rows),
            "mean_other_actual_opponent_miss_rate": mean(p["other_actual_opponent_miss_rate"] for p in rows),
            "mean_target_intent_opponent_miss_rate": mean(p["target_intent_opponent_miss_rate"] for p in rows),
            "mean_other_intent_opponent_miss_rate": mean(p["other_intent_opponent_miss_rate"] for p in rows),
            "mean_first20_target_intent_fraction": mean(p["first20_target_intent_fraction"] for p in rows),
            "mean_last20_target_intent_fraction": mean(p["last20_target_intent_fraction"] for p in rows),
            "adaptation_latencies": [p["adaptation_latency_episodes"] for p in rows],
        }
        lat = [x for x in out[phase]["adaptation_latencies"] if x is not None]
        out[phase]["mean_adaptation_latency_episodes"] = mean(lat) if lat else None
        out[phase]["adapted_runs"] = len(lat)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="A->B->A hidden opponent-behaviour switch test for SEM.")
    p.add_argument("--checkpoint", default="checkpoints/sem_40runs.json.gz")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--episodes-per-phase", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--max-frames", type=int, default=900000)
    p.add_argument("--report", default="results/opponent_switch_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    runs = []
    for i in range(args.seeds):
        seed = args.seed + i * 1009
        run = run_one(checkpoint, seed, args.episodes_per_phase, args.max_frames)
        runs.append(run)
        pr = {p["phase"]: p for p in run["phase_reports"]}
        print(
            f"seed {seed}: "
            f"A1 {pr['A1']['last20_target_intent_fraction']:.2f}  "
            f"B {pr['B']['last20_target_intent_fraction']:.2f}  "
            f"A2 {pr['A2']['last20_target_intent_fraction']:.2f}  "
            f"lat={pr['A1']['adaptation_latency_episodes']}/{pr['B']['adaptation_latency_episodes']}/{pr['A2']['adaptation_latency_episodes']}"
        )

    agg = aggregate(runs)
    report = {
        "status": "SEM_HIDDEN_OPPONENT_SWITCH_TEST_COMPLETE",
        "checkpoint": str(checkpoint),
        "protocol": {
            "phases": ["A1: weak_up", "B: weak_down", "A2: weak_up"],
            "episodes_per_phase": args.episodes_per_phase,
            "seeds": args.seeds,
            "strategy_reset_at_start": True,
            "memory_reset_between_phases": False,
            "switch_signal_exposed_to_sem": False,
            "note": "Opponent is a controlled benchmark with direction-dependent response latency; it is not a human model.",
        },
        "aggregate": agg,
        "runs": runs,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAggregate")
    for phase in ("A1", "B", "A2"):
        a = agg[phase]
        print(
            f"{phase}: first20={a['mean_first20_target_intent_fraction']:.3f}  "
            f"last20={a['mean_last20_target_intent_fraction']:.3f}  "
            f"target_all={a['mean_intended_target_fraction']:.3f}  "
            f"opp_miss={a['mean_opponent_miss_rate']:.3f}  "
            f"lat={a['mean_adaptation_latency_episodes']}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
