from __future__ import annotations

import argparse
import json
import math
import random
import types
from pathlib import Path
from statistics import mean

from pong_core import PongConfig, PongEnv
from sem_agent import AdaptiveSEMAgent

ROOT = Path(__file__).resolve().parent


def sign_of(x: float, dead: float = 0.08) -> int:
    if x > dead:
        return 1
    if x < -dead:
        return -1
    return 0


def install_mode(sem: AdaptiveSEMAgent, mode: str):
    if mode == "full":
        return
    if mode != "global_only":
        raise ValueError(mode)

    def global_only_value(self, ctx, action):
        g = self.global_stats[action]
        base = 0.70 * g.fast + 0.30 * g.slow
        total_n = sum(s.n for s in self.global_stats.values()) + 1
        bonus = 0.15 * math.sqrt(math.log(total_n + 1) / (g.n + 1))
        return base + bonus

    sem._strategy_value = types.MethodType(global_only_value, sem)


class ConditionalCueCoach:
    """Trialized benchmark opponent with context-dependent vulnerability.

    A visible cue is the opponent paddle's vertical location *before* SEM chooses
    a contact intent. The weakness is conditional on that cue:

      high cue -> vulnerable to positive/downward SEM contact
      low cue  -> vulnerable to negative/upward SEM contact

    The cue/target mapping is NOT exposed to SEM. The coach only changes its
    response dynamics after SEM's shot. Global action statistics are therefore
    contradictory across the balanced contexts, while context-specific memory
    can separate them.
    """

    HIGH_Y_FRAC = 0.20
    LOW_Y_FRAC = 0.80

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.cue = "high"
        self.target_sign = 1
        self.cached_velocity = 0.0
        self.hold = 0
        self.desired_contact = 0.0
        self.return_counter = 0
        self.vulnerable = False
        self.last_vx_sign = None

    def set_trial(self, env: PongEnv, cue: str):
        if cue not in {"high", "low"}:
            raise ValueError(cue)
        self.cue = cue
        self.target_sign = 1 if cue == "high" else -1
        env.left_y = env.cfg.height * (self.HIGH_Y_FRAC if cue == "high" else self.LOW_Y_FRAC)
        self.cached_velocity = 0.0
        self.hold = 0
        self.desired_contact = 0.0
        self.vulnerable = False
        self.last_vx_sign = 1  # trial starts with ball moving toward SEM

    def cue_y(self, obs):
        return obs["height"] * (self.HIGH_Y_FRAC if self.cue == "high" else self.LOW_Y_FRAC)

    def _pick_return_contact(self):
        # Balanced fixed repertoire, independent of cue.
        opts = (-0.50, -0.22, 0.0, 0.22, 0.50)
        c = opts[self.return_counter % len(opts)]
        self.return_counter += 1
        return c

    def act(self, obs):
        vx_sign = -1 if obs["ball_vx"] < 0 else 1
        if vx_sign != self.last_vx_sign:
            self.last_vx_sign = vx_sign
            self.hold = 0
            if vx_sign < 0:
                # Latch whether SEM's ACTUAL trajectory belongs to the weakness
                # family. This is benchmark-side scoring logic, not SEM logic.
                shot_sign = sign_of(obs["ball_vy"], dead=20.0)
                self.vulnerable = shot_sign == self.target_sign
                self.desired_contact = self._pick_return_contact()

        if self.hold > 0:
            self.hold -= 1
            return self.cached_velocity

        if obs["ball_vx"] > 0:
            # While SEM is receiving the ball, hold the visible context cue.
            target_y = self.cue_y(obs)
            v = (target_y - obs["left_y"]) * 8.0
            self.cached_velocity = max(-900.0, min(900.0, v))
            self.hold = 1
            return self.cached_velocity

        # Ball is coming back to the benchmark opponent.
        if self.vulnerable:
            reaction_frames = 10
            gain = 3.2
            max_speed = 430.0
            # Hesitation is stochastic but independent of SEM internals.
            if self.rng.random() < 0.34:
                v = 0.0
            else:
                target_y = obs["ball_y"] - self.desired_contact * (obs["paddle_h"] / 2.0)
                v = (target_y - obs["left_y"]) * gain
        else:
            reaction_frames = 1
            gain = 11.5
            max_speed = 1050.0
            target_y = obs["ball_y"] - self.desired_contact * (obs["paddle_h"] / 2.0)
            v = (target_y - obs["left_y"]) * gain

        self.cached_velocity = max(-max_speed, min(max_speed, v))
        self.hold = reaction_frames - 1
        return self.cached_velocity


def prime_visible_context(sem: AdaptiveSEMAgent, obs, repeats: int = 8):
    """Give several static observation frames before a trial begins.

    This prevents the artificial teleport between trial cue positions from being
    misread as opponent motion. No action is selected and no memory is updated.
    It models the opponent already occupying the visible cue position before the
    incoming ball arrives.
    """
    for _ in range(repeats):
        sem.update_frame(obs)


def run_one(checkpoint: Path, seed: int, trials: int, mode: str, max_frames: int):
    sem = AdaptiveSEMAgent.load_checkpoint(checkpoint, seed=seed, reset_strategy=True)
    sem.set_intent_mode("strategy")
    sem.set_eval_mode(False)
    install_mode(sem, mode)

    env = PongEnv(PongConfig(seed=seed ^ 0xA51C, win_score=999))
    coach = ConditionalCueCoach(seed ^ 0xC017)
    cue_rng = random.Random(seed ^ 0xBADA55)

    rows = []
    frame_count = 0
    cue_counts = {"high": 0, "low": 0}

    for trial in range(1, trials + 1):
        # Stratified-random cue: every pair contains one high and one low, with
        # randomized order. This keeps global statistics maximally conflicting.
        if trial % 2 == 1:
            pair = ["high", "low"]
            cue_rng.shuffle(pair)
            pending_pair = pair
        cue = pending_pair[(trial - 1) % 2]
        cue_counts[cue] += 1
        target_sign = 1 if cue == "high" else -1

        env.reset_round(direction=1)  # ball starts toward SEM
        coach.set_trial(env, cue)
        # Keep trial speed distribution fixed and avoid cross-trial opponent
        # motion artefacts in the context representation.
        obs0 = env.observe()
        prime_visible_context(sem, obs0)

        before = sem.shots_learned
        trial_frames = 0
        while sem.shots_learned == before and frame_count < max_frames:
            obs = env.observe()
            sem_v = sem.act(obs)
            coach_v = coach.act(obs)
            obs, events = env.step(coach_v, sem_v, 1 / env.cfg.fps)
            sem.on_events(obs, events)
            frame_count += 1
            trial_frames += 1
            if trial_frames > 5000:
                raise RuntimeError("trial did not settle")

        if sem.shots_learned == before:
            break

        ep = dict(sem.strategy_episodes[-1])
        ctx = tuple(ep["ctx"])
        vals = {a: sem._strategy_value(ctx, a) for a in sem.CONTACT_ACTIONS}
        preferred = max(vals, key=vals.get)
        rows.append({
            "trial": trial,
            "cue": cue,
            "target_sign": target_sign,
            "ctx": list(ctx),
            "intended_contact": float(ep["intended_contact"]),
            "actual_contact": float(ep["actual_contact"]),
            "actual_contact_bin": float(ep["actual_contact_bin"]),
            "outcome": ep["outcome"],
            "reward": float(ep["reward"]),
            "preferred_action": float(preferred),
            "strategy_explore": float(sem.strategy_explore),
        })

    def summarize(chunk):
        if not chunk:
            return {}
        return {
            "n": len(chunk),
            "intent_target_fraction": mean(sign_of(r["intended_contact"]) == r["target_sign"] for r in chunk),
            "actual_target_fraction": mean(sign_of(r["actual_contact"]) == r["target_sign"] for r in chunk),
            "preferred_target_fraction": mean(sign_of(r["preferred_action"]) == r["target_sign"] for r in chunk),
            "opponent_miss_rate": mean(r["outcome"] == "opponent_miss" for r in chunk),
            "mean_reward": mean(r["reward"] for r in chunk),
        }

    last_n = min(100, len(rows))
    last_rows = rows[-last_n:]
    high = [r for r in last_rows if r["cue"] == "high"]
    low = [r for r in last_rows if r["cue"] == "low"]
    target_actual = [r for r in rows if sign_of(r["actual_contact"]) == r["target_sign"]]
    other_actual = [r for r in rows if sign_of(r["actual_contact"]) != r["target_sign"]]

    return {
        "seed": seed,
        "mode": mode,
        "trials_requested": trials,
        "trials_completed": len(rows),
        "frames": frame_count,
        "cue_counts": cue_counts,
        "overall": summarize(rows),
        "first100": summarize(rows[:100]),
        "last100": summarize(last_rows),
        "last100_high": summarize(high),
        "last100_low": summarize(low),
        "reward_landscape": {
            "target_actual_opponent_miss_rate": mean(r["outcome"] == "opponent_miss" for r in target_actual) if target_actual else 0.0,
            "other_actual_opponent_miss_rate": mean(r["outcome"] == "opponent_miss" for r in other_actual) if other_actual else 0.0,
            "target_actual_n": len(target_actual),
            "other_actual_n": len(other_actual),
        },
        "motor_hit_rate": sem.motor_hits / max(1, sem.motor_hits + sem.motor_misses),
        "rows": rows,
    }


def aggregate(runs):
    keys = ["intent_target_fraction", "actual_target_fraction", "preferred_target_fraction", "opponent_miss_rate", "mean_reward"]
    out = {}
    for section in ["overall", "first100", "last100", "last100_high", "last100_low"]:
        out[section] = {k: mean(r[section][k] for r in runs) for k in keys}
    out["reward_landscape"] = {
        "target_actual_opponent_miss_rate": mean(r["reward_landscape"]["target_actual_opponent_miss_rate"] for r in runs),
        "other_actual_opponent_miss_rate": mean(r["reward_landscape"]["other_actual_opponent_miss_rate"] for r in runs),
    }
    out["mean_motor_hit_rate"] = mean(r["motor_hit_rate"] for r in runs)
    return out


def main():
    p = argparse.ArgumentParser(description="Context-dependent conflicting-contingency benchmark for SEM strategy memory.")
    p.add_argument("--checkpoint", default="checkpoints/sem_40runs.json.gz")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--trials", type=int, default=320)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--max-frames", type=int, default=1200000)
    p.add_argument("--report", default="results/context_conditional_benchmark.json")
    args = p.parse_args()

    cp = Path(args.checkpoint)
    if not cp.is_absolute():
        cp = ROOT / cp

    modes = {"full": [], "global_only": []}
    for i in range(args.seeds):
        seed = args.seed + i * 1009
        for mode in modes:
            modes[mode].append(run_one(cp, seed, args.trials, mode, args.max_frames))
        f = modes["full"][-1]["last100"]
        g = modes["global_only"][-1]["last100"]
        print(
            f"seed {seed}: last100 target-intent full={f['intent_target_fraction']:.3f} global={g['intent_target_fraction']:.3f} | "
            f"preferred full={f['preferred_target_fraction']:.3f} global={g['preferred_target_fraction']:.3f}"
        )

    agg = {m: aggregate(rs) for m, rs in modes.items()}
    paired = {
        "last100_intent_target_advantage_full_minus_global": mean(
            f["last100"]["intent_target_fraction"] - g["last100"]["intent_target_fraction"]
            for f, g in zip(modes["full"], modes["global_only"])
        ),
        "last100_preferred_target_advantage_full_minus_global": mean(
            f["last100"]["preferred_target_fraction"] - g["last100"]["preferred_target_fraction"]
            for f, g in zip(modes["full"], modes["global_only"])
        ),
        "full_better_intent_seeds": sum(
            f["last100"]["intent_target_fraction"] > g["last100"]["intent_target_fraction"]
            for f, g in zip(modes["full"], modes["global_only"])
        ),
        "full_better_preferred_seeds": sum(
            f["last100"]["preferred_target_fraction"] > g["last100"]["preferred_target_fraction"]
            for f, g in zip(modes["full"], modes["global_only"])
        ),
    }

    report = {
        "status": "SEM_CONTEXT_CONDITIONAL_BENCHMARK_COMPLETE",
        "protocol": {
            "paired_seeds": True,
            "seeds": args.seeds,
            "trials_per_seed": args.trials,
            "balanced_contexts": True,
            "context_cue_exposed": "opponent paddle vertical location only",
            "target_rule_exposed_to_sem": False,
            "high_context_target": "positive/downward contact",
            "low_context_target": "negative/upward contact",
            "memory_reset_between_trials": False,
            "full": "global + context-specific fast/slow memory",
            "global_only": "same strategy learner with context statistics ignored",
            "note": "Trial preparation primes a static visible cue only to avoid teleport-induced opponent-motion artifacts; it does not select actions or update value memory.",
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

    print("\nAggregate last100")
    for mode in ("full", "global_only"):
        a = agg[mode]["last100"]
        print(mode, json.dumps(a, indent=2))
    print("paired", json.dumps(paired, indent=2))
    print("reward landscape", json.dumps(agg["full"]["reward_landscape"], indent=2))
    print("report", out)


if __name__ == "__main__":
    main()
