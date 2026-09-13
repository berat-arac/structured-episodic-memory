from __future__ import annotations

import ast
from collections import defaultdict, deque
import gzip
import json
import math
from pathlib import Path
import random

from sem_core import CortexTable, EligibilityTrace, ValueStat


class AdaptiveSEMAgent:
    """Oracle-free, non-neural memory-based Pong agent.

    Survival/interception and contact aiming are learned by two associative
    memories. Neither computes a future ball path. The survival memory learns
    which velocity action tends to produce a later hit. The aiming memory learns
    a residual preference for actions that make the actual hit contact match a
    chosen contact intent. Opponent memory chooses that intent online.
    """

    CHECKPOINT_VERSION = 3
    MOTOR_ACTIONS = (-1.0, -0.62, -0.28, 0.0, 0.28, 0.62, 1.0)
    CONTACT_ACTIONS = (-0.66, -0.44, -0.22, 0.0, 0.22, 0.44, 0.66)

    def __init__(self, side="right", seed=123):
        assert side == "right"
        self.side = side
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        # Survival motor memory + contact-aim residual memory.
        self.motor = CortexTable(self.MOTOR_ACTIONS, seed=self.seed ^ 0x51A7)
        self.aim_motor = CortexTable(self.MOTOR_ACTIONS, seed=self.seed ^ 0xA19B)
        self.motor_trace = EligibilityTrace(decay=0.88, max_items=84)
        self.aim_trace = EligibilityTrace(decay=0.70, max_items=48)
        self.motor_episodes = deque(maxlen=1000)
        self.motor_outcomes = 0
        self.motor_hits = 0
        self.motor_misses = 0
        self.motor_decisions = 0
        self.motor_exploratory_decisions = 0
        self.motor_explore = 0.34
        self.motor_surprise = 0.0
        self.last_motor_values = {}
        self.last_base_keys = ()
        self.last_aim_keys = ()
        self.last_motor_action = 0.0

        # Generic actuator continuity. No ball-target logic lives here.
        self.decision_hold_frames = 5
        self.decision_frames_left = 0
        self.command_action = 0.0
        self.applied_velocity = 0.0
        self.velocity_smoothing = 0.28
        self.switch_cost = 0.055
        self.aim_weight = 1.05

        self.incoming_active = False
        self.current_contact_intent = 0.0
        self.incoming_strategy_ctx = None
        self.intent_mode = "strategy"  # strategy | neutral | random
        self.eval_mode = False
        self.intent_errors = deque(maxlen=400)

        # Opponent-response / tactic memory.
        self.global_stats = {a: ValueStat() for a in self.CONTACT_ACTIONS}
        self.ctx_stats = defaultdict(lambda: {a: ValueStat() for a in self.CONTACT_ACTIONS})
        self.strategy_episodes = deque(maxlen=800)
        self.pending_shot = None
        self.shots_learned = 0
        self.strategy_explore = 0.42
        self.strategy_surprise = 0.0
        self.last_strategy_values = {a: 0.0 for a in self.CONTACT_ACTIONS}
        self.last_strategy_choice = 0.0
        self.last_opp_y = None
        self.opp_vy = 0.0
        self.last_contact = 0.0

        # Runtime-only motor audit. This never affects action selection or learning.
        # It exists solely to explain failures after the fact.
        self.incoming_decision_audit = deque(maxlen=64)
        self.miss_audits = deque(maxlen=120)
        self.total_miss_audits = 0

    @staticmethod
    def _bin(value, edges):
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    @classmethod
    def _nearest_contact_action(cls, contact):
        return min(cls.CONTACT_ACTIONS, key=lambda a: abs(a - float(contact)))

    def _speed_bin(self, speed):
        return self._bin(speed, (420.0, 560.0, 760.0, 940.0))

    def _opp_bin(self, y, h):
        return self._bin(y / max(h, 1.0), (0.25, 0.45, 0.60, 0.78))

    def _opp_motion_bin(self):
        if self.opp_vy < -120:
            return -2
        if self.opp_vy < -35:
            return -1
        if self.opp_vy > 120:
            return 2
        if self.opp_vy > 35:
            return 1
        return 0

    def strategy_context(self, obs):
        return (
            self._opp_bin(obs["left_y"], obs["height"]),
            self._speed_bin(obs["ball_speed"]),
            self._opp_motion_bin(),
        )

    def update_frame(self, obs):
        if self.last_opp_y is not None:
            measured = (obs["left_y"] - self.last_opp_y) * 60.0
            self.opp_vy = 0.86 * self.opp_vy + 0.14 * measured
        self.last_opp_y = obs["left_y"]

    def _features(self, obs):
        w = max(float(obs["width"]), 1.0)
        h = max(float(obs["height"]), 1.0)
        ph = max(float(obs["paddle_h"]), 1.0)
        distance = (w - float(obs["ball_x"])) / w
        rel_y = (float(obs["ball_y"]) - float(obs["right_y"])) / ph
        vy_norm = float(obs["ball_vy"]) / max(float(obs["ball_speed"]), 1.0)
        paddle_y = float(obs["right_y"]) / h
        ball_y = float(obs["ball_y"]) / h
        return dict(
            xf=self._bin(distance, (0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.90)),
            xc=self._bin(distance, (0.12, 0.30, 0.55, 0.80)),
            ef=self._bin(rel_y, (-1.8, -1.15, -0.72, -0.36, -0.12, 0.12, 0.36, 0.72, 1.15, 1.8)),
            ec=self._bin(rel_y, (-1.05, -0.42, 0.42, 1.05)),
            vf=self._bin(vy_norm, (-0.55, -0.18, 0.18, 0.55)),
            vs=-1 if vy_norm < -0.10 else (1 if vy_norm > 0.10 else 0),
            py=self._bin(paddle_y, (0.18, 0.36, 0.55, 0.74, 0.90)),
            by=self._bin(ball_y, (0.16, 0.34, 0.52, 0.70, 0.86)),
            sp=self._speed_bin(obs["ball_speed"]),
        )

    def _base_motor_keys(self, obs):
        f = self._features(obs)
        return (
            ("f", f["xf"], f["ef"], f["vf"], f["py"], f["by"], f["sp"]),
            ("m", f["xc"], f["ef"], f["vf"], f["py"], f["sp"]),
            ("r", f["xc"], f["ec"], f["vs"], f["sp"]),
            ("y", f["ef"], f["vf"], f["xc"]),
            ("b", f["ec"], f["vs"]),
        )

    def _aim_motor_keys(self, obs):
        f = self._features(obs)
        ib = self._nearest_contact_action(self.current_contact_intent)
        # Current geometric goal error only. At this instant, 2*relative_y is
        # the contact coordinate the ball would have against the paddle centre.
        # This is a representation feature, not an action rule or future solver.
        ph = max(float(obs["paddle_h"]), 1.0)
        rel_y = (float(obs["ball_y"]) - float(obs["right_y"])) / ph
        goal_error = 2.0 * rel_y - float(self.current_contact_intent)
        gf = self._bin(goal_error, (-1.8, -1.15, -0.72, -0.38, -0.16, 0.16, 0.38, 0.72, 1.15, 1.8))
        gc = self._bin(goal_error, (-0.90, -0.35, 0.35, 0.90))
        return (
            ("af", ib, f["xf"], f["ef"], f["vf"], f["py"], f["by"], f["sp"]),
            ("ag", ib, f["xc"], gf, f["vf"], f["sp"]),
            ("am", ib, f["xc"], f["ef"], f["vf"], f["sp"]),
            ("ar", ib, f["xc"], gc, f["vs"], f["sp"]),
            ("ab", ib, gc, f["vs"]),
        )

    def _motor_epsilon(self):
        if self.eval_mode:
            return 0.0
        return max(0.020, 0.34 * math.exp(-self.motor_outcomes / 170.0))

    def set_intent_mode(self, mode):
        if mode not in {"strategy", "neutral", "random"}:
            raise ValueError("intent mode must be strategy, neutral or random")
        self.intent_mode = mode

    def set_eval_mode(self, enabled=True):
        self.eval_mode = bool(enabled)

    # ------------------------- strategy selection -------------------------
    def _strategy_value(self, ctx, action):
        g = self.global_stats[action]
        c = self.ctx_stats[ctx][action]
        cw = min(0.72, c.n / 7.0)
        base = (1 - cw) * (0.62 * g.fast + 0.38 * g.slow) + cw * (0.72 * c.fast + 0.28 * c.slow)
        total_n = sum(s.n for s in self.global_stats.values()) + 1
        bonus = 0.15 * math.sqrt(math.log(total_n + 1) / (g.n + 1))
        return base + bonus

    def preferred_contact(self, obs, explore=False):
        ctx = self.strategy_context(obs)
        vals = {a: self._strategy_value(ctx, a) for a in self.CONTACT_ACTIONS}
        eps = 0.0 if self.eval_mode else self.strategy_explore
        if explore and self.rng.random() < eps:
            choice = self.rng.choice(self.CONTACT_ACTIONS)
        else:
            m = max(vals.values())
            best = [a for a, v in vals.items() if abs(v - m) < 1e-12]
            choice = self.rng.choice(best)
        self.last_strategy_values = vals
        self.last_strategy_choice = choice
        return choice, vals

    def _choose_incoming_intent(self, obs):
        if self.intent_mode == "neutral":
            self.last_strategy_choice = 0.0
            return 0.0
        if self.intent_mode == "random":
            c = self.rng.choice(self.CONTACT_ACTIONS)
            self.last_strategy_choice = c
            return c
        c, _ = self.preferred_contact(obs, explore=True)
        return c

    def _begin_incoming(self, obs):
        self.incoming_active = True
        self.motor_trace.clear()
        self.aim_trace.clear()
        self.incoming_decision_audit.clear()
        self.decision_frames_left = 0
        self.incoming_strategy_ctx = self.strategy_context(obs)
        self.current_contact_intent = self._choose_incoming_intent(obs)

    def _smooth_velocity(self, desired):
        self.applied_velocity += self.velocity_smoothing * (float(desired) - self.applied_velocity)
        if abs(self.applied_velocity) < 1.0:
            self.applied_velocity = 0.0
        return self.applied_velocity

    def _audit_motor_decision(self, obs, features, action, exploratory, epsilon,
                              base_vals, aim_vals, combined_vals, stable_vals):
        # Evidence is descriptive only; _estimate does not mutate the table.
        base_est = self.motor._estimate(self.last_base_keys, action) if self.last_base_keys else (0.0, 0, 0.0)
        aim_est = self.aim_motor._estimate(self.last_aim_keys, action) if self.last_aim_keys else (0.0, 0, 0.0)
        ordered = sorted(stable_vals.items(), key=lambda kv: kv[1], reverse=True)
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 0.0
        self.incoming_decision_audit.append({
            "decision_index": int(self.motor_decisions + 1),
            "exploratory": bool(exploratory),
            "epsilon": float(epsilon),
            "action": float(action),
            "previous_command": float(self.command_action),
            "applied_velocity_before": float(self.applied_velocity),
            "intent": float(self.current_contact_intent),
            "obs": {
                "ball_x": float(obs["ball_x"]),
                "ball_y": float(obs["ball_y"]),
                "ball_vx": float(obs["ball_vx"]),
                "ball_vy": float(obs["ball_vy"]),
                "ball_speed": float(obs["ball_speed"]),
                "right_y": float(obs["right_y"]),
            },
            "features": dict(features),
            "selected_base_value": float(base_vals[action]),
            "selected_aim_value": float(aim_vals[action]),
            "selected_combined_value": float(combined_vals[action]),
            "selected_stable_value": float(stable_vals[action]),
            "decision_margin": float(margin),
            "base_evidence": int(base_est[1]),
            "aim_evidence": int(aim_est[1]),
            "base_disagreement": float(base_est[2]),
            "aim_disagreement": float(aim_est[2]),
        })

    def _classify_miss(self):
        ds = list(self.incoming_decision_audit)
        if not ds:
            return "no_decision_trace"
        recent = ds[-4:]
        if any(d["exploratory"] for d in recent):
            return "recent_exploration"
        if any(d["base_evidence"] < 8 for d in recent):
            return "low_evidence_state"
        if any(d["decision_margin"] < 0.035 for d in recent):
            return "ambiguous_policy_values"
        speeds = [d["obs"]["ball_speed"] for d in recent]
        if speeds and max(speeds) >= 940.0:
            return "high_speed_generalization"
        return "learned_policy_error_or_timing"

    # ---------------------------- motor policy ----------------------------
    def act(self, obs):
        self.update_frame(obs)
        toward = obs["ball_vx"] > 0
        if not toward:
            self.incoming_active = False
            self.command_action = 0.0
            self.last_motor_action = 0.0
            return self._smooth_velocity(0.0)

        if not self.incoming_active:
            self._begin_incoming(obs)

        if self.decision_frames_left <= 0:
            self.motor_trace.step()
            self.aim_trace.step()
            features = self._features(obs)
            base_keys = self._base_motor_keys(obs)
            aim_keys = self._aim_motor_keys(obs)
            base_vals = self.motor.values(base_keys)
            aim_vals = self.aim_motor.raw_values(aim_keys)
            vals = {a: base_vals[a] + self.aim_weight * aim_vals[a] for a in self.MOTOR_ACTIONS}
            stable = {a: vals[a] - self.switch_cost * abs(a - self.command_action) for a in self.MOTOR_ACTIONS}

            epsilon = self._motor_epsilon()
            exploratory = self.rng.random() < epsilon
            if exploratory:
                action = self.rng.choice(self.MOTOR_ACTIONS)
            else:
                best_score = max(stable.values())
                best = [a for a, v in stable.items() if abs(v - best_score) < 1e-12]
                action = self.rng.choice(best)

            self.motor_trace.add(base_keys, action)
            self.aim_trace.add(aim_keys, action)
            self.last_base_keys = base_keys
            self.last_aim_keys = aim_keys
            self.last_motor_values = vals
            self.last_motor_action = action
            self._audit_motor_decision(obs, features, action, exploratory, epsilon,
                                       base_vals, aim_vals, vals, stable)
            self.command_action = action
            self.decision_frames_left = self.decision_hold_frames
            self.motor_decisions += 1
            if exploratory:
                self.motor_exploratory_decisions += 1
        self.decision_frames_left -= 1

        desired = self.command_action * float(obs.get("sem_max_speed", 1050.0))
        return self._smooth_velocity(desired)

    def _settle_motor(self, outcome, actual_contact=None):
        if outcome == "hit":
            survival_reward = 1.0
            err = abs(float(actual_contact) - float(self.current_contact_intent))
            closeness = max(0.0, 1.0 - err / 1.05)
            # Contact execution is a separate sparse objective. Wrong contact can
            # be negative for the aim memory while the survival hit stays positive.
            aim_reward = 2.0 * closeness - 1.0
            self.intent_errors.append(err)
            self.motor_hits += 1
        else:
            survival_reward = -1.15
            aim_reward = -0.70
            err = None
            self.motor_misses += 1

        updated = self.motor.apply_sparse(self.motor_trace, survival_reward)
        aim_updated = self.aim_motor.apply_sparse(self.aim_trace, aim_reward)
        self.motor_outcomes += 1
        self.motor_explore = self._motor_epsilon()
        if self.last_base_keys:
            self.motor_surprise = self.motor.surprise(self.last_base_keys, self.last_motor_action)

        ep = {
            "outcome": outcome,
            "survival_reward": survival_reward,
            "aim_reward": aim_reward,
            "trace_updates": updated,
            "aim_trace_updates": aim_updated,
            "motor_action": self.last_motor_action,
            "explore": self.motor_explore,
            "intent": self.current_contact_intent,
        }
        if actual_contact is not None:
            ep["actual_contact"] = float(actual_contact)
            ep["intent_error"] = err
        self.motor_episodes.append(ep)
        if outcome == "miss":
            decisions = list(self.incoming_decision_audit)
            audit = {
                "miss_index": int(self.motor_misses),
                "motor_outcome_index": int(self.motor_outcomes),
                "likely_cause": self._classify_miss(),
                "intent": float(self.current_contact_intent),
                "num_decisions": len(decisions),
                "num_exploratory_decisions": sum(1 for d in decisions if d["exploratory"]),
                "max_ball_speed": max((d["obs"]["ball_speed"] for d in decisions), default=0.0),
                "last_decisions": decisions[-12:],
            }
            self.miss_audits.append(audit)
            self.total_miss_audits += 1
        self.motor_trace.clear()
        self.aim_trace.clear()
        self.incoming_decision_audit.clear()
        self.incoming_active = False
        self.decision_frames_left = 0

    # --------------------------- event learning ---------------------------
    def _settle_strategy(self, reward, response, outcome):
        p = self.pending_shot
        if not p:
            return
        a = p["actual_contact_bin"]
        ctx = p["ctx"]
        self.global_stats[a].update(reward)
        self.ctx_stats[ctx][a].update(reward)
        self.shots_learned += 1
        self.strategy_explore = max(0.080, 0.42 * math.exp(-self.shots_learned / 48.0))
        self.strategy_surprise = abs(self.global_stats[a].fast - self.global_stats[a].slow)
        self.strategy_episodes.append({
            "ctx": ctx,
            "intended_contact": p.get("intended_contact", 0.0),
            "actual_contact_bin": a,
            "actual_contact": p["actual_contact"],
            "reward": float(reward),
            "response": float(response),
            "outcome": outcome,
            "fast": self.global_stats[a].fast,
            "slow": self.global_stats[a].slow,
        })
        self.pending_shot = None

    def on_events(self, obs, events):
        for ev in events:
            if ev["type"] == "hit" and ev["side"] == "right":
                actual = float(ev["contact"])
                self.last_contact = actual
                self._settle_motor("hit", actual_contact=actual)
                self.pending_shot = {
                    "ctx": self.incoming_strategy_ctx or self.strategy_context(obs),
                    "intended_contact": self.current_contact_intent,
                    "actual_contact": actual,
                    "actual_contact_bin": self._nearest_contact_action(actual),
                }
            elif ev["type"] == "hit" and ev["side"] == "left":
                if self.pending_shot:
                    response = min(1.0, abs(float(ev["contact"])))
                    self._settle_strategy(0.12 + 0.30 * response, response, "returned")
            elif ev["type"] == "score":
                if ev["side"] == "left":
                    if self.incoming_active or self.motor_trace.items():
                        self._settle_motor("miss")
                    if self.pending_shot:
                        self._settle_strategy(0.0, 0.0, "other_score")
                else:
                    if self.pending_shot:
                        self._settle_strategy(1.0, 1.0, "opponent_miss")

    # ------------------------------ memory I/O ----------------------------
    def reset_opponent_memory(self):
        self.global_stats = {a: ValueStat() for a in self.CONTACT_ACTIONS}
        self.ctx_stats = defaultdict(lambda: {a: ValueStat() for a in self.CONTACT_ACTIONS})
        self.strategy_episodes.clear()
        self.pending_shot = None
        self.shots_learned = 0
        self.strategy_explore = 0.42
        self.strategy_surprise = 0.0
        self.last_strategy_values = {a: 0.0 for a in self.CONTACT_ACTIONS}
        self.last_strategy_choice = 0.0
        self.last_opp_y = None
        self.opp_vy = 0.0

    def _strategy_to_dict(self):
        rows = []
        for ctx, amap in self.ctx_stats.items():
            used = {str(a): s.to_dict() for a, s in amap.items() if s.n}
            if used:
                rows.append({"ctx": repr(ctx), "actions": used})
        return {
            "global": {str(a): s.to_dict() for a, s in self.global_stats.items()},
            "ctx_rows": rows,
            "shots_learned": self.shots_learned,
            "strategy_explore": self.strategy_explore,
        }

    def _load_strategy_dict(self, data):
        self.reset_opponent_memory()
        for a_str, d in data.get("global", {}).items():
            a = float(a_str)
            if a in self.global_stats:
                self.global_stats[a] = ValueStat.from_dict(d)
        for row in data.get("ctx_rows", []):
            ctx = ast.literal_eval(row["ctx"])
            amap = self.ctx_stats[ctx]
            for a_str, d in row.get("actions", {}).items():
                a = float(a_str)
                if a in amap:
                    amap[a] = ValueStat.from_dict(d)
        self.shots_learned = int(data.get("shots_learned", 0))
        self.strategy_explore = float(data.get("strategy_explore", max(0.080, 0.42 * math.exp(-self.shots_learned / 48.0))))

    def save_checkpoint(self, path, include_strategy=True, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.CHECKPOINT_VERSION,
            "seed": self.seed,
            "motor": self.motor.to_dict(),
            "aim_motor": self.aim_motor.to_dict(),
            "motor_state": {
                "outcomes": self.motor_outcomes,
                "hits": self.motor_hits,
                "misses": self.motor_misses,
                "decisions": self.motor_decisions,
                "exploratory_decisions": self.motor_exploratory_decisions,
            },
            "strategy": self._strategy_to_dict() if include_strategy else None,
            "metadata": metadata or {},
        }
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        if str(path).endswith(".gz"):
            with gzip.open(path, "wb") as f:
                f.write(payload)
        else:
            path.write_bytes(payload)
        return path

    @classmethod
    def load_checkpoint(cls, path, seed=None, reset_strategy=False):
        path = Path(path)
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(seed=int(data.get("seed", 123) if seed is None else seed))
        obj.motor = CortexTable.from_dict(data["motor"], seed=obj.seed ^ 0x51A7)
        if "aim_motor" in data:
            obj.aim_motor = CortexTable.from_dict(data["aim_motor"], seed=obj.seed ^ 0xA19B)
        ms = data.get("motor_state", {})
        obj.motor_outcomes = int(ms.get("outcomes", 0))
        obj.motor_hits = int(ms.get("hits", 0))
        obj.motor_misses = int(ms.get("misses", 0))
        obj.motor_decisions = int(ms.get("decisions", 0))
        obj.motor_exploratory_decisions = int(ms.get("exploratory_decisions", 0))
        obj.motor_explore = obj._motor_epsilon()
        if data.get("strategy") and not reset_strategy:
            obj._load_strategy_dict(data["strategy"])
        else:
            obj.reset_opponent_memory()
        return obj

    def summary(self):
        strategy_vals = {
            str(a): {"n": s.n, "mean": s.mean, "fast": s.fast, "slow": s.slow}
            for a, s in self.global_stats.items()
        }
        observed = [a for a in self.CONTACT_ACTIONS if self.global_stats[a].n]
        best = max(observed, key=lambda a: self.global_stats[a].fast) if observed else 0.0
        return {
            "oracle_free": True,
            "motor_outcomes": self.motor_outcomes,
            "motor_hits": self.motor_hits,
            "motor_misses": self.motor_misses,
            "motor_hit_rate": self.motor_hits / max(1, self.motor_hits + self.motor_misses),
            "motor_decisions": self.motor_decisions,
            "motor_exploratory_decisions": self.motor_exploratory_decisions,
            "motor_explore": self._motor_epsilon(),
            "motor_surprise": self.motor_surprise,
            "last_motor_action": self.last_motor_action,
            "applied_velocity": self.applied_velocity,
            "current_contact_intent": self.current_contact_intent,
            "mean_intent_error": sum(self.intent_errors) / max(1, len(self.intent_errors)),
            "shots_learned": self.shots_learned,
            "strategy_explore": self.strategy_explore,
            "strategy_surprise": self.strategy_surprise,
            "strategy_choice": self.last_strategy_choice,
            "best_observed_contact": best,
            "last_contact": self.last_contact,
            "actions": strategy_vals,
            "total_miss_audits": self.total_miss_audits,
            "last_miss_audit": (self.miss_audits[-1] if self.miss_audits else None),
            "miss_audits": list(self.miss_audits),
            "motor_episodes": list(self.motor_episodes),
            "strategy_episodes": list(self.strategy_episodes),
        }
