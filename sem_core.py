from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import math
import random
from typing import Deque, Dict, Hashable, Iterable, List, Tuple


@dataclass
class ValueStat:
    """Fast/slow value estimate used by the non-neural cortex table."""

    n: int = 0
    fast: float = 0.0
    slow: float = 0.0
    mean: float = 0.0

    def update(self, target: float) -> None:
        self.n += 1
        self.mean += (target - self.mean) / self.n
        if self.n == 1:
            self.fast = self.slow = float(target)
            return
        # Fast memory follows changed contingencies; slow memory stabilises them.
        self.fast = 0.66 * self.fast + 0.34 * target
        self.slow = 0.945 * self.slow + 0.055 * target

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ValueStat":
        return cls(
            n=int(d.get("n", 0)),
            fast=float(d.get("fast", 0.0)),
            slow=float(d.get("slow", 0.0)),
            mean=float(d.get("mean", 0.0)),
        )


class EligibilityTrace:
    """Bounded replacing eligibility trace over *motor decisions*.

    A trace step happens only when the motor layer makes a new decision, not on
    every rendered frame. That keeps temporal credit from rewarding hundreds of
    contradictory micro-actions during one incoming ball.
    """

    def __init__(self, decay: float = 0.88, max_items: int = 96):
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.decay = float(decay)
        self.max_items = int(max_items)
        self._values: Dict[Tuple[Hashable, float], float] = {}
        self._order: Deque[Tuple[Hashable, float]] = deque()

    def clear(self) -> None:
        self._values.clear()
        self._order.clear()

    def step(self) -> None:
        dead = []
        for item, value in list(self._values.items()):
            nv = value * self.decay
            if nv < 0.045:
                dead.append(item)
            else:
                self._values[item] = nv
        for item in dead:
            self._values.pop(item, None)

    def add(self, keys: Iterable[Hashable], action: float) -> None:
        for key in keys:
            item = (key, float(action))
            if item not in self._values:
                self._order.append(item)
            self._values[item] = 1.0
        while len(self._order) > self.max_items:
            old = self._order.popleft()
            self._values.pop(old, None)

    def items(self) -> List[Tuple[Hashable, float, float]]:
        return [(k, a, e) for (k, a), e in self._values.items()]


class CortexTable:
    """Non-neural associative value memory with overlapping abstractions."""

    def __init__(self, actions: Iterable[float], seed: int = 0):
        self.actions = tuple(float(a) for a in actions)
        self.rng = random.Random(seed)
        self.table = defaultdict(lambda: {a: ValueStat() for a in self.actions})
        self.total_updates = 0

    def _estimate(self, keys: Iterable[Hashable], action: float) -> Tuple[float, int, float]:
        weighted = 0.0
        weight_sum = 0.0
        evidence = 0
        disagreement = 0.0
        for level, key in enumerate(keys):
            stat = self.table[key][action]
            specificity = 1.0 / (1.0 + 0.20 * level)
            confidence = min(1.0, stat.n / 7.0)
            w = specificity * (0.25 + 0.75 * confidence)
            value = 0.70 * stat.fast + 0.30 * stat.slow
            weighted += w * value
            weight_sum += w
            evidence += stat.n
            disagreement = max(disagreement, abs(stat.fast - stat.slow))
        return (weighted / max(weight_sum, 1e-9), evidence, disagreement)

    def values(self, keys: Iterable[Hashable]) -> Dict[float, float]:
        keys = tuple(keys)
        total = max(2, self.total_updates + 2)
        out = {}
        for a in self.actions:
            base, evidence, _ = self._estimate(keys, a)
            # Symmetric uncertainty bonus. No direction is privileged.
            bonus = 0.17 * math.sqrt(math.log(total) / (1.0 + evidence))
            out[a] = base + bonus
        return out

    def raw_values(self, keys: Iterable[Hashable]) -> Dict[float, float]:
        """Value estimates without the uncertainty bonus (for residual memories)."""
        keys = tuple(keys)
        return {a: self._estimate(keys, a)[0] for a in self.actions}

    def choose(self, keys: Iterable[Hashable], epsilon: float) -> Tuple[float, Dict[float, float], bool]:
        keys = tuple(keys)
        vals = self.values(keys)
        exploratory = self.rng.random() < epsilon
        if exploratory:
            return self.rng.choice(self.actions), vals, True
        m = max(vals.values())
        best = [a for a, v in vals.items() if abs(v - m) < 1e-12]
        return self.rng.choice(best), vals, False

    def apply_sparse(self, trace: EligibilityTrace, reward: float) -> int:
        count = 0
        for key, action, eligibility in trace.items():
            self.table[key][action].update(float(reward) * float(eligibility))
            count += 1
        self.total_updates += count
        return count

    def surprise(self, keys: Iterable[Hashable], action: float) -> float:
        vals = []
        for key in keys:
            stat = self.table[key][action]
            if stat.n:
                vals.append(abs(stat.fast - stat.slow))
        return max(vals) if vals else 0.0

    def to_dict(self) -> dict:
        rows = []
        for key, amap in self.table.items():
            if not any(stat.n for stat in amap.values()):
                continue
            rows.append({
                "key": repr(key),
                "actions": {str(a): stat.to_dict() for a, stat in amap.items() if stat.n},
            })
        return {
            "actions": list(self.actions),
            "total_updates": int(self.total_updates),
            "rows": rows,
        }

    @classmethod
    def from_dict(cls, data: dict, seed: int = 0) -> "CortexTable":
        obj = cls(data.get("actions", ()), seed=seed)
        obj.total_updates = int(data.get("total_updates", 0))
        for row in data.get("rows", []):
            key = ast.literal_eval(row["key"])
            amap = obj.table[key]
            for a_str, stat_d in row.get("actions", {}).items():
                a = float(a_str)
                if a in amap:
                    amap[a] = ValueStat.from_dict(stat_d)
        return obj
