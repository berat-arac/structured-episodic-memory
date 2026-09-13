from __future__ import annotations

import random


class DiverseCoach:
    """Strong but varied hand-written training opponent.

    This opponent is allowed to be engineered; it exists only to generate a
    diverse curriculum. It uses the *current* ball position and tries to meet it
    at different parts of its paddle, which creates varied return angles. No
    information from this controller is exposed to SEM.
    """

    STYLE_CONTACTS = {
        "center": (0.0, 0.0, -0.18, 0.18),
        "spinner": (-0.72, -0.52, -0.28, 0.28, 0.52, 0.72),
        "top_bias": (-0.72, -0.58, -0.42, -0.20, 0.0),
        "bottom_bias": (0.0, 0.20, 0.42, 0.58, 0.72),
        "mixed": (-0.74, -0.50, -0.25, 0.0, 0.25, 0.50, 0.74),
        "alternator": (-0.68, 0.68, -0.44, 0.44, -0.20, 0.20),
    }

    def __init__(self, seed: int, style: str = "mixed", max_speed: float = 1020.0,
                 gain: float = 8.5, reaction_frames: int = 2, noise: float = 0.0):
        if style not in self.STYLE_CONTACTS:
            raise ValueError(f"unknown style: {style}")
        self.rng = random.Random(seed)
        self.style = style
        self.max_speed = float(max_speed)
        self.gain = float(gain)
        self.reaction_frames = max(1, int(reaction_frames))
        self.noise = float(noise)
        self.hold = 0
        self.cached_velocity = 0.0
        self.last_vx_sign = None
        self.return_index = 0
        self.desired_contact = 0.0

    def _pick_contact(self):
        options = self.STYLE_CONTACTS[self.style]
        if self.style == "alternator":
            c = options[self.return_index % len(options)]
        else:
            c = self.rng.choice(options)
        self.return_index += 1
        return float(c)

    def act(self, obs):
        vx_sign = -1 if obs["ball_vx"] < 0 else 1
        if vx_sign != self.last_vx_sign:
            self.last_vx_sign = vx_sign
            if vx_sign < 0:
                self.desired_contact = self._pick_contact()
            self.hold = 0

        if self.hold > 0:
            self.hold -= 1
            return self.cached_velocity

        if obs["ball_vx"] < 0:
            # To make contact at +c, the paddle centre should be below the ball
            # by c * half-paddle. This is based on current geometry only.
            target_y = obs["ball_y"] - self.desired_contact * (obs["paddle_h"] / 2.0)
            error = target_y - obs["left_y"]
            v = error * self.gain
            if self.noise:
                v += self.rng.uniform(-self.noise, self.noise)
        else:
            # Re-centre while waiting for the next incoming ball.
            error = obs["height"] * 0.5 - obs["left_y"]
            v = error * min(4.2, self.gain * 0.50)

        v = max(-self.max_speed, min(self.max_speed, v))
        self.cached_velocity = v
        self.hold = self.reaction_frames - 1
        return v


STYLE_ORDER = ("center", "spinner", "top_bias", "bottom_bias", "mixed", "alternator")


def build_coach(seed: int, run_index: int) -> DiverseCoach:
    """Deterministically vary style, reaction time, gain and speed per run."""
    rng = random.Random((seed + 1) * 1009 + run_index * 9176)
    style = STYLE_ORDER[run_index % len(STYLE_ORDER)]
    max_speed = rng.uniform(900.0, 1080.0)
    gain = rng.uniform(7.2, 10.2)
    reaction_frames = rng.choice((1, 2, 2, 3, 4))
    noise = rng.choice((0.0, 0.0, 10.0, 18.0))
    return DiverseCoach(
        seed=seed ^ (run_index * 7919),
        style=style,
        max_speed=max_speed,
        gain=gain,
        reaction_frames=reaction_frames,
        noise=noise,
    )
