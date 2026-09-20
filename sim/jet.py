"""Octopus locomotion animator: jet propulsion with reactive arms.

An octopus doesn't glide like a fish: it fires a mantle pulse (a short
thrust), then glides while the momentum decays, then fires again. The
arms are not decoration — they react to the motion phase: streaming
behind during a thrust, relaxing outward during the glide, gathering
under the body at rest.

This module owns the *visual* motion only. The Q-brain still decides
where to go; here we turn its steady swimming into pulses and glides.
"""

import math
from dataclasses import dataclass, field

from .creature import Creature, Vec
from .geometry import lerp, wrap_angle

PULSE_PERIOD = 1.1  # seconds between thrust pulses at full effort
THRUST_TIME = 0.25  # seconds of active thrust inside one period
THRUST_EASE = 0.35  # per-frame share of the thrust velocity to adopt
GLIDE_DECAY = 2.2  # per-second exponential decay of glide velocity
TURN_RATE = 0.10  # per-frame share of the heading gap to close
BEND_RATE = 0.10  # per-frame share of the body-bend gap to close
MAX_BEND = 0.45  # radians: the rear never bends more than this
PULSE_BOOST = 1.6  # peak visual speed multiplier during a thrust


@dataclass
class JetAnimator:
    """Smoothed render state for one octopus: pulse phase + eased pose."""

    pos: Vec = field(default_factory=lambda: Vec(0.0, 0.0))
    heading: float = 0.0
    bend: float = 0.0
    # motion phase
    pulse_timer: float = 0.0  # seconds since the last thrust started
    thrust_vel: Vec = field(default_factory=lambda: Vec(0.0, 0.0))
    effort: float = 0.0  # 0..1, how hard the last pulse fired (for the arms)
    initialized: bool = False

    @property
    def pulse_phase(self) -> float:
        """0..1 through the current pulse cycle."""
        return min(1.0, self.pulse_timer / PULSE_PERIOD)

    @property
    def thrusting(self) -> bool:
        """Is the mantle actively pushing right now?"""
        return self.pulse_timer < THRUST_TIME

    def update(self, creature: Creature, dt: float) -> None:
        """Advance the pulse cycle and ease the pose toward the sim truth."""
        if not self.initialized:
            self.pos = creature.pos
            self.heading = creature.heading
            self.initialized = True
            return
        self._advance_pulse(creature, dt)
        self._ease_pose(creature, dt)

    def _advance_pulse(self, creature: Creature, dt: float) -> None:
        """Fire pulses on rhythm; decay glide velocity between them."""
        self.pulse_timer += dt
        if self.pulse_timer >= PULSE_PERIOD:
            self.pulse_timer = 0.0
            # thrust along the sim's heading, scaled by its actual speed
            speed = math.hypot(creature.pos.x - self.pos.x,
                               creature.pos.y - self.pos.y) / max(dt, 1e-6)
            effort = min(1.0, speed / 0.15) if speed > 0.01 else 0.0
            self.effort = effort
            self.thrust_vel = Vec(
                x=math.cos(creature.heading) * effort * PULSE_BOOST,
                y=math.sin(creature.heading) * effort * PULSE_BOOST,
            )
        decay = math.exp(-GLIDE_DECAY * dt)
        self.thrust_vel = Vec(x=self.thrust_vel.x * decay,
                              y=self.thrust_vel.y * decay)
        drift = self.thrust_vel.x * dt, self.thrust_vel.y * dt
        self.pos = Vec(x=self.pos.x + drift[0], y=self.pos.y + drift[1])

    def _ease_pose(self, creature: Creature, dt: float) -> None:
        """Pull the rendered pose toward the sim's truth."""
        rate = 1.0 - math.exp(-dt / 0.08)
        self.pos = Vec(
            x=lerp(self.pos.x, creature.pos.x, rate),
            y=lerp(self.pos.y, creature.pos.y, rate),
        )
        gap = wrap_angle(creature.heading - self.heading)
        self.heading = wrap_angle(self.heading + gap * rate)
        target_bend = max(-MAX_BEND, min(MAX_BEND, gap * 6.0))
        self.bend = lerp(self.bend, target_bend, BEND_RATE)
