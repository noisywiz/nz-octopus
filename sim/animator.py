"""Per-creature animation state: smooths the sim's discrete jumps into
continuous on-screen motion.

The simulation moves and turns in whole ticks (up to 7.5 deg per tick),
which makes the rendered body snap around its axis every frame. The
animator keeps its own slowly-converging copy of position and heading
and eases toward the sim's truth each frame, so the body glides and the
tail end lags behind the nose like a real soft body.
"""

import math
from dataclasses import dataclass, field

from .creature import Creature, Vec
from .geometry import lerp, wrap_angle

TURN_SMOOTHING = 0.12  # per-frame share of the heading gap to close
TURN_LAG = 0.35  # how far behind the nose the tail end trails (radians)
BEND_RATE = 0.10  # per-frame share of the body-bend gap to close
MAX_BEND = 0.45  # radians: the tail end never bends more than this


@dataclass
class Animator:
    """Smoothed render state for one creature. Owns no sim truth."""

    pos: Vec = field(default_factory=lambda: Vec(0.0, 0.0))
    heading: float = 0.0
    bend: float = 0.0  # signed bend of the tail half, radians
    initialized: bool = False

    def update(self, creature: Creature, dt: float) -> None:
        """Ease toward the sim's position and heading; bend from turning."""
        rate = 1.0 - math.exp(-dt / 0.08)  # ~80 ms time constant
        if not self.initialized:
            self.pos = creature.pos
            self.heading = creature.heading
            self.initialized = True
            return
        self.pos = Vec(
            x=lerp(self.pos.x, creature.pos.x, rate),
            y=lerp(self.pos.y, creature.pos.y, rate),
        )
        gap = wrap_angle(creature.heading - self.heading)
        self.heading = wrap_angle(self.heading + gap * rate)
        # turning bows the tail outward: the bend follows the recent turn
        target_bend = max(-MAX_BEND, min(MAX_BEND, gap * 6.0))
        self.bend = lerp(self.bend, target_bend, BEND_RATE)
