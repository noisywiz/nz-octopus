"""Chemosensory system: the creature's ONLY food sense is scent."""

import math
from dataclasses import dataclass
from math import cos, sin

from .creature import Body, Vec

PLUME_SPREAD = 8.0  # world units: how fast scent dilutes (linear in d, not d^2:
# a quadratic plume saturates near the food, the gradient flattens and then
# inverts, and the creature orbits a piece it can almost touch)
RECEPTOR_AHEAD = 0.8  # receptor offset straight ahead of the body; must be
# smaller than EAT_RADIUS (2.0), or the "nose" senses past the "mouth"
RECEPTOR_SIDE = 0.6  # forward and lateral offsets of the side receptors

GRAD_RISING, GRAD_FALLING, GRAD_FLAT = 0, 1, 2
STEER_LEFT, STEER_RIGHT, STEER_SYMMETRIC = 0, 1, 2
N_GRAD, N_STEER, N_INTENSITY = 3, 3, 4
N_STATES = N_GRAD * N_STEER * N_INTENSITY

_RISE_RATIO = 1.005  # ahead must exceed here by 0.5% to count as "rising"
_STEER_RATIO = 1.03  # one side must exceed the other by 3% to count as steering
_INTENSITY_THRESHOLDS = (0.75, 0.55, 0.35)  # linear plume: 0.75 ~ d<2.7, 0.55 ~ d<6, 0.35 ~ d<13


@dataclass(frozen=True)
class Receptors:
    """Scent intensity at the body and at three receptor points around it."""

    here: float
    ahead: float
    left: float
    right: float


@dataclass(frozen=True)
class Reading:
    """Discretized chemosensory reading — the creature's entire food perception."""

    gradient: int  # scent rising / falling / flat along the heading
    steer: int  # food to the left / right / symmetric
    intensity: int  # absolute closeness, coarse (4 levels)

    def index(self) -> int:
        """Pack into one Q-table state index."""
        return (self.gradient * N_STEER + self.steer) * N_INTENSITY + self.intensity


def scent_at(point: Vec, food: list[Vec]) -> float:
    """Scent concentration at a point: sum of plumes from all food.

    Falls off linearly in distance (1 / (1 + d/k)): contrast between two
    nearby points stays proportional to their distance gap all the way to
    the source. A quadratic law saturates, killing the contrast close-in.
    """
    return sum(
        1.0 / (1.0 + math.hypot(f.x - point.x, f.y - point.y) / PLUME_SPREAD)
        for f in food
    )


def receptor_readings(body: Body, food: list[Vec]) -> Receptors:
    """Smell at the body and at ahead/left/right receptor points."""
    ca, sa = cos(body.heading), sin(body.heading)
    px, py = body.pos.x, body.pos.y
    ahead = Vec(px + ca * RECEPTOR_AHEAD, py + sa * RECEPTOR_AHEAD)
    left = Vec(px + ca * RECEPTOR_SIDE - sa * RECEPTOR_SIDE,
               py + sa * RECEPTOR_SIDE + ca * RECEPTOR_SIDE)
    right = Vec(px + ca * RECEPTOR_SIDE + sa * RECEPTOR_SIDE,
                py + sa * RECEPTOR_SIDE - ca * RECEPTOR_SIDE)
    return Receptors(
        here=scent_at(body.pos, food),
        ahead=scent_at(ahead, food),
        left=scent_at(left, food),
        right=scent_at(right, food),
    )


def read(receptors: Receptors) -> Reading:
    """Compress receptor intensities into a discrete reading."""
    if receptors.ahead > receptors.here * _RISE_RATIO:
        gradient = GRAD_RISING
    elif receptors.ahead < receptors.here / _RISE_RATIO:
        gradient = GRAD_FALLING
    else:
        gradient = GRAD_FLAT
    return Reading(
        gradient=gradient,
        steer=steer_side(receptors.left, receptors.right),
        intensity=_intensity(receptors.here),
    )


def steer_side(left: float, right: float) -> int:
    """Which side smells stronger (with a dead zone for symmetry)."""
    if left > right * _STEER_RATIO:
        return STEER_LEFT
    if right > left * _STEER_RATIO:
        return STEER_RIGHT
    return STEER_SYMMETRIC


def _intensity(here: float) -> int:
    """Coarse distance proxy from the absolute scent at the body."""
    for level, threshold in enumerate(_INTENSITY_THRESHOLDS):
        if here > threshold:
            return level
    return N_INTENSITY - 1
