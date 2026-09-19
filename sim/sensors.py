"""Chemosensory system: the creature's ONLY food sense is scent."""

from dataclasses import dataclass
from math import cos, sin

from .creature import Body, Vec

PLUME_SPREAD = 30.0  # world units^2: how fast scent dilutes
RECEPTOR_AHEAD = 2.0  # receptor offset straight ahead of the body
RECEPTOR_SIDE = 1.5  # forward and lateral offsets of the side receptors

GRAD_RISING, GRAD_FALLING, GRAD_FLAT = 0, 1, 2
STEER_LEFT, STEER_RIGHT, STEER_SYMMETRIC = 0, 1, 2
N_GRAD, N_STEER, N_INTENSITY = 3, 3, 4
N_STATES = N_GRAD * N_STEER * N_INTENSITY

_RISE_RATIO = 1.01  # ahead must exceed here by 1% to count as "rising"
_STEER_RATIO = 1.05  # one side must exceed the other by 5% to count as steering
_INTENSITY_THRESHOLDS = (0.5, 0.15, 0.04)


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
    """Scent concentration at a point: sum of plumes from all food."""
    return sum(
        1.0 / (1.0 + ((f.x - point.x) ** 2 + (f.y - point.y) ** 2) / PLUME_SPREAD)
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
