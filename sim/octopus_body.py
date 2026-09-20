"""Functional octopus arms: joint chains that reach, grab and carry.

Arms are reflexes, not learned policy: the Q-brain steers the body by
scent, while an arm simply extends its tip toward the nearest piece
within reach, grips it, and passes it to the mouth. Eating happens only
at the mouth, so the learning signal stays where the reward belongs.

Each arm is a chain of joints solved follow-the-leader style (one
FABRIK-like pass): the tip moves a bounded step toward its target, then
every joint is pulled back to segment length with the base pinned to the
mantle. Bounded steps and length constraints keep the motion smooth.
"""

import math
from dataclasses import dataclass

from . import food as fd
from .creature import Vec
from .geometry import distance

N_ARMS = 8
JOINTS = 8  # joints per arm, base included
REACH = 4.0  # full arm length in world units
SEGMENT = REACH / (JOINTS - 1)
TIP_SPEED = 0.08  # max tip movement per tick: no snapping
LUNGE_SPEED = 0.30  # final stretch: must outpace a swimming body's retreat
LUNGE_DISTANCE = 1.5  # tip-to-food distance that triggers the lunge
CARRY_SPEED = 0.2  # a gripping arm folds faster than it gropes
GRAB_RADIUS = 0.35  # tip-to-food distance that counts as a grip
MOUTH_RADIUS = 1.2  # tip-to-mouth distance that counts as a swallow
MOUTH_OFFSET = 2.0  # mouth sits this far ahead of the body center
BASE_OFFSET = 1.2  # arm bases sit this far behind the body center
BASE_SPREAD = 1.1  # radians of fan covered by the bases
REST_REACH = 2.4  # how far a resting arm stretches from its base
REST_SWAY = 0.25  # radians of idle sway
REST_RATE = 0.05  # sway speed, radians per tick
GRASP_MARGIN = 1.15  # food is targeted within this multiple of REACH


@dataclass
class Arm:
    """One arm: joints[0] is the base, joints[-1] the tip."""

    index: int  # position in the fan, 0..N_ARMS-1
    joints: list[Vec]
    holding: fd.Food | None = None  # piece gripped by the tip
    carry: Vec | None = None  # current position of the held piece
    target_fid: int | None = None  # fid of the piece this arm reaches for


@dataclass
class Arms:
    """All eight arms of one octopus."""

    limbs: list[Arm]


def mouth_point(pos: Vec, heading: float) -> Vec:
    """The mouth: ahead of the body center along the heading."""
    return Vec(
        x=pos.x + math.cos(heading) * MOUTH_OFFSET,
        y=pos.y + math.sin(heading) * MOUTH_OFFSET,
    )


def base_point(pos: Vec, heading: float, index: int) -> Vec:
    """Where arm `index` attaches, fanned around the rear of the mantle."""
    fan = (index / (N_ARMS - 1) - 0.5) * BASE_SPREAD
    h = heading + math.pi + fan
    return Vec(
        x=pos.x + math.cos(h) * BASE_OFFSET,
        y=pos.y + math.sin(h) * BASE_OFFSET,
    )


def rest_tip(base: Vec, heading: float, index: int, ticks: int) -> Vec:
    """Where an idle arm's tip drifts: a loose fan with a slow sway."""
    fan = (index / (N_ARMS - 1) - 0.5) * 2.2
    h = heading + math.pi + fan + math.sin(ticks * REST_RATE + index * 0.9) * REST_SWAY
    return Vec(
        x=base.x + math.cos(h) * REST_REACH,
        y=base.y + math.sin(h) * REST_REACH,
    )


def tip_of(arm: Arm) -> Vec:
    """The arm's tip joint."""
    return arm.joints[-1]


def _straight_chain(base: Vec, heading: float, index: int) -> list[Vec]:
    """Initial chain pointing away from the body, joints evenly spaced."""
    fan = (index / (N_ARMS - 1) - 0.5) * BASE_SPREAD
    h = heading + math.pi + fan
    return [
        Vec(
            x=base.x + math.cos(h) * SEGMENT * i,
            y=base.y + math.sin(h) * SEGMENT * i,
        )
        for i in range(JOINTS)
    ]


def create(pos: Vec, heading: float) -> Arms:
    """Eight fresh arms for a newborn octopus."""
    return Arms(limbs=[
        Arm(index=i, joints=_straight_chain(base_point(pos, heading, i), heading, i))
        for i in range(N_ARMS)
    ])


def _pull(pivot: Vec, point: Vec, seg: float) -> Vec:
    """Move `point` to exactly `seg` away from `pivot`, keeping direction."""
    d = distance(pivot.x, pivot.y, point.x, point.y)
    if d < 1e-6:
        return point
    return Vec(
        x=pivot.x + (point.x - pivot.x) / d * seg,
        y=pivot.y + (point.y - pivot.y) / d * seg,
    )


def solve_chain(joints: list[Vec], base: Vec) -> list[Vec]:
    """One FABRIK-like pass: backward from the tip, forward from the base."""
    chain = list(joints)
    for i in range(len(chain) - 2, -1, -1):
        chain[i] = _pull(chain[i + 1], chain[i], SEGMENT)
    chain[0] = base
    for i in range(1, len(chain)):
        chain[i] = _pull(chain[i - 1], chain[i], SEGMENT)
    return chain


def move_tip(arm: Arm, base: Vec, target: Vec, speed: float = TIP_SPEED) -> list[Vec]:
    """Advance the tip toward `target` by at most `speed`, re-solve."""
    t = tip_of(arm)
    d = distance(t.x, t.y, target.x, target.y)
    if d <= LUNGE_DISTANCE:
        speed = max(speed, LUNGE_SPEED)  # final stretch outpaces a swimming body
    if d <= 1e-6:
        new_tip = t
    else:
        step = min(speed, d)
        new_tip = Vec(
            x=t.x + (target.x - t.x) / d * step,
            y=t.y + (target.y - t.y) / d * step,
        )
    return solve_chain(arm.joints[:-1] + [new_tip], base)


def nearest_food(point: Vec, food: list[fd.Food]) -> tuple[int, float] | None:
    """Index and distance of the piece closest to `point`, if any."""
    best: tuple[int, float] | None = None
    for i, f in enumerate(food):
        d = distance(point.x, point.y, f.pos.x, f.pos.y)
        if best is None or d < best[1]:
            best = (i, d)
    return best


def claim_food(arms: Arms, food: list[fd.Food]) -> None:
    """Assign each free piece to the arm whose base is closest to it.

    Assignment happens once, when a piece enters reach: the arm keeps
    chasing its piece until the piece is gripped or leaves reach. Pieces
    are tracked by stable `fid`, because physics re-creates the objects
    every tick.
    """
    for piece in food:
        if any(a.target_fid == piece.fid for a in arms.limbs):
            continue  # already owned, keep the commitment
        taker = min(arms.limbs,
                    key=lambda a: distance(a.joints[0].x, a.joints[0].y,
                                           piece.pos.x, piece.pos.y))
        taker.target_fid = piece.fid
