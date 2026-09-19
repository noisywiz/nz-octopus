"""Mutable state of the creature plus all pure functions over it."""

import math
import random
from dataclasses import dataclass, field

from .geometry import TWO_PI, turn_toward

N_DIRECTIONS = 8  # action space: 8 headings + rest
SWIM_SPEED = 0.15
REST_DRIFT = 0.05
MAX_TURN = math.pi / 24  # 7.5 deg per tick: no snapping between sectors
WALL_PUSH = 0.2  # physical push-off, must exceed SWIM_SPEED
WALL_TURN = 2.5 * MAX_TURN  # reflex turn away from a wall


@dataclass(frozen=True)
class Vec:
    """A 2D point/velocity in world units."""

    x: float
    y: float


@dataclass(frozen=True)
class Body:
    """Everything physical about the creature at one instant."""

    pos: Vec
    heading: float  # radians, full 2D
    starving: float  # 0..1 weakness factor from hunger


@dataclass
class Creature:
    """The living state; only world.step() mutates it, via the functions below."""

    pos: Vec
    heading: float
    hunger: float
    starving: float
    bumping: bool = False
    trail: list[Vec] = field(default_factory=list)

    @property
    def body(self) -> Body:
        """Read-only snapshot for sensors and renderers."""
        return Body(pos=self.pos, heading=self.heading, starving=self.starving)

    def clamp_trail(self, length: int) -> None:
        """Keep only the last `length` trail points."""
        del self.trail[:-length]


def action_heading(action: int) -> float:
    """Target heading of a swim action (0..7)."""
    return action * (TWO_PI / N_DIRECTIONS)


def weaken(speed: float, starving: float) -> float:
    """Hunger slows swimming down to 40% of full speed."""
    return speed * (1.0 - 0.6 * starving)


def move(creature: Creature, action: int) -> None:
    """Apply one action: turn smoothly, then swim (or drift, if resting)."""
    if action < N_DIRECTIONS:
        creature.heading = turn_toward(creature.heading, action_heading(action), MAX_TURN)
        speed = weaken(SWIM_SPEED, creature.starving)
    elif creature.starving > 0.0:
        speed = REST_DRIFT * creature.starving  # cannot afford to stand still
    else:
        speed = 0.0
    creature.pos = Vec(
        x=creature.pos.x + math.cos(creature.heading) * speed,
        y=creature.pos.y + math.sin(creature.heading) * speed,
    )
    if speed > 0.0:
        creature.trail.append(creature.pos)
        creature.clamp_trail(TRAIL_LENGTH)


TRAIL_LENGTH = 8


def push_off_wall(point: Vec, low: float, high: float, axis: int, inward: float) -> Vec:
    """Nudge a point inside [low, high] on one axis by `inward` units.

    `inward` must point away from the wall the point is touching: the caller
    decides the sign, this function just applies and clamps it.
    """
    if axis == 0:
        return Vec(min(high, max(low, point.x + inward)), point.y)
    return Vec(point.x, min(high, max(low, point.y + inward)))


def at_wall(point: Vec, margin: float, width: float, height: float) -> tuple[bool, bool]:
    """Is the point touching a wall? Returns (at_vertical, at_horizontal)."""
    at_v = point.x <= margin or point.x >= width - margin
    at_h = point.y <= margin or point.y >= height - margin
    return at_v, at_h


def wall_escape(creature: Creature, margin: float, width: float, height: float) -> None:
    """Physical reflex while touching a wall: push off + turn away.

    The push must be stronger than the strongest swim action, otherwise the
    policy can cancel it and the creature sticks to the wall forever.
    """
    at_v, at_h = at_wall(creature.pos, margin, width, height)
    if at_v:
        inward = WALL_PUSH if creature.pos.x <= margin else -WALL_PUSH
        outward = 0.0 if creature.pos.x <= margin else math.pi  # 0 = +x = away from left wall
        creature.pos = push_off_wall(creature.pos, margin, width - margin, axis=0, inward=inward)
        creature.heading = turn_toward(creature.heading, outward, WALL_TURN)
    if at_h:
        inward = WALL_PUSH if creature.pos.y <= margin else -WALL_PUSH
        outward = math.pi / 2 if creature.pos.y <= margin else -math.pi / 2  # +y is downward on screen
        creature.pos = push_off_wall(creature.pos, margin, height - margin, axis=1, inward=inward)
        creature.heading = turn_toward(creature.heading, outward, WALL_TURN)


def reset_bump_flag(creature: Creature, at_v: bool, at_h: bool) -> None:
    """Remember wall contact between ticks so bumps are counted once."""
    creature.bumping = at_v or at_h


def touch(point: Vec, food: Vec, radius: float) -> bool:
    """Is a piece of food within eating range?"""
    return math.hypot(food.x - point.x, food.y - point.y) < radius
