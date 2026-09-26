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
    bloat: float = 0.0  # 0..1 fullness: overeating swells the body into a ball
    stall_anchor: Vec | None = None  # position STALL_WINDOW ticks ago
    stall_ticks: int = 0  # ticks spent within STALL_RADIUS of stall_anchor
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


BLOAT_SLOWDOWN = 0.92  # a stuffed ball swims at 8% speed: it just hangs there


def bloat_reach(creature: Creature) -> float:
    """Half of the contact distance; the swollen ball pushes neighbors from afar."""
    return (CONTACT_RADIUS / 2) * (1.0 + 1.2 * creature.bloat)


def move(creature: Creature, action: int) -> None:
    """Apply one action: turn smoothly, then swim (or drift, if resting)."""
    if action < N_DIRECTIONS:
        creature.heading = turn_toward(creature.heading, action_heading(action), MAX_TURN)
        speed = weaken(SWIM_SPEED, creature.starving)
    elif creature.starving > 0.0:
        speed = REST_DRIFT * creature.starving  # cannot afford to stand still
    else:
        speed = 0.0
    speed *= 1.0 - BLOAT_SLOWDOWN * creature.bloat  # a ball of food barely moves
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


CORNER_PUSH = 0.15  # gentle draft away from a corner, on top of per-axis push
STALL_WINDOW = 120  # ticks between displacement checks
STALL_RADIUS = 1.5  # net displacement below this over the window = stuck
STUCK_SURGE = 4.0  # units of the forced launch toward the center


def wall_escape(creature: Creature, margin: float, width: float, height: float,
                floor_line: float | None = None) -> None:
    """Physical reflex while touching a wall: push off + turn away.

    The push must be stronger than the strongest swim action, otherwise the
    policy can cancel it and the creature sticks to the wall forever.
    Stuck detection is deliberately position-agnostic (wall, corner, or open
    water): if net displacement over STALL_WINDOW ticks is tiny, the policy
    is jittering in place and a forced surge toward the center breaks the
    loop. The per-axis push alone cannot fix this: the policy re-picks
    "toward the wall" every tick.

    `floor_line` is the world y of the dune surface under the creature, when
    known: touching it counts as the bottom wall, so the same push-off and
    turn-away reflex applies to dunes.
    """
    at_v, at_h = at_wall(creature.pos, margin, width, height)
    if floor_line is not None and creature.pos.y >= floor_line:
        at_h = True
    if at_v:
        inward = WALL_PUSH if creature.pos.x <= margin else -WALL_PUSH
        outward = 0.0 if creature.pos.x <= margin else math.pi  # 0 = +x = away from left wall
        creature.pos = push_off_wall(creature.pos, margin, width - margin, axis=0, inward=inward)
        creature.heading = turn_toward(creature.heading, outward, WALL_TURN)
    if at_h:
        below_top = floor_line if (
            floor_line is not None and creature.pos.y >= floor_line
        ) else height - margin
        inward = WALL_PUSH if creature.pos.y <= margin else -WALL_PUSH
        outward = math.pi / 2 if creature.pos.y <= margin else -math.pi / 2  # +y is downward on screen
        creature.pos = push_off_wall(
            creature.pos, margin, below_top, axis=1, inward=inward,
        )
        creature.heading = turn_toward(creature.heading, outward, WALL_TURN)
    _unstuck(creature, margin, width, height)


def _unstuck(creature: Creature, margin: float, width: float, height: float) -> None:
    """Watch net displacement and launch the creature free when it stalls."""
    if creature.stall_anchor is None:
        creature.stall_anchor = creature.pos
        return
    anchor = creature.stall_anchor
    moved = math.hypot(creature.pos.x - anchor.x, creature.pos.y - anchor.y)
    if moved >= STALL_RADIUS:
        creature.stall_anchor = creature.pos
        creature.stall_ticks = 0
        return
    creature.stall_ticks += 1
    if creature.stall_ticks >= STALL_WINDOW:
        cx, cy = width / 2, height / 2
        d = math.hypot(cx - creature.pos.x, cy - creature.pos.y)
        if d > 1e-6:
            step = min(STUCK_SURGE, d)
            creature.pos = Vec(
                x=creature.pos.x + (cx - creature.pos.x) / d * step,
                y=creature.pos.y + (cy - creature.pos.y) / d * step,
            )
        creature.stall_anchor = creature.pos
        creature.stall_ticks = 0


def reset_bump_flag(creature: Creature, at_v: bool, at_h: bool) -> None:
    """Remember wall contact between ticks so bumps are counted once."""
    creature.bumping = at_v or at_h


CONTACT_RADIUS = 3.2  # centers closer than this = soft bodies touching;
# matches the rendered body: discs of r=5 art px ≈ 1.67 units, so sprites
# kiss at ~3.3 — anything smaller and bodies visibly overlap while "colliding"
CONTACT_PUSH = 0.4  # max mutual push per tick; head-on pairs settle at
# overlap = CONTACT_RADIUS * SWIM_SPEED / CONTACT_PUSH = 1.2 (a visible squish)


def separate(a: Creature, b: Creature, reach_a: float | None = None,
             reach_b: float | None = None) -> bool:
    """Soft body collision: push both apart in proportion to the overlap.

    Purely physical: no heading change and no reward, so the policy can
    neither sense nor cancel it — a body simply cannot overlap a neighbor.
    The push scales with overlap, so a head-on pair squishes to a stop
    instead of vibrating (a fixed push equal to SWIM_SPEED would cancel
    the approach exactly and freeze them nose-to-nose). Reaches default
    to the normal body size; a bloated ball passes its inflated reach so
    neighbors bounce off the ball, not its former silhouette.
    """
    ra = bloat_reach(a) if reach_a is None else reach_a
    rb = bloat_reach(b) if reach_b is None else reach_b
    dx, dy = b.pos.x - a.pos.x, b.pos.y - a.pos.y
    d = math.hypot(dx, dy)
    limit = ra + rb
    if d >= limit:
        return False
    if d < 1e-6:  # exactly on top of each other: part along +x
        dx, dy, d = 1.0, 0.0, 1.0
    push = CONTACT_PUSH * (limit - d) / limit
    ux, uy = dx / d, dy / d
    a.pos = Vec(a.pos.x - ux * push, a.pos.y - uy * push)
    b.pos = Vec(b.pos.x + ux * push, b.pos.y + uy * push)
    return True


def touch(point: Vec, food: Vec, radius: float) -> bool:
    """Is a piece of food within eating range?"""
    return math.hypot(food.x - point.x, food.y - point.y) < radius
