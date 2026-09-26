"""Sinking food: pieces drop from the surface one by one and dissolve
on the floor.

Food is the environment's half of the loop, so its physics live in pure
functions here: the world only swaps immutable pieces in and out. Pieces
appear at the surface in a random column, sink slowly while wobbling
sideways (a smooth sine plus a tiny random jitter — no snapping), and
dissolve when they touch the bottom, so food never waits on the floor.
"""

import itertools
import math
import random
from dataclasses import dataclass, field

from .creature import Vec

FOOD_ENERGY = 40.0
SINK_SPEED = 0.02  # world units per tick: ~25 s of screen time per piece
WOBBLE_AMPLITUDE = 0.3  # small sway: real sinking food flutters, not swings
WOBBLE_RATE = 0.08  # radians per tick: faster flutter than the old big sine
JITTER = 0.02  # random per-tick drift of the column: paths never repeat
SURFACE_Y = 1.0  # spawn height, below the top wall
FLOOR_PAD = 1.0  # the piece dissolves this high above the bottom wall
SPAWN_PAD = 6.0  # columns never spawn this close to a wall
SPAWN_INTERVAL = 300  # ticks between two pieces entering the tank
MAX_PIECES = 3  # hard cap: the tank never floods with food


@dataclass(frozen=True)
class Food:
    """A piece of food: its wobble phase, and the energy it restores.

    `base_x` is the slowly drifting center column; the smelled position is
    `base_x` plus the sine wobble, so the plume sways with the piece.
    `fid` is a stable identity: physics re-creates the object every tick,
    so holders (octopus arms) track pieces by fid, not by object identity.
    """

    base_x: float
    y: float
    phase: float
    energy: float = FOOD_ENERGY
    fid: int = field(default=-1, compare=False)

    @property
    def pos(self) -> Vec:
        """Current position: the world point the scent plume radiates from."""
        return Vec(
            x=self.base_x + math.sin(self.phase) * WOBBLE_AMPLITUDE,
            y=self.y,
        )


_ids = itertools.count()


def _next_id() -> int:
    """A fresh stable id for a newly spawned piece."""
    return next(_ids)


def spawn(rng: random.Random, width: float) -> Food:
    """A new piece at the surface, in a column kept away from the walls."""
    return Food(
        base_x=rng.uniform(SPAWN_PAD, width - SPAWN_PAD),
        y=SURFACE_Y,
        phase=rng.uniform(0, 2.0 * math.pi),
        fid=_next_id(),
    )


def sunk(food: Food, rng: random.Random, width: float, height: float) -> Food | None:
    """One tick of physics: sink, sway, jitter — or dissolve, if landed."""
    if landed(food, height):
        return None
    base_x = food.base_x + rng.uniform(-JITTER, JITTER)
    base_x = min(width - SPAWN_PAD, max(SPAWN_PAD, base_x))
    return Food(
        base_x=base_x,
        y=min(height - FLOOR_PAD, food.y + SINK_SPEED),
        phase=food.phase + WOBBLE_RATE,
        energy=food.energy,
        fid=food.fid,
    )


def landed(food: Food, height: float) -> bool:
    """Has the piece reached the floor (and should dissolve)?"""
    return food.y >= height - FLOOR_PAD


def spawn_due(piece_count: int, ticks_since_last: int, cap: float = MAX_PIECES,
              interval: float = SPAWN_INTERVAL) -> bool:
    """Time for the next piece: one at a time, on a steady rhythm.

    The cap and interval are parameters so a narrow tank can receive
    food more slowly than the default: the same rhythm in a sliver of
    water piles every piece at the top.
    """
    return piece_count < cap and ticks_since_last >= interval
