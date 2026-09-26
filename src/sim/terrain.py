"""Tank floor terrain: a heightmap both the simulation and the renderer use.

The floor is not flat: dunes rise toward the surface in a few wide mounds.
Height is a pure deterministic function of x (fixed seed), so the physics
(clamping, food landing) and the pixel art always agree without sharing
mutable state.
"""

import math
import random

BASE_STEP = 1  # the floor baseline sits this many art pixels below floor_y
MAX_STEP = 7  # the tallest dune rises this many art pixels above baseline
MIN_STEP = -2  # trenches dip this far below the baseline
SURF_INTERVAL = (6, 18)  # level changes every this many art columns
PIX = 4  # art pixel size in world-to-screen pixels; see canvas/aquarium

# World-space constants derived from the art grid: the renderer works in
# art pixels, the sim in world units, so the shared unit is one art column.
ART_TO_WORLD = 0.25  # one art pixel column = this many world units


def _level_sequence(columns: int, seed: int) -> list[int]:
    """Random walk of dune levels, one entry per art column.

    The walk moves at most one art pixel per column, so dune slopes are
    never steeper than 45 degrees: no vertical cliffs.
    """
    rng = random.Random(seed)
    levels: list[int] = []
    step = 0
    until_change = 0
    for _ in range(columns):
        if until_change <= 0:
            until_change = rng.randrange(*SURF_INTERVAL)
            target = rng.choice((-2, -1, 1, 2, 3, 4))
            step += max(-1, min(1, target - step))
            step = max(MIN_STEP, min(MAX_STEP, step))
        until_change -= 1
        levels.append(step)
    return levels


def build(columns: int, seed: int = 20260920) -> list[int]:
    """Dune heights in art pixels above the baseline, indexed by art column."""
    return [step + BASE_STEP for step in _level_sequence(columns, seed)]


def height_at_world(terrain: list[int], x: float, width: float) -> float:
    """Floor height at world x, in world units above the bottom wall."""
    columns = len(terrain)
    col = min(columns - 1, max(0, int(x / width * columns)))
    return terrain[col] * ART_TO_WORLD
