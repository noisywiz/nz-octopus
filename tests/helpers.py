"""Shared builders for the tests: fresh brains and small worlds."""

import random

from src.sim import world as wd
from src.sim.brain import QBrain
from src.sim.creature import N_DIRECTIONS
from src.sim.sensors import N_STATES
from src.sim.world import CreatureBrain


def make_brain() -> QBrain:
    """An empty Q-table sized for the real sensor/action space."""
    return QBrain(N_STATES, N_DIRECTIONS)


def make_world(creatures: list[CreatureBrain], seed: int = 7) -> wd.World:
    """A deterministic world for reproducible assertions."""
    return wd.World(creatures=creatures, rng=random.Random(seed))
