"""nz-octopus: a single creature that lives and learns in a 2D tank."""

from .brain import BrainConfig, QBrain
from .creature import N_DIRECTIONS, Body, Creature, Vec
from .sensors import N_STATES, Reading
from .geometry import angle_sector
from .world import Food, World

DIR_ARROW_BY_SECTOR = ("→", "↘", "↓", "↙", "←", "↖", "↑", "↗")

__all__ = [
    "Body",
    "BrainConfig",
    "Creature",
    "DIR_ARROW_BY_SECTOR",
    "Food",
    "N_DIRECTIONS",
    "N_STATES",
    "QBrain",
    "Reading",
    "Vec",
    "World",
    "angle_sector",
]
