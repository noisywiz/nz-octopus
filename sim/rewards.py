"""Pure reward functions. Every lesson the creature learns lives here."""

from dataclasses import dataclass

from .metabolism import MetabolicReport

WALL_BUMP_PENALTY = -2.0
FOOD_REWARD = 10.0


@dataclass(frozen=True)
class Reward:
    """Total reward for one tick, split by source (for stats and debugging)."""

    total: float
    food: float
    wall: float
    metabolism: float


def wall_bump(bumped: bool) -> float:
    """Penalty on the FIRST tick of wall contact only; sliding is free."""
    return WALL_BUMP_PENALTY if bumped else 0.0


def food_reward(ate: bool) -> float:
    """One-shot reward for each piece of food eaten this tick."""
    return FOOD_REWARD if ate else 0.0


def combine(food_r: float, wall_r: float, metabolic: MetabolicReport) -> Reward:
    """Sum all reward sources into one tick's signal."""
    total = food_r + wall_r + metabolic.reward
    return Reward(total=total, food=food_r, wall=wall_r, metabolism=metabolic.reward)
