"""Metabolism: hunger grows, eating heals, extreme starvation weakens but never kills."""

from dataclasses import dataclass

HUNGER_RATE = 0.05  # growth per tick
HUNGER_MAX = 100.0
WEAKEN_THRESHOLD = 75.0  # above this, the creature grays out and slows down
STARVE_EVENT = 99.95  # crossing this triggers the one-shot starvation penalty
STARVE_EVENT_PENALTY = -20.0
GROWTH_PENALTY = 0.3  # reward cost per unit of hunger growth
BLOAT_THRESHOLD = 12.0  # hunger below this = stuffed full of food
BLOAT_INFLATE = 0.01  # per-tick share when puffing up: a slow, visible swell
BLOAT_DEFLATE = 0.05  # per-tick share when digesting back down
BLOAT_PENALTY = -0.15  # reward per tick at full bloat: gorging has a cost


@dataclass(frozen=True)
class MetabolicReport:
    """Reward signal and events produced by one metabolic tick."""

    reward: float
    starvation_event: bool


def grown(hunger: float) -> float:
    """Hunger after one tick, capped at HUNGER_MAX."""
    return min(HUNGER_MAX, hunger + HUNGER_RATE)


def starving_factor(hunger: float) -> float:
    """Weakness in 0..1: 0 below WEAKEN_THRESHOLD, 1 at HUNGER_MAX."""
    if hunger <= WEAKEN_THRESHOLD:
        return 0.0
    return (hunger - WEAKEN_THRESHOLD) / (HUNGER_MAX - WEAKEN_THRESHOLD)


def metabolize(hunger: float, just_crossed_starve: bool, bloat: float = 0.0) -> MetabolicReport:
    """One metabolic tick: reward penalizes hunger GROWTH, not its level.

    Penalizing the absolute level taught the creature to idle in a corner;
    the growth penalty keeps it hunting. Crossing STARVE_EVENT is a one-shot
    event (counted outside), after which misery lingers via slow crawling.
    A stuffed creature also pays a small per-tick cost while it is a ball:
    gorging past the threshold should feel like a mistake, not a strategy.
    """
    new_hunger = grown(hunger)
    reward = -GROWTH_PENALTY * (new_hunger - hunger) + BLOAT_PENALTY * bloat
    if just_crossed_starve:
        reward += STARVE_EVENT_PENALTY
    return MetabolicReport(reward=reward, starvation_event=just_crossed_starve)


def crosses_starve(before: float, after: float) -> bool:
    """Did this tick move hunger across the starvation threshold?"""
    return before < STARVE_EVENT <= after


def bloat_target(hunger: float) -> float:
    """Fullness in 0..1: 1 right after a feast, 0 once hunger passes the threshold."""
    if hunger >= BLOAT_THRESHOLD:
        return 0.0
    return 1.0 - hunger / BLOAT_THRESHOLD


def ease_bloat(current: float, target: float) -> float:
    """Fullness drifts toward the target: slow puff-up, quicker deflate.

    Asymmetric rates on purpose — popping into a ball in a second reads as
    a glitch, while digesting back down should not keep the creature
    helpless for long.
    """
    rate = BLOAT_INFLATE if target > current else BLOAT_DEFLATE
    return current + (target - current) * rate


def fed(hunger: float, energy: float) -> float:
    """Hunger after eating a piece of food."""
    return max(0.0, hunger - energy)
