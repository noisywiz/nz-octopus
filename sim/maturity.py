"""Growth assessment: has the creature reached its learning plateau?

Competence is measured against an oracle with identical kinematics that
always knows the exact position of the nearest food (x-ray vision, no
exploration noise). The gap between the creature and the oracle is purely
the cost of imperfect sensing and imperfect policy — exactly what learning
should close. When the competence ratio stops improving, the brain is done.
"""

import math
import random
from dataclasses import dataclass, field

from . import food as fd
from .brain import QBrain
from .creature import MAX_TURN, SWIM_SPEED, Vec
from .geometry import turn_toward

# Oracle kinematics — deliberately the same limits as the creature (creature.py).
EAT_RADIUS = 2.0  # keep in sync with world.EAT_RADIUS
TANK_WIDTH, TANK_HEIGHT = 80.0, 30.0

ORACLE_PIECES = 30  # pieces the oracle "eats" per estimation run
ORACLE_RUNS = 8  # Monte-Carlo runs averaged into the estimate
ORACLE_MAX_TICKS = 200_000  # safety valve against pathological geometry

WINDOW_TICKS = 25_000  # small windows are pure noise: 0–4 food each
PLATEAU_WINDOWS = 2  # consecutive stable window-pairs required
PLATEAU_TOLERANCE = 0.15  # max relative drift between the two buffer halves
MIN_COVERAGE = 0.6  # share of *occurring* states that must be well-visited
MIN_COMPETENCE = 0.4  # best sustained rate must reach this share of the oracle
WELL_VISITED_CELLS = 3  # a state counts as explored after ~this many nonzero
# Q-cells: most of its actions were tried at least once, so the policy there
# was actually learned, not just glanced at


@dataclass(frozen=True)
class MaturityConfig:
    """Thresholds that decide when learning is considered finished."""

    window: int = WINDOW_TICKS
    plateau_windows: int = PLATEAU_WINDOWS
    tolerance: float = PLATEAU_TOLERANCE
    min_coverage: float = MIN_COVERAGE
    min_competence: float = MIN_COMPETENCE


@dataclass
class Window:
    """Performance counters accumulated over one window of ticks."""

    ticks: int = 0
    food: int = 0

    def complete(self, size: int) -> bool:
        """Has this window gathered its full share of ticks?"""
        return self.ticks >= size


@dataclass
class PlateauTracker:
    """Rolling history of window rates, used for plateau detection.

    A plateau means the creature sustains a performance level close to its
    own all-time best for several consecutive windows — it has converged,
    whether the level itself is good or bad. Whether it is good enough is
    the competence ratio's job (assess()), not this tracker's.
    """

    config: MaturityConfig = field(default_factory=MaturityConfig)
    current: Window = field(default_factory=Window)
    last_window_food: int = 0  # food in the most recent completed window
    rates: list[float] = field(default_factory=list)  # food per 1k ticks
    best_rate: float = 0.0  # all-time best completed window
    stable_streak: int = 0

    def record(self, ticks: int, food: int) -> None:
        """Accumulate one step's counters and close a window when full."""
        self.current.ticks += ticks
        self.current.food += food
        if self.current.complete(self.config.window):
            size = max(self.current.ticks, 1)
            self.last_window_food = self.current.food
            rate = self.current.food * 1000.0 / size
            self.rates.append(rate)
            self.best_rate = max(self.best_rate, rate)
            self._update_streak()
            self.current = Window()

    def _update_streak(self) -> None:
        """Count consecutive windows within tolerance of the all-time best."""
        recent = self.rates[-self.config.plateau_windows:]
        if len(recent) < self.config.plateau_windows:
            self.stable_streak = 0
            return
        floor = self.best_rate * (1.0 - self.config.tolerance)
        if all(r >= floor for r in recent):
            self.stable_streak = len(recent)
        else:
            self.stable_streak = 0


@dataclass(frozen=True)
class MaturityReport:
    """Verdict on the brain's growth, with the numbers behind it."""

    mature: bool
    competence: float  # best sustained rate / oracle rate, 0..1+
    best_rate: float  # best completed window, food per 1k ticks
    oracle_rate: float  # food per 1k ticks, ideal
    coverage: int  # visited Q-states
    total_states: int
    epsilon: float
    plateau: bool
    coverage_ok: bool
    competence_ok: bool

    def summary(self) -> str:
        """One-line human-readable status."""
        pct = f"{self.competence * 100:.0f}%"
        return (f"competence {pct} of oracle "
                f"(best {self.best_rate:.2f} vs {self.oracle_rate:.2f} food/1k), "
                f"states {self.coverage}/{self.total_states}, "
                f"eps {self.epsilon:.2f}"
                + ("  → MATURE" if self.mature else ""))

    def stage(self) -> str:
        """Life-phase label: is the brain still growing, stalled, or done?

        - MATURE: full verdict passed (plateau + competence + coverage).
        - PLATEAU: performance is stable but below the competence floor —
          the brain has converged on its sensory ceiling.
        - GROWING: best window rate is still being pushed upward.
        """
        if self.mature:
            return "MATURE — apex reached"
        if self.plateau and self.coverage_ok:
            return "PLATEAU — sensory ceiling"
        if self.plateau:
            return "PLATEAU — stalled"
        return "GROWING"


def ideal_rate() -> float:
    """Oracle food per 1k ticks: x-ray vision, perfect turns, same speed.

    Estimated by Monte-Carlo so respawn randomness is averaged out; cached
    by the caller if needed (it is deterministic and cheap to recompute).
    """
    rng = random.Random(20260919)  # fixed seed: the estimate must be reproducible
    total_ticks = 0
    for _ in range(ORACLE_RUNS):
        total_ticks += _oracle_run(rng)
    eaten = ORACLE_RUNS * ORACLE_PIECES
    return eaten * 1000.0 / total_ticks


def _oracle_run(rng: random.Random) -> int:
    """Ticks for the oracle to eat ORACLE_PIECES pieces. Returns ticks.

    Mirrors the world's food economy: pieces drop one at a time on the
    spawn rhythm and dissolve on the floor. The first piece is already
    due at tick 0, same as the world's initial spawn timer.
    """
    pos = Vec(TANK_WIDTH / 2, TANK_HEIGHT / 2)
    heading = 0.0
    food = [_random_food(rng)]
    ticks = 0
    since_spawn = fd.SPAWN_INTERVAL
    eaten = 0
    while eaten < ORACLE_PIECES and ticks < ORACLE_MAX_TICKS:
        food = [s for f in food if (s := fd.sunk(f, rng, TANK_WIDTH, TANK_HEIGHT)) is not None]
        since_spawn += 1
        if fd.spawn_due(len(food), since_spawn):
            food.append(_random_food(rng))
            since_spawn = 0
        if not food:  # all pieces dissolved in transit: wait for the next drop
            ticks += 1
            continue
        target = _nearest(pos, food).pos
        heading = turn_toward(heading, math.atan2(target.y - pos.y, target.x - pos.x), MAX_TURN)
        pos = Vec(pos.x + math.cos(heading) * SWIM_SPEED,
                  pos.y + math.sin(heading) * SWIM_SPEED)
        ticks += 1
        survivors: list[fd.Food] = []
        for f in food:
            if math.hypot(f.pos.x - pos.x, f.pos.y - pos.y) < EAT_RADIUS:
                eaten += 1
            else:
                survivors.append(f)
        food = survivors
    return ticks


def _nearest(pos: Vec, food: list[fd.Food]) -> fd.Food:
    """Closest food to a point."""
    return min(food, key=lambda f: (f.pos.x - pos.x) ** 2 + (f.pos.y - pos.y) ** 2)


def _random_food(rng: random.Random) -> fd.Food:
    """A new piece at the surface, same physics as the world's."""
    return fd.spawn(rng, TANK_WIDTH)


def state_coverage(brain: QBrain) -> tuple[int, int]:
    """Coverage as (well-visited, occurring) states.

    The 36 sensory states are not equally reachable: symmetry and receptor
    geometry make many combinations rare or impossible, so a fixed fraction
    of N_STATES can never be met. Instead: "occurring" = states the creature
    has ever entered (nonzero row), "well-visited" = entered with most
    actions explored. Coverage = well-visited / occurring, so the bar
    measures learning depth in the world it actually lives in.
    """
    counts = brain.state_counts()
    occurring = len(counts)
    well = sum(1 for n in counts.values() if n >= WELL_VISITED_CELLS)
    return well, occurring


def coverage(brain: QBrain) -> int:
    """How many Q-states the creature has actually experienced."""
    return len(brain.q)


def assess(brain: QBrain, tracker: PlateauTracker) -> MaturityReport:
    """Combine plateau, competence and coverage into one verdict.

    Maturity = the creature sustains a window rate near its own all-time
    best (plateau) AND that best is a meaningful fraction of the oracle
    (competence floor) AND most of its sensory states have been explored.
    The competence floor is what rules out a "converged on doing nothing"
    verdict: a plateau at zero must never count as maturity.
    """
    oracle = ideal_rate()
    best = tracker.best_rate
    competence = best / oracle if oracle > 0 else 0.0
    plateau = tracker.stable_streak >= tracker.config.plateau_windows
    well, occurring = state_coverage(brain)
    coverage_ok = occurring > 0 and well >= tracker.config.min_coverage * occurring
    competent_enough = competence >= tracker.config.min_competence
    return MaturityReport(
        mature=plateau and coverage_ok and competent_enough,
        competence=competence,
        best_rate=best,
        oracle_rate=oracle,
        coverage=well,
        total_states=occurring,
        epsilon=brain.config.epsilon,
        plateau=plateau,
        coverage_ok=coverage_ok,
        competence_ok=competent_enough,
    )
