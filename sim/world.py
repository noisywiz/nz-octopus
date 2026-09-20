"""Aquarium world: ties brain, sensors, movement and metabolism into one loop.

The World class is a thin orchestrator; every actual decision is a pure
function in sibling modules, so the logic stays testable without pygame.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

from . import creature as cr
from . import food as fd
from . import metabolism as meta
from . import rewards as rw
from . import sensors as sn
from . import terrain as tn
from .brain import QBrain
from .geometry import distance

WIDTH, HEIGHT = 80.0, 30.0
ART_COLUMNS = int(WIDTH / tn.ART_TO_WORLD)  # tank width in art columns


def screen_size() -> tuple[float, float]:
    """Tank dimensions as (width, height), for renderers."""
    return WIDTH, HEIGHT
WALL_MARGIN = 1.0
EAT_RADIUS = 2.0  # mouth with margin: near-miss arcs used to sail past food
REWARD_WINDOW = 500  # ticks in the rolling average-reward window


@dataclass
class World:
    """One creature in one tank. step() advances the simulation one tick."""

    brain: QBrain = field(default_factory=lambda: QBrain(sn.N_STATES, cr.N_DIRECTIONS))
    rng: random.Random = field(default_factory=random.Random)
    terrain: list[int] = field(default_factory=lambda: tn.build(ART_COLUMNS))
    creature: cr.Creature = field(init=False)
    food: list[fd.Food] = field(init=False)
    ticks: int = 0
    ticks_since_spawn: int = fd.SPAWN_INTERVAL  # first piece drops immediately
    food_eaten: int = 0
    wall_bumps: int = 0
    starvations: int = 0
    episode_reward: float = 0.0
    recent_rewards: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.creature = self._spawn_creature()
        self.food = []  # the tank starts empty; pieces drop in one by one

    # --- construction helpers -------------------------------------------------

    def _spawn_creature(self) -> cr.Creature:
        """A rested creature in the middle, facing a random direction."""
        return cr.Creature(
            pos=cr.Vec(WIDTH / 2, HEIGHT / 2),
            heading=self.rng.uniform(0, 6.283185307179586),
            hunger=30.0,
            starving=0.0,
        )

    def _spawn_food(self) -> fd.Food:
        """A new piece dropping from the surface."""
        return fd.spawn(self.rng, WIDTH)

    # --- senses ----------------------------------------------------------------

    def food_points(self) -> list[cr.Vec]:
        """Positions of all food, for the scent field."""
        return [f.pos for f in self.food]

    def read_sensors(self) -> sn.Reading:
        """The creature's current discrete perception."""
        body = self.creature.body
        return sn.read(sn.receptor_readings(body, self.food_points()))

    # --- simulation step ---------------------------------------------------------

    def step(self) -> None:
        """One tick: act, move, collide, eat, metabolize, learn."""
        reading = self.read_sensors()
        state = reading.index()
        action = self.brain.act(state, self.rng)

        cr.move(self.creature, action)
        wall_reward = self._handle_walls()
        self._advance_food()
        food_reward = self._handle_eating()
        metabolic = self._handle_metabolism()

        reward = rw.combine(food_reward, wall_reward, metabolic)
        next_state = self.read_sensors().index()
        self.brain.learn(state, action, reward.total, next_state)

        self.ticks += 1
        self.episode_reward += reward.total
        self._remember(reward.total)

    def floor_height(self, x: float) -> float:
        """Dune height in world units at world x (0 = flat baseline)."""
        return tn.height_at_world(self.terrain, x, WIDTH)

    def _handle_walls(self) -> float:
        """Clamp position, count first-contact bumps, run the escape reflex."""
        c = self.creature
        floor_line = HEIGHT - self.floor_height(c.pos.x) - WALL_MARGIN * 0.5
        c.pos = cr.Vec(
            x=min(WIDTH - WALL_MARGIN, max(WALL_MARGIN, c.pos.x)),
            y=min(HEIGHT - WALL_MARGIN, max(WALL_MARGIN, min(floor_line, c.pos.y))),
        )
        at_v, at_h = cr.at_wall(c.pos, WALL_MARGIN, WIDTH, HEIGHT)
        at_h = at_h or c.pos.y >= floor_line
        bumped = (at_v or at_h) and not c.bumping
        c.bumping = at_v or at_h
        if bumped:
            self.wall_bumps += 1
        if at_v or at_h:
            cr.wall_escape(c, WALL_MARGIN, WIDTH, HEIGHT, floor_line)
        return rw.wall_bump(bumped)

    def _advance_food(self) -> None:
        """Sink every piece one tick, drop dissolved ones, spawn on rhythm."""
        alive: list[fd.Food] = []
        for f in self.food:
            sunk = fd.sunk(f, self.rng, WIDTH,
                           HEIGHT - self.floor_height(f.pos.x))
            if sunk is not None:
                alive.append(sunk)
        self.food = alive
        self.ticks_since_spawn += 1
        if fd.spawn_due(len(self.food), self.ticks_since_spawn):
            self.food.append(self._spawn_food())
            self.ticks_since_spawn = 0

    def _handle_eating(self) -> float:
        """Eat every piece of food within reach; each respawns at the surface."""
        ate = False
        for i, f in enumerate(self.food):
            if cr.touch(self.creature.pos, f.pos, EAT_RADIUS):
                self.food[i] = self._spawn_food()
                self.creature.hunger = meta.fed(self.creature.hunger, f.energy)
                self.food_eaten += 1
                ate = True
        return rw.food_reward(ate)

    def _handle_metabolism(self) -> meta.MetabolicReport:
        """Grow hunger, track weakness, penalize crossing the starve threshold."""
        c = self.creature
        before = c.hunger
        c.hunger = meta.grown(before)
        c.starving = meta.starving_factor(c.hunger)
        event = meta.crosses_starve(before, c.hunger)
        if event:
            self.starvations += 1
        return meta.metabolize(before, event)

    def _remember(self, reward: float) -> None:
        """Keep the rolling reward window bounded."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > REWARD_WINDOW:
            del self.recent_rewards[:-REWARD_WINDOW]

    # --- stats -----------------------------------------------------------------

    @property
    def recent_avg_reward(self) -> float:
        """Mean reward over the last REWARD_WINDOW ticks (0 if empty)."""
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    @property
    def trail(self) -> list[cr.Vec]:
        """Recent positions, for rendering."""
        return self.creature.trail

    @property
    def position(self) -> cr.Vec:
        """Where the creature is right now."""
        return self.creature.pos

    @property
    def heading(self) -> float:
        """Where the creature is facing right now."""
        return self.creature.heading

    @property
    def hunger(self) -> float:
        """Current hunger in 0..100."""
        return self.creature.hunger

    @property
    def starving(self) -> float:
        """Weakness factor in 0..1 (drives gray color and slowness)."""
        return self.creature.starving

    def food_distances(self) -> list[float]:
        """Distance from the creature to every piece of food."""
        return [distance(self.creature.pos.x, self.creature.pos.y, f.pos.x, f.pos.y)
                for f in self.food]

    def save_brain(self, path: Path) -> None:
        """Persist the Q-table."""
        self.brain.save(path)
