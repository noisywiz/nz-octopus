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
from .brain import QBrain, conjugate, save_all
from .creature import Creature
from .geometry import distance

WIDTH, HEIGHT = 80.0, 30.0
ART_COLUMNS = int(WIDTH / tn.ART_TO_WORLD)  # tank width in art columns
WALL_MARGIN = 1.0
EAT_RADIUS = 2.0  # mouth with margin: near-miss arcs used to sail past food
POUR_COUNT = 6  # pieces per pour keypress: a handful, not a flood
REWARD_WINDOW = 500  # ticks in the rolling average-reward window


@dataclass
class CreatureBrain:
    """One living creature: its body, brain and lifetime counters."""

    body: cr.Creature
    brain: QBrain
    food_eaten: int = 0
    wall_bumps: int = 0

    def stats(self) -> tuple[float, int, int]:
        """(hunger, food_eaten, wall_bumps) lifetime counters."""
        return self.body.hunger, self.food_eaten, self.wall_bumps


@dataclass
class World:
    """A tank with any number of creatures sharing one food economy."""

    rng: random.Random = field(default_factory=random.Random)
    width: float = WIDTH
    height: float = HEIGHT
    terrain: list[int] = field(default_factory=lambda: tn.build(ART_COLUMNS))
    creatures: list[CreatureBrain] = field(default_factory=list)
    next_id: int = 0
    food: list[fd.Food] = field(init=False)
    ticks: int = 0
    ticks_since_spawn: int = fd.SPAWN_INTERVAL  # first piece drops immediately
    food_eaten: int = 0
    wall_bumps: int = 0
    starvations: int = 0
    conjugations: int = 0  # contact episodes that exchanged brain knowledge
    touching: dict[tuple[int, int], bool] = field(default_factory=dict)
    episode_reward: float = 0.0
    recent_rewards: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.food = []  # the tank starts empty; pieces drop in one by one
        self._rebuild_terrain()
        if not self.creatures:
            self.creatures.append(self._new_creature(
                cr.Vec(self.width / 2, self.height / 2),
            ))
        self._adopt_ids()

    def resize(self, width: float, height: float) -> None:
        """Grow or shrink the tank to match the window (same zoom).

        The dune heightmap is rebuilt for the new width — it is a pure
        function of the column count, so physics and pixel art stay in
        agreement. Bodies and food outside the new bounds are clamped in.
        """
        self.width, self.height = width, height
        self._rebuild_terrain()
        for cb in self.creatures:
            self._clamp_body(cb.body)
        self.food = [f for f in self.food
                     if WALL_MARGIN <= f.pos.x <= width - WALL_MARGIN]

    def _rebuild_terrain(self) -> None:
        """Regenerate the dune heightmap for the current tank width."""
        columns = max(1, int(self.width / tn.ART_TO_WORLD))
        if len(self.terrain) != columns:
            self.terrain = tn.build(columns)

    def _adopt_ids(self) -> None:
        """Make creature ids unique and move next_id past all of them.

        Viewers build creatures from brains loaded off disk, but next_id
        starts at 0 — the first spawned creature then reused id 0, and
        two bodies shared one animator (the veteran visually jumped to
        the spawn point). Saves written before this fix hold duplicate
        ids; colliding brains are renumbered here. An id carries no
        learned content, so renumbering is safe anytime.
        """
        seen: set[int] = set()
        for cb in self.creatures:
            if cb.brain.id in seen:
                cb.brain.id = self.next_id
            seen.add(cb.brain.id)
            self.next_id = max(self.next_id, cb.brain.id + 1)

    # --- construction helpers -------------------------------------------------

    def _new_creature(self, pos: cr.Vec) -> CreatureBrain:
        """A rested creature with an empty brain, facing a random direction."""
        brain = QBrain(sn.N_STATES, cr.N_DIRECTIONS)
        brain.id = self.next_id
        self.next_id += 1
        heading = self.rng.uniform(0, 6.283185307179586)
        return CreatureBrain(
            body=cr.Creature(
                pos=pos,
                heading=heading,
                hunger=30.0,
                starving=0.0,
            ),
            brain=brain,
        )

    def floor_height(self, x: float) -> float:
        """Dune height in world units at world x (0 = flat baseline)."""
        return tn.height_at_world(self.terrain, x, self.width)

    def spawn_at(self, x: float, y: float) -> CreatureBrain:
        """Birth by click: a new creature with an empty brain at a point."""
        floor_line = self.height - self.floor_height(x) - WALL_MARGIN * 0.5
        pos = cr.Vec(
            x=min(self.width - WALL_MARGIN, max(WALL_MARGIN, x)),
            y=min(floor_line, max(WALL_MARGIN, y)),
        )
        creature = self._new_creature(pos)
        self.creatures.append(creature)
        return creature

    def pour_food(self, count: int = POUR_COUNT) -> None:
        """Sprinkle a handful of pieces from the surface (the f key).

        Deliberately ignores MAX_PIECES: the cap throttles the passive
        rhythm, while a pour is an explicit feeding decision.
        """
        for _ in range(count):
            self.food.append(self._spawn_food())

    def _spawn_food(self) -> fd.Food:
        """A new piece dropping from the surface."""
        return fd.spawn(self.rng, self.width)

    # --- senses ----------------------------------------------------------------

    def food_points(self) -> list[cr.Vec]:
        """Positions of all food, for the scent field."""
        return [f.pos for f in self.food]

    def read_sensors(self, body: cr.Body) -> sn.Reading:
        """One creature's current discrete perception."""
        return sn.read(sn.receptor_readings(body, self.food_points()))

    # --- simulation step ---------------------------------------------------------

    def step(self) -> None:
        """One tick: every creature acts, moves, eats, metabolizes, learns."""
        for cb in self.creatures:
            self._step_creature(cb)
        self._resolve_contacts()
        self._advance_food()
        self.ticks += 1
        self.episode_reward += sum(
            cb.brain.config.epsilon for cb in self.creatures
        )

    def _step_creature(self, cb: CreatureBrain) -> None:
        """One creature's tick: sense, act, move, collide, eat, learn."""
        body = cb.body
        state = self.read_sensors(body.body).index()
        action = cb.brain.act(state, self.rng)

        cr.move(body, action)
        wall_reward = self._handle_walls(body)
        food_reward = self._eat(cb)
        metabolic = self._handle_metabolism(body)

        reward = rw.combine(food_reward, wall_reward, metabolic)
        next_state = self.read_sensors(body.body).index()
        cb.brain.learn(state, action, reward.total, next_state)
        self._remember(reward.total)

    def _clamp_body(self, c: cr.Creature) -> None:
        """Keep a body inside the tank; bump counting lives in _handle_walls."""
        floor_line = self.height - self.floor_height(c.pos.x) - WALL_MARGIN * 0.5
        c.pos = cr.Vec(
            x=min(self.width - WALL_MARGIN, max(WALL_MARGIN, c.pos.x)),
            y=min(self.height - WALL_MARGIN,
                  max(WALL_MARGIN, min(floor_line, c.pos.y))),
        )

    def _handle_walls(self, c: cr.Creature) -> float:
        """Clamp position, count first-contact bumps, run the escape reflex."""
        self._clamp_body(c)
        floor_line = self.height - self.floor_height(c.pos.x) - WALL_MARGIN * 0.5
        at_v, at_h = cr.at_wall(c.pos, WALL_MARGIN, self.width, self.height)
        at_h = at_h or c.pos.y >= floor_line
        bumped = (at_v or at_h) and not c.bumping
        c.bumping = at_v or at_h
        if bumped:
            self.wall_bumps += 1
        if at_v or at_h:
            cr.wall_escape(c, WALL_MARGIN, self.width, self.height, floor_line)
        return rw.wall_bump(bumped)

    def _resolve_contacts(self) -> None:
        """Pairwise soft collisions, plus conjugation while bodies touch.

        Runs after every creature has moved: separation is symmetric, so
        both bodies of a pair must already be at their final positions.
        A push can shove a body past a wall, hence the re-clamp afterwards.
        Reaches come from the bodies themselves, so a bloated ball shoves
        neighbors with its inflated silhouette.
        """
        for i, a in enumerate(self.creatures):
            for b in self.creatures[i + 1:]:
                touching = cr.separate(a.body, b.body,
                                       cr.bloat_reach(a.body), cr.bloat_reach(b.body))
                key = (a.brain.id, b.brain.id)
                if touching:
                    conjugate(a.brain, b.brain)
                    conjugate(b.brain, a.brain)
                    if not self.touching.get(key, False):
                        self.conjugations += 1
                self.touching[key] = touching
        for cb in self.creatures:
            self._clamp_body(cb.body)

    def _advance_food(self) -> None:
        """Sink every piece one tick, drop dissolved ones, spawn on rhythm."""
        alive: list[fd.Food] = []
        for f in self.food:
            sunk = fd.sunk(f, self.rng, self.width,
                           self.height - self.floor_height(f.pos.x))
            if sunk is not None:
                alive.append(sunk)
        self.food = alive
        self.ticks_since_spawn += 1
        if fd.spawn_due(len(self.food), self.ticks_since_spawn):
            self.food.append(self._spawn_food())
            self.ticks_since_spawn = 0

    def _eat(self, cb: CreatureBrain) -> float:
        """Feeding: the bacterium swallows every piece within body reach."""
        return self._contact_eating(cb.body)

    def _contact_eating(self, c: cr.Creature) -> float:
        """Bacterium: eat every piece within body reach.

        The reach grows with bloat — a stuffed ball hoovers everything
        nearby, which is exactly how it got stuffed in the first place.
        """
        ate = False
        radius = EAT_RADIUS + 2.0 * c.bloat
        for i, f in enumerate(self.food):
            if cr.touch(c.pos, f.pos, radius):
                self.food[i] = self._spawn_food()
                c.hunger = meta.fed(c.hunger, f.energy)
                self.food_eaten += 1
                ate = True
        return rw.food_reward(ate)

    def _handle_metabolism(self, c: cr.Creature) -> meta.MetabolicReport:
        """Grow hunger, track weakness, swell into a ball after a feast."""
        before = c.hunger
        c.hunger = meta.grown(before)
        c.starving = meta.starving_factor(c.hunger)
        c.bloat = meta.ease_bloat(c.bloat, meta.bloat_target(c.hunger))
        event = meta.crosses_starve(before, c.hunger)
        if event:
            self.starvations += 1
        return meta.metabolize(before, event, c.bloat)

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
        """All creatures' recent trails, for rendering."""
        return [p for cb in self.creatures for p in cb.body.trail]

    @property
    def position(self) -> cr.Vec:
        """First creature's position (single-creature viewers)."""
        return self.creatures[0].body.pos

    @property
    def heading(self) -> float:
        """First creature's heading (single-creature viewers)."""
        return self.creatures[0].body.heading

    @property
    def hunger(self) -> float:
        """First creature's hunger in 0..100."""
        return self.creatures[0].body.hunger

    @property
    def starving(self) -> float:
        """First creature's weakness factor in 0..1."""
        return self.creatures[0].body.starving

    @property
    def brain(self) -> QBrain:
        """First creature's brain (single-creature viewers)."""
        return self.creatures[0].brain

    def food_distances(self) -> list[float]:
        """Distance from the first creature to every piece of food."""
        pos = self.creatures[0].body.pos
        return [distance(pos.x, pos.y, f.pos.x, f.pos.y) for f in self.food]

    def save_brain(self, path: Path) -> None:
        """Persist every creature's brain."""
        save_all([cb.brain for cb in self.creatures], path)
