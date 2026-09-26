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
from . import octopus_body as ob
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
REWARD_WINDOW = 500  # ticks in the rolling average-reward window


@dataclass
class CreatureBrain:
    """One living creature: its body, brain and lifetime counters."""

    body: cr.Creature
    brain: QBrain
    food_eaten: int = 0
    wall_bumps: int = 0
    arms: ob.Arms | None = None  # octopus morphology; None for the bacterium

    def stats(self) -> tuple[float, int, int]:
        """(hunger, food_eaten, wall_bumps) lifetime counters."""
        return self.body.hunger, self.food_eaten, self.wall_bumps


@dataclass
class World:
    """A tank with any number of creatures sharing one food economy."""

    rng: random.Random = field(default_factory=random.Random)
    terrain: list[int] = field(default_factory=lambda: tn.build(ART_COLUMNS))
    creatures: list[CreatureBrain] = field(default_factory=list)
    octopus: bool = False  # morphology of newly spawned creatures
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
        if not self.creatures:
            self.creatures.append(self._new_creature(
                cr.Vec(WIDTH / 2, HEIGHT / 2),
                octopus=self.octopus,
            ))
        self._adopt_ids()

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

    def _new_creature(self, pos: cr.Vec, octopus: bool = False) -> CreatureBrain:
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
            arms=ob.create(pos, heading) if octopus else None,
        )

    def floor_height(self, x: float) -> float:
        """Dune height in world units at world x (0 = flat baseline)."""
        return tn.height_at_world(self.terrain, x, WIDTH)

    def spawn_at(self, x: float, y: float) -> CreatureBrain:
        """Birth by click: a new creature with an empty brain at a point."""
        floor_line = HEIGHT - self.floor_height(x) - WALL_MARGIN * 0.5
        pos = cr.Vec(
            x=min(WIDTH - WALL_MARGIN, max(WALL_MARGIN, x)),
            y=min(floor_line, max(WALL_MARGIN, y)),
        )
        creature = self._new_creature(pos, octopus=self.octopus)
        self.creatures.append(creature)
        return creature

    def _spawn_food(self) -> fd.Food:
        """A new piece dropping from the surface."""
        return fd.spawn(self.rng, WIDTH)

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
        floor_line = HEIGHT - self.floor_height(c.pos.x) - WALL_MARGIN * 0.5
        c.pos = cr.Vec(
            x=min(WIDTH - WALL_MARGIN, max(WALL_MARGIN, c.pos.x)),
            y=min(HEIGHT - WALL_MARGIN, max(WALL_MARGIN, min(floor_line, c.pos.y))),
        )

    def _handle_walls(self, c: cr.Creature) -> float:
        """Clamp position, count first-contact bumps, run the escape reflex."""
        self._clamp_body(c)
        floor_line = HEIGHT - self.floor_height(c.pos.x) - WALL_MARGIN * 0.5
        at_v, at_h = cr.at_wall(c.pos, WALL_MARGIN, WIDTH, HEIGHT)
        at_h = at_h or c.pos.y >= floor_line
        bumped = (at_v or at_h) and not c.bumping
        c.bumping = at_v or at_h
        if bumped:
            self.wall_bumps += 1
        if at_v or at_h:
            cr.wall_escape(c, WALL_MARGIN, WIDTH, HEIGHT, floor_line)
        return rw.wall_bump(bumped)

    def _resolve_contacts(self) -> None:
        """Pairwise soft collisions, plus conjugation while bodies touch.

        Runs after every creature has moved: separation is symmetric, so
        both bodies of a pair must already be at their final positions.
        A push can shove a body past a wall, hence the re-clamp afterwards.
        """
        for i, a in enumerate(self.creatures):
            for b in self.creatures[i + 1:]:
                touching = cr.separate(a.body, b.body)
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
            sunk = fd.sunk(f, self.rng, WIDTH,
                           HEIGHT - self.floor_height(f.pos.x))
            if sunk is not None:
                alive.append(sunk)
        self.food = alive
        self.ticks_since_spawn += 1
        if fd.spawn_due(len(self.food), self.ticks_since_spawn):
            self.food.append(self._spawn_food())
            self.ticks_since_spawn = 0

    def _eat(self, cb: CreatureBrain) -> float:
        """Feeding for the creature's morphology.

        The bacterium swallows on body contact; the octopus must catch a
        piece with an arm tip and carry it to the mouth.
        """
        if cb.arms is not None:
            return self._arms_feeding(cb)
        return self._contact_eating(cb.body)

    def _contact_eating(self, c: cr.Creature) -> float:
        """Bacterium: eat every piece within body reach."""
        ate = False
        for i, f in enumerate(self.food):
            if cr.touch(c.pos, f.pos, EAT_RADIUS):
                self.food[i] = self._spawn_food()
                c.hunger = meta.fed(c.hunger, f.energy)
                self.food_eaten += 1
                ate = True
        return rw.food_reward(ate)

    def _arms_feeding(self, cb: CreatureBrain) -> float:
        """Octopus: arms reach for food, grip it, and feed it to the mouth."""
        assert cb.arms is not None  # checked by the caller
        body = cb.body
        mouth = ob.mouth_point(body.pos, body.heading)
        self._step_arms(cb.arms, body.pos, body.heading, mouth)
        return self._swallow(cb, mouth)

    def _step_arms(self, arms: ob.Arms, pos: cr.Vec, heading: float,
                   mouth: cr.Vec) -> None:
        """Advance every arm one tick: carry, reach, or sway at rest."""
        ob.claim_food(arms, self.food)
        for arm in arms.limbs:
            base = ob.base_point(pos, heading, arm.index)
            if arm.holding is not None:
                target = mouth
                speed = ob.CARRY_SPEED
            else:
                target = self._reach_target(arm, base, pos, heading)
                speed = ob.TIP_SPEED
            arm.joints = ob.move_tip(arm, base, target, speed)
            self._grip(arm)

    def _reach_target(self, arm: ob.Arm, base: cr.Vec, pos: cr.Vec,
                      heading: float) -> cr.Vec:
        """Where a free arm wants its tip: its claimed piece, or a rest pose.

        A claimed piece is chased until it is gripped or drifts beyond
        reach*1.3 — no per-tick re-evaluation, so the tip never oscillates
        between two goals.
        """
        piece = self._piece_by_fid(arm.target_fid)
        if piece is not None:
            d = distance(pos.x, pos.y, piece.pos.x, piece.pos.y)
            if d <= ob.REACH * ob.GRASP_MARGIN * 1.3:
                return piece.pos
            arm.target_fid = None  # out of reach: rest until re-claimed
            return ob.rest_tip(base, heading, arm.index, self.ticks)
        if arm.target_fid is not None:
            arm.target_fid = None  # piece dissolved: release the claim
        return ob.rest_tip(base, heading, arm.index, self.ticks)

    def _piece_by_fid(self, fid: int | None) -> fd.Food | None:
        """The current object for a piece id, or None if it's gone."""
        if fid is None:
            return None
        return next((f for f in self.food if f.fid == fid), None)

    def _grip(self, arm: ob.Arm) -> None:
        """Grab a piece with the tip, or track the one already held.

        A gripped piece leaves the world's food list: the arm owns it now,
        so it stops sinking, stops smelling and can't be retargeted. The
        grab itself is claim-agnostic: a tip that touches a piece grabs it
        even if the claim was lost a tick earlier — otherwise near-misses
        would starve a creature whose tip is literally on the food.
        """
        tip = ob.tip_of(arm)
        if arm.holding is not None:
            arm.carry = tip
            return
        near = ob.nearest_food(tip, self.food)
        if near is not None and near[1] <= ob.GRAB_RADIUS:
            piece = self.food[near[0]]
            arm.holding = piece
            arm.carry = tip
            arm.target_fid = piece.fid
            self.food.remove(piece)

    def _swallow(self, cb: CreatureBrain, mouth: cr.Vec) -> float:
        """Consume any piece an arm has delivered to the mouth."""
        assert cb.arms is not None  # checked by the caller
        ate = False
        for arm in cb.arms.limbs:
            if arm.holding is None or arm.carry is None:
                continue
            if distance(arm.carry.x, arm.carry.y, mouth.x, mouth.y) > ob.MOUTH_RADIUS:
                continue
            cb.body.hunger = meta.fed(cb.body.hunger, arm.holding.energy)
            self.food_eaten += 1
            ate = True
            arm.holding = None
            arm.carry = None
            arm.target_fid = None
        return rw.food_reward(ate)

    def _handle_metabolism(self, c: cr.Creature) -> meta.MetabolicReport:
        """Grow hunger, track weakness, penalize crossing the starve threshold."""
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
