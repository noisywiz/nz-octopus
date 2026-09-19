"""Aquarium world: a continuous 2D space with food, walls and one octopus."""

import math
import random
from dataclasses import dataclass

from .brain import QBrain

# --- discrete sensor encoding -------------------------------------------------
DIRS = 8  # 8 directions around the octopus


def dir_index(angle: float) -> int:
    """Map an angle (radians) to one of 8 sectors (0..7)."""
    sector = int((angle + math.pi / DIRS) // (2 * math.pi / DIRS)) % DIRS
    return sector


@dataclass
class Food:
    x: float
    y: float
    energy: float = 12.0


class World:
    WIDTH = 80.0
    HEIGHT = 30.0
    VISION = 25.0     # how far the octopus can sense food
    EAT_RADIUS = 1.5

    HUNGER_LEVELS = 5

    def __init__(self, brain: QBrain | None = None, n_food: int = 4):
        self.n_food = n_food
        # state space: food dir (9: 8 dirs + none) * dist (3). Hunger is felt
        # through rewards, not encoded in the state — fewer states, faster learning.
        n_states = (DIRS + 1) * 3
        n_actions = DIRS + 1  # 8 directions + rest
        if brain is None:
            brain = QBrain(n_states, n_actions)
        elif brain.n_actions == 0:
            brain.n_states, brain.n_actions = n_states, n_actions
        self.brain = brain
        self.reset()

    def reset(self, keep_stats: bool = False):
        self.octopus_x = self.WIDTH / 2
        self.octopus_y = self.HEIGHT / 2
        self.octopus_dir = random.uniform(0, 2 * math.pi)
        self.hunger = 50.0          # 0 = full, 100 = starving
        self.foods = [self._spawn_food() for _ in range(self.n_food)]
        self.last_state = self.sense()
        self.ticks = 0
        self.episode_reward = 0.0
        self.trail: list[tuple[float, float]] = []   # recent positions, for rendering
        self.recent_rewards: list[float] = []   # rolling window of rewards
        if not keep_stats:
            # lifetime counters: this creature is one and only, stats span its whole life
            self.food_eaten = 0
            self.wall_bumps = 0
            self.starvations = 0

    def _spawn_food(self) -> Food:
        return Food(
            x=random.uniform(3, self.WIDTH - 3),
            y=random.uniform(3, self.HEIGHT - 3),
        )

    # --- senses ---------------------------------------------------------------

    def sense(self) -> int:
        """Compress the world into one discrete state index."""
        nearest, dist = self._nearest_food()
        if nearest is None or dist > self.VISION:
            food_part = 0            # "don't see anything"
            dist_part = 0
        else:
            food_part = 1 + dir_index(math.atan2(nearest.y - self.octopus_y,
                                                 nearest.x - self.octopus_x))
            dist_part = 0 if dist < self.VISION / 3 else (1 if dist < 2 * self.VISION / 3 else 2)
        return food_part * 3 + dist_part

    def _nearest_food(self) -> tuple[Food | None, float]:
        best, best_d = None, float("inf")
        for f in self.foods:
            d = math.hypot(f.x - self.octopus_x, f.y - self.octopus_y)
            if d < best_d:
                best, best_d = f, d
        return best, best_d

    # --- simulation step --------------------------------------------------------

    def step(self):
        state = self.last_state
        action = self.brain.act(state)

        # perform action
        if action < DIRS:
            self.octopus_dir = action * (2 * math.pi / DIRS)
            speed = 0.8
        else:
            speed = 0.0  # rest
        self.octopus_x += math.cos(self.octopus_dir) * speed
        self.octopus_y += math.sin(self.octopus_dir) * speed
        if speed > 0:
            self.trail.append((self.octopus_x, self.octopus_y))
            if len(self.trail) > 8:
                self.trail.pop(0)

        # walls: sliding along a wall is fine, only a real bump (moving into it) hurts
        reward = 0.0
        bumped = False
        if not (1 < self.octopus_x < self.WIDTH - 1):
            self.octopus_x = min(max(self.octopus_x, 1), self.WIDTH - 1)
            if abs(math.cos(self.octopus_dir)) > 0.01:
                bumped = True
        if not (1 < self.octopus_y < self.HEIGHT - 1):
            self.octopus_y = min(max(self.octopus_y, 1), self.HEIGHT - 1)
            if abs(math.sin(self.octopus_dir)) > 0.01:
                bumped = True
        if bumped:
            reward -= 2.0
            self.wall_bumps += 1

        # eating
        for f in list(self.foods):
            if math.hypot(f.x - self.octopus_x, f.y - self.octopus_y) < self.EAT_RADIUS:
                self.foods.remove(f)
                self.foods.append(self._spawn_food())
                self.hunger = max(0.0, self.hunger - f.energy)
                reward += 10.0
                self.food_eaten += 1

        # metabolism: starving hurts, and full starvation kills (hard reset = big penalty).
        # Only penalize the *growth* of hunger, not its absolute level, otherwise
        # the octopus is better off staying idle in a corner than hunting.
        prev_hunger = self.hunger
        self.hunger = min(100.0, self.hunger + 0.15)
        reward -= 0.3 * (self.hunger - prev_hunger)
        if self.hunger >= 100.0:
            reward -= 50.0
            self.starvations += 1
            self.reset(keep_stats=True)

        next_state = self.sense()
        self.brain.learn(state, action, reward, next_state)
        self.last_state = next_state

        self.ticks += 1
        self.episode_reward += reward
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 500:
            self.recent_rewards.pop(0)

    @property
    def recent_avg_reward(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)
