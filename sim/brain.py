"""Tabular Q-learning brain. One creature, learning across its whole life."""

import json
import random
from pathlib import Path


class QBrain:
    def __init__(self, n_states: int, n_actions: int, load_path: Path | None = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = 0.2          # learning rate (decays as experience grows)
        self.lr_min = 0.02
        self.gamma = 0.9       # discount: how much it cares about the future
        self.epsilon = 1.0     # exploration: 1.0 = fully random at birth
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995

        self.q: dict[int, list[float]] = {}
        self.experience = 0    # total learning updates ever made
        if load_path and load_path.exists():
            self.load(load_path)

    def _row(self, state: int) -> list[float]:
        row = self.q.get(state)
        if row is None:
            row = [0.0] * self.n_actions
            self.q[state] = row
        return row

    def act(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        row = self._row(state)
        best = max(row)
        # break ties randomly so the creature doesn't get stuck in habits
        return random.choice([a for a, v in enumerate(row) if v == best])

    def learn(self, state: int, action: int, reward: float, next_state: int):
        row = self._row(state)
        next_max = max(self._row(next_state))
        lr = max(self.lr_min, self.lr / (1.0 + self.experience / 20000))
        row[action] += lr * (reward + self.gamma * next_max - row[action])
        self.experience += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def known_states(self) -> int:
        return len(self.q)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "epsilon": self.epsilon,
            "experience": self.experience,
            "q": {str(k): v for k, v in self.q.items()},
        }
        path.write_text(json.dumps(data))

    def load(self, path: Path):
        data = json.loads(path.read_text())
        self.epsilon = data["epsilon"]
        self.experience = data["experience"]
        self.q = {int(k): v for k, v in data["q"].items()}
