"""Tabular Q-learning brain. One creature, learning across its whole life."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SAVE_FORMAT = 3  # bump when the save layout changes -> old saves are rejected


@dataclass
class BrainConfig:
    """Learning hyperparameters, all tuned by hand over Development sessions."""

    lr: float = 0.3
    lr_min: float = 0.05
    lr_decay_horizon: int = 20000  # updates after which lr halves toward lr_min
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay: float = 0.9999


@dataclass
class QBrain:
    """A single Q-table plus the ε-greedy / decaying-lr update rule."""

    n_states: int
    n_actions: int
    config: BrainConfig = field(default_factory=BrainConfig)
    q: dict[int, list[float]] = field(default_factory=dict)
    experience: int = 0
    id: int = 0  # creature identity in the shared save file

    def row(self, state: int) -> list[float]:
        """Q-values for one state, created zero-filled on first touch."""
        row = self.q.get(state)
        if row is None:
            row = [0.0] * self.n_actions
            self.q[state] = row
        return row

    def act(self, state: int, rng: random.Random) -> int:
        """ε-greedy action; ties in the greedy branch are broken randomly."""
        cfg = self.config
        if rng.random() < cfg.epsilon:
            return rng.randrange(self.n_actions)
        row = self.row(state)
        best = max(row)
        return rng.choice([a for a, v in enumerate(row) if v == best])

    def learning_rate(self) -> float:
        """Effective lr for the current update, decaying with experience."""
        cfg = self.config
        decayed = cfg.lr / (1.0 + self.experience / cfg.lr_decay_horizon)
        return max(cfg.lr_min, decayed)

    def learn(self, state: int, action: int, reward: float, next_state: int) -> None:
        """One Q-update: also decays lr (with experience) and epsilon (per update)."""
        cfg = self.config
        row = self.row(state)
        target = reward + cfg.gamma * max(self.row(next_state))
        row[action] += self.learning_rate() * (target - row[action])
        self.experience += 1
        cfg.epsilon = max(cfg.epsilon_min, cfg.epsilon * cfg.epsilon_decay)

    @property
    def known_states(self) -> int:
        """How many distinct states the creature has ever observed."""
        return len(self.q)

    def state_counts(self) -> dict[int, int]:
        """Visits per state: any nonzero row means the state was observed,
        and the number of nonzero cells in a row approximates visit volume
        (a row with all actions explored was visited much more than one
        touched once). Not an exact counter, but enough for coverage."""
        return {s: sum(1 for v in row if v != 0.0) for s, row in self.q.items()}

    def save(self, path: Path) -> None:
        """Persist the table; tagged with a format version, not per-world params."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": SAVE_FORMAT,
            "epsilon": self.config.epsilon,
            "experience": self.experience,
            "q": {str(k): v for k, v in self.q.items()},
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path, n_states: int, n_actions: int) -> "QBrain":
        """Rebuild a brain from disk; a stale save returns an empty brain instead."""
        brain = cls(n_states=n_states, n_actions=n_actions)
        if not path.exists():
            return brain
        data = json.loads(path.read_text())
        if data.get("format") != SAVE_FORMAT:
            return brain  # incompatible encoding: start fresh rather than crash
        brain.q = {int(k): list(v) for k, v in data["q"].items()}
        brain.experience = int(data["experience"])
        brain.config.epsilon = float(data["epsilon"])
        return brain


CONJUGATION_RATE = 0.05  # fraction of a donor Q-value blended per contact tick


def conjugate(donor: QBrain, receiver: QBrain, rate: float = CONJUGATION_RATE) -> int:
    """Blend the donor's explored (state, action) values into the receiver.

    Bacterial conjugation as social learning: on body contact knowledge
    flows with no sensor and no reward signal. Only cells the donor has
    actually explored (nonzero) are shared, so an untouched action never
    drags the receiver's explored value toward zero. Returns the number
    of cells shared.
    """
    shared = 0
    for state, row in donor.q.items():
        rec = receiver.row(state)
        for action, value in enumerate(row):
            if value != 0.0:
                rec[action] += rate * (value - rec[action])
                shared += 1
    return shared


def save_all(brains: list["QBrain"], path: Path) -> None:
    """Persist every creature's brain into one file, keyed by creature id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SAVE_FORMAT,
        "creatures": [
            {
                "id": brain.id,
                "epsilon": brain.config.epsilon,
                "experience": brain.experience,
                "q": {str(k): v for k, v in brain.q.items()},
            }
            for brain in brains
        ],
    }
    path.write_text(json.dumps(payload))


def load_all(path: Path, n_states: int, n_actions: int) -> list["QBrain"]:
    """Rebuild all brains from disk; a stale save returns one empty brain."""
    if not path.exists():
        return [QBrain(n_states=n_states, n_actions=n_actions)]
    data = json.loads(path.read_text())
    if data.get("format") != SAVE_FORMAT:
        return [QBrain(n_states=n_states, n_actions=n_actions)]
    brains: list[QBrain] = []
    for entry in data["creatures"]:
        brain = QBrain(n_states=n_states, n_actions=n_actions)
        brain.id = int(entry["id"])
        brain.q = {int(k): list(v) for k, v in entry["q"].items()}
        brain.experience = int(entry["experience"])
        brain.config.epsilon = float(entry["epsilon"])
        brains.append(brain)
    return brains or [QBrain(n_states=n_states, n_actions=n_actions)]
