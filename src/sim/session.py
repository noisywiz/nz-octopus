"""Shared session plumbing: argument parsing, brain load/save."""

import argparse
from pathlib import Path

from .brain import QBrain, load_all, save_all
from .creature import N_DIRECTIONS
from .sensors import N_STATES

# the package lives in src/, the data files stay at the project root
BRAIN_PATH = Path(__file__).resolve().parents[2] / "brains.json"


def parse_args(description: str, default_speed: int) -> argparse.Namespace:
    """Arguments common to both viewers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--speed", type=int, default=default_speed,
                        help="sim ticks per frame")
    parser.add_argument("--brain", action="store_true",
                        help="resume from brains.json and save on exit; "
                             "without it the run starts fresh and touches nothing")
    return parser.parse_args()


def open_brains(load: bool) -> list[QBrain]:
    """One fresh brain by default; with the flag, resume the saved colony."""
    if not load:
        return [QBrain(N_STATES, N_DIRECTIONS)]
    return load_all(BRAIN_PATH, N_STATES, N_DIRECTIONS)


def save_brains(brains: list[QBrain]) -> None:
    """Persist all brains on exit."""
    save_all(brains, BRAIN_PATH)
