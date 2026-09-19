"""Shared session plumbing: argument parsing, brain load/save lifecycle."""

import argparse
from pathlib import Path

from .brain import QBrain
from .sensors import N_STATES
from .creature import N_DIRECTIONS

BRAIN_PATH = Path(__file__).resolve().parent.parent / "brain.json"


def parse_args(description: str, default_speed: int) -> argparse.Namespace:
    """Arguments common to both viewers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--speed", type=int, default=default_speed,
                        help="sim ticks per frame")
    parser.add_argument("--fresh", action="store_true",
                        help="start with an empty brain")
    return parser.parse_args()


def open_brain(fresh: bool) -> QBrain:
    """Load the brain from disk, or start empty with --fresh."""
    if fresh and BRAIN_PATH.exists():
        BRAIN_PATH.unlink()
    return QBrain.load(BRAIN_PATH, N_STATES, N_DIRECTIONS)


def save_brain(brain: QBrain) -> None:
    """Persist the brain on exit."""
    brain.save(BRAIN_PATH)
