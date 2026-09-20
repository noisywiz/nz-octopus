"""Shared session plumbing: argument parsing, brain load/save, progress log."""

import argparse
from pathlib import Path

from .brain import QBrain
from .creature import N_DIRECTIONS
from .maturity import MaturityConfig, MaturityReport, PlateauTracker, assess, ideal_rate
from .sensors import N_STATES

BRAIN_PATH = Path(__file__).resolve().parent.parent / "brain.json"
PROGRESS_PATH = BRAIN_PATH.with_name("progress.tsv")

LOG_HEADER = "tick\trate\tbest\tcompetence\tstates\tmature"


class ProgressMonitor:
    """Feeds world counters into the maturity tracker once per sim tick and
    appends one TSV line per completed window, so learning progress survives
    across runs. `latest` holds the freshest report for the HUD."""

    def __init__(self, brain: QBrain, config: MaturityConfig = MaturityConfig(),
                 path: Path = PROGRESS_PATH) -> None:
        self.brain = brain
        self.tracker = PlateauTracker(config)
        self.config = config
        self.path = path
        self.oracle_rate = ideal_rate()  # Monte-Carlo, cached: not cheap per frame
        self.latest: MaturityReport | None = None
        self.ticks = 0
        self._last_food = 0
        if not path.exists():
            path.write_text(LOG_HEADER + "\n")
        else:
            self._restore()

    def _restore(self) -> None:
        """Rebuild window history from the TSV log, so the HUD shows a stage
        immediately instead of waiting 25k ticks for the first fresh window.

        The plateau streak is deliberately not restored: a plateau must be
        confirmed by windows closed in *this* session, otherwise a stale
        verdict from an old run would stick around forever.
        """
        rows = self.path.read_text().splitlines()[1:]
        for row in rows:
            parts = row.split("\t")
            if len(parts) >= 3:
                self.tracker.rates.append(float(parts[1]))
                self.ticks = int(parts[0])
        self.tracker.best_rate = max(self.tracker.rates, default=0.0)
        if self.tracker.rates:
            self.latest = assess(self.brain, self.tracker)

    def snapshot(self) -> str:
        """Cheap one-line status: current window vs the all-time best.

        Deliberately shows nothing monotonic: "best" only ever grows, so
        watching it feels like endless progress. The interesting signal is
        how the *current* window compares to that best — rising toward it
        (still improving) or hovering just under it (plateau).
        """
        t = self.tracker
        size = self.config.window
        projected = (t.current.food * 1000.0 / size) if t.current.ticks < size \
            else (t.current.food * 1000.0 / max(t.current.ticks, 1))
        if t.best_rate > 0.0:
            ratio = projected / t.best_rate
            phase = "near best" if ratio >= 0.85 else "below best"
            core = f"window {projected:.2f} food/1k ({phase}, {ratio * 100:.0f}% of best)"
        else:
            core = f"window {projected:.2f} food/1k (no best yet)"
        return f"{core}, states {self.latest.coverage if self.latest else 0}/{self.latest.total_states if self.latest else 0} explored"

    def step(self, world_food: int) -> None:
        """Account one sim tick; log when a window just closed."""
        self.tracker.record(1, world_food - self._last_food)
        self._last_food = world_food
        self.ticks += 1
        if self.tracker.current.ticks == 0 and self.tracker.rates:
            self._log_closed_window()

    def _log_closed_window(self) -> None:
        """Assess the brain and append the window's row to the TSV log."""
        report = assess(self.brain, self.tracker)
        self.latest = report
        row = (f"{self.ticks}\t{self.tracker.rates[-1]:.3f}\t{report.best_rate:.3f}\t"
               f"{report.competence:.3f}\t{report.coverage}\t{int(report.mature)}\n")
        with self.path.open("a") as fh:
            fh.write(row)

    def hud_line(self) -> str:
        """Full HUD status: live window progress, plus the last verdict if any."""
        line = f"growth: {self.snapshot()}"
        if self.latest is not None:
            line += (f" | {self.latest.stage()}, "
                     f"{self.latest.competence * 100:.0f}% of oracle")
        return line


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
