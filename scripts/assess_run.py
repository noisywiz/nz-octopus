"""Headless learning run with periodic maturity assessment. Debug tool."""

import random
import time

from sim import world as wd
from sim.brain import QBrain
from sim.creature import N_DIRECTIONS
from sim.maturity import MaturityConfig, PlateauTracker, assess, ideal_rate
from sim.sensors import N_STATES

TICKS = 150_000
SEED = 3


def main() -> None:
    rng = random.Random(SEED)
    world = wd.World(brain=QBrain(N_STATES, N_DIRECTIONS), rng=rng)
    tracker = PlateauTracker(config=MaturityConfig())
    oracle = ideal_rate()
    last_len = -1
    start = time.time()
    while world.ticks < TICKS:
        food_before = world.food_eaten
        world.step()
        tracker.record(1, world.food_eaten - food_before)
        if tracker.current.ticks == 0 and len(tracker.rates) > last_len:
            last_len = len(tracker.rates)
            rep = assess(world.brain, tracker)
            print(f"tick {world.ticks:>7}: {rep.summary()}", flush=True)
    rep = assess(world.brain, tracker)
    print(f"\nfinal: {rep.summary()}")
    print(f"mature={rep.mature} plateau={rep.plateau} "
          f"coverage_ok={rep.coverage_ok} competence_ok={rep.competence_ok}")
    print(f"walltime {time.time() - start:.0f}s, bumps={world.wall_bumps}")


if __name__ == "__main__":
    main()
