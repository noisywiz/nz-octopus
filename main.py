"""Entry point: parse args, load the brains, run the canvas, save the brains."""

import random

from src.sim import World, session
from src.sim.creature import Creature, Vec
from src.sim.world import WIDTH, HEIGHT, CreatureBrain

from src import canvas


def main() -> None:
    """One session: load brains, live, persist brains on any exit."""
    args = session.parse_args("nz-octopus", default_speed=1)
    brains = session.open_brains(args.brain)
    world = World(creatures=[CreatureBrain(
        body=Creature(
            pos=Vec(WIDTH / 2, HEIGHT / 2),
            heading=random.uniform(0, 6.283185307179586),
            hunger=30.0,
            starving=0.0,
        ),
        brain=brain,
    ) for brain in brains])
    try:
        canvas.run(world, max(1, args.speed))
    finally:
        if args.brain:  # default runs stay disposable: nothing is written
            session.save_brains([cb.brain for cb in world.creatures])


if __name__ == "__main__":
    main()
