"""Headless check: octopus arms catch food and swallow it at the mouth."""

import os
import random

os.environ["SDL_VIDEODRIVER"] = "dummy"

from sim import food as fd  # noqa: E402
from sim import octopus_body as ob  # noqa: E402
from sim.world import World  # noqa: E402


def main() -> None:
    world = World(octopus=True, rng=random.Random(7))
    cb = world.creatures[0]
    assert cb.arms is not None

    # one piece just within grasp margin of the body
    piece = fd.Food(base_x=cb.body.pos.x + 3.0, y=cb.body.pos.y, phase=0.0)
    world.food = [piece]
    world.ticks_since_spawn = 10_000  # keep the spawner quiet

    grabbed = False
    eaten = False
    for _ in range(3000):
        world.step()
        if not grabbed and any(a.holding is not None for a in cb.arms.limbs):
            grabbed = True
            print(f"grabbed at tick {world.ticks}, "
                  f"tip={ob.tip_of(cb.arms.limbs[0])}")
        if world.food_eaten > 0:
            eaten = True
            print(f"eaten at tick {world.ticks}, hunger={cb.body.hunger:.1f}")
            break
    assert grabbed, "no arm ever gripped the piece"
    assert eaten, "piece was never delivered to the mouth"
    print("OK: reach -> grip -> carry -> swallow")


if __name__ == "__main__":
    main()
