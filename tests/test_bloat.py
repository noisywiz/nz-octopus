"""Overeating bloat: hunger near zero swells the body into a slow ball."""

import math
import unittest

from src.sim import metabolism as meta
from src.sim.creature import Creature, Vec, bloat_reach, move, separate
from src.sim.world import CreatureBrain
from tests.helpers import make_brain, make_world


class BloatTargetTest(unittest.TestCase):
    def test_full_when_hunger_is_zero(self) -> None:
        self.assertAlmostEqual(meta.bloat_target(0.0), 1.0)

    def test_empty_above_threshold(self) -> None:
        self.assertEqual(meta.bloat_target(meta.BLOAT_THRESHOLD), 0.0)
        self.assertEqual(meta.bloat_target(50.0), 0.0)

    def test_ease_moves_toward_target(self) -> None:
        self.assertGreater(meta.ease_bloat(0.0, 1.0), 0.0)
        self.assertLess(meta.ease_bloat(1.0, 0.0), 1.0)

    def test_inflation_slower_than_deflation(self) -> None:
        up = meta.ease_bloat(0.0, 1.0)
        down = 1.0 - meta.ease_bloat(1.0, 0.0)
        self.assertLess(up, down, "puffing up must be slower than digesting down")


class BloatMoveTest(unittest.TestCase):
    def _moved(self, bloat: float) -> float:
        c = Creature(pos=Vec(40.0, 15.0), heading=0.0, hunger=0.0, starving=0.0)
        c.bloat = bloat
        start = c.pos
        for _ in range(10):
            move(c, 0)
        return math.hypot(c.pos.x - start.x, c.pos.y - start.y)

    def test_bloated_creature_crawls(self) -> None:
        slim, fat = self._moved(0.0), self._moved(1.0)
        self.assertGreater(slim, 0.0)
        self.assertLess(fat, slim * 0.2,
                        f"full bloat must slow to a crawl: slim={slim:.3f} fat={fat:.3f}")

    def test_bloated_reach_is_wider(self) -> None:
        c = Creature(pos=Vec(40.0, 15.0), heading=0.0, hunger=0.0, starving=0.0)
        c.bloat = 1.0
        self.assertGreater(bloat_reach(c), bloat_reach(Creature(
            pos=Vec(40.0, 15.0), heading=0.0, hunger=30.0, starving=0.0,
        )))

    def test_ball_pushes_neighbor_from_afar(self) -> None:
        slim = Creature(pos=Vec(40.0, 15.0), heading=0.0, hunger=30.0, starving=0.0)
        ball = Creature(pos=Vec(43.5, 15.0), heading=math.pi,
                        hunger=0.0, starving=0.0)
        ball.bloat = 1.0
        self.assertTrue(separate(slim, ball),
                        "a bloated ball must shove a neighbor outside the normal radius")


class WorldBloatTest(unittest.TestCase):
    def test_feasting_swell_and_recovery(self) -> None:
        cb = CreatureBrain(body=Creature(pos=Vec(40.0, 15.0), heading=0.0,
                                         hunger=0.0, starving=0.0),
                           brain=make_brain())
        w = make_world([cb])
        for _ in range(50):
            w.step()
        self.assertGreater(cb.body.bloat, 0.3,
                           "a freshly fed creature must swell up")
        cb.body.hunger = meta.BLOAT_THRESHOLD + 20.0  # digest it all away
        for _ in range(400):
            w.step()
        self.assertLess(cb.body.bloat, 0.05, "the ball must deflate once digested")

    def test_bloat_penalized(self) -> None:
        stuffed = meta.metabolize(5.0, False, bloat=1.0)
        slim = meta.metabolize(5.0, False, bloat=0.0)
        self.assertLess(stuffed.reward, slim.reward)


if __name__ == "__main__":
    unittest.main()
