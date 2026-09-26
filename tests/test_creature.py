"""Soft collision: bodies push apart, distant bodies ignore each other."""

import math
import unittest

from src.sim.creature import CONTACT_RADIUS, Creature, Vec, separate


class SeparateTest(unittest.TestCase):
    def test_pushes_apart(self) -> None:
        a = Creature(pos=Vec(10.0, 10.0), heading=0.0, hunger=30.0, starving=0.0)
        b = Creature(pos=Vec(11.0, 10.0), heading=math.pi,
                     hunger=30.0, starving=0.0)
        self.assertTrue(separate(a, b), "bodies 1.0 apart must count as touching")
        d = math.hypot(b.pos.x - a.pos.x, b.pos.y - a.pos.y)
        self.assertGreater(d, 1.0, f"expected push apart, got d={d:.3f}")

    def test_distant_bodies_ignored(self) -> None:
        a = Creature(pos=Vec(10.0, 10.0), heading=0.0, hunger=30.0, starving=0.0)
        b = Creature(pos=Vec(10.0 + CONTACT_RADIUS + 5.0, 10.0), heading=0.0,
                     hunger=30.0, starving=0.0)
        self.assertFalse(separate(a, b), "distant bodies must not interact")

    def test_exact_overlap_resolves(self) -> None:
        a = Creature(pos=Vec(5.0, 5.0), heading=0.0, hunger=30.0, starving=0.0)
        b = Creature(pos=Vec(5.0, 5.0), heading=0.0, hunger=30.0, starving=0.0)
        self.assertTrue(separate(a, b))
        d = math.hypot(b.pos.x - a.pos.x, b.pos.y - a.pos.y)
        self.assertGreater(d, 0.0, f"exact overlap must resolve, got d={d}")


if __name__ == "__main__":
    unittest.main()
