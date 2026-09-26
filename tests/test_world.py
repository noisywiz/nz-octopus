"""World-level behavior: contact, conjugation, id hygiene, food pouring."""

import math
import random
import unittest

from src.sim import world as wd
from src.sim import terrain as tn
from src.sim.creature import Creature, Vec
from src.sim.world import CreatureBrain
from tests.helpers import make_brain, make_world


class ContactTest(unittest.TestCase):
    def test_contact_and_conjugation(self) -> None:
        brains = [make_brain(), make_brain()]
        brains[0].row(5)[1] = 2.0
        brains[0].row(5)[3] = -0.5
        cbs = [
            CreatureBrain(body=Creature(pos=Vec(40.0, 15.0), heading=0.0,
                                        hunger=30.0, starving=0.0),
                          brain=brains[0]),
            CreatureBrain(body=Creature(pos=Vec(41.0, 15.0), heading=math.pi,
                                        hunger=30.0, starving=0.0),
                          brain=brains[1]),
        ]
        w = make_world(cbs)
        min_d = math.inf
        for _ in range(300):
            w.step()
            a, b = w.creatures[0].body.pos, w.creatures[1].body.pos
            min_d = min(min_d, math.hypot(b.x - a.x, b.y - a.y))
            for cb in w.creatures:  # never outside the tank
                self.assertGreaterEqual(cb.body.pos.x, wd.WALL_MARGIN - 1e-6)
                self.assertLessEqual(cb.body.pos.x, wd.WIDTH - wd.WALL_MARGIN + 1e-6)
                self.assertGreaterEqual(cb.body.pos.y, wd.WALL_MARGIN - 1e-6)
                self.assertLessEqual(cb.body.pos.y, wd.HEIGHT - wd.WALL_MARGIN + 1e-6)
        self.assertGreaterEqual(w.conjugations, 1,
                                "bodies born touching must conjugate at least once")
        self.assertGreater(min_d, 0.5, f"bodies must not fully overlap, min_d={min_d:.3f}")
        rec_row = w.creatures[1].brain.row(5)
        self.assertTrue(any(v != 0.0 for v in rec_row),
                        "receiver Q-row must receive knowledge")

    def test_single_creature_regression(self) -> None:
        w = make_world([CreatureBrain(
            body=Creature(pos=Vec(wd.WIDTH / 2, wd.HEIGHT / 2),
                          heading=random.Random(3).uniform(0, 6.283185307179586),
                          hunger=30.0, starving=0.0),
            brain=make_brain(),
        )])
        for _ in range(2000):
            w.step()
        self.assertEqual(w.conjugations, 0)


class IdsTest(unittest.TestCase):
    def test_spawn_ids_stay_unique(self) -> None:
        """A viewer-built world: loaded brain id 0, next_id still at default 0."""
        brain = make_brain()
        brain.id = 0
        w = make_world([CreatureBrain(
            body=Creature(pos=Vec(40.0, 15.0), heading=0.0,
                          hunger=30.0, starving=0.0),
            brain=brain,
        )])
        spawned = w.spawn_at(40.0, 15.0)
        self.assertNotEqual(spawned.brain.id, w.creatures[0].brain.id,
                            "spawned brain must not reuse the loaded id")
        self.assertGreater(w.next_id, spawned.brain.id)

    def test_duplicate_loaded_ids_renumbered(self) -> None:
        """A save written before the fix holds duplicate ids; World must heal it."""
        brains = [make_brain(), make_brain(), make_brain()]
        brains[0].id = 0
        brains[1].id = 0  # the corrupted first spawn
        brains[2].id = 1  # collides with the renumbered duplicate
        w = make_world([
            CreatureBrain(body=Creature(pos=Vec(10.0, 10.0), heading=0.0,
                                        hunger=30.0, starving=0.0), brain=b)
            for b in brains
        ])
        ids = [cb.brain.id for cb in w.creatures]
        self.assertEqual(len(set(ids)), len(ids), f"ids must be unique, got {ids}")
        self.assertGreater(w.next_id, max(ids))


class PourFoodTest(unittest.TestCase):
    def test_pour_adds_pieces_past_the_cap(self) -> None:
        """The f key: a handful of pieces at once, past the passive cap."""
        w = make_world([CreatureBrain(
            body=Creature(pos=Vec(wd.WIDTH / 2, wd.HEIGHT / 2), heading=0.0,
                          hunger=30.0, starving=0.0),
            brain=make_brain(),
        )])
        before = len(w.food)
        w.pour_food()
        self.assertEqual(len(w.food), before + wd.POUR_COUNT,
                         "pour must add exactly POUR_COUNT pieces")
        w.pour_food(2)
        self.assertEqual(len(w.food), before + wd.POUR_COUNT + 2)


class ResizeTest(unittest.TestCase):
    def test_resize_grows_shrinks_and_clamps(self) -> None:
        w = make_world([CreatureBrain(
            body=Creature(pos=Vec(40.0, 15.0), heading=0.0,
                          hunger=30.0, starving=0.0),
            brain=make_brain(),
        )])
        w.pour_food(3)
        w.resize(160.0, 60.0)
        self.assertEqual(w.width, 160.0)
        self.assertEqual(len(w.terrain), int(160.0 / tn.ART_TO_WORLD),
                         "terrain must be rebuilt for the new width")
        w.step()  # sim must run in the bigger tank without escaping it
        for cb in w.creatures:
            self.assertLessEqual(cb.body.pos.x, w.width - wd.WALL_MARGIN)
        w.resize(20.0, 10.0)
        for cb in w.creatures:  # shrink: bodies clamped back inside
            self.assertGreaterEqual(cb.body.pos.x, wd.WALL_MARGIN)
            self.assertLessEqual(cb.body.pos.x, w.width - wd.WALL_MARGIN)
            self.assertGreaterEqual(cb.body.pos.y, wd.WALL_MARGIN)
        self.assertTrue(all(wd.WALL_MARGIN <= f.pos.x <= w.width - wd.WALL_MARGIN
                            for f in w.food),
                        "food outside the shrunk tank must be dropped")


if __name__ == "__main__":
    unittest.main()
