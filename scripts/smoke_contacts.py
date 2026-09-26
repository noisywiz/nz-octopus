"""Headless smoke test: soft collision + conjugation. Debug tool."""
import math
import random

from sim import world as wd
from sim.brain import QBrain, conjugate
from sim.creature import Creature, CONTACT_RADIUS, N_DIRECTIONS, Vec, separate
from sim.sensors import N_STATES
from sim.world import CreatureBrain


def make_brain() -> QBrain:
    return QBrain(N_STATES, N_DIRECTIONS)


def make_world(creatures: list[CreatureBrain], seed: int = 7) -> wd.World:
    return wd.World(creatures=creatures, rng=random.Random(seed))


def test_separate_pushes_apart() -> None:
    a = Creature(pos=Vec(10.0, 10.0), heading=0.0, hunger=30.0, starving=0.0)
    b = Creature(pos=Vec(11.0, 10.0), heading=math.pi, hunger=30.0, starving=0.0)
    assert separate(a, b), "bodies 1.0 apart must count as touching"
    d = math.hypot(b.pos.x - a.pos.x, b.pos.y - a.pos.y)
    assert d > 1.0, f"expected push apart, got d={d:.3f}"
    far = Creature(pos=Vec(10.0, 10.0), heading=0.0, hunger=30.0, starving=0.0)
    far2 = Creature(pos=Vec(10.0 + CONTACT_RADIUS + 5.0, 10.0), heading=0.0,
                    hunger=30.0, starving=0.0)
    assert not separate(far, far2), "distant bodies must not interact"
    print("separate: ok")


def test_separate_exact_overlap() -> None:
    a = Creature(pos=Vec(5.0, 5.0), heading=0.0, hunger=30.0, starving=0.0)
    b = Creature(pos=Vec(5.0, 5.0), heading=0.0, hunger=30.0, starving=0.0)
    assert separate(a, b)
    d = math.hypot(b.pos.x - a.pos.x, b.pos.y - a.pos.y)
    assert d > 0.0, f"exact overlap must resolve, got d={d}"
    print("separate overlap: ok")


def test_conjugate_blends_known_cells_only() -> None:
    donor = make_brain()
    donor.row(3)[2] = 5.0
    donor.row(3)[4] = -1.0
    receiver = make_brain()
    receiver.row(3)[2] = 1.0  # receiver already knows this cell
    shared = conjugate(donor, receiver)
    assert shared == 2, f"donor has 2 explored cells, shared={shared}"
    assert abs(receiver.row(3)[2] - (1.0 + 0.05 * 4.0)) < 1e-9
    # untouched donor cells (0.0) must not drag receiver values to zero
    receiver2 = make_brain()
    receiver2.row(9)[0] = 3.0
    conjugate(donor, receiver2)
    assert receiver2.row(9)[0] == 3.0, "zero donor cells must not touch receiver"
    print("conjugate: ok")


def test_world_contact_and_conjugation() -> None:
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
            assert wd.WALL_MARGIN - 1e-6 <= cb.body.pos.x <= wd.WIDTH - wd.WALL_MARGIN + 1e-6
            assert wd.WALL_MARGIN - 1e-6 <= cb.body.pos.y <= wd.HEIGHT - wd.WALL_MARGIN + 1e-6
    assert w.conjugations >= 1, "bodies born touching must conjugate at least once"
    assert min_d > 0.5, f"bodies must not fully overlap, min_d={min_d:.3f}"
    # the newborn must have caught the veteran's known cells
    rec_row = w.creatures[1].brain.row(5)
    assert any(v != 0.0 for v in rec_row), "receiver Q-row must receive knowledge"
    print(f"world contact: ok (conjugations={w.conjugations}, min_d={min_d:.2f})")


def test_single_creature_regression() -> None:
    w = make_world([CreatureBrain(
        body=Creature(pos=Vec(wd.WIDTH / 2, wd.HEIGHT / 2),
                      heading=random.Random(3).uniform(0, 6.283185307179586),
                      hunger=30.0, starving=0.0),
        brain=make_brain(),
    )])
    for _ in range(2000):
        w.step()
    assert w.conjugations == 0
    print("single creature: ok")


def test_canvas_render() -> None:
    """Draw one frame with two touching creatures on a dummy SDL driver."""
    import os

    import pygame

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import canvas as cv

    brains = [make_brain(), make_brain()]
    cbs = [
        CreatureBrain(body=Creature(pos=Vec(40.0, 15.0), heading=0.0,
                                    hunger=30.0, starving=0.0),
                      brain=brains[0]),
        CreatureBrain(body=Creature(pos=Vec(41.0, 15.0), heading=math.pi,
                                    hunger=30.0, starving=0.0),
                      brain=brains[1]),
    ]
    w = make_world(cbs)
    pygame.init()
    try:
        screen = pygame.display.set_mode(cv.SCREEN)
        font = pygame.font.SysFont("monospace", 14)
        animators: dict[int, cv.Animator] = {}
        w.step()  # populate world.touching so the halo path is exercised
        for cb in w.creatures:  # populate like run() does before drawing
            animators.setdefault(cb.brain.id, cv.Animator()).update(
                cb.body, 1.0 / cv.FPS,
            )
        cv.draw_conjugations(screen, w, animators, t=0.5)
        cv.draw_all(screen, w, t=0.5, animators=animators)
        cv.draw_stats(screen, font, w, speed=1, paused=False, monitor=None)
        assert w.conjugations == 1  # one step of contact: drawing must not add more
        print("canvas render: ok")
    finally:
        pygame.quit()


if __name__ == "__main__":
    test_separate_pushes_apart()
    test_separate_exact_overlap()
    test_conjugate_blends_known_cells_only()
    test_world_contact_and_conjugation()
    test_single_creature_regression()
    test_canvas_render()
    print("ALL OK")
