"""Canvas rendering: one frame with two touching creatures, headless."""

import os
import unittest

from src.sim.creature import Creature, Vec
from src.sim.world import CreatureBrain
from tests.helpers import make_brain, make_world


class CanvasRenderTest(unittest.TestCase):
    def test_draw_one_frame(self) -> None:
        """Draw one frame with two touching creatures on a dummy SDL driver."""
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame

        from src import canvas as cv
        from src.sim.animator import Animator

        cbs = [
            CreatureBrain(body=Creature(pos=Vec(40.0, 15.0), heading=0.0,
                                        hunger=30.0, starving=0.0),
                          brain=make_brain()),
            CreatureBrain(body=Creature(pos=Vec(41.0, 15.0), heading=3.141592653589793,
                                        hunger=30.0, starving=0.0),
                          brain=make_brain()),
        ]
        w = make_world(cbs)
        pygame.init()
        try:
            screen = pygame.display.set_mode(cv.SCREEN)
            font = pygame.font.SysFont("monospace", 14)
            animators: dict[int, Animator] = {}
            w.step()  # populate world.touching so the halo path is exercised
            for cb in w.creatures:  # populate like run() does before drawing
                animators.setdefault(cb.brain.id, Animator()).update(
                    cb.body, 1.0 / cv.FPS,
                )
            cv.draw_conjugations(screen, w, animators, t=0.5)
            cv.draw_all(screen, w, t=0.5, animators=animators)
            cv.draw_stats(screen, font, speed=1, paused=False)
            self.assertEqual(w.conjugations, 1,
                             "one step of contact: drawing must not add more")
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
