"""Conjugation: Q-knowledge flows only through cells the donor has explored."""

import unittest

from src.sim.brain import QBrain, conjugate
from src.sim.creature import N_DIRECTIONS
from src.sim.sensors import N_STATES
from tests.helpers import make_brain


class ConjugateTest(unittest.TestCase):
    def test_blends_known_cells_only(self) -> None:
        donor = make_brain()
        donor.row(3)[2] = 5.0
        donor.row(3)[4] = -1.0
        receiver = make_brain()
        receiver.row(3)[2] = 1.0  # receiver already knows this cell
        shared = conjugate(donor, receiver)
        self.assertEqual(shared, 2, "donor has 2 explored cells")
        self.assertAlmostEqual(receiver.row(3)[2], 1.0 + 0.05 * 4.0)

    def test_zero_donor_cells_do_not_erase_receiver(self) -> None:
        donor = QBrain(N_STATES, N_DIRECTIONS)
        receiver = QBrain(N_STATES, N_DIRECTIONS)
        receiver.row(9)[0] = 3.0
        conjugate(donor, receiver)
        self.assertEqual(receiver.row(9)[0], 3.0,
                         "zero donor cells must not touch receiver")


if __name__ == "__main__":
    unittest.main()
