"""Transverse, single-edge vertex moves on a small embedded polygon."""
import unittest

import numpy as np

from equistick.crossing_change import find_single_crossing, propose_moves, propose_nearest_moves
from equistick.geometry import min_dist, safe_move


SIX = np.array([[-1, -1, 0], [-1, 0, 1], [-1, 1, 0],
                [0, -.25, -1], [0, -.25, 2], [2, 2, 4]], dtype=np.float64)


class CrossingChangeTests(unittest.TestCase):
    def test_one_transverse_edge_piercing_two_embedded_endpoints(self):
        V = SIX.copy()
        destination = np.array([1., 0., 1.])
        W = V.copy()
        W[1] = destination
        self.assertGreater(min_dist(V), 1e-6)
        self.assertGreater(min_dist(W), 1e-6)
        self.assertFalse(safe_move(V, 1, destination, 1e-9))
        crossing = find_single_crossing(V, 1, destination)
        self.assertIsNotNone(crossing)
        self.assertEqual(crossing['edge'], 3)
        self.assertAlmostEqual(crossing['time'], 2 / 3, places=7)

    def test_finite_grid_includes_each_vertex_edge_and_reaches_a_crossing(self):
        proposals = list(propose_moves(SIX, fractions=(.5,), weights=(.25,), times=(.5,)))
        self.assertEqual(len(proposals), 36)
        hits = [(move, find_single_crossing(SIX, move['vertex'], move['destination']))
                for move in proposals]
        self.assertTrue(any(move['vertex'] == 1 and move['target_edge'] == 3 and
                            crossing is not None and crossing['edge'] == 3
                            for move, crossing in hits))

    def test_nearest_edge_grid_finds_the_transverse_fixture(self):
        proposals = list(propose_nearest_moves(SIX))
        self.assertGreater(len(proposals), 0)
        self.assertTrue(any(move['vertex'] == 1 and move['target_edge'] == 3 and
                            (event := find_single_crossing(SIX, 1, move['destination'])) is not None
                            and event['edge'] == 3 for move in proposals))


if __name__ == '__main__':
    unittest.main()
