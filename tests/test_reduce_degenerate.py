"""Degenerate candidate triangles must not be treated as deletable."""
import unittest
import numpy as np
from equistick.reduce import tri_pierce_weight


class DegeneratePenaltyTest(unittest.TestCase):
    def test_near_collinear_triangle_never_divides_by_zero(self):
        a = np.array([0., 0., 0.])
        b = np.array([1., 0., 0.])
        c = np.array([1., 1e-12, 0.])
        p = np.array([0.5, 0., 1.])
        q = np.array([0.5, 0., -1.])
        self.assertGreater(tri_pierce_weight(p, q, a, b, c), 0.0)
