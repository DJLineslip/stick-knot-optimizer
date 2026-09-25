"""
01_ladder_check.py: test the ladder lemma on published equilateral data.

Ladder lemma.  Let K have bridge index b and let P be a (2b+2)-gon of type
K.  Every generic direction sees at least b local maxima; Milnor's formula
(total curvature = 2 pi x average number of maxima) then gives
sum(turning angles) >= 2 pi b, i.e. sum(interior angles) <= 2 pi.
If all edges have length 1, |v_{i+1} - v_{i-1}| = 2 sin(beta_i/2) <= beta_i,
so the closed polygons through the even and the odd vertices ("rails")
have total length <= 2 pi, while 2b+2 unit "rungs" zigzag between them.

Checked here on every bridge-tight equilateral polygon in Eddy's data.
Expected: all angle sums and rail totals below 2 pi.
"""
import numpy as np
from equistick.data import load_eddy
from equistick.geometry import angle_sum, rail_lengths

# (knot, bridge index) with 2b+2 = number of sticks in Eddy's file
CASES = [('3_1', 2), ('8_19', 3), ('8_20', 3), ('K13n592', 4), ('K15n41127', 4)]
if __name__ == '__main__':
    print('knot       n  b  angle sum  rails(even+odd)   2pi = %.3f' % (2 * np.pi))
    for name, b in CASES:
        V = load_eddy(name)
        assert len(V) == 2 * b + 2
        le, lo = rail_lengths(V / np.linalg.norm(V[1] - V[0]))
        print('%-9s %2d %2d  %8.3f  %8.3f' % (name, len(V), b, angle_sum(V), le + lo))
