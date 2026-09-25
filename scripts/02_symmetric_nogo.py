"""
02_symmetric_nogo.py: the symmetry no-go, numerically.

Polygons invariant under a rotation advancing every vertex two steps are
the shape of the classical torus-knot constructions.  Equal lengths force
each odd vertex into the vertical plane bisecting its neighbours; then
every even/odd crossing seen down the axis is a real intersection.  The
scan samples all such equilateral polygons with 2m sticks and every
angular step k, and counts embedded and knotted ones.
Expected: knotted = 0 everywhere.  (Replaces legacy/sym.py + scan.py,
which did the same with pyknotid on a grid.)
"""
from equistick.torus import symmetric_equilateral_check

if __name__ == '__main__':
    for m in (3, 4, 5, 6):
        res = symmetric_equilateral_check(m, trials=400)
        emb = sum(v[0] for v in res.values())
        kn = sum(v[1] for v in res.values())
        print('%2d sticks: %5d embedded samples over all k, %d knotted' % (2 * m, emb, kn), flush=True)
