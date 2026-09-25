"""
03_false_collapse.py: reproduce the false obstruction signal.

Equalizer #1 (path lifting, equistick.flows) started near the symmetric
T(4,5) construction.  Watch mu fall in lockstep with the length defect.
This looks exactly like the collapse Rawdon and Scharein saw for 8_19, but
it is the flow heading to the symmetric equilateral point, which is always
singular; 04_torus_family.py certifies T(4,5) anyway.
"""
import numpy as np
from equistick.torus import torus_poly, STARTS
from equistick.geometry import normalize, min_dist, mr_ratio
from equistick.flows import agitate, step

if __name__ == '__main__':
    rng = np.random.default_rng(3)
    V = normalize(torus_poly(4, *STARTS[4]))
    agitate(V, rng, 0.02 * min_dist(V), 50)
    print(' it   defect      mu     mu/defect')
    for it in range(41):
        step(V, rng, 0.2, 0.3)
        V[:] = normalize(V)
        r, defect, mu = mr_ratio(V)
        if it % 5 == 0:
            print('%3d  %.3e  %.3e  %.2f' % (it, defect, mu, mu / defect))
