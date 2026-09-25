"""
05_tenstick.py: equilateral 10-stick versions of the 4-bridge knots with
stick number exactly 10.

Pipeline per knot: take Eddy's equilateral 11- or 12-stick polygon ->
anneal down to 10 sticks (equistick.reduce), check SnapPy identifies the
right knot -> fatten -> safe homotopy to equal lengths (equalizer #3) ->
Millett-Rawdon certificate in 40 digits -> SnapPy identification again.
Results are appended to results/tenstick.log and coordinates saved.

usage: python 05_tenstick.py K11n71,K11n75,... [seconds_per_knot]
"""
import sys, time
import numpy as np
from equistick.data import load_eddy, eddy_available, seed_for
from equistick.geometry import normalize, mr_ratio
from equistick.invariants import identify
from equistick.reduce import reduce_to
from equistick.optimize import fatten, homotopy_equalize
from equistick.certify import mr_certificate_mp

def run(name, budget=240, target=10, out='../results'):
    if not eddy_available(name):
        return 'no starting data in Eddy repository'
    rng = np.random.default_rng(seed_for(name))
    V0 = load_eddy(name)
    t0, nreal = time.time(), 0
    while time.time() - t0 < budget:
        V = reduce_to(V0.copy(), target, rng)
        if V is None or identify(V) != name:
            continue
        nreal += 1
        Vf = fatten(V, rng, 400)
        for floor in (0.9, 0.5, 0.2):
            E, t, mu0, done = homotopy_equalize(Vf, mu_floor=floor, tlimit=120)
            if done and mr_ratio(E)[0] < 1 and identify(E) == name:
                d, mu, b, ok = mr_certificate_mp(E)
                np.savetxt('%s/%s_equilateral_%dsticks.txt' % (out, name, target), normalize(E), fmt='%.17g')
                return ('FOUND equilateral %d-stick (defect %.1e, mu %.4f, bound %.1e, certified %s), %d reduction(s), %.0fs'
                        % (target, float(d), float(mu), float(b), ok, nreal, time.time() - t0))
    return 'not found: %d %d-stick realization(s) in %.0fs' % (nreal, target, time.time() - t0)

if __name__ == '__main__':
    names = sys.argv[1].split(',')
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    with open('../results/tenstick.log', 'a') as log:
        for nm in names:
            msg = '%-9s %s' % (nm, run(nm, budget))
            print(msg, flush=True)
            log.write(msg + '\n')
