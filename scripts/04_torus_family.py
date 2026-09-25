"""
04_torus_family.py: equilateral minimal polygons of T(p, p+1).

For each p, start from the symmetric construction, break symmetry with
random safe moves, and run the clearance-floor solver (equalizer #2) at a
decreasing sequence of floors mu0.  A run counts if the Millett-Rawdon
ratio is < 1 and the Alexander polynomial still matches T(p, p+1).  The
first certified polygon at the highest successful floor is polished to
exactly equal lengths, verified (Alexander in 4 projections, knot Floer
homology, 40-digit certificate) and saved to results/.

usage: python 04_torus_family.py p [trials]
"""
import sys, time
import numpy as np
from equistick.torus import torus_poly, STARTS
from equistick.geometry import normalize, min_dist, mr_ratio, angle_sum, lengths
from equistick.invariants import is_torus
from equistick.flows import agitate
from equistick.optimize import clearance_floor_solve
from equistick.certify import polish, mr_certificate_mp, verify_torus

FLOORS = {3: [0.04, 0.02, 0.01], 4: [0.02, 0.01, 0.005], 5: [0.01, 0.005, 0.0025],
          6: [0.005, 0.0025, 0.001], 7: [0.0025, 0.001, 0.0005],
          8: [0.001, 0.0005, 0.00025], 9: [0.0005, 0.00025, 0.0001]}

def run(p, trials=8, out='../results'):
    rng = np.random.default_rng(p)
    V0 = torus_poly(p, *STARTS[p]) if p in STARTS else None
    if V0 is None:
        from equistick.torus import symmetric_scan
        r, h, phif, _ = symmetric_scan(p, trials=1500)[0]
        V0 = torus_poly(p, 1.0, r, h, phif)
    for mu0 in FLOORS[p]:
        cert = []
        for _ in range(trials):
            V = normalize(V0.copy())
            agitate(V, rng, 0.15 * min_dist(V), 60 * len(V))
            W = clearance_floor_solve(V, mu0, maxiter=500)
            if mr_ratio(W)[0] < 1 and is_torus(W, p, p + 1):
                cert.append(W)
        print('T(%d,%d) %d sticks  floor %.5f: %d/%d certified' % (p, p + 1, 2 * p + 2, mu0, len(cert), trials), flush=True)
        if cert:
            W = polish(cert[0])
            d, mu, b, ok = mr_certificate_mp(W)
            alex, h, c = verify_torus(W, p, p + 1)
            L = lengths(W)
            print('   polished: spread %.1e, 40-digit certificate %s (defect %.1e < %.2e, mu %.4g)'
                  % (L.max() - L.min(), ok, float(d), float(b), float(mu)))
            print('   Alexander x4 %s; HFK genus %d fibred %s L-space %s tau %d rank %d; %d crossings; angle sum %.3f'
                  % (alex, h['seifert_genus'], h['fibered'], h['L_space_knot'], h['tau'], h['total_rank'], c, angle_sum(W)))
            np.savetxt('%s/T%d_%d_equilateral_%dsticks.txt' % (out, p, p + 1, 2 * p + 2), W, fmt='%.17g')
            return mu0
    return None

if __name__ == '__main__':
    p = int(sys.argv[1])
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(p, trials)
