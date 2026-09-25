import numpy as np, time, sys, json
from sk import *
from eq import agitate
from nlp import solve, polish
from reduce import reduce_once
from load import load

knots = sys.argv[1].split(',')
budget_red = float(sys.argv[2])   # seconds per knot for reductions
nreal = int(sys.argv[3])          # target number of independent 10-stick realizations
mus = [0.01, 0.005, 0.0025, 0.001]
out = open('batch19.log', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()

for name in knots:
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    V0 = load(name)
    reals = []
    t0 = time.time()
    while time.time() - t0 < budget_red and len(reals) < nreal:
        V = V0.copy(); ok = True
        while len(V) > 10:
            W, k = reduce_once(V, rng, steps=40000)
            if W is None:
                ok = False; break
            V = W
        if ok and len(V) == 10:
            idn = identify(V)
            if idn == name:
                reals.append(V); np.save('ten_%s_%d.npy' % (name, len(reals)), V)
    log('%s: %d ten-stick realizations in %.0fs' % (name, len(reals), time.time() - t0))
    best_overall = None
    for ri, V10 in enumerate(reals):
        for mu0 in mus:
            nc = 0; dmin = np.inf
            for trial in range(6):
                V = normalize(V10.copy()); agitate(V, rng, 0.1 * min_dist(V), 300)
                W = solve(V, mu0, maxiter=400)
                r, defect, mu = mr_ratio(W)
                if r < 1:
                    if identify(W) == name:
                        nc += 1
                        np.save('eq_%s_r%d_mu%g_%d.npy' % (name, ri, mu0, trial), W)
                        dmin = min(dmin, defect)
                        continue
                if identify(W) == name:
                    dmin = min(dmin, defect)
            log('   %s real%d mu0=%.4f certified %d/6 min defect (right type) %.2e' % (name, ri, mu0, nc, dmin))
            if nc > 0:
                best_overall = (ri, mu0)
                break
    log('%s RESULT: %s' % (name, ('EQUILATERAL 10-STICK FOUND (real%d, mu0=%g)' % best_overall) if best_overall else 'no equilateral 10-stick found'))
