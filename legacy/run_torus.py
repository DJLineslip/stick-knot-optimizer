import numpy as np, time, sys
from sk import *
from eq import agitate
from nlp import *
from torus import torus_poly, is_torus

starts = {5: (1.0, 1.286, 1.040, 0.457), 6: (1.0, 1.197, 0.886, 0.535), 7: (1.0, 0.897, 0.898, 0.470)}
p = int(sys.argv[1]); mus = [float(x) for x in sys.argv[2].split(',')]; ntr = int(sys.argv[3])
rng = np.random.default_rng(p)
V0 = torus_poly(p, *starts[p])
t0 = time.time()
for mu0 in mus:
    res = []
    for trial in range(ntr):
        V = normalize(V0.copy()); agitate(V, rng, 0.15 * min_dist(V), 60 * len(V))
        W = solve(V, mu0, maxiter=500)
        ok = is_torus(W, p, p + 1)
        r, defect, mu = mr_ratio(W)
        res.append((defect, mu, r, ok))
        if ok and r < 1:
            np.save('T%d%d_cert_mu%g_%d.npy' % (p, p + 1, mu0, trial), W)
    good = [x for x in res if x[3]]
    print('T(%d,%d) n=%d mu0=%.4f kept %d/%d  min defect %s  certified %d   (%.0fs)' % (
        p, p + 1, 2 * p + 2, mu0, len(good), ntr,
        ('%.2e' % min(x[0] for x in good)) if good else '-', sum(1 for x in good if x[2] < 1), time.time() - t0), flush=True)
