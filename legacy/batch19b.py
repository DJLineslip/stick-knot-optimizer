import numpy as np, time, sys
from sk import *
from reduce import reduce_once
from homo import fatten, homotopy_equalize
from load import load
from verify import mr_certificate_mp

knots = sys.argv[1].split(','); budget = float(sys.argv[2])
out = open('batch19b.log', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()
for name in knots:
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    V0 = load(name); t0 = time.time(); found = None; nreal = 0
    while time.time() - t0 < budget and found is None:
        V = V0.copy(); ok = True
        while len(V) > 10:
            W, k = reduce_once(V, rng, steps=40000)
            if W is None:
                ok = False; break
            V = W
        if not ok or identify(V) != name:
            continue
        nreal += 1
        Vf = fatten(V, rng, 400)
        for floor in (0.9, 0.5, 0.2):
            E, t, mu0, done = homotopy_equalize(Vf, mu_floor=floor, tlimit=120)
            r, defect, mu = mr_ratio(E)
            if done and r < 1 and identify(E) == name:
                d, mmu, b, okc = mr_certificate_mp(E)
                found = (E, float(d), float(mmu), float(b), okc)
                np.savetxt('eq10_%s.txt' % name, normalize(E), fmt='%.17g')
                break
    if found:
        log('%-9s equilateral 10-stick FOUND  (defect %.1e, mu %.4f, MR bound %.1e, certified %s)  after %d reduction(s), %.0fs' % (name, found[1], found[2], found[3], found[4], nreal, time.time() - t0))
    else:
        log('%-9s not found: %d ten-stick realization(s) in %.0fs' % (name, nreal, time.time() - t0))
