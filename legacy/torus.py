import numpy as np
from sk import *

def torus_poly(p, R, r, h, phif, h2=None):
    q = p + 1
    a = 2 * np.pi * p / q
    phi = phif * a
    h2 = -h if h2 is None else h2
    V = []
    for j in range(q):
        V.append([R*np.cos(j*a), R*np.sin(j*a), h])
        V.append([r*np.cos(j*a+phi), r*np.sin(j*a+phi), h2])
    return np.array(V)

def is_torus(V, p, q, ts=None):
    ts = np.exp(1j*np.array([0.37, 0.91, 1.43, 2.07, 2.71])) if ts is None else ts
    rng = np.random.default_rng(0)
    pd = pd_code(V, rng=rng)
    if not pd:
        return False
    a = alex_abs(pd, ts); b = torus_alex_abs(p, q, ts)
    return np.allclose(a, b, rtol=1e-6, atol=1e-8)

if __name__ == '__main__':
    import sys
    rng = np.random.default_rng(0)
    for p in [3, 4, 5, 6, 7]:
        hits = []
        for trial in range(4000):
            R = 1.0; r = rng.uniform(0.2, 3.0); h = rng.uniform(0.02, 1.5)
            phif = rng.uniform(0.3, 0.7)
            V = torus_poly(p, R, r, h, phif)
            if min_dist(V) < 1e-3:
                continue
            if is_torus(V, p, p+1):
                hits.append((r, h, phif, min_dist(normalize(V))))
        print(p, len(hits))
        if hits:
            hits.sort(key=lambda x: -x[3])
            for hh in hits[:3]:
                print('   r=%.3f h=%.3f phif=%.3f mu=%.4f' % hh)
