import numpy as np, mpmath as mp
from sk import *
from torus import is_torus
mp.mp.dps = 40

def mp_segdist(p1, q1, p2, q2, N=None):
    # exact-ish segment distance in high precision (Ericson), mp vectors
    d1 = [q1[k]-p1[k] for k in range(3)]; d2 = [q2[k]-p2[k] for k in range(3)]; r = [p1[k]-p2[k] for k in range(3)]
    dot = lambda a,b: sum(a[k]*b[k] for k in range(3))
    a = dot(d1,d1); e = dot(d2,d2); f = dot(d2,r); c = dot(d1,r); b = dot(d1,d2)
    cl = lambda x: mp.mpf(0) if x < 0 else (mp.mpf(1) if x > 1 else x)
    den = a*e - b*b
    s = cl((b*f - c*e)/den) if den > 0 else mp.mpf(0)
    t = (b*s + f)/e
    if t < 0: t = mp.mpf(0); s = cl(-c/a)
    elif t > 1: t = mp.mpf(1); s = cl((b-c)/a)
    diff = [p1[k]+d1[k]*s - p2[k]-d2[k]*t for k in range(3)]
    return mp.sqrt(dot(diff, diff))

def mr_certificate_mp(V):
    n = len(V)
    P = [[mp.mpf(float(x)) for x in row] for row in V]
    L = [mp.sqrt(sum((P[(i+1)%n][k]-P[i][k])**2 for k in range(3))) for i in range(n)]
    Lbar = sum(L)/n
    P = [[x/Lbar for x in row] for row in P]
    L = [l/Lbar for l in L]
    defect = max(abs(l-1) for l in L)
    mu = min(mp_segdist(P[i], P[(i+1)%n], P[j], P[(j+1)%n]) for i in range(n) for j in range(i+2, n) if not (i == 0 and j == n-1))
    bound = min(mu/n, mu*mu/4)
    return defect, mu, bound, defect < bound

def verify_torus(V, p, q, nproj=4):
    rng = np.random.default_rng(123)
    ts = np.exp(1j*np.array([0.37, 0.91, 1.43, 2.07, 2.71]))
    alex = all(np.allclose(alex_abs(pd_code(V, rng=rng), ts), torus_alex_abs(p, q, ts), rtol=1e-6) for _ in range(nproj))
    h, c = hfk(V, rng=rng)
    return alex, h, c
