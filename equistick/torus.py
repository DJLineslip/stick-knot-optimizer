"""
equistick.torus
===============

Minimal-stick polygons for the torus knots T(p, p+1), and the symmetric
family behind the classical constructions.

T(p, p+1) has bridge index p and stick number exactly 2p+2 (Jin 1997;
Adams, Brennan, Greilsheimer and Woo 1997), so it is "bridge-tight":
s = 2b + 2, the smallest value the superbridge inequality b < sb <= s/2
allows.  These were the flagship candidates in the ladder-lemma plan.

torus_poly(p, R, r, h, phif)
    2(p+1) vertices alternating between an outer ring (radius R, height +h)
    and an inner ring (radius r, height -h).  Every two steps the polygon
    rotates by alpha = 2*pi*p/(p+1), so it has (p+1)-fold symmetry and winds
    p times around the axis.  The odd vertices are offset by phi = phif*alpha.
    For a large open set of parameters this is T(p, p+1).

symmetric_scan(p, trials)
    Random parameter search; returns parameter sets giving T(p, p+1).

star_poly, symmetric_equilateral_check(m, trials)
    The symmetry no-go experiment.  In this family, equal edge lengths
    force phi = alpha/2 (mod pi) (odd vertex in the vertical plane bisecting
    its neighbours).  The reflection in that plane then swaps each odd edge
    with an even one at the same heights, so every even/odd crossing seen
    down the axis is a real intersection.  This function samples phi =
    alpha/2 polygons and reports how many are embedded and knotted
    (expected: none knotted).
"""
import numpy as np

from .geometry import min_dist, normalize, lengths
from .invariants import is_torus, pd_code, alexander_abs, DEFAULT_TS


def torus_poly(p, R, r, h, phif, h2=None):
    q = p + 1
    a = 2 * np.pi * p / q
    phi = phif * a
    h2 = -h if h2 is None else h2
    V = []
    for j in range(q):
        V.append([R * np.cos(j * a), R * np.sin(j * a), h])
        V.append([r * np.cos(j * a + phi), r * np.sin(j * a + phi), h2])
    return np.array(V)


# parameter sets found by symmetric_scan (seed 0) and used for all runs
STARTS = {
    3: (1.0, 0.910, 0.858, 0.430),
    4: (1.0, 1.055, 0.761, 0.550),
    5: (1.0, 1.286, 1.040, 0.457),
    6: (1.0, 1.197, 0.886, 0.535),
    7: (1.0, 0.897, 0.898, 0.470),
}


def symmetric_scan(p, trials=4000, seed=0):
    """Return [(r, h, phif, mu)] giving T(p, p+1), fattest first (R = 1)."""
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(trials):
        r = rng.uniform(0.2, 3.0)
        h = rng.uniform(0.02, 1.5)
        phif = rng.uniform(0.3, 0.7)
        V = torus_poly(p, 1.0, r, h, phif)
        if min_dist(V) < 1e-3:
            continue
        if is_torus(V, p, p + 1):
            hits.append((r, h, phif, min_dist(normalize(V))))
    hits.sort(key=lambda x: -x[3])
    return hits


def star_poly(m, k, r, H):
    """The equilateral symmetric family of the early scan (legacy/sym.py):
    2m vertices, vertex j at angle j*pi*k/m, even ones at radius 1 and
    height +H, odd ones at radius r (sign allowed) and height -H.  All edges
    have the same length by construction."""
    j = np.arange(2 * m)
    ang = j * np.pi * k / m
    rho = np.where(j % 2 == 0, 1.0, r)
    z = np.where(j % 2 == 0, H, -H)
    return np.stack([rho * np.cos(ang), rho * np.sin(ang), z], 1)


def symmetric_equilateral_check(m, trials=1500, seed=0):
    """Sample star_poly(m, k, r, H) over all k and random (r, H).

    Returns {k: (embedded, knotted)}; knotted means a nontrivial Alexander
    polynomial.  The no-go lemma predicts knotted == 0 throughout."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(1, 2 * m):
        emb = knot = 0
        for _ in range(trials):
            r = rng.uniform(-3.0, 3.0)
            H = rng.uniform(0.02, 2.0)
            V = star_poly(m, k, r, H)
            L = lengths(V)
            if min_dist(V) < 1e-7 * L.mean():
                continue
            emb += 1
            pd = pd_code(V, rng=rng)
            if pd and not np.allclose(alexander_abs(pd, DEFAULT_TS), 1.0, atol=1e-8):
                knot += 1
        out[k] = (emb, knot)
    return out
