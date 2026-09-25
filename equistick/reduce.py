"""
equistick.reduce
================

Lowering the stick count by one, keeping the knot type.

A vertex v_i can be deleted exactly when the triangle (v_{i-1}, v_i,
v_{i+1}) is not pierced by any other edge (geometry.deletable).  Starting
from an (n+1)-stick polygon we anneal with random knot-type-safe vertex
moves on the energy

    E(V) = min_i P_i,   P_i = sum over edges piercing triangle i of
                              (barycentric weight of the apex v_i at the
                               piercing point) + 1e-3

P_i measures how deep inside triangle i the piercing edges sit; an edge can
only leave through the base v_{i-1} v_{i+1} (the other two sides are polygon
edges), and the apex weight goes to 0 exactly there.  Every 50 steps any
vertex with P_i = 0 is tested with `deletable` and removed if possible.

This reproduced the Cantarella-Rechnitzer-Schumacher-Shonkwiler 10-stick
realisations of most of their nineteen 4-bridge knots from the 11-stick
equilateral data in Eddy's repository, typically within seconds.
"""
import math
import numpy as np
from numba import njit

from .geometry import cross3, safe_move, deletable, min_dist, normalize


@njit(cache=True)
def tri_pierce_weight(p, q, a, b, c):
    """If segment [p,q] crosses triangle (a, b, c) (b = apex), return the
    apex barycentric weight of the crossing point (+1e-3), else 0."""
    e1 = b - a
    e2 = c - a
    nrm = cross3(e1, e2)
    nn = math.sqrt(nrm @ nrm)
    if nn < 1e-14:
        return 0.0
    nh = nrm / nn
    dp = (p - a) @ nh
    dq = (q - a) @ nh
    if (dp > 0 and dq > 0) or (dp < 0 and dq < 0) or dp == dq:
        return 0.0
    t = dp / (dp - dq)
    x = p + (q - p) * t
    v2 = x - a
    d00 = e1 @ e1
    d01 = e1 @ e2
    d11 = e2 @ e2
    d20 = v2 @ e1
    d21 = v2 @ e2
    den = d00 * d11 - d01 * d01
    vv = (d11 * d20 - d01 * d21) / den      # apex weight
    ww = (d00 * d21 - d01 * d20) / den
    uu = 1.0 - vv - ww
    if uu >= 0 and vv >= 0 and ww >= 0:
        return vv + 1e-3
    return 0.0


@njit(cache=True)
def penalties(V):
    """P_i for every vertex (0 means: probably deletable)."""
    n = V.shape[0]
    P = np.zeros(n)
    for i in range(n):
        im = (i - 1) % n
        ip = (i + 1) % n
        s = 0.0
        for j in range(n):
            if j == im or j == i or j == (i - 2) % n or j == ip:
                continue
            s += tri_pierce_weight(V[j], V[(j + 1) % n], V[im], V[i], V[ip])
        P[i] = s
    return P


def reduce_once(V, rng, steps=40000, T0=0.05):
    """Anneal until one vertex can be deleted.  Returns (W, steps_used) with
    W the (n-1)-gon, or (None, steps) on failure."""
    V = normalize(V.copy())
    n = len(V)
    P = penalties(V)
    E = P.min()
    sig, acc, tried = 0.05, 0, 0
    for k in range(steps):
        T = T0 * (1 - k / steps) + 1e-4
        if k % 50 == 0:
            for i in np.argsort(P):
                if P[i] > 0:
                    break
                if deletable(V, i, 1e-9):
                    W = np.delete(V, i, 0)
                    if min_dist(W) > 1e-6:
                        return normalize(W), k
        i = rng.integers(n)
        P2 = V[i] + rng.normal(size=3) * sig
        tried += 1
        if not safe_move(V, i, P2, 1e-9):
            if tried % 200 == 0:
                sig = max(sig * 0.9, 1e-4)
            continue
        old = V[i].copy()
        V[i] = P2
        Pn = penalties(V)
        En = Pn.min()
        if En <= E or rng.random() < math.exp(-(En - E) / T):
            P, E = Pn, En
            acc += 1
            if acc % 50 == 0:
                sig = min(sig * 1.1, 0.3)
        else:
            V[i] = old
        if k % 2000 == 0:
            V[:] = normalize(V)
    return None, steps


def reduce_to(V, target, rng, steps=40000):
    """Repeat reduce_once until `target` sticks; None on failure."""
    while len(V) > target:
        W, _ = reduce_once(V, rng, steps=steps)
        if W is None:
            return None
        V = W
    return V
