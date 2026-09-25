"""
equistick.certify
=================

Turning a numerically near-equilateral polygon into a trustworthy claim.

mr_certificate_mp(V)  recompute the Millett-Rawdon test in 40-digit
                      arithmetic from the float coordinates
polish(V)             Newton projection onto exactly equal edge lengths
                      (minimal-norm steps); knot type must be rechecked
verify_torus(V,p,q)   Alexander polynomial in several projections plus a
                      knot Floer homology summary

The Millett-Rawdon theorem (Millett and Rawdon, J. Comput. Phys. 186
(2003), as quoted in Rawdon and Scharein 2002): if a polygon with average
edge length 1 satisfies max_i |L_i - 1| < min(mu/n, mu^2/4), where mu is the
minimum distance between non-adjacent edges, then an exactly equilateral
polygon of the same knot type exists nearby.  Our certificates are
floating-point coordinates checked in high precision; a fully rigorous
version would use interval arithmetic throughout.
"""
import numpy as np
import mpmath as mp

from .geometry import lengths, normalize
from .invariants import is_torus, hfk

mp.mp.dps = 40


def _mp_segdist(p1, q1, p2, q2):
    d1 = [q1[k] - p1[k] for k in range(3)]
    d2 = [q2[k] - p2[k] for k in range(3)]
    r = [p1[k] - p2[k] for k in range(3)]
    dot = lambda a, b: sum(a[k] * b[k] for k in range(3))
    a, e, f, c, b = dot(d1, d1), dot(d2, d2), dot(d2, r), dot(d1, r), dot(d1, d2)
    cl = lambda x: mp.mpf(0) if x < 0 else (mp.mpf(1) if x > 1 else x)
    den = a * e - b * b
    s = cl((b * f - c * e) / den) if den > 0 else mp.mpf(0)
    t = (b * s + f) / e
    if t < 0:
        t = mp.mpf(0)
        s = cl(-c / a)
    elif t > 1:
        t = mp.mpf(1)
        s = cl((b - c) / a)
    diff = [p1[k] + d1[k] * s - p2[k] - d2[k] * t for k in range(3)]
    return mp.sqrt(dot(diff, diff))


def mr_certificate_mp(V):
    """High-precision Millett-Rawdon test.

    Returns (defect, mu, bound, certified) as mpf numbers / bool, after
    rescaling to mean edge length exactly 1."""
    n = len(V)
    P = [[mp.mpf(float(x)) for x in row] for row in V]
    L = [mp.sqrt(sum((P[(i + 1) % n][k] - P[i][k]) ** 2 for k in range(3))) for i in range(n)]
    Lbar = sum(L) / n
    P = [[x / Lbar for x in row] for row in P]
    L = [l / Lbar for l in L]
    defect = max(abs(l - 1) for l in L)
    mu = min(_mp_segdist(P[i], P[(i + 1) % n], P[j], P[(j + 1) % n])
             for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1))
    bound = min(mu / n, mu * mu / 4)
    return defect, mu, bound, defect < bound


def length_jacobian(V):
    """Operators for the edge-length map L(V).

    Returns (L, U, J, JT, M): lengths, unit edge vectors, J (dV -> dL),
    its transpose JT (dL -> dV), and the Gram matrix M = J J^T, which is
    cyclic tridiagonal with 2 on the diagonal and -U_i.U_{i+1} beside it.
    The length map is a submersion (M invertible) unless all edges are
    parallel."""
    n = len(V)
    E = np.roll(V, -1, 0) - V
    L = np.linalg.norm(E, axis=1)
    U = E / L[:, None]

    def J(dV):
        return np.einsum('ij,ij->i', U, np.roll(dV, -1, 0) - dV)

    def JT(x):
        G = -x[:, None] * U
        G += np.roll(x[:, None] * U, 1, 0)
        return G

    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = 2.0
        c = -U[i] @ U[(i + 1) % n]
        M[i, (i + 1) % n] += c
        M[(i + 1) % n, i] += c
    return L, U, J, JT, M


def polish(V, iters=30):
    """Newton projection to exactly equal lengths: repeatedly apply the
    minimal-norm vertex displacement that fixes the length residual.
    Converges quadratically from a certified polygon; the moves are not
    checked for safety, so re-verify the knot type afterwards."""
    V = normalize(V)
    for _ in range(iters):
        L, U, J, JT, M = length_jacobian(V)
        r = L - 1
        if np.abs(r).max() < 1e-15:
            break
        V = V + JT(np.linalg.solve(M, -r))
    return V


def verify_torus(V, p, q, nproj=4):
    """(alexander_ok, hfk_summary, crossings_after_simplification)."""
    alex = is_torus(V, p, q, nproj=nproj, seed=123)
    h, c = hfk(V)
    return alex, h, c
