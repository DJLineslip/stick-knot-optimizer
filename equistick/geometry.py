"""
equistick.geometry
==================

Low-level geometry of closed polygons in R^3, compiled with numba.

A polygon is stored as an (n, 3) float array V of vertices; edge i joins
V[i] to V[(i+1) % n].  Two edges are *adjacent* if they share a vertex;
only non-adjacent pairs can intersect in an embedded polygon.

Contents
--------
seg_seg          exact distance between two segments (+ closest-point params)
min_dist         mu(P): minimum distance between non-adjacent edges
pair_dists       all non-adjacent pairwise edge distances (n x n matrix)
clearance_grad   value and gradient of U = sum d_ij^(-power)  (a repulsion)
seg_tri          conservative segment / triangle intersection test
safe_move        can one vertex slide in a straight line without the
                 polygon passing through itself?  (knot type preserved)
deletable        can a vertex be deleted without changing the knot type?
lengths, normalize, mr_ratio, angle_sum, rail_lengths
                 plain numpy helpers

Why "safe moves" preserve knot type
-----------------------------------
Sliding vertex v_i linearly from P to P2 sweeps the two triangles
(v_{i-1}, P, P2) and (P, P2, v_{i+1}).  If no other edge meets these
triangles, the polygon stays embedded throughout the motion, so the
motion is an ambient isotopy and the knot type cannot change.  This is
Reidemeister's classical triangle (Delta) move.  Deleting v_i is the special
case where the vertex collapses onto the segment v_{i-1} v_{i+1}: it is
safe iff the triangle (v_{i-1}, v_i, v_{i+1}) is not pierced.

All tests are deliberately *conservative*: near-degenerate or coplanar
situations are reported as unsafe.  A false "unsafe" only costs a rejected
move; a false "safe" could silently change the knot type.
"""
import math
import numpy as np
from numba import njit


# --------------------------------------------------------------------------
# segment / segment distance
# --------------------------------------------------------------------------
@njit(cache=True)
def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True)
def seg_seg(p1, q1, p2, q2):
    """Distance between segments [p1,q1] and [p2,q2].

    Returns (d, s, t) where the closest points are p1 + s(q1-p1) and
    p2 + t(q2-p2), s, t in [0,1].  Algorithm from Ericson, *Real-Time
    Collision Detection*, sec. 5.1.9.  Assumes both segments have positive
    length (true for any polygon edge we produce).
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    c = d1 @ r
    b = d1 @ d2
    den = a * e - b * b
    s = 0.0
    if den > 1e-300:                 # not parallel
        s = _clamp01((b * f - c * e) / den)
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = _clamp01(-c / a)
    elif t > 1.0:
        t = 1.0
        s = _clamp01((b - c) / a)
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    dv = c1 - c2
    return math.sqrt(dv @ dv), s, t


@njit(cache=True)
def min_dist(V):
    """mu(P): the minimum distance between non-adjacent edges.

    mu > 0 iff the polygon is embedded.  This is the quantity in the
    Millett-Rawdon criterion."""
    n = V.shape[0]
    m = 1e300
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:      # edges n-1 and 0 are adjacent
                continue
            d, s, t = seg_seg(V[i], V[(i + 1) % n], V[j], V[(j + 1) % n])
            if d < m:
                m = d
    return m


@njit(cache=True)
def pair_dists(V):
    """Symmetric n x n matrix of non-adjacent edge distances (1e300 elsewhere)."""
    n = V.shape[0]
    out = np.full((n, n), 1e300)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            d, s, t = seg_seg(V[i], V[(i + 1) % n], V[j], V[(j + 1) % n])
            out[i, j] = d
            out[j, i] = d
    return out


@njit(cache=True)
def clearance_grad(V, power):
    """Repulsive energy U = sum over non-adjacent pairs of d_ij^(-power),
    and its gradient with respect to the vertices.

    The derivative of d_ij moves the closest points apart along the unit
    vector joining them, split between segment endpoints in proportion to
    the closest-point parameters.  Internal forces of this kind have zero
    net force and zero net torque, so -grad U contains no rigid motion.
    """
    n = V.shape[0]
    G = np.zeros_like(V)
    U = 0.0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            i1 = (i + 1) % n
            j1 = (j + 1) % n
            d, s, t = seg_seg(V[i], V[i1], V[j], V[j1])
            c1 = V[i] + (V[i1] - V[i]) * s
            c2 = V[j] + (V[j1] - V[j]) * t
            u = (c1 - c2) / d
            U += d ** (-power)
            fp = -power * d ** (-power - 1.0)
            G[i] += fp * (1 - s) * u
            G[i1] += fp * s * u
            G[j] -= fp * (1 - t) * u
            G[j1] -= fp * t * u
    return U, G


# --------------------------------------------------------------------------
# segment / triangle tests (the heart of knot-type safety)
# --------------------------------------------------------------------------
@njit(cache=True)
def cross3(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


@njit(cache=True)
def seg_tri(p, q, a, b, c, eps):
    """Conservative test: does segment [p,q] meet the closed triangle abc?

    Returns True (= "blocked") for degenerate triangles, for segments lying
    (nearly) in the triangle's plane, and for crossing points within a
    barycentric slack `eps` of the triangle.  Used with eps ~ 1e-9."""
    e1 = b - a
    e2 = c - a
    nrm = cross3(e1, e2)
    nn = math.sqrt(nrm @ nrm)
    if nn < 1e-14:
        return True
    nh = nrm / nn
    dp = (p - a) @ nh
    dq = (q - a) @ nh
    L = math.sqrt((q - p) @ (q - p))
    tol = 1e-12 * (1.0 + L)
    if (dp > tol and dq > tol) or (dp < -tol and dq < -tol):
        return False                       # both endpoints strictly on one side
    if abs(dp - dq) < tol:
        return True                        # (nearly) coplanar: be conservative
    tt = dp / (dp - dq)
    x = p + (q - p) * tt                   # where the segment meets the plane
    v2 = x - a
    d00 = e1 @ e1
    d01 = e1 @ e2
    d11 = e2 @ e2
    d20 = v2 @ e1
    d21 = v2 @ e2
    den = d00 * d11 - d01 * d01
    if den < 1e-300:
        return True
    vv = (d11 * d20 - d01 * d21) / den
    ww = (d00 * d21 - d01 * d20) / den
    uu = 1.0 - vv - ww
    return uu >= -eps and vv >= -eps and ww >= -eps


@njit(cache=True)
def _adjacent_ok(shared, other, a2, a3):
    """An edge from `shared` to `other` touches the triangle (shared, a2, a3)
    at `shared`; it can meet the triangle anywhere else only if it lies in
    the triangle's plane.  Return True if it is safely non-coplanar."""
    e1 = a2 - shared
    e2 = a3 - shared
    nrm = cross3(e1, e2)
    nn = math.sqrt(nrm @ nrm)
    w = other - shared
    wl = math.sqrt(w @ w)
    if nn < 1e-14 or wl < 1e-14:
        return False
    return abs(w @ nrm) / (nn * wl) > 1e-9


@njit(cache=True)
def safe_move(V, i, P2, eps):
    """True if vertex i can slide in a straight line from V[i] to P2 without
    the polygon passing through itself (so the knot type is unchanged).

    Checks every other edge against the two swept triangles
    T1 = (V[i-1], V[i], P2) and T2 = (V[i], P2, V[i+1]).  The two edges that
    share a vertex with one of the triangles are handled by the coplanarity
    test `_adjacent_ok`."""
    n = V.shape[0]
    im = (i - 1) % n
    ip = (i + 1) % n
    A = V[im]
    P = V[i]
    B = V[ip]
    if math.sqrt((P2 - A) @ (P2 - A)) < 1e-9 or math.sqrt((P2 - B) @ (P2 - B)) < 1e-9:
        return False                       # would create a zero-length edge
    for j in range(n):
        if j == im or j == i:              # the two moving edges themselves
            continue
        a = V[j]
        b = V[(j + 1) % n]
        if j == (i - 2) % n:               # edge (V[i-2], A) shares A with T1
            if not _adjacent_ok(A, a, P, P2):
                return False
            if seg_tri(a, b, P, P2, B, eps):
                return False
        elif j == ip:                      # edge (B, V[i+2]) shares B with T2
            if not _adjacent_ok(B, b, P, P2):
                return False
            if seg_tri(a, b, A, P, P2, eps):
                return False
        else:
            if seg_tri(a, b, A, P, P2, eps) or seg_tri(a, b, P, P2, B, eps):
                return False
    return True


@njit(cache=True)
def deletable(V, i, eps):
    """True if vertex i can be removed (the triangle V[i-1], V[i], V[i+1] is
    unpierced), which lowers the stick count by one without changing type."""
    n = V.shape[0]
    im = (i - 1) % n
    ip = (i + 1) % n
    A = V[im]
    P = V[i]
    B = V[ip]
    for j in range(n):
        if j == im or j == i:
            continue
        a = V[j]
        b = V[(j + 1) % n]
        if j == (i - 2) % n:
            if not _adjacent_ok(A, a, P, B):
                return False
        elif j == ip:
            if not _adjacent_ok(B, b, A, P):
                return False
        else:
            if seg_tri(a, b, A, P, B, eps):
                return False
    return True


# --------------------------------------------------------------------------
# plain numpy helpers
# --------------------------------------------------------------------------
def lengths(V):
    """Edge lengths |V[i+1] - V[i]|."""
    return np.linalg.norm(np.roll(V, -1, 0) - V, axis=1)


def normalize(V):
    """Translate to centroid 0 and scale to mean edge length 1.
    (Similarities preserve knot type; the Millett-Rawdon test assumes mean 1.)"""
    V = V - V.mean(0)
    return V / lengths(V).mean()


def mr_ratio(V):
    """Millett-Rawdon ratio.  With mean edge length 1, the theorem guarantees
    an exactly equilateral polygon of the same knot type nearby if

        max_i |L_i - 1|  <  min( mu/n , mu^2/4 ).

    Returns (ratio, defect, mu); the polygon is certified iff ratio < 1."""
    W = normalize(V)
    L = lengths(W)
    n = len(W)
    mu = min_dist(W)
    defect = np.abs(L - 1).max()
    return defect / min(mu / n, mu * mu / 4), defect, mu


def interior_angles(V):
    """Interior angle beta_i at each vertex (pi = straight, 0 = hairpin)."""
    n = len(V)
    out = np.empty(n)
    for i in range(n):
        a = V[i - 1] - V[i]
        c = V[(i + 1) % n] - V[i]
        out[i] = math.acos(np.clip(a @ c / np.linalg.norm(a) / np.linalg.norm(c), -1, 1))
    return out


def angle_sum(V):
    """Sum of interior angles.  Ladder lemma: for a (2b+2)-gon whose knot has
    bridge index b, this sum is at most 2*pi."""
    return interior_angles(V).sum()


def rail_lengths(V):
    """Lengths of the closed 'rails' through the even- and odd-indexed
    vertices (n even).  Ladder lemma: for an equilateral bridge-tight
    polygon these two lengths sum to at most 2*pi."""
    n = len(V)
    assert n % 2 == 0
    ev = V[0::2]
    od = V[1::2]
    le = np.linalg.norm(np.roll(ev, -1, 0) - ev, axis=1).sum()
    lo = np.linalg.norm(np.roll(od, -1, 0) - od, axis=1).sum()
    return le, lo
