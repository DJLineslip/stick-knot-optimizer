"""
equistick.optimize
==================

Equalizers #2 and #3.

#2  clearance_floor_solve(V, mu0)
    Nonlinear program:   minimise  sum_i (L_i - 1)^2
                         subject to d_ij(V) >= mu0 for all non-adjacent i, j
    solved with SLSQP.  The floor mu0 keeps the polygon embedded with room
    to spare, which is exactly what the Millett-Rawdon test rewards.
    Scanning mu0 gives the diagnostic from the plan: if the best defect
    only reaches zero as mu0 -> 0, that is the collapse signature; if it
    reaches ~1e-10 at positive mu0, the knot has an equilateral version.

    Caveat: SLSQP iterates can jump, so the knot type is NOT preserved by
    construction.  Results are only trusted after the final polygon's type
    is re-identified.  This worked well from near-equilateral starts (the
    torus knots), but from lopsided starts it jumped straight to an
    equilateral unknot, which is why #3 exists.

#3  homotopy_equalize(V, mu_floor)
    Move target lengths from the current ones to all-ones in stages; solve
    each stage with the floor fixed at mu_floor x (starting clearance); keep
    a stage only if the vertex-by-vertex straight-line transition passes
    `safe_move` (subdivided into substeps).  The knot type is therefore
    preserved by construction.  `fatten` first raises the clearance of
    lopsided polygons with safe repulsion steps.
"""
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize

from .geometry import seg_seg, safe_move, min_dist, clearance_grad, lengths, normalize


@njit(cache=True)
def dist_jac(V):
    """All non-adjacent edge distances and their Jacobian (m x 3n)."""
    n = V.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            pairs.append((i, j))
    m = len(pairs)
    D = np.zeros(m)
    Jm = np.zeros((m, n * 3))
    for k in range(m):
        i, j = pairs[k]
        i1 = (i + 1) % n
        j1 = (j + 1) % n
        d, s, t = seg_seg(V[i], V[i1], V[j], V[j1])
        c1 = V[i] + (V[i1] - V[i]) * s
        c2 = V[j] + (V[j1] - V[j]) * t
        u = (c1 - c2) / max(d, 1e-300)
        D[k] = d
        for a in range(3):
            Jm[k, 3 * i + a] += (1 - s) * u[a]
            Jm[k, 3 * i1 + a] += s * u[a]
            Jm[k, 3 * j + a] -= (1 - t) * u[a]
            Jm[k, 3 * j1 + a] -= t * u[a]
    return D, Jm


def _length_objective(n, target):
    def f(x):
        W = x.reshape(n, 3)
        E = np.roll(W, -1, 0) - W
        L = np.linalg.norm(E, axis=1)
        U = E / L[:, None]
        r = L - target
        g = -2 * r[:, None] * U + np.roll(2 * r[:, None] * U, 1, 0)
        return r @ r, g.ravel()
    return f


def _floor_constraint(n, mu0):
    return {'type': 'ineq',
            'fun': lambda x: dist_jac(x.reshape(n, 3))[0] - mu0,
            'jac': lambda x: dist_jac(x.reshape(n, 3))[1]}


def clearance_floor_solve(V0, mu0, maxiter=300):
    """Equalizer #2 (see module docstring).  Returns the final polygon;
    re-identify its knot type before trusting it."""
    n = len(V0)
    x0 = normalize(V0).ravel()
    res = minimize(_length_objective(n, np.ones(n)), x0, jac=True, method='SLSQP',
                   constraints=[_floor_constraint(n, mu0)],
                   options={'maxiter': maxiter, 'ftol': 1e-16})
    return res.x.reshape(n, 3)


def stage_solve(V, target, mu0, maxiter=200):
    """Minimise sum (L_i - target_i)^2 subject to the clearance floor."""
    n = len(V)
    res = minimize(_length_objective(n, target), V.ravel(), jac=True, method='SLSQP',
                   constraints=[_floor_constraint(n, mu0)],
                   options={'maxiter': maxiter, 'ftol': 1e-18})
    return res.x.reshape(n, 3)


def safe_path(V, W, sub=4):
    """Is the straight-line vertex-by-vertex motion V -> W, in `sub`
    substeps, knot-type safe?  (Conservative.)"""
    U = V.copy()
    n = len(V)
    D = (W - V) / sub
    for _ in range(sub):
        for i in range(n):
            P2 = U[i] + D[i]
            if not safe_move(U, i, P2, 1e-10):
                return False
            U[i] = P2
    return True


def fatten(V, rng, iters=300):
    """Safe repulsion ascent with a weak pull toward unit lengths: raises mu
    of lopsided polygons (e.g. fresh from vertex deletion) before the
    homotopy.  Knot type preserved (every move is a safe move)."""
    V = normalize(V.copy())
    n = len(V)
    for _ in range(iters):
        mu = min_dist(V)
        _, G = clearance_grad(V, 2.0)
        d = -G
        d = 0.2 * mu * d / np.linalg.norm(d, axis=1).max()
        L = lengths(V)
        E = np.roll(V, -1, 0) - V
        Uu = E / L[:, None]
        pull = -(L - 1)[:, None] * Uu
        pull = pull - np.roll(pull, 1, 0)
        d = d + 0.05 * mu * (-pull) / max(np.abs(pull).max(), 1e-12)
        for i in rng.permutation(n):
            if safe_move(V, i, V[i] + d[i], 1e-10):
                V[i] = V[i] + d[i]
        V = normalize(V)
    return V


def homotopy_equalize(V0, mu_floor=0.5, dt0=0.1, tlimit=120):
    """Equalizer #3.  Returns (V, t_reached, mu0_used, completed)."""
    t0 = time.time()
    V = normalize(V0.copy())
    L0 = lengths(V)
    mu0 = mu_floor * min_dist(V)
    t, dt = 0.0, dt0
    while t < 1.0:
        if time.time() - t0 > tlimit:
            return V, t, mu0, False
        tn = min(1.0, t + dt)
        target = (1 - tn) * L0 + tn * 1.0
        W = stage_solve(V, target, mu0)
        if min_dist(W) >= 0.999 * mu0 and safe_path(V, W):
            V, t, dt = W, tn, min(dt * 1.5, 0.25)
        else:
            dt /= 2
            if dt < 1e-4:
                return V, t, mu0, False
    for _ in range(3):                      # final push to exact unit lengths
        W = stage_solve(V, np.ones(len(V)), mu0, maxiter=400)
        if safe_path(V, W, sub=8):
            V = W
    return V, 1.0, mu0, True
