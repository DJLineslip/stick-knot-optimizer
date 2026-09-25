"""
equistick.flows
===============

Equalizer #1: path lifting through the edge-length map, with every vertex
move checked for knot-type safety.

Idea.  The map V -> (edge lengths) is a submersion away from collinear
polygons.  So we can ask for a small length change dL = -eta (L - mean(L))
and realise it with the minimal-norm displacement dV = J^T (J J^T)^(-1) dL.
The fibre of the length map has dimension 2n-6 (mod rigid motions), and we
use that freedom to push non-adjacent edges apart: the repulsion gradient
is projected onto ker J (to first order it does not change lengths).

Moves are applied vertex by vertex with `safe_move`, so the knot type is
preserved by construction.

What happened.  On 8_19 = T(3,4) this sometimes certifies, with tiny
clearance.  On T(4,5), started near the symmetric construction, it
produced a clean "collapse signature": mu fell in lockstep with the length
defect (mu ~ 2 x defect over 40 steps).  That looked like an obstruction,
but it was the flow heading for the symmetric equilateral point, which the
symmetry no-go shows is always singular.  The clearance-floor solver
(equistick.nlp) later certified T(4,5).  Kept here because the false alarm
is instructive.

agitate      random knot-type-safe vertex moves (explores the knot's region)
step         one path-lifting step
equalize     the flow, with agitation when stalled
fiber_ascent move inside a length fibre to increase clearance
equalize2    flow with fibre ascent before each length step
"""
import time
import numpy as np

from .geometry import safe_move, min_dist, clearance_grad, normalize, mr_ratio
from .certify import length_jacobian


def agitate(V, rng, amp, k, eps=1e-9):
    """k random Gaussian vertex displacements of scale `amp`, each kept only
    if safe.  Modifies V in place; returns the number accepted."""
    n = len(V)
    acc = 0
    for _ in range(k):
        i = rng.integers(n)
        P2 = V[i] + rng.normal(size=3) * amp
        if safe_move(V, i, P2, eps):
            V[i] = P2
            acc += 1
    return acc


def step(V, rng, eta, kappa, power=2.0, capf=0.3):
    """One path-lifting step (in place).  eta: fraction of the length
    residual to remove; kappa: weight of the in-fibre repulsion; capf: cap
    on vertex displacement as a fraction of mu.  Returns moves accepted."""
    n = len(V)
    L, U, J, JT, M = length_jacobian(V)
    Minv = np.linalg.pinv(M)
    dV = JT(Minv @ (-eta * (L - L.mean())))
    mu = min_dist(V)
    if kappa > 0:
        _, G = clearance_grad(V, power)
        g = -G
        gN = g - JT(Minv @ J(g))            # projection onto ker J
        m = np.linalg.norm(gN, axis=1).max()
        if m > 0:
            dV = dV + kappa * mu * gN / m
    cap = capf * mu
    m = np.linalg.norm(dV, axis=1).max()
    if m > cap:
        dV *= cap / m
    moved = 0
    for i in rng.permutation(n):
        for f in (1.0, 0.5, 0.25):
            P2 = V[i] + f * dV[i]
            if safe_move(V, i, P2, 1e-9):
                V[i] = P2
                moved += 1
                break
    return moved


def equalize(V0, rng, iters=4000, eta=0.2, kappa=0.3, agit=0.05, tlimit=None):
    """Run the flow; return ((best_ratio, V, defect, mu), log)."""
    V = normalize(V0.copy())
    n = len(V)
    t0 = time.time()
    best = (np.inf, None, None, None)
    log = []
    stall, prev = 0, np.inf
    for it in range(iters):
        step(V, rng, eta, kappa)
        V[:] = normalize(V)
        r, defect, mu = mr_ratio(V)
        log.append((it, defect, mu, r))
        if r < best[0]:
            best = (r, V.copy(), defect, mu)
        if r < 1:
            break
        stall = stall + 1 if defect > 0.98 * prev else max(0, stall - 1)
        prev = defect
        if stall > 25:
            agitate(V, rng, agit * mu, 5 * n)
            V[:] = normalize(V)
            stall = 0
        if tlimit and time.time() - t0 > tlimit:
            break
    return best, log


def fiber_ascent(V, rng, nsteps=4, s=0.2, power=2.0):
    """Increase mu while (to first order) keeping every edge length:
    repulsion step projected onto ker J, then Newton correction back onto
    the fibre, accepted only if mu actually increases.  In place."""
    n = len(V)
    for _ in range(nsteps):
        L, U, J, JT, M = length_jacobian(V)
        Minv = np.linalg.pinv(M)
        mu0 = min_dist(V)
        _, G = clearance_grad(V, power)
        g = -G
        gN = g - JT(Minv @ J(g))
        m = np.linalg.norm(gN, axis=1).max()
        if m == 0:
            return
        d = s * mu0 * gN / m
        W = V.copy()
        for i in rng.permutation(n):
            if safe_move(W, i, W[i] + d[i], 1e-9):
                W[i] = W[i] + d[i]
        for _k in range(2):
            L2, U2, J2, JT2, M2 = length_jacobian(W)
            corr = JT2(np.linalg.pinv(M2) @ (L - L2))
            for i in range(n):
                if safe_move(W, i, W[i] + corr[i], 1e-9):
                    W[i] = W[i] + corr[i]
        if min_dist(W) > mu0:
            V[:] = W
            s = min(s * 1.3, 0.5)
        else:
            s *= 0.5


def equalize2(V0, rng, iters=600, eta=0.15, asc=4, tlimit=None):
    """Flow with `asc` fibre-ascent steps before each length step."""
    V = normalize(V0.copy())
    t0 = time.time()
    best = (np.inf, None, None, None)
    log = []
    for it in range(iters):
        fiber_ascent(V, rng, nsteps=asc)
        step(V, rng, eta, 0.0)
        V[:] = normalize(V)
        r, defect, mu = mr_ratio(V)
        log.append((defect, mu, r))
        if r < best[0]:
            best = (r, V.copy(), defect, mu)
        if r < 1:
            break
        if tlimit and time.time() - t0 > tlimit:
            break
    return best, log
