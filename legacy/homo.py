import numpy as np, time
from scipy.optimize import minimize
from sk import *
from nlp import dist_jac

def stage_solve(V, target, mu0, maxiter=200):
    n = len(V); x0 = V.ravel()
    def f(x):
        W = x.reshape(n, 3)
        E = np.roll(W, -1, 0) - W; L = np.linalg.norm(E, axis=1)
        U = E / L[:, None]; r = L - target
        g = -2 * r[:, None] * U + np.roll(2 * r[:, None] * U, 1, 0)
        return r @ r, g.ravel()
    cons = {'type': 'ineq', 'fun': lambda x: dist_jac(x.reshape(n, 3))[0] - mu0,
            'jac': lambda x: dist_jac(x.reshape(n, 3))[1]}
    res = minimize(f, x0, jac=True, method='SLSQP', constraints=[cons],
                   options={'maxiter': maxiter, 'ftol': 1e-18})
    return res.x.reshape(n, 3)

def safe_path(V, W, sub=4):
    U = V.copy(); n = len(V)
    D = (W - V) / sub
    for s in range(sub):
        for i in range(n):
            P2 = U[i] + D[i]
            if not safe_move(U, i, P2, 1e-10):
                return False
            U[i] = P2
    return True

def fatten(V, rng, iters=300):
    # safe repulsion ascent (not length preserving) to raise clearance
    V = normalize(V.copy()); n = len(V)
    for it in range(iters):
        mu = min_dist(V)
        _, G = clearance_grad(V, 2.0)
        d = -G; m = np.linalg.norm(d, axis=1).max()
        d = 0.2 * mu * d / m
        # keep edge lengths from collapsing: add a weak pull toward length 1
        L = lengths(V); E = np.roll(V, -1, 0) - V; Uu = E / L[:, None]
        pull = -(L - 1)[:, None] * Uu; pull = pull - np.roll(pull, 1, 0)   # gradient-ish of sum (L-1)^2 /2 negative
        d = d + 0.05 * mu * (-pull) / max(np.abs(pull).max(), 1e-12)
        for i in rng.permutation(n):
            if safe_move(V, i, V[i] + d[i], 1e-10):
                V[i] = V[i] + d[i]
        V = normalize(V)
    return V

def homotopy_equalize(V0, mu_floor=0.5, dt0=0.1, verbose=False, tlimit=120):
    t0 = time.time()
    V = normalize(V0.copy())
    L0 = lengths(V)
    mu_start = min_dist(V)
    mu0 = mu_floor * mu_start
    t = 0.0; dt = dt0
    while t < 1.0:
        if time.time() - t0 > tlimit:
            return V, t, mu0, False
        tn = min(1.0, t + dt)
        target = (1 - tn) * L0 + tn * 1.0
        W = stage_solve(V, target, mu0)
        if min_dist(W) >= 0.999 * mu0 and safe_path(V, W):
            V = W; t = tn; dt = min(dt * 1.5, 0.25)
            if verbose:
                print('  t=%.3f defect %.2e mu %.2e' % (t, np.abs(lengths(V) - target).max(), min_dist(V)), flush=True)
        else:
            dt /= 2
            if dt < 1e-4:
                return V, t, mu0, False
    # final polish toward exact ones with same floor
    for _ in range(3):
        W = stage_solve(V, np.ones(len(V)), mu0, maxiter=400)
        if safe_path(V, W, sub=8):
            V = W
    return V, 1.0, mu0, True
