import numpy as np, math, time
from sk import *

def jac_ops(V):
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

def agitate(V, rng, amp, k, eps=1e-9):
    n = len(V)
    acc = 0
    for _ in range(k):
        i = rng.integers(n)
        P2 = V[i] + rng.normal(size=3) * amp
        if safe_move(V, i, P2, eps):
            V[i] = P2; acc += 1
    return acc

def step(V, rng, eta, kappa, power=2.0, capf=0.3):
    n = len(V)
    L, U, J, JT, M = jac_ops(V)
    Minv = np.linalg.pinv(M)
    dl = -eta * (L - L.mean())
    dV = JT(Minv @ dl)
    mu = min_dist(V)
    if kappa > 0:
        _, G = clearance_grad(V, power)
        g = -G
        gN = g - JT(Minv @ J(g))
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
                V[i] = P2; moved += 1
                break
    return moved

def equalize(V0, rng, iters=4000, eta=0.2, kappa=0.3, agit=0.05, verbose=False, tlimit=None):
    V = normalize(V0.copy())
    n = len(V)
    t0 = time.time()
    best = (np.inf, None, None, None)
    log = []
    stall = 0
    prev = np.inf
    for it in range(iters):
        moved = step(V, rng, eta, kappa)
        V[:] = normalize(V)
        r, defect, mu = mr_ratio(V)
        log.append((it, defect, mu, r))
        if r < best[0]:
            best = (r, V.copy(), defect, mu)
        if r < 1:
            break
        if defect > 0.98 * prev:
            stall += 1
        else:
            stall = max(0, stall - 1)
        prev = defect
        if stall > 25:
            agitate(V, rng, agit * mu, 5 * n)
            V[:] = normalize(V)
            stall = 0
        if verbose and it % 200 == 0:
            print(it, 'defect %.2e mu %.2e MR %.2e' % (defect, mu, r), flush=True)
        if tlimit and time.time() - t0 > tlimit:
            break
    return best, log

def fiber_ascent(V, rng, nsteps=4, s=0.2, power=2.0):
    # move inside the current length fiber to increase clearance
    n = len(V)
    for _ in range(nsteps):
        L, U, J, JT, M = jac_ops(V)
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
            P2 = W[i] + d[i]
            if safe_move(W, i, P2, 1e-9):
                W[i] = P2
        # restore lengths to first order (project back onto the fiber)
        for _k in range(2):
            L2, U2, J2, JT2, M2 = jac_ops(W)
            corr = JT2(np.linalg.pinv(M2) @ (L - L2))
            W2 = W.copy()
            okc = True
            for i in range(n):
                if safe_move(W2, i, W2[i] + corr[i], 1e-9):
                    W2[i] = W2[i] + corr[i]
                else:
                    okc = False
            W = W2
        if min_dist(W) > mu0:
            V[:] = W
            s = min(s * 1.3, 0.5)
        else:
            s *= 0.5

def equalize2(V0, rng, iters=600, eta=0.15, asc=4, verbose=False, tlimit=None):
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
        if verbose and it % 25 == 0:
            print(it, 'defect %.3e mu %.3e MR %.2e' % (defect, mu, r), flush=True)
        if tlimit and time.time() - t0 > tlimit:
            break
    return best, log
