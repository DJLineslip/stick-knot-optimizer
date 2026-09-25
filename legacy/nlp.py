import numpy as np, math
from scipy.optimize import minimize
from sk import *
from numba import njit

@njit(cache=True)
def dist_jac(V):
    n = V.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            pairs.append((i, j))
    m = len(pairs)
    D = np.zeros(m); Jm = np.zeros((m, n * 3))
    for k in range(m):
        i, j = pairs[k]
        i1 = (i + 1) % n; j1 = (j + 1) % n
        d, s, t = seg_seg(V[i], V[i1], V[j], V[j1])
        c1 = V[i] + (V[i1] - V[i]) * s; c2 = V[j] + (V[j1] - V[j]) * t
        u = (c1 - c2) / max(d, 1e-300)
        D[k] = d
        for a in range(3):
            Jm[k, 3*i+a] += (1-s)*u[a]; Jm[k, 3*i1+a] += s*u[a]
            Jm[k, 3*j+a] -= (1-t)*u[a]; Jm[k, 3*j1+a] -= t*u[a]
    return D, Jm

def solve(V0, mu0, maxiter=300):
    n = len(V0)
    x0 = normalize(V0).ravel()
    def f(x):
        V = x.reshape(n, 3)
        E = np.roll(V, -1, 0) - V; L = np.linalg.norm(E, axis=1)
        U = E / L[:, None]; r = L - 1
        g = -2 * r[:, None] * U + np.roll(2 * r[:, None] * U, 1, 0)
        return (r @ r), g.ravel()
    def cons(x):
        D, Jm = dist_jac(x.reshape(n, 3)); return D - mu0
    def cjac(x):
        D, Jm = dist_jac(x.reshape(n, 3)); return Jm
    res = minimize(f, x0, jac=True, method='SLSQP',
                   constraints=[{'type': 'ineq', 'fun': cons, 'jac': cjac}],
                   options={'maxiter': maxiter, 'ftol': 1e-16})
    V = res.x.reshape(n, 3)
    return V

def polish(V, iters=30):
    # Newton projection to exact equal lengths (minimal norm), no safety needed: type rechecked after
    V = normalize(V)
    from eq import jac_ops
    for _ in range(iters):
        L, U, J, JT, M = jac_ops(V)
        r = L - 1
        if np.abs(r).max() < 1e-15:
            break
        V = V + JT(np.linalg.solve(M, -r))
    return V
