import numpy as np, math, time, sys
from numba import njit
from sk import *
from sk import _cross

@njit(cache=True)
def tri_pierce_weight(p, q, a, b, c):
    # if segment pq crosses triangle (a=apex? no: a=V[i-1], b=V[i] apex, c=V[i+1]) return barycentric weight of apex b
    e1 = b - a; e2 = c - a
    nrm = _cross(e1, e2); nn = math.sqrt(nrm @ nrm)
    if nn < 1e-14:
        return 0.0
    nh = nrm / nn
    dp = (p - a) @ nh; dq = (q - a) @ nh
    if (dp > 0 and dq > 0) or (dp < 0 and dq < 0) or dp == dq:
        return 0.0
    t = dp / (dp - dq)
    x = p + (q - p) * t
    v0 = e1; v1 = e2; v2 = x - a
    d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1; d20 = v2 @ v0; d21 = v2 @ v1
    den = d00 * d11 - d01 * d01
    vv = (d11 * d20 - d01 * d21) / den   # weight of b (apex)
    ww = (d00 * d21 - d01 * d20) / den   # weight of c
    uu = 1.0 - vv - ww
    if uu >= 0 and vv >= 0 and ww >= 0:
        return vv + 1e-3
    return 0.0

@njit(cache=True)
def penalties(V):
    n = V.shape[0]
    P = np.zeros(n)
    for i in range(n):
        im = (i - 1) % n; ip = (i + 1) % n
        s = 0.0
        for j in range(n):
            if j == im or j == i or j == (i - 2) % n or j == ip:
                continue
            s += tri_pierce_weight(V[j], V[(j + 1) % n], V[im], V[i], V[ip])
        P[i] = s
    return P

def reduce_once(V, rng, steps=200000, T0=0.05, verbose=False):
    V = normalize(V.copy()); n = len(V)
    P = penalties(V); E = P.min()
    sig = 0.05; acc = 0; tried = 0
    for k in range(steps):
        T = T0 * (1 - k / steps) + 1e-4
        # try deletions
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
        old = V[i].copy(); V[i] = P2
        Pn = penalties(V); En = Pn.min()
        if En <= E or rng.random() < math.exp(-(En - E) / T):
            P, E = Pn, En; acc += 1
            if acc % 50 == 0:
                sig = min(sig * 1.1, 0.3)
        else:
            V[i] = old
        if k % 2000 == 0:
            V[:] = normalize(V)
            if verbose:
                print(k, 'E %.3f sig %.3g mu %.3g' % (E, sig, min_dist(V)), flush=True)
    return None, steps

if __name__ == '__main__':
    from load import load
    name = sys.argv[1]; budget = float(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(seed)
    V0 = load(name)
    t0 = time.time(); tries = 0
    while time.time() - t0 < budget:
        tries += 1
        W, k = reduce_once(V0, rng, steps=40000)
        if W is not None:
            idn = identify(W)
            print(name, 'reduced to', len(W), 'sticks after', k, 'steps; id =', idn, '(%.0fs, try %d)' % (time.time() - t0, tries), flush=True)
            if idn == name:
                np.save('%s_%d.npy' % (name, len(W)), W)
                break
    else:
        print(name, 'no reduction found in %.0fs (%d tries)' % (time.time() - t0, tries))
