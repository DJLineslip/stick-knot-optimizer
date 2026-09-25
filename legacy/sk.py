import numpy as np
from numba import njit
import math, cmath, warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- geometry
@njit(cache=True)
def _cl(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

@njit(cache=True)
def seg_seg(p1, q1, p2, q2):
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = d1 @ d1; e = d2 @ d2; f = d2 @ r
    c = d1 @ r
    b = d1 @ d2
    den = a * e - b * b
    s = 0.0
    if den > 1e-300:
        s = _cl((b * f - c * e) / den)
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0; s = _cl(-c / a)
    elif t > 1.0:
        t = 1.0; s = _cl((b - c) / a)
    c1 = p1 + d1 * s; c2 = p2 + d2 * t
    dv = c1 - c2
    return math.sqrt(dv @ dv), s, t

@njit(cache=True)
def pair_dists(V):
    n = V.shape[0]
    out = np.full((n, n), 1e300)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            d, s, t = seg_seg(V[i], V[(i + 1) % n], V[j], V[(j + 1) % n])
            out[i, j] = d; out[j, i] = d
    return out

@njit(cache=True)
def min_dist(V):
    n = V.shape[0]
    m = 1e300
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            d, s, t = seg_seg(V[i], V[(i + 1) % n], V[j], V[(j + 1) % n])
            if d < m:
                m = d
    return m

@njit(cache=True)
def clearance_grad(V, power):
    # gradient of U = sum d_ij^(-power) over non-adjacent edge pairs
    n = V.shape[0]
    G = np.zeros_like(V)
    U = 0.0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            i1 = (i + 1) % n; j1 = (j + 1) % n
            d, s, t = seg_seg(V[i], V[i1], V[j], V[j1])
            c1 = V[i] + (V[i1] - V[i]) * s
            c2 = V[j] + (V[j1] - V[j]) * t
            u = (c1 - c2) / d
            U += d ** (-power)
            fp = -power * d ** (-power - 1.0)
            G[i] += fp * (1 - s) * u; G[i1] += fp * s * u
            G[j] -= fp * (1 - t) * u; G[j1] -= fp * t * u
    return U, G

@njit(cache=True)
def _cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

@njit(cache=True)
def seg_tri(p, q, a, b, c, eps):
    # conservative: True if segment pq meets closed triangle abc (with slack eps)
    e1 = b - a; e2 = c - a
    nrm = _cross(e1, e2); nn = math.sqrt(nrm @ nrm)
    if nn < 1e-14:
        return True
    nh = nrm / nn
    dp = (p - a) @ nh; dq = (q - a) @ nh
    L = math.sqrt((q - p) @ (q - p))
    tol = 1e-12 * (1.0 + L)
    if (dp > tol and dq > tol) or (dp < -tol and dq < -tol):
        return False
    if abs(dp - dq) < tol:
        # coplanar (or nearly): conservative test via distances
        return True
    tt = dp / (dp - dq)
    x = p + (q - p) * tt
    # barycentric
    v0 = e1; v1 = e2; v2 = x - a
    d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1; d20 = v2 @ v0; d21 = v2 @ v1
    den = d00 * d11 - d01 * d01
    if den < 1e-300:
        return True
    vv = (d11 * d20 - d01 * d21) / den
    ww = (d00 * d21 - d01 * d20) / den
    uu = 1.0 - vv - ww
    return uu >= -eps and vv >= -eps and ww >= -eps

@njit(cache=True)
def _adj_ok(shared, other, a2, a3):
    # segment from shared vertex toward 'other'; triangle (shared, a2, a3).
    # safe unless coplanar
    e1 = a2 - shared; e2 = a3 - shared
    nrm = _cross(e1, e2); nn = math.sqrt(nrm @ nrm)
    w = other - shared; wl = math.sqrt(w @ w)
    if nn < 1e-14 or wl < 1e-14:
        return False
    return abs(w @ nrm) / (nn * wl) > 1e-9

@njit(cache=True)
def safe_move(V, i, P2, eps):
    # can vertex i move in a straight line to P2 without passing through the polygon?
    n = V.shape[0]
    im = (i - 1) % n; ip = (i + 1) % n
    A = V[im]; P = V[i]; B = V[ip]
    # degenerate new edges
    if math.sqrt((P2 - A) @ (P2 - A)) < 1e-9 or math.sqrt((P2 - B) @ (P2 - B)) < 1e-9:
        return False
    for j in range(n):
        if j == im or j == i:
            continue
        a = V[j]; b = V[(j + 1) % n]
        # triangle T1 = (A, P, P2), T2 = (P, P2, B)
        if j == (i - 2) % n:
            # edge (V[i-2], A) shares A with T1
            if not _adj_ok(A, a, P, P2):
                return False
            if seg_tri(a, b, P, P2, B, eps):
                return False
        elif j == ip:
            # edge (B, V[i+2]) shares B with T2
            if not _adj_ok(B, b, P, P2):
                return False
            if seg_tri(a, b, A, P, P2, eps):
                return False
        else:
            if seg_tri(a, b, A, P, P2, eps) or seg_tri(a, b, P, P2, B, eps):
                return False
    return True

@njit(cache=True)
def deletable(V, i, eps):
    # can vertex i be deleted (triangle (V[i-1],V[i],V[i+1]) unpierced)?
    n = V.shape[0]
    im = (i - 1) % n; ip = (i + 1) % n
    A = V[im]; P = V[i]; B = V[ip]
    for j in range(n):
        if j == im or j == i:
            continue
        a = V[j]; b = V[(j + 1) % n]
        if j == (i - 2) % n:
            if not _adj_ok(A, a, P, B):
                return False
        elif j == ip:
            if not _adj_ok(B, b, A, P):
                return False
        else:
            if seg_tri(a, b, A, P, B, eps):
                return False
    return True

def lengths(V):
    return np.linalg.norm(np.roll(V, -1, 0) - V, axis=1)

def normalize(V):
    V = V - V.mean(0)
    return V / lengths(V).mean()

def mr_ratio(V):
    # Millett-Rawdon: certified if ratio < 1 (average edge length normalised to 1)
    W = normalize(V)
    L = lengths(W); n = len(W)
    mu = min_dist(W)
    return np.abs(L - 1).max() / min(mu / n, mu * mu / 4), np.abs(L - 1).max(), mu

def angle_sum(V):
    n = len(V); S = 0.0
    for i in range(n):
        a = V[i - 1] - V[i]; c = V[(i + 1) % n] - V[i]
        S += math.acos(np.clip(a @ c / np.linalg.norm(a) / np.linalg.norm(c), -1, 1))
    return S

# ---------------------------------------------------------------- diagrams
def pd_code(V, direction=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if direction is None:
        direction = rng.normal(size=3)
    d = direction / np.linalg.norm(direction)
    e1 = np.cross(d, rng.normal(size=3)); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    n = len(V)
    X = np.stack([V @ e1, V @ e2], 1); H = V @ d
    ev = []   # (edge, param, crossing id, is_over)
    cr = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p, r = X[i], X[(i + 1) % n] - X[i]
            q, s = X[j], X[(j + 1) % n] - X[j]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-14:
                continue
            w = q - p
            t = (w[0] * s[1] - w[1] * s[0]) / den
            u = (w[0] * r[1] - w[1] * r[0]) / den
            if 0 < t < 1 and 0 < u < 1:
                hi = H[i] + t * (H[(i + 1) % n] - H[i])
                hj = H[j] + u * (H[(j + 1) % n] - H[j])
                k = len(cr)
                over_i = hi > hj
                cr.append((i, j, over_i, r, s))
                ev.append((i, t, k, over_i))
                ev.append((j, u, k, not over_i))
    if not cr:
        return []
    ev.sort(key=lambda x: (x[0], x[1]))
    m = len(ev)
    pos = {}
    for idx, (e, t, k, ov) in enumerate(ev):
        pos[(k, ov)] = idx
    pd = []
    for k, (i, j, over_i, r, s) in enumerate(cr):
        ui = pos[(k, False)]; oi = pos[(k, True)]
        du = r if not over_i else s
        do = s if not over_i else r
        u_in, u_out = ui, (ui + 1) % m
        o_in, o_out = oi, (oi + 1) % m
        if du[0] * do[1] - du[1] * do[0] > 0:
            pd.append((u_in, o_in, u_out, o_out))
        else:
            pd.append((u_in, o_out, u_out, o_in))
    return pd

def alex_abs(pd, ts):
    # |Alexander polynomial| at points on the unit circle, from a PD code
    c = len(pd)
    if c == 0:
        return np.ones(len(ts))
    m = 2 * c
    # arcs: new arc starts after each under-passage (label u_out)
    starts = sorted(x[2] for x in pd)
    arc_of = {}
    arc = 0
    lab_start = set(starts)
    # find a label that starts an arc
    first = starts[0]
    for k in range(m):
        lab = (first + k) % m
        if lab in lab_start and k > 0:
            arc += 1
        arc_of[lab] = arc
    na = arc + 1
    out = []
    for t in ts:
        M = np.zeros((c, na), dtype=complex)
        for row, (a, b, cc, dd) in enumerate(pd):
            ki = arc_of[b]; ii = arc_of[a]; jj = arc_of[cc]
            pos = ((dd - b) % m == 1)  # over strand goes b -> dd ?
            # sign bookkeeping is irrelevant for |det| up to units except via t vs 1/t;
            M[row, ki] += 1 - t
            if pos:
                M[row, ii] += t; M[row, jj] += -1
            else:
                M[row, ii] += -1; M[row, jj] += t
        out.append(abs(np.linalg.det(M[1:, 1:])))
    return np.array(out)

def torus_alex_abs(p, q, ts):
    ts = np.asarray(ts)
    num = (ts ** (p * q) - 1) * (ts - 1)
    den = (ts ** p - 1) * (ts ** q - 1)
    return np.abs(num / den)

def identify(V, tries=3):
    import snappy, spherogram
    rng = np.random.default_rng(1)
    for _ in range(tries):
        pd = pd_code(V, rng=rng)
        if not pd:
            return 'unknot'
        try:
            L = spherogram.Link(pd)
            L.simplify('global')
            if len(L.crossings) == 0:
                return 'unknot'
            ids = L.exterior().identify()
            if ids:
                return str(ids[-1]).split('(')[0]
        except Exception as ex:
            pass
    return None

def hfk(V, rng=None):
    import snappy, spherogram
    pd = pd_code(V, rng=rng)
    L = spherogram.Link(pd)
    L.simplify('global')
    h = L.knot_floer_homology()
    return {k: v for k, v in h.items() if k != 'ranks'}, len(L.crossings)
