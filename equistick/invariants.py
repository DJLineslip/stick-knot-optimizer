"""
equistick.invariants
====================

Turning a polygon into a knot diagram and identifying its knot type.

pd_code(V)              project along a random direction and build a
                        planar-diagram (PD) code in the KnotTheory /
                        spherogram convention
alexander_abs(pd, ts)   |Alexander polynomial| at points t on the unit
                        circle, from the Wirtinger / Fox-calculus matrix
torus_alexander_abs     the same for the torus knot T(p,q), from the
                        closed formula (t^pq - 1)(t - 1)/((t^p - 1)(t^q - 1))
identify(V)             SnapPy census identification of the exterior
                        (hyperbolic knots only; e.g. 'K11n71')
hfk(V)                  knot Floer homology summary (genus, fibredness,
                        L-space property, tau, total rank) via SnapPy's
                        HFK calculator

Why |Delta| on the unit circle is enough
----------------------------------------
The Alexander polynomial is defined only up to units +-t^k, and a knot and
its mirror have Delta(t) and Delta(1/t).  On |t| = 1 all of these have the
same absolute value, and two symmetric Laurent polynomials with equal
|values| on the circle agree up to sign.  Comparing at a handful of generic
points therefore compares Alexander polynomials, ignoring chirality.

Conventions for pd_code
-----------------------
Arcs are labelled 0..2c-1 in order along the polygon, a new label starting
after every crossing passage.  Each crossing is X[a, b, c, d] with a the
incoming under-strand and labels read counterclockwise as seen by the
viewer; c is the outgoing under-strand.  If the convention were globally
reversed we would get the mirror image, which none of our identifications
care about.
"""
import numpy as np


def pd_code(V, direction=None, rng=None):
    """PD code of the projection of polygon V along `direction`
    (random if None).  Returns a list of 4-tuples, [] for a crossingless
    projection."""
    rng = np.random.default_rng() if rng is None else rng
    if direction is None:
        direction = rng.normal(size=3)
    d = direction / np.linalg.norm(direction)
    e1 = np.cross(d, rng.normal(size=3))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)                     # (e1, e2, d) right-handed
    n = len(V)
    X = np.stack([V @ e1, V @ e2], 1)        # projected coordinates
    H = V @ d                                # height toward the viewer
    events = []                              # (edge, parameter, crossing id, is_over)
    crossings = []
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
                k = len(crossings)
                over_i = hi > hj
                crossings.append((over_i, r, s))
                events.append((i, t, k, over_i))
                events.append((j, u, k, not over_i))
    if not crossings:
        return []
    events.sort(key=lambda x: (x[0], x[1]))
    m = len(events)
    pos = {}
    for idx, (e, t, k, ov) in enumerate(events):
        pos[(k, ov)] = idx                   # label entering the passage
    pd = []
    for k, (over_i, r, s) in enumerate(crossings):
        ui = pos[(k, False)]
        oi = pos[(k, True)]
        du = r if not over_i else s          # under-strand direction
        do = s if not over_i else r          # over-strand direction
        u_in, u_out = ui, (ui + 1) % m
        o_in, o_out = oi, (oi + 1) % m
        if du[0] * do[1] - du[1] * do[0] > 0:
            pd.append((u_in, o_in, u_out, o_out))
        else:
            pd.append((u_in, o_out, u_out, o_in))
    return pd


def alexander_abs(pd, ts):
    """|Delta(t)| for each t in ts (complex, on the unit circle).

    Builds the c x c Alexander matrix: one row per crossing, one column per
    over-arc (arcs end at under-passages).  Row entries are (1-t) at the
    over-arc and (t, -1) or (-1, t) at the incoming/outgoing under-arcs
    according to the crossing sign (Fox calculus on the Wirtinger
    relations).  Deleting a row and a column gives Delta up to units."""
    c = len(pd)
    if c == 0:
        return np.ones(len(ts))
    m = 2 * c
    starts = set(x[2] for x in pd)           # labels that begin an over-arc
    first = min(starts)
    arc_of, arc = {}, 0
    for k in range(m):
        lab = (first + k) % m
        if lab in starts and k > 0:
            arc += 1
        arc_of[lab] = arc
    na = arc + 1
    out = []
    for t in ts:
        M = np.zeros((c, na), dtype=complex)
        for row, (a, b, cc, dd) in enumerate(pd):
            over, u_in, u_out = arc_of[b], arc_of[a], arc_of[cc]
            b_is_incoming_over = ((dd - b) % m == 1)
            M[row, over] += 1 - t
            if b_is_incoming_over:
                M[row, u_in] += t
                M[row, u_out] += -1
            else:
                M[row, u_in] += -1
                M[row, u_out] += t
        out.append(abs(np.linalg.det(M[1:, 1:])))
    return np.array(out)


def torus_alexander_abs(p, q, ts):
    """|Delta_{T(p,q)}(t)| from the closed formula."""
    ts = np.asarray(ts)
    return np.abs((ts ** (p * q) - 1) * (ts - 1) / ((ts ** p - 1) * (ts ** q - 1)))


DEFAULT_TS = np.exp(1j * np.array([0.37, 0.91, 1.43, 2.07, 2.71]))


def is_torus(V, p, q, nproj=1, seed=0, ts=DEFAULT_TS):
    """True if |Delta| of V matches T(p,q) in `nproj` random projections."""
    rng = np.random.default_rng(seed)
    for _ in range(nproj):
        pd = pd_code(V, rng=rng)
        if not pd:
            return False
        if not np.allclose(alexander_abs(pd, ts), torus_alexander_abs(p, q, ts),
                           rtol=1e-6, atol=1e-8):
            return False
    return True


def _link(V, rng):
    import snappy  # noqa: F401  (spherogram.Link.exterior needs snappy loaded)
    import spherogram
    pd = pd_code(V, rng=rng)
    if not pd:
        return None
    L = spherogram.Link(pd)
    L.simplify('global')
    return L


def matches_census_name(identified, expected):
    """Check a SnapPy identifier against every census alias of a table knot."""
    if identified == expected:
        return True
    if identified is None:
        return False
    import snappy
    try:
        aliases = {str(m).split('(')[0] for m in snappy.Manifold(expected).identify()}
    except Exception:
        return False
    return identified in aliases


def identify_pd(pd):
    """Identify a 0-based PD code up to mirror image, where SnapPy can.

    Returns None for nonhyperbolic complements, which need another check.
    Prefer the Rolfsen name when SnapPy supplies one for a small knot.
    """
    if not pd:
        return 'unknot'
    import re
    import snappy  # noqa: F401  (spherogram.Link.exterior needs snappy loaded)
    import spherogram

    link = spherogram.Link(pd)
    link.simplify('global')
    if not link.crossings:
        return 'unknot'
    names = [str(m).split('(')[0] for m in link.exterior().identify()]
    if not names:
        return None
    return next((name for name in names if re.fullmatch(r'\d+_\d+', name)),
                names[-1])


def identify(V, tries=3, seed=1):
    """SnapPy census name of the knot of polygon V ('unknot' if trivial,
    None if SnapPy cannot identify it, e.g. for torus knots).

    Uses the hyperbolic structure of the exterior, so it identifies knots up
    to isometry of complements, ignoring chirality.  The HTW-table name
    (like 'K13n592') is returned when available, else another census name."""
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        try:
            L = _link(V, rng)
            if L is None or len(L.crossings) == 0:
                return 'unknot'
            names = [str(m).split('(')[0] for m in L.exterior().identify()]
            if names:
                import re
                htw = [x for x in names if re.fullmatch(r'K\d+[an]\d+', x)]
                return htw[0] if htw else names[-1]
        except Exception:
            pass
    return None


def hfk(V, seed=123):
    """Knot Floer homology summary of polygon V, plus the crossing number of
    the simplified diagram.  For torus knots T(p,q): L-space knot, fibred,
    genus (p-1)(q-1)/2, |tau| = genus, total rank = number of nonzero
    Alexander coefficients."""
    rng = np.random.default_rng(seed)
    L = _link(V, rng)
    h = L.knot_floer_homology()
    return {k: v for k, v in h.items() if k != 'ranks'}, len(L.crossings)
