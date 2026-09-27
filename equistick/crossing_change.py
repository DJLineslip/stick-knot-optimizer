"""Check whether sliding one polygon vertex makes exactly one crossing change.

An interior, transverse pierce of one swept triangle by exactly one
nonmoving edge is one double point of the moving polygon. Degeneracy,
additional contacts and nonembedded endpoints are conservatively rejected.
"""
import numpy as np

from .geometry import _adjacent_ok, min_dist, seg_seg, seg_tri


def _strict_pierce(p, q, anchor, start, finish, eps):
    """Return (motion time, edge fraction, barycentric margin), or None."""
    u = start - anchor
    v = finish - anchor
    normal = np.cross(u, v)
    norm = np.linalg.norm(normal)
    if norm < eps:
        return None
    normal /= norm
    da = float(np.dot(p - anchor, normal))
    db = float(np.dot(q - anchor, normal))
    if da * db >= 0 or min(abs(da), abs(db)) <= eps:
        return None
    fraction = da / (da - db)
    point = p + fraction * (q - p)
    w = point - anchor
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(w @ u), float(w @ v)
    den = a * c - b * b
    if den <= eps * a * c:
        return None
    beta = (c * d - b * e) / den
    gamma = (a * e - b * d) / den
    alpha = 1 - beta - gamma
    if min(alpha, beta, gamma) <= eps:
        return None
    time = gamma / (beta + gamma)
    if not eps < time < 1 - eps:
        return None
    return time, fraction, min(alpha, beta, gamma)


def find_single_crossing(vertices, vertex, destination, eps=1e-9):
    """A one-crossing motion certificate, or None (including uncertain)."""
    V = np.asarray(vertices, dtype=np.float64)
    P2 = np.asarray(destination, dtype=np.float64)
    n = len(V)
    if (V.shape != (n, 3) or n < 5 or not 0 <= vertex < n or P2.shape != (3,)
            or not np.isfinite(V).all() or not np.isfinite(P2).all()):
        return None
    scale = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1).mean()
    if scale < eps:
        return None
    W = V.copy()
    W[vertex] = P2
    if min_dist(V) <= eps * scale or min_dist(W) <= eps * scale:
        return None
    i = vertex
    im, ip = (i - 1) % n, (i + 1) % n
    A, P, B = V[im], V[i], V[ip]
    events = []
    for j in range(n):
        if j in (im, i):
            continue
        p, q = V[j], V[(j + 1) % n]
        if j == (i - 2) % n:
            if not _adjacent_ok(A, p, P, P2):
                return None
            triangles = [(1, B, P, P2)]
        elif j == ip:
            if not _adjacent_ok(B, q, P, P2):
                return None
            triangles = [(0, A, P, P2)]
        else:
            triangles = [(0, A, P, P2), (1, B, P, P2)]
        for side, anchor, start, finish in triangles:
            if not seg_tri(p, q, anchor, start, finish, eps):
                continue
            event = _strict_pierce(p, q, anchor, start, finish, eps * scale)
            if event is None:
                return None
            events.append((j, side, *event))
            if len(events) > 1:
                return None
    if len(events) != 1:
        return None
    edge, triangle, time, fraction, margin = events[0]
    return {'vertex': i, 'edge': edge, 'triangle': triangle,
            'time': time, 'edge_fraction': fraction,
            'barycentric_margin': margin, 'endpoint_clearance': float(min_dist(W))}


def propose_moves(vertices, fractions=(.25, .5, .75), weights=(.25, .5), times=(.5,)):
    """Finite, reproducible vertex trajectories through edge-interior points.

    Q is a point on a target edge. The endpoint is solved so Q lies in the
    interior of one swept triangle at the specified motion time. All
    trajectories are suggestions; find_single_crossing must validate them.
    """
    V = np.asarray(vertices, dtype=np.float64)
    n = len(V)
    for i in range(n):
        P = V[i]
        im, ip = (i - 1) % n, (i + 1) % n
        for j in range(n):
            if j in (im, i):
                continue
            if j == (i - 2) % n:
                anchors = ((1, V[ip]),)
            elif j == ip:
                anchors = ((0, V[im]),)
            else:
                anchors = ((0, V[im]), (1, V[ip]))
            for side, anchor in anchors:
                for fraction in fractions:
                    Q = (1 - fraction) * V[j] + fraction * V[(j + 1) % n]
                    for weight in weights:
                        for time in times:
                            beta = (1 - weight) * (1 - time)
                            gamma = (1 - weight) * time
                            destination = (Q - weight * anchor - beta * P) / gamma
                            yield {'vertex': i, 'target_edge': j, 'triangle': side,
                                   'edge_fraction': fraction, 'anchor_weight': weight,
                                   'planned_time': time, 'destination': destination}


def propose_nearest_moves(vertices, offsets=(-.15, 0., .15), times=(.7, .9)):
    """Build near-minimal moves around closest points on incident/target edges.

    The closest-point parameters guide a finite grid; no candidate is
    accepted without the independent single-crossing predicate.
    """
    V = np.asarray(vertices, dtype=np.float64)
    n = len(V)
    for i in range(n):
        P = V[i]
        im, ip = (i - 1) % n, (i + 1) % n
        for j in range(n):
            if j in (im, i):
                continue
            if j == (i - 2) % n:
                anchors = ((1, V[ip]),)
            elif j == ip:
                anchors = ((0, V[im]),)
            else:
                anchors = ((0, V[im]), (1, V[ip]))
            for side, anchor in anchors:
                _, along, across = seg_seg(anchor, P, V[j], V[(j + 1) % n])
                for delta_along in offsets:
                    s = np.clip(along + delta_along, .1, .9)
                    for delta_across in offsets:
                        fraction = np.clip(across + delta_across, .1, .9)
                        Q = (1 - fraction) * V[j] + fraction * V[(j + 1) % n]
                        for time in times:
                            weight = 1 - s
                            beta = s * (1 - time)
                            gamma = s * time
                            destination = (Q - weight * anchor - beta * P) / gamma
                            yield {'vertex': i, 'target_edge': j, 'triangle': side,
                                   'edge_fraction': float(fraction), 'anchor_weight': float(weight),
                                   'planned_time': time, 'destination': destination,
                                   'grid_type': 'nearest'}
