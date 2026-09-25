"""Interval geometric Millett-Rawdon checker for exact decimal vertex data.

All segment minima are solved as convex quadratic problems over the unit
square using exact rational arithmetic. Square roots, mean lengths, defects
and the theorem bound use outward-rounded mpmath intervals. No optimizer or
floating-point branch decides a certificate. This does not identify knot type.
"""
from __future__ import annotations

import hashlib
import re
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import mpmath
from mpmath.ctx_iv import MPIntervalContext
from mpmath.libmp import to_rational


DECIMAL = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _clamp(x):
    return max(Fraction(0), min(Fraction(1), x))


def squared_segment_distance(p, q, x, y):
    """Exact minimum of ||p+s(q-p)-x-t(y-x)||^2 for 0<=s,t<=1.

    A positive semidefinite quadratic reaches its minimum at the interior
    stationary point or on a boundary. Each of the four boundary restrictions
    is a one-dimensional convex quadratic, minimized by a clamped projection.
    The parallel and point-segment cases need only those boundaries.
    """
    u, v, r = _sub(q, p), _sub(y, x), _sub(p, x)
    a, e, b = _dot(u, u), _dot(v, v), _dot(u, v)
    c, f = _dot(u, r), _dot(v, r)
    base = _dot(r, r)

    def value(s, t):
        return base + a*s*s + e*t*t + 2*c*s - 2*f*t - 2*b*s*t

    candidates = []
    for s in (Fraction(0), Fraction(1)):
        t = _clamp((b*s + f)/e) if e else Fraction(0)
        candidates.append(value(s, t))
    for t in (Fraction(0), Fraction(1)):
        s = _clamp((b*t - c)/a) if a else Fraction(0)
        candidates.append(value(s, t))
    determinant = a*e - b*b
    if determinant > 0:
        s = (b*f - c*e)/determinant
        t = (a*f - b*c)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(value(s, t))
    answer = min(candidates)
    assert answer >= 0  # Exact arithmetic, not a numerical tolerance.
    return answer


def _endpoint(x, side):
    numerator, denominator = to_rational(x._mpi_[side])
    return Fraction(numerator, denominator)


def _interval_fraction(ctx, value):
    # Integers may exceed working precision; both conversions and division
    # enclose the exact rational, including for negative decimal coordinates.
    return ctx.mpf(str(value.numerator)) / ctx.mpf(str(value.denominator))


def _parse_vertices(raw):
    rows = []
    for line in raw.decode('ascii').splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3 or any(len(field) > 128 or not DECIMAL.fullmatch(field) for field in fields):
            raise ValueError('expected three finite decimal coordinates per row')
        decimals = [Decimal(field) for field in fields]
        if any(abs(value.as_tuple().exponent) > 1000 for value in decimals):
            raise ValueError('decimal exponent outside bounded range')
        rows.append(tuple(Fraction(value) for value in decimals))
    if len(rows) < 4 or len(rows) > 64:
        raise ValueError('expected 4 to 64 vertices')
    return rows


def certify_file(path, *, precision_bits=160, max_pairs=1000, timeout_s=30):
    """Certify strict MR inequality or return inconclusive within pair/time budget.

    Exact rational strings in the output are the outward interval endpoints;
    they must not be replaced by rounded decimal text for verification.
    """
    if precision_bits < 53 or precision_bits > 4096:
        raise ValueError('precision_bits must be between 53 and 4096')
    if max_pairs < 0 or timeout_s < 0:
        raise ValueError('budgets must be nonnegative')
    path = Path(path)
    if path.stat().st_size > 30000:
        raise ValueError('coordinate file exceeds size limit')
    raw = path.read_bytes()
    vertices = _parse_vertices(raw)
    n = len(vertices)
    record = {'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
              'sticks': n, 'precision_bits': precision_bits,
              'backend': 'mpmath.iv', 'mpmath_version': mpmath.__version__,
              'pair_count': 0, 'status': 'inconclusive'}
    edges = [(_sub(vertices[(i+1) % n], vertices[i])) for i in range(n)]
    length_sq = [_dot(edge, edge) for edge in edges]
    if any(sq == 0 for sq in length_sq):
        record['reason'] = 'zero-length edge'
        return record
    deadline = time.monotonic() + timeout_s
    minimum = None
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if record['pair_count'] >= max_pairs or time.monotonic() >= deadline:
                record['reason'] = 'pair or time budget exhausted'
                return record
            sq = squared_segment_distance(vertices[i], vertices[(i+1) % n],
                                          vertices[j], vertices[(j+1) % n])
            record['pair_count'] += 1
            minimum = sq if minimum is None else min(minimum, sq)
    ctx = MPIntervalContext()
    ctx.prec = precision_bits
    lengths = [ctx.sqrt(_interval_fraction(ctx, sq)) for sq in length_sq]
    mean = sum(lengths, ctx.mpf(0)) / n
    defect = max((abs(length / mean - 1) for length in lengths),
                 key=lambda z: _endpoint(z, 1))
    mu = ctx.sqrt(_interval_fraction(ctx, minimum)) / mean
    bound1, bound2 = mu/n, mu*mu/4
    record.update(defect_lower=str(_endpoint(defect, 0)),
                  defect_upper=str(_endpoint(defect, 1)),
                  mu_lower=str(_endpoint(mu, 0)),
                  mu_upper=str(_endpoint(mu, 1)),
                  threshold_lower=str(min(_endpoint(bound1, 0), _endpoint(bound2, 0))),
                  threshold_upper=str(min(_endpoint(bound1, 1), _endpoint(bound2, 1))))
    if _endpoint(defect, 1) < Fraction(record['threshold_lower']):
        record['status'] = 'certified'
    else:
        record['reason'] = 'strict inequality not established'
    return record
