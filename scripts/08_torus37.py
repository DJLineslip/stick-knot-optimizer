"""Bounded 12-stick searches for T(3,7) and T(3,8). Numerical certificates only.

Run from the repository root with the project Python, for example:
    python scripts/08_torus37.py --budget 1800 --workers 2
The supervisor enforces a hard wall deadline including spawn and validation.
A timeout or stalled optimizer is only a search log, never evidence of e > s.
"""
import os

# Set before importing numpy or scipy, including in spawned interpreters.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import importlib
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from equistick.data import seed_for
from equistick.geometry import lengths, min_dist, mr_ratio, normalize
from equistick.invariants import is_torus
from equistick.certify import mr_certificate_mp, polish, verify_torus
from equistick.optimize import fatten, homotopy_equalize
from equistick.reduce import reduce_to

SUPERVISOR = importlib.import_module('07_parallel')
KNOTS = {'T3_7': 7, 'T3_8': 8}
# Checked against SnapPy HFK on 24-sample parametrized polygons, not inferred
# from the Alexander polynomial alone. Mirror signs of tau are immaterial.
HFK_RANK = {7: 9, 8: 11}
TARGET = 12  # Known stick number for these superbridge-tight torus knots.


class CandidateRejected(ValueError):
    """A geometric or invariant check failed, not a numerical obstruction."""


def torus_start(q, n=24):
    """Sample t -> ((R+r cos(qt))cos(3t), (R+r cos(qt))sin(3t), r sin(qt)).

    R=2.5 and r=1 keep the smooth core torus embedded. The 24-stick
    polygon is accepted only after independent polygonal invariant checks.
    """
    if q not in (7, 8) or n < 24:
        raise ValueError('only T(3,7) and T(3,8) starts with at least 24 samples')
    t = 2 * np.pi * np.arange(n) / n + 0.013
    radial = 2.5 + np.cos(q * t)
    return np.column_stack((radial * np.cos(3 * t),
                            radial * np.sin(3 * t), np.sin(q * t)))


def _check_type(V, q):
    alex, h, crossings = verify_torus(V, 3, q, nproj=4)
    genus = q - 1
    if not (alex and h['seifert_genus'] == genus and
            h['fibered'] and h['L_space_knot'] and
            abs(h['tau']) == genus and h['total_rank'] == HFK_RANK[q] and
            crossings >= 2 * q):
        raise CandidateRejected(f'T(3,{q}) four-projection Alexander/HFK mismatch: '
                                f'alex={alex}, HFK={h}, crossings={crossings}')
    return h, crossings


def validate_start(V, q):
    """Confirm type of the polygonal sampling, not merely the smooth curve."""
    if q not in (7, 8) or V.ndim != 2 or V.shape[1] != 3 or not np.isfinite(V).all():
        raise CandidateRejected('invalid start')
    if np.min(lengths(V)) <= 1e-9:
        raise CandidateRejected('starting polygon has a zero-length edge')
    if min_dist(V) <= 1e-6:
        raise CandidateRejected('starting polygon is not well embedded')
    _check_type(V, q)
    return True


def validate_candidate(V, q):
    """Numerically certify and re-identify exactly the supplied saved floats."""
    if q not in (7, 8) or V.shape != (TARGET, 3) or not np.isfinite(V).all():
        raise CandidateRejected('invalid 12-stick coordinates')
    if np.min(lengths(V)) <= 1e-9:
        raise CandidateRejected('zero-length edge')
    ratio, defect, mu = mr_ratio(V)
    if not (np.isfinite(ratio) and mu > 0 and ratio < 1):
        raise CandidateRejected('float Millett and Rawdon ratio is not below 1')
    cert = mr_certificate_mp(V)
    if not cert[3]:
        raise CandidateRejected('40-digit Millett and Rawdon certificate failed')
    h, crossings = _check_type(V, q)
    return dict(ratio=float(ratio), defect=str(cert[0]), mu=str(cert[1]),
                bound=str(cert[2]), mp_certified=bool(cert[3]),
                alexander_projections=4, hfk=h, crossings=crossings)


def run_search(q, budget, stage):
    """Try independent seeded reductions; no Eddy or Dataverse data needed."""
    name = f'T3_{q}'
    rng = np.random.default_rng(seed_for(name))
    start = torus_start(q)
    validate_start(start, q)
    t0 = time.monotonic()
    print(f'{name}: seed={seed_for(name)} budget={budget:g}s start=24 target=12 '
          f'initial_mu={min_dist(start):.6g}; four-projection Alexander and HFK checked', flush=True)
    attempts = 0
    realizations = 0
    while time.monotonic() - t0 < budget:
        attempts += 1
        print(f'{name}: reduction attempt {attempts} at {time.monotonic()-t0:.2f}s', flush=True)
        V = reduce_to(start.copy(), TARGET, rng, steps=40000)
        if V is None:
            print(f'{name}: reduction attempt {attempts} failed', flush=True)
            continue
        if not is_torus(V, 3, q, nproj=4, seed=123):
            print(f'{name}: reduction invariant mismatch, rejected', flush=True)
            continue
        try:
            _check_type(V, q)
        except CandidateRejected as exc:
            print(f'{name}: reduced polygon rejected: {exc}', flush=True)
            continue
        realizations += 1
        print(f'{name}: 12-stick type checked, fattening, mu={min_dist(V):.6g}', flush=True)
        V = fatten(V, rng, iters=300)
        for floor in (0.9, 0.5, 0.2):
            remaining = budget - (time.monotonic() - t0)
            if remaining <= 0:
                break
            E, progress, mu0, done = homotopy_equalize(
                V, mu_floor=floor, tlimit=min(120, remaining))
            print(f'{name}: floor={floor} progress={progress:.4g} '
                  f'mu0={mu0:.6g} done={done}', flush=True)
            if not done:
                continue
            # Polish is not path-safe. Validate the polished saved file afresh.
            final = normalize(polish(E))
            filename = f'{name}_equilateral_{TARGET}sticks.txt'
            path = Path(stage) / filename
            np.savetxt(path, final, fmt='%.17g')
            try:
                details = validate_candidate(np.loadtxt(path), q)
            except CandidateRejected as exc:
                path.unlink()
                print(f'{name}: saved candidate rejected: {exc}', flush=True)
                continue
            print(f'{name}: saved candidate validated: {details}', flush=True)
            return f'FOUND numerical 12-stick certificate after {attempts} reductions, {realizations} realizations'
    return f'not found within budget ({budget:g}s); {attempts} reductions, {realizations} 12-stick realizations'


def search_worker(job):
    """Only write a certified outcome after reopening and checking staged bytes."""
    q = KNOTS[job['name']]
    stage = Path(job['stage'])
    message = run_search(q, job['budget'], stage)
    path = stage / f"{job['name']}_equilateral_{TARGET}sticks.txt"
    if path.exists():
        details = validate_candidate(np.loadtxt(path), q)
        result = dict(status='certified', message=message, certificate=details)
    else:
        result = dict(status='not_found_within_budget', message=message)
    result['completed_monotonic'] = time.monotonic()
    SUPERVISOR.atomic_json(stage / 'outcome.json', result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knots', default='T3_7,T3_8', help='comma-separated subset of T3_7,T3_8')
    parser.add_argument('--budget', type=float, default=1800, help='hard wall seconds per knot, including spawn and validation')
    parser.add_argument('--workers', type=int, default=2, help='maximum spawned workers, capped by CPU quota')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args(argv)
    args.knots = args.knots.split(',')
    if not args.knots or any(n not in KNOTS for n in args.knots) or len(set(args.knots)) != len(args.knots):
        parser.error('knots must be a unique subset of T3_7,T3_8')
    if not math.isfinite(args.budget) or args.budget <= 0 or args.budget > 1800:
        parser.error('budget must be finite and in (0, 1800] seconds per knot')
    if args.workers < 1:
        parser.error('workers must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    SUPERVISOR.warmup()
    manifest = SUPERVISOR.run_batch(
        args.knots, args.budget, args.workers, args.out, worker=search_worker,
        target=TARGET, log_name='torus37.log',
        extra_provenance=dict(stick_number_source='12 sticks: superbridge-tight T(3,7) and T(3,8), see AGENTS.md',
                              projection_seeds=[123], start='explicit torus R=2.5 r=1 p=3 q=7/8 n=24 phase=0.013',
                              search_seed_rule='seed_for(T3_q), one persistent RNG per knot',
                              interval_certificate=False))
    print('Manifest:', manifest['manifest'], flush=True)
    return int(any(r['status'] == 'error' for r in manifest['results']))


if __name__ == '__main__':
    raise SystemExit(main())
