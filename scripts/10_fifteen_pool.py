"""Bounded one-process-per-effective-core search of Eddy's 15-crossing pool.

Run from repository root: .venv/bin/python scripts/10_fifteen_pool.py
Each knot gets at most 1200 wall seconds, including startup and validation.
Failures to find a polygon are search logs, not mathematical obstructions.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'BLIS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import snappy  # noqa: F401
import spherogram
from equistick.certify import mr_certificate_mp
from equistick.data import load_eddy, seed_for
from equistick.geometry import mr_ratio, normalize
from equistick.invariants import pd_code
from equistick.interval_certificate import certify_file
from equistick.optimize import fatten, homotopy_equalize
from equistick.reduce import reduce_to

folder = ROOT / 'results' / 'ten_new_stick_knots'
parallel = importlib.import_module('07_parallel')


def table_isometries(V, name, seeds=range(1, 13), needed=3):
    """Collect decisive projection checks, retrying undecidable SnapPy calls."""
    reference = spherogram.Link(name).exterior()
    matches = []
    for seed in seeds:
        try:
            outcome = spherogram.Link(pd_code(V, rng=np.random.default_rng(seed))).exterior().is_isometric_to(reference)
        except RuntimeError as exc:
            print('isometry indeterminate at seed', seed, ':', exc, flush=True)
            continue
        matches.append(bool(outcome))
        if len(matches) == needed:
            break
    return matches


def validate(path, name):
    V = np.loadtxt(path)
    if V.shape != (10, 3) or not np.isfinite(V).all():
        raise ValueError('wrong shape or nonfinite coordinates')
    if not mr_ratio(V)[0] < 1 or not mr_certificate_mp(V)[3]:
        raise ValueError('float or high precision geometric check failed')
    interval = certify_file(path)
    if interval['status'] != 'certified':
        raise ValueError('interval certificate failed: ' + str(interval.get('reason')))
    matches = table_isometries(V, name)
    if len(matches) != 3 or not all(matches):
        raise ValueError('table complement isometry failed or indeterminate in three projections')
    return dict(interval=interval, projections=matches)


def pool_worker(job):
    name = job['name']
    stage = Path(job['stage'])
    budget = job['budget']
    started = time.monotonic()
    if not parallel.Path(parallel.provenance([name], budget, 1)['data_path'], 'stick_number', 'mseq_knots', name + '.txt').is_file():
        result = dict(status='missing_data', message='Eddy starting polygon missing')
    else:
        V0 = load_eddy(name)
        if V0.shape != (11, 3) or table_isometries(V0, name, seeds=range(1, 9), needed=1) != [True]:
            raise ValueError('Eddy input is not a positively identified 11-gon of this table knot')
        rng = np.random.default_rng(seed_for(name))
        attempts = 0
        realisations = 0
        result = None
        while time.monotonic() - started < budget:
            attempts += 1
            print(name, 'reduction attempt', attempts, 'elapsed', round(time.monotonic() - started, 2), flush=True)
            V = reduce_to(V0.copy(), 10, rng)
            if V is None:
                continue
            if table_isometries(V, name, seeds=range(1, 9), needed=1) != [True]:
                print('reduced polygon failed or indeterminate identity check', flush=True)
                continue
            realisations += 1
            Vf = fatten(V, rng, 400)
            for floor in (0.9, 0.5, 0.2):
                remaining = budget - (time.monotonic() - started)
                if remaining <= 1:
                    break
                E, t, mu0, done = homotopy_equalize(Vf, mu_floor=floor, tlimit=min(120, remaining))
                print('homotopy', floor, done, 'elapsed', round(time.monotonic() - started, 2), flush=True)
                if not done:
                    continue
                candidate = stage / f'{name}_equilateral_10sticks.txt'
                np.savetxt(candidate, normalize(E), fmt='%.17g')
                try:
                    checks = validate(candidate, name)
                except ValueError as exc:
                    print('candidate rejected:', exc, flush=True)
                    candidate.unlink()
                    continue
                result = dict(status='certified', message=f'10-stick equal-length polygon, {attempts} reduction attempts',
                              attempts=attempts, realisations=realisations, checks=checks)
                break
            if result:
                break
        if result is None:
            result = dict(status='not_found_within_budget',
                          message=f'no 10-stick polygon in {budget / 60:g} min ({attempts} reduction attempts)',
                          attempts=attempts, realisations=realisations)
    result['completed_monotonic'] = time.monotonic()
    parallel.atomic_json(stage / 'outcome.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=float, default=1200)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--knots', help='comma-separated subset of the 29 for a smoke test')
    parser.add_argument('--out', type=Path, default=folder)
    args = parser.parse_args()
    if args.budget <= 0 or not np.isfinite(args.budget):
        parser.error('budget must be finite and positive')
    if args.workers is not None and args.workers < 1:
        parser.error('workers must be positive')
    preflight = json.loads((folder / 'verification.json').read_text())
    if not all(x['passed'] for x in preflight['six']):
        parser.error('six-knot preflight failed')
    names = preflight['pool']
    if len(names) != 29 or len(set(names)) != 29:
        parser.error('expected exactly 29 distinct unresolved knots')
    if args.knots:
        chosen = args.knots.split(',')
        if len(set(chosen)) != len(chosen) or not set(chosen) <= set(names):
            parser.error('knots must be a distinct subset of the verified pool')
        names = chosen
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers or parallel.effective_cpus(), parallel.effective_cpus())
    print('Pool:', len(names), 'workers:', workers, 'hard per-knot budget:', args.budget, flush=True)
    parallel.warmup()
    manifest = parallel.run_batch(names, args.budget, workers, out, worker=pool_worker)
    output = out / 'pool_results.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Completed:', len(manifest['results']), output, flush=True)
    return int(len(manifest['results']) != len(names) or
               any(row['status'] in ('error', 'missing_data') for row in manifest['results']))


if __name__ == '__main__':
    raise SystemExit(main())
