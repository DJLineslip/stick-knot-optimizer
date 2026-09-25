"""Bounded task-2 searches from CRSS low-stick inputs (local only).

The published candidate is the exact staged decimal file that passed numerical
identity, high-precision geometry and an outward interval geometric check.
A timeout or a failed lift is never an obstruction to equalisation.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Set these before scientific imports, including in spawned workers.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from equistick.certify import mr_certificate_mp
from equistick.crss import crss_path, load_crss, load_crss_pd
from equistick.data import TEN_STICK_19, exact_stick_numbers, seed_for
from equistick.geometry import mr_ratio, normalize
from equistick.invariants import identify, identify_pd, matches_census_name
from equistick.interval_certificate import certify_file
from equistick.optimize import fatten, homotopy_equalize

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {'9_29': 9, 'K13n586': 10, 'K13n593': 10}


class CandidateRejected(ValueError):
    """A tested candidate failed acceptance, not a negative mathematical claim."""


def validate_candidate(path, name, target):
    """Check the saved 17-digit file, independently of the optimizer memory."""
    path = Path(path)
    V = np.loadtxt(path)
    if V.shape != (target, 3) or not np.isfinite(V).all():
        raise CandidateRejected('invalid coordinate shape or nonfinite values')
    known = 10 if name in TEN_STICK_19 else exact_stick_numbers().get(name)
    if known != target:
        raise CandidateRejected('target is not the known exact stick number')
    ratio, defect, mu = mr_ratio(V)
    if not np.isfinite(ratio) or ratio >= 1:
        raise CandidateRejected('Millett-Rawdon ratio is not below one')
    mp_result = mr_certificate_mp(V)
    if not mp_result[3]:
        raise CandidateRejected('high precision geometry check failed')
    ids = [identify(V, seed=run) for run in (1, 2, 3)]
    if not all(matches_census_name(label, name) for label in ids):
        raise CandidateRejected('independent knot identity disagrees with input')
    interval = certify_file(path)
    if interval['status'] != 'certified':
        raise CandidateRejected('outward interval geometry was inconclusive')
    return dict(ratio=float(ratio), defect=float(defect), mu=float(mu),
                mp_certificate=[str(x) for x in mp_result[:3]] + [bool(mp_result[3])],
                coordinate_ids=ids, interval=interval)


def verify_input(V, name):
    """Check the source's PD code against independent coordinate projections."""
    pd_label = identify_pd(load_crss_pd(name))
    coord_labels = [identify(V, seed=seed_for(name) + run) for run in range(3)]
    if not matches_census_name(pd_label, name) or not all(
            matches_census_name(label, name) for label in coord_labels):
        raise ValueError(f'{name}: source identity mismatch (PD {pd_label}, coords {coord_labels})')
    return dict(pd_id=pd_label, coordinate_ids=coord_labels)


def close_worker(job):
    parallel = importlib.import_module('07_parallel')
    name, target, budget = job['name'], job['target'], job['budget']
    start = time.monotonic()
    V = load_crss(name)
    if len(V) != target or TARGETS.get(name) != target:
        raise ValueError(f'{name}: wrong source stick count')
    input_check = verify_input(V, name)
    stage = Path(job['stage'])
    candidate = stage / f'{name}_equilateral_{target}sticks.txt'
    history = []
    V = normalize(V)

    def try_candidate(W, method):
        np.savetxt(candidate, W, fmt='%.17g')
        try:
            certificate = validate_candidate(candidate, name, target)
        except CandidateRejected as exc:
            candidate.unlink()
            history.append(dict(method=method, rejected=str(exc)))
            return None
        return dict(status='certified', message=f'certified {method}',
                    method=method, seed=seed_for(name), input_identity=input_check,
                    certificate=certificate, attempts=history)

    result = try_candidate(V, 'unchanged equal-stick source')
    if result is None:
        rng = np.random.default_rng(seed_for(name))
        expanded = fatten(V, rng, iters=400)
        for floor in (0.9, 0.5, 0.2):
            remaining = budget - (time.monotonic() - start) - 10
            if remaining <= 0:
                break
            W, reached, mu_floor, done = homotopy_equalize(
                expanded, mu_floor=floor, tlimit=min(120, remaining))
            attempt = dict(method='fatten plus safe homotopy', floor=floor,
                           reached=float(reached), mu_floor=float(mu_floor),
                           completed=bool(done), elapsed_seconds=round(time.monotonic() - start, 2))
            history.append(attempt)
            print(name, json.dumps(attempt), flush=True)
            if done:
                result = try_candidate(normalize(W), f'fatten plus homotopy at floor {floor}')
                if result is not None:
                    break
    if result is None:
        result = dict(status='not_found_within_budget',
                      message=f'not found within {budget:g}s from source polygon',
                      seed=seed_for(name), input_identity=input_check, attempts=history)
    result['completed_monotonic'] = time.monotonic()
    parallel.atomic_json(stage / 'outcome.json', result)
    print(json.dumps(result, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knots', default='K13n586,K13n593,9_29')
    parser.add_argument('--budget', type=float, default=300)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    names = args.knots.split(',')
    if len(set(names)) != len(names) or any(n not in TARGETS for n in names):
        parser.error('knots must be distinct members of K13n586,K13n593,9_29')
    if not math.isfinite(args.budget) or args.budget <= 0 or args.workers < 1:
        parser.error('budget must be finite positive and workers positive')
    parallel = importlib.import_module('07_parallel')
    parallel.warmup()
    digest = hashlib.sha256(crss_path().read_bytes()).hexdigest()
    manifests = []
    for target in (9, 10):
        subset = [name for name in names if TARGETS[name] == target]
        if subset:
            manifests.append(parallel.run_batch(
                subset, args.budget, min(args.workers, 2), ROOT / 'results',
                worker=close_worker, target=target, log_name='crss_close_known.log',
                extra_provenance=dict(crss_sha256=digest, input_doi='10.7910/DVN/NFJIII',
                                      optimizer='fatten plus homotopy_equalize',
                                      interval_checker='exact decimal outward geometric bounds')))
    for manifest in manifests:
        print('Manifest', manifest['manifest'],
              [(row['name'], row['status']) for row in manifest['results']], flush=True)
    return int(any(row['status'] == 'error' for m in manifests for row in m['results']))


if __name__ == '__main__':
    sys.exit(main())
