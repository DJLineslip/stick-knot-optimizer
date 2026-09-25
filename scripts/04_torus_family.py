"""Search T(p,p+1) from deterministic symmetric starts.

Use 08_torus_batch.py for a hard per-knot wall deadline. A failure to find a
polygon within a budget is a search log, not evidence of an obstruction.
"""
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from equistick.data import seed_for
from equistick.torus import torus_poly, STARTS, symmetric_scan
from equistick.geometry import normalize, min_dist, mr_ratio
from equistick.invariants import is_torus
from equistick.flows import agitate
from equistick.optimize import clearance_floor_solve
from equistick.certify import polish, mr_certificate_mp, verify_torus

FLOORS = {3: [0.04, 0.02, 0.01], 4: [0.02, 0.01, 0.005], 5: [0.01, 0.005, 0.0025],
          6: [0.005, 0.0025, 0.001], 7: [0.0025, 0.001, 0.0005],
          8: [0.001, 0.0005, 0.00025], 9: [0.0005, 0.00025, 0.0001]}


def torus_alexander_rank(p, q):
    """Count nonzero coefficients of the exact T(p,q) Alexander polynomial."""
    degree = (p - 1) * (q - 1)
    numerator = {0: 1, 1: -1, p*q: -1, p*q + 1: 1}
    coefficients = []
    for k in range(degree + 1):
        value = numerator.get(k, 0)
        if k >= p:
            value += coefficients[k - p]
        if k >= q:
            value += coefficients[k - q]
        if k >= p + q:
            value -= coefficients[k - p - q]
        coefficients.append(value)
    return sum(value != 0 for value in coefficients)


def validate_candidate(V, p, q):
    """Validate the actual, serialized coordinates, not just an optimizer path.

    Numerical evidence only: Alexander in four projections and knot Floer
    invariants do not constitute a formal knot identification proof.
    """
    if q != p + 1 or V.shape != (2 * q, 3) or not np.isfinite(V).all():
        raise ValueError('wrong shape, torus parameters or nonfinite coordinates')
    ratio, defect, mu = mr_ratio(V)
    if not np.isfinite(ratio) or ratio >= 1 or mu <= 0:
        raise ValueError(f'MR float ratio rejected: {ratio}')
    certificate = mr_certificate_mp(V)
    if not certificate[3]:
        raise ValueError('40-digit MR certificate rejected')
    alex, h, crossings = verify_torus(V, p, q, nproj=4)
    genus = (p - 1) * (q - 1) // 2
    if (not alex or h['seifert_genus'] != genus or abs(h['tau']) != genus
            or not h['fibered'] or not h['L_space_knot']
            or h['total_rank'] != torus_alexander_rank(p, q)
            or crossings < (p - 1) * q):
        raise ValueError(f'T({p},{q}) four-projection/HFK mismatch: {alex}, {h}')
    return certificate, h, crossings, (ratio, defect, mu)


def publish_candidate(V, p, out):
    """Round-trip the 17-digit file and publish only after full validation."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = f'T{p}_{p+1}_equilateral_{2*p+2}sticks.txt'
    with tempfile.NamedTemporaryFile(mode='w', dir=out, prefix='.torus_candidate_',
                                     suffix='.txt', delete=False) as f:
        temporary = Path(f.name)
        np.savetxt(f, V, fmt='%.17g')
    try:
        details = validate_candidate(np.loadtxt(temporary), p, p + 1)
        target = out / name
        if target.exists():
            if target.read_bytes() != temporary.read_bytes():
                raise ValueError('existing certified coordinates differ; refusing overwrite')
        else:
            os.replace(temporary, target)
        return details
    finally:
        temporary.unlink(missing_ok=True)


def run(p, trials=8, out='../results', scan_trials=1500, run_index=0):
    if p not in FLOORS or trials < 1 or scan_trials < 1 or run_index < 0:
        raise ValueError('invalid p, trial count or run index')
    name = f'T{p}_{p+1}'
    seed = seed_for(name) + run_index
    print(f'{name} seed={seed} run_index={run_index} trials={trials} scan_trials={scan_trials}', flush=True)
    rng = np.random.default_rng(seed)
    if p in STARTS:
        params = STARTS[p]
        print(f'{name} start=STARTS {params}', flush=True)
    else:
        hits = symmetric_scan(p, trials=scan_trials, seed=seed)
        print(f'{name} symmetric_scan hits={len(hits)}', flush=True)
        if not hits:
            print(f'{name}: no symmetric start found; not found within budget', flush=True)
            return None
        r, h, phif, mu = hits[0]
        params = (1.0, r, h, phif)
        print(f'{name} start={params} normalized_mu={mu:.6g}', flush=True)
    V0 = torus_poly(p, *params)
    for mu0 in FLOORS[p]:
        for trial in range(trials):
            V = normalize(V0.copy())
            agitate(V, rng, 0.15 * min_dist(V), 60 * len(V))
            W = clearance_floor_solve(V, mu0, maxiter=500)
            ratio, defect, mu = mr_ratio(W)
            eligible = bool(np.isfinite(ratio) and ratio < 1 and is_torus(W, p, p + 1))
            print(f'{name} floor={mu0:g} trial={trial} ratio={ratio:.6g} '
                  f'defect={defect:.6g} mu={mu:.6g} eligible={eligible}', flush=True)
            if not eligible:
                continue
            try:
                details = publish_candidate(polish(W), p, out)
            except ValueError as exc:
                print(f'{name} floor={mu0:g} trial={trial} rejected: {exc}', flush=True)
                continue
            print(f'{name} FOUND floor={mu0:g} cert={details[0][3]} '
                  f'genus={details[1]["seifert_genus"]} tau={details[1]["tau"]} '
                  f'crossings={details[2]}', flush=True)
            return mu0
        print(f'{name} floor={mu0:g} not found in {trials} trials', flush=True)
    print(f'{name}: not found within budget; no obstruction inferred', flush=True)
    return None


if __name__ == '__main__':
    p = int(sys.argv[1])
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(p, trials)
