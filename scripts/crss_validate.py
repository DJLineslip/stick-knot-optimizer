"""Audit the user-provided CRSS source and re-identify a stratified sample.

From scripts/: PYTHONPATH=.. EQUISTICK_DATA=/path/to/stick-knot-gen python crss_validate.py
The file's crossings scalar is the number of crossings in its chosen diagram,
not the table crossing number encoded in the group name.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter

import numpy as np
import spherogram

from equistick.certify import verify_torus
from equistick.crss import (crss_index, crss_path, knot_crossings,
                            load_crss, load_crss_pd, open_crss)
from equistick.data import MSEQ, TEN_STICK_19, exact_stick_numbers, seed_for
from equistick.geometry import min_dist
from equistick.invariants import (DEFAULT_TS, alexander_abs, identify, identify_pd,
                                  matches_census_name, torus_alexander_abs)

PAPER_TABLE = {
    3: {6: 1}, 4: {7: 1}, 5: {8: 2}, 6: {8: 3}, 7: {9: 7},
    8: {8: 2, 9: 4, 10: 15}, 9: {9: 14, 10: 35},
    10: {10: 94, 11: 71}, 11: {10: 160, 11: 362, 12: 30},
    12: {10: 122, 11: 1156, 12: 898},
    13: {10: 108, 11: 2397, 12: 6509, 13: 974},
}
PAPER_URL = 'https://arxiv.org/html/2508.18263v1'
TORUS = {
    '3_1': (2, 3), '5_1': (2, 5), '7_1': (2, 7),
    '8_19': (3, 4), '9_1': (2, 9), '10_124': (3, 5),
    'K11a367': (2, 11), 'K13a4878': (2, 13),
}


def stratified_sample(index, count=300, seed=None):
    """Reproducible round-robin strata, independent of HDF5 group ordering."""
    if count < 0 or count > len(index):
        raise ValueError('sample size outside dataset bounds')
    rng = random.Random(seed)
    strata = defaultdict(list)
    for name, (crossings, sticks) in sorted(index.items()):
        strata[crossings].append(name)
    for names in strata.values():
        rng.shuffle(names)
    sample = []
    while len(sample) < count:
        for crossing in sorted(strata):
            if len(sample) >= count:
                break
            if strata[crossing]:
                sample.append(strata[crossing].pop())
    return sample


def missing_in_range_exact(index, exact):
    """Only exact-table knots in this dataset's crossing range are required."""
    maximum = max(crossing for crossing, sticks in index.values())
    return sorted(name for name in exact if name != '0_1' and name not in index
                  and knot_crossings(name) <= maximum)


def table_discrepancies(observed):
    expected = {(c, sticks): n for c, counts in PAPER_TABLE.items()
                for sticks, n in counts.items()}
    return {key: dict(paper=expected.get(key, 0), observed=observed.get(key, 0))
            for key in sorted(set(expected) | set(observed))
            if expected.get(key, 0) != observed.get(key, 0)}


def check_identity(name, coords, pd):
    """Numerical independent re-identification, never a topological proof."""
    if name in TORUS:
        p, q = TORUS[name]
        alex, hfk, simplified_crossings = verify_torus(coords, p, q, nproj=4)
        pd_alex = bool(np.allclose(alexander_abs(pd, DEFAULT_TS),
                                  torus_alexander_abs(p, q, DEFAULT_TS),
                                  rtol=1e-6, atol=1e-8))
        diagram_hfk = spherogram.Link(pd).knot_floer_homology()
        genus = (p - 1) * (q - 1) // 2
        hfk_ok = all(h.get('L_space_knot') and h.get('fibered') and
                     h.get('seifert_genus') == genus and abs(h.get('tau', -999)) == genus
                     for h in (hfk, diagram_hfk))
        status = 'matched' if alex and pd_alex and hfk_ok else 'mismatch'
        return dict(knot=name, kind='torus', status=status, pd_id=None,
                    coordinate_ids=[], alexander_match=bool(alex),
                    pd_alexander_match=pd_alex, hfk_match=bool(hfk_ok),
                    crossings_after_simplification=simplified_crossings)
    pd_id = identify_pd(pd)
    coord_ids = [identify(coords, tries=1, seed=seed_for(name) + run)
                 for run in range(3)]
    wrong = (pd_id is not None and not matches_census_name(pd_id, name)) or any(
        label is not None and not matches_census_name(label, name) for label in coord_ids)
    unresolved = pd_id is None or any(label is None for label in coord_ids)
    status = 'mismatch' if wrong else ('pending_nonhyperbolic' if unresolved else 'matched')
    return dict(knot=name, kind='hyperbolic_or_other', status=status,
                pd_id=pd_id, coordinate_ids=coord_ids)


def source_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    """Write portable LF-delimited audit records."""
    with path.open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    start = perf_counter()
    path = crss_path()
    with open_crss() as source:
        index = crss_index(dataset=source)
        if len(index) != 12965:
            raise ValueError(f'Expected 12965 prime groups, got {len(index)}')
        exact = exact_stick_numbers()
        missing_exact = missing_in_range_exact(index, exact)
        if missing_exact:
            raise ValueError(f'Exact table names missing from CRSS: {missing_exact}')
        rows = []
        frequency = Counter()
        projected_different = 0
        names = sorted(index, key=lambda name: (index[name][0], name))
        for step, name in enumerate(names, 1):
            crossing, sticks = index[name]
            V = load_crss(name, dataset=source)
            pd = load_crss_pd(name, dataset=source)
            mu = float(min_dist(V))
            if not np.isfinite(mu) or mu <= 0:
                raise ValueError(f'{name}: nonpositive or nonfinite input min_dist: {mu}')
            projected_different += len(pd) != crossing
            frequency[crossing, sticks] += 1
            rows.append(dict(knot=name, crossings=crossing, sticks=sticks,
                             input_min_dist=format(mu, '.17g')))
            if step % 2000 == 0:
                print(f'Geometry {step}/{len(index)} checked', flush=True)
        differences = table_discrepancies(frequency)
        base = stratified_sample(index, 300, seed_for('crss-validation'))
        extras = set(TEN_STICK_19) | {'9_29'} | set(TORUS)
        if extras - set(index):
            raise ValueError(f'Required identity knots absent: {sorted(extras - set(index))}')
        audited = sorted(set(base) | extras, key=lambda name: (index[name][0], name))
        identities = []
        for step, name in enumerate(audited, 1):
            V = load_crss(name, dataset=source)
            pd = load_crss_pd(name, dataset=source)
            try:
                result = check_identity(name, V, pd)
            except Exception as exc:
                result = dict(knot=name, kind='unknown', status='error',
                              reason=f'{type(exc).__name__}: {exc}')
            result['stratified'] = name in base
            result['ten_stick'] = name in TEN_STICK_19
            identities.append(result)
            if step % 50 == 0:
                print(f'Identity {step}/{len(audited)} checked', flush=True)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / 'crss_index.csv'
    write_csv(out, ('knot', 'crossings', 'sticks', 'input_min_dist'), rows)
    unresolved = [x for x in identities if x['status'] == 'pending_nonhyperbolic']
    mismatches = [x for x in identities if x['status'] in ('mismatch', 'error')]
    mismatch_file = root / 'results' / 'sweep' / 'input_mismatches.csv'
    mismatch_file.parent.mkdir(parents=True, exist_ok=True)
    mismatch_fields = ('knot', 'status', 'pd_id', 'coordinate_ids', 'reason')
    write_csv(mismatch_file, mismatch_fields,
              ({key: json.dumps(item.get(key)) if key == 'coordinate_ids'
                else item.get(key, '') for key in mismatch_fields}
               for item in mismatches))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                     text=True).strip()
    report = dict(source_file=path.name, source_sha256=source_hash(path),
                  citation='doi:10.7910/DVN/NFJIII', paper_table=PAPER_URL,
                  groups=len(index), by_crossing=dict(sorted(Counter(c for c, n in index.values()).items())),
                  scalar_layout='variables (sticks and crossings); fallback attributes supported',
                  coords_layout='float64 N x 3, unnormalised source scale',
                  pd_layout='int64 diagram_crossings x 4, zero-based labels; validated with spherogram',
                  netcdf_global_attributes={},
                  projected_crossings_different_from_table=projected_different,
                  table_1_match=not differences,
                  table_1_discrepancies=[dict(crossings=c, sticks=n, **counts)
                                         for (c, n), counts in differences.items()],
                  stick_frequencies=[dict(crossings=c, sticks=n, knots=count)
                                     for (c, n), count in sorted(frequency.items())],
                  eddy_name_matches=len(set(index) & {p.stem for p in Path(MSEQ).glob('*.txt')}),
                  exact_table_matches=len(set(index) & set(exact)),
                  stratified_seed=seed_for('crss-validation'),
                  stratified_count=len(base),
                  stratified_by_crossing=dict(sorted(Counter(index[n][0] for n in base).items())),
                  ten_stick_count=len(TEN_STICK_19), torus_count=len(TORUS),
                  identity_total=len(audited), identity_matched=sum(x['status'] == 'matched' for x in identities),
                  identity_pending=len(unresolved), identity_mismatches=len(mismatches),
                  identity_details=identities,
                  command='PYTHONPATH=.. EQUISTICK_DATA=/workspace/repos/stick-knot-optimizer/stick-knot-gen python crss_validate.py',
                  git_base_commit=commit,
                  code_sha256={p: source_hash(root / p) for p in
                               ('equistick/crss.py', 'equistick/invariants.py', 'scripts/crss_validate.py')},
                  packages={name: version(name) for name in ('numpy', 'netCDF4', 'h5py', 'snappy')},
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(perf_counter() - start, 2))
    (data_dir / 'crss_validation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('Audit', report['groups'], 'groups; Table 1', 'match' if report['table_1_match'] else 'MISMATCH',
          '; projected crossings unlike name', projected_different)
    print('Identities', len(audited), 'matched', report['identity_matched'],
          'pending', len(unresolved), 'mismatch_or_error', len(mismatches))
    if differences or mismatches:
        raise SystemExit('Audit found table discrepancies or input identity mismatches; see report')


if __name__ == '__main__':
    main()
