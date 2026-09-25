"""Join CRSS stick bounds with known exact and equilateral stick counts.

Run from scripts/ with PYTHONPATH=.. and EQUISTICK_DATA set to Eddy's
stick-knot-gen clone. Eddy-only polygons are edge-length checked here, not
numerically re-identified or interval-certified by this census. A positive
reported e_ub - s_ub is not a gap proof.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from equistick.crss import crss_index, crss_path
from equistick.data import MSEQ, TEN_STICK_19, exact_stick_numbers, seed_for
from crss_validate import TORUS, stratified_sample

FIELDS = ('knot', 'crossings', 's_ub', 'exact_s', 'e_ub',
          'e_ub_minus_s_ub', 'source_sticks', 'input_status', 'e_status')
SAVED_NAME = re.compile(r'(.+)_equilateral_(\d+)sticks\.txt\Z')
PRODUCTION_GROUP_COUNT = 12965


def eddy_bounds(index):
    """Only Eddy's reported mseq polygons, not its bound tables."""
    counts = {}
    for path in sorted(Path(MSEQ).glob('*.txt')):
        if path.stem not in index:
            continue
        coords = np.loadtxt(path)
        if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
            raise ValueError(f'{path}: invalid equilateral coordinates')
        lengths = np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1)
        mean = float(np.mean(lengths))
        if mean <= 0 or np.max(np.abs(lengths - mean)) > 1e-8 * mean:
            raise ValueError(f'{path}: polygon is not equilateral')
        counts[path.stem] = len(coords)
    return counts


def saved_bounds(index, results_dir):
    """Count only local polygons backed by identity and interval verification."""
    summary = {}
    with (results_dir / 'summary.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            name = row['knot']
            if name in summary:
                raise ValueError(f'duplicate saved summary entry: {name}')
            if row['certified'] == 'True' and row['type_confirmed'] == 'True':
                summary[name] = int(row['sticks'])
    report = json.loads((results_dir / 'interval_certificates.json').read_text())
    counts = {}
    for item in report['files']:
        if item['status'] != 'certified':
            continue
        match = SAVED_NAME.fullmatch(item['file'])
        if match is None:
            raise ValueError(f"invalid saved polygon file: {item['file']}")
        name, sticks = match.group(1), int(match.group(2))
        if name not in index or summary.get(name) != sticks:
            continue
        if int(item['sticks']) != sticks:
            raise ValueError(f'{name}: certificate stick count mismatch')
        file = results_dir / item['file']
        if hashlib.sha256(file.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'{file}: SHA256 disagrees with interval certificate')
        counts[name] = min(counts.get(name, sticks), sticks)
    return counts


def validation_statuses(index, report_path):
    """Bind source identity findings to the exact NetCDF file in this census."""
    extras = set(TEN_STICK_19) | {'9_29'} | set(TORUS)
    if len(index) >= PRODUCTION_GROUP_COUNT:
        missing = extras - index.keys()
        if missing:
            raise ValueError(f'Required identity knots absent: {sorted(missing)}')
    else:
        extras &= index.keys()  # Tiny synthetic fixtures need not have every special knot.
    report = json.loads(report_path.read_text())
    if report['groups'] != len(index):
        raise ValueError('source validation group count disagrees with census')
    digest = hashlib.sha256()
    with crss_path().open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if report['source_sha256'] != digest.hexdigest():
        raise ValueError('source validation SHA256 disagrees with NetCDF file')
    details = report['identity_details']
    if not isinstance(details, list) or report['identity_total'] != len(details):
        raise ValueError('identity_total disagrees with identity_details')
    statuses = {}
    for item in details:
        name, status = item['knot'], item['status']
        if name not in index or name in statuses or status not in (
                'matched', 'mismatch', 'error', 'pending_nonhyperbolic'):
            raise ValueError(f'invalid or repeated input identity: {name}')
        statuses[name] = status
    base = stratified_sample(index, min(300, len(index)), seed_for('crss-validation'))
    required = set(base) | extras
    if statuses.keys() != required:
        raise ValueError(f'identity_details disagrees with required audited names: '
                         f'missing={sorted(required - statuses.keys())}, '
                         f'unexpected={sorted(statuses.keys() - required)}')
    counts = Counter(statuses.values())
    expected = dict(identity_matched=counts['matched'],
                    identity_pending=counts['pending_nonhyperbolic'],
                    identity_mismatches=counts['mismatch'] + counts['error'])
    for field, actual in expected.items():
        if report[field] != actual:
            raise ValueError(f'{field} disagrees with identity_details')
    return statuses


def census_rows(index, exact, eddy, saved, statuses):
    """One unique row per group, sorted by crossing number then table name."""
    rows = []
    for knot, (crossings, sticks) in sorted(index.items(), key=lambda pair: (pair[1][0], pair[0])):
        bounds = [source[knot] for source in (eddy, saved) if knot in source]
        best = min(bounds) if bounds else None
        e_status = ('checked' if best is not None and saved.get(knot) == best else
                    'reported_only' if best is not None else 'none')
        status = statuses.get(knot, 'unchecked')
        valid_bound = status in ('matched', 'unchecked')
        rows.append(dict(knot=knot, crossings=crossings,
                         s_ub=sticks if valid_bound else '', source_sticks=sticks,
                         input_status=status, e_status=e_status,
                         exact_s=exact.get(knot, ''), e_ub=best if best is not None else '',
                         e_ub_minus_s_ub=best - sticks if best is not None and valid_bound else ''))
    return rows


def crossing_summary(rows):
    """Aggregate known bounds without interpreting missing bounds as failures."""
    summary = defaultdict(lambda: dict(knots=0, exact_s_known=0, e_ub_known=0,
                                       e_ub_le_s_ub=0, e_ub_gt_s_ub=0, e_ub_unknown=0))
    for row in rows:
        group = summary[row['crossings']]
        group['knots'] += 1
        group['exact_s_known'] += row['exact_s'] != ''
        if row['e_ub'] == '':
            group['e_ub_unknown'] += 1
        else:
            group['e_ub_known'] += 1
            if row['s_ub'] != '':
                group['e_ub_le_s_ub' if row['e_ub'] <= row['s_ub'] else 'e_ub_gt_s_ub'] += 1
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('../results/gap_census.csv'))
    parser.add_argument('--results-dir', type=Path, default=Path('../results'))
    parser.add_argument('--validation-report', type=Path,
                        default=Path('../data/crss_validation.json'))
    parser.add_argument('--expected-count', type=int, default=PRODUCTION_GROUP_COUNT)
    args = parser.parse_args()
    index = crss_index()
    if len(index) != args.expected_count:
        raise ValueError(f'CRSS groups: expected {args.expected_count}, got {len(index)}')
    statuses = validation_statuses(index, args.validation_report)
    rows = census_rows(index, exact_stick_numbers(), eddy_bounds(index),
                       saved_bounds(index, args.results_dir), statuses)
    names = [row['knot'] for row in rows]
    if len(names) != len(set(names)) or len(rows) != args.expected_count:
        raise ValueError('census has missing or duplicate knot names')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('crossings knots exact_s_known e_ub_known e_ub_le_s_ub e_ub_gt_s_ub e_ub_unknown')
    for crossing, counts in sorted(crossing_summary(rows).items()):
        print(crossing, *(counts[field] for field in
                          ('knots', 'exact_s_known', 'e_ub_known', 'e_ub_le_s_ub',
                           'e_ub_gt_s_ub', 'e_ub_unknown')))
    print('Input identity statuses:',
          {status: sum(row['input_status'] == status for row in rows)
           for status in ('matched', 'unchecked', 'mismatch', 'error',
                          'pending_nonhyperbolic')})
    print('Equilateral evidence:',
          {status: sum(row['e_status'] == status for row in rows)
           for status in ('checked', 'reported_only', 'none')})
    print(f'Wrote {len(rows)} unique, sorted knots to {args.output}')


if __name__ == '__main__':
    main()
