"""Verify exact-stick coverage using in-repo and external Eddy polygons.

Eddy's raw coordinates remain in its clone; this script commits only hashes,
interval geometric records and independent numerical type checks.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from equistick.certify import mr_certificate_mp, verify_torus
from equistick.data import MSEQ, ROOT as EDDY_ROOT, exact_stick_numbers
from equistick.geometry import mr_ratio
from equistick.invariants import identify, is_torus, matches_census_name
from equistick.interval_certificate import certify_file

ROOT = Path(__file__).resolve().parents[1]
TORUS_ALIASES = {
    'K14n21881': 'T(3,7)',
    'K15n41185': 'T(4,5)',
    'K16n783154': 'T(3,8)',
}
TORUS = {
    '3_1': (2, 3), '5_1': (2, 5), '7_1': (2, 7),
    '8_19': (3, 4), '9_1': (2, 9), '10_124': (3, 5),
    'K11a367': (2, 11), 'K13a4878': (2, 13),
}


def select_source(name, sticks, saved, eddy_dir):
    """Choose one known exact-stick polygon, without changing its name."""
    candidate = TORUS_ALIASES.get(name, name)
    if saved.get(candidate) == sticks:
        return 'in_repo', candidate
    file = Path(eddy_dir) / (name + '.txt')
    if file.is_file():
        return 'eddy', name
    return 'missing', None


def verify_eddy(name, sticks, path):
    """Read an unchanged external decimal file and independently check it."""
    path = Path(path)
    V = np.loadtxt(path)
    if V.shape != (sticks, 3) or not np.isfinite(V).all():
        raise ValueError(f'{name}: invalid saved polygon shape or coordinates')
    ratio, defect, mu = mr_ratio(V)
    high_precision = mr_certificate_mp(V)
    interval = certify_file(path)
    result = dict(knot=name, sticks=sticks, source='eddy',
                  source_path=f'stick_number/mseq_knots/{name}.txt',
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  ratio=float(ratio), defect=float(defect), mu=float(mu),
                  high_precision_certificate=bool(high_precision[3]),
                  interval=interval)
    if name in TORUS:
        p, q = TORUS[name]
        if not is_torus(V, p, q, nproj=4, seed=123):
            result.update(torus=[p, q], torus_alexander=False,
                          numerical_type_match=False, status='unverified')
            return result
        alex, hfk, crossings = verify_torus(V, p, q)
        genus = (p - 1) * (q - 1) // 2
        result.update(torus=[p, q], torus_alexander=bool(alex),
                      hfk_genus=hfk['seifert_genus'],
                      hfk_rank=hfk.get('total_rank'),
                      simplified_crossings=crossings,
                      numerical_type_match=bool(alex and hfk['fibered'] and hfk['L_space_knot']
                                                and hfk['seifert_genus'] == genus
                                                and abs(hfk['tau']) == genus))
    else:
        ids = [identify(V, seed=i) for i in (1, 2, 3)]
        result['coordinate_ids'] = ids
        result['numerical_type_match'] = all(matches_census_name(x, name) for x in ids)
    result['status'] = ('certified' if ratio < 1 and high_precision[3]
                        and interval['status'] == 'certified' and result['numerical_type_match']
                        else 'unverified')
    return result


def verify_saved(name, alias, sticks, summary, intervals, results):
    """Link existing independent reports to the same file bytes."""
    row = summary[alias]
    stem = alias.replace('(', '').replace(',', '_').replace(')', '')
    file = results / f'{stem}_equilateral_{sticks}sticks.txt'
    interval = intervals[file.name]
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if (row['certified'] != 'True' or row['type_confirmed'] != 'True'
            or int(row['sticks']) != sticks or interval['sha256'] != digest
            or interval['status'] != 'certified'):
        raise ValueError(f'{name}: saved file does not match certified reports')
    return dict(knot=name, sticks=sticks, source='in_repo', saved_name=alias,
                source_path=f'results/{file.name}', source_sha256=digest,
                status='certified', numerical_type_match=True,
                interval=interval)


def main():
    start = perf_counter()
    exact = exact_stick_numbers()
    results = ROOT / 'results'
    with (results / 'summary.csv').open(newline='') as f:
        summary = {row['knot']: row for row in csv.DictReader(f)}
    intervals_report = json.loads((results / 'interval_certificates.json').read_text())
    if intervals_report['file_count'] != intervals_report['certified_count']:
        raise ValueError('interval report has unverified saved polygons')
    intervals = {row['file']: row for row in intervals_report['files']}
    saved = {name: int(row['sticks']) for name, row in summary.items()
             if row['certified'] == 'True' and row['type_confirmed'] == 'True'}
    eddy_dir = Path(MSEQ)
    rows = []
    for name, sticks in sorted(exact.items()):
        if name == '0_1':
            rows.append(dict(knot=name, sticks=sticks, source='trivial', status='not_needed'))
            continue
        source, alias = select_source(name, sticks, saved, eddy_dir)
        if source == 'in_repo':
            row = verify_saved(name, alias, sticks, summary, intervals, results)
        elif source == 'eddy':
            row = verify_eddy(name, sticks, eddy_dir / (name + '.txt'))
        else:
            row = dict(knot=name, sticks=sticks, source='missing', status='missing')
        rows.append(row)
        print(name, row['source'], row['status'], flush=True)
    counts = dict(Counter((row['source'], row['status']) for row in rows))
    eddy_commit = subprocess.check_output(['git', '-C', str(EDDY_ROOT), 'rev-parse', 'HEAD'],
                                         text=True).strip()
    report = dict(citation_eddy='https://github.com/thomaseddy/stick-knot-gen',
                  eddy_commit=eddy_commit, eddy_license='MIT, copyright 2019 Thomas D. Eddy',
                  exact_table_total=len(exact),
                  exact_nontrivial_total=len(exact) - 1,
                  certified=sum(row['status'] == 'certified' for row in rows),
                  source_counts=[dict(source=source, status=status, count=n)
                                 for (source, status), n in sorted(counts.items())],
                  results=rows, command='PYTHONPATH=.. EQUISTICK_DATA=/path/to/stick-knot-gen python crss_verify_exact.py',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interval_checker_sha256=hashlib.sha256((ROOT / 'equistick/interval_certificate.py').read_bytes()).hexdigest(),
                  packages={name: version(name) for name in ('numpy', 'snappy', 'snappy-15-knots', 'mpmath')},
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(perf_counter() - start, 2))
    output = ROOT / 'data' / 'exact_stick_coverage.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('Exact nontrivial coverage', report['certified'], '/', report['exact_nontrivial_total'],
          'sources', report['source_counts'], 'report', output)
    return 0 if report['certified'] == report['exact_nontrivial_total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
