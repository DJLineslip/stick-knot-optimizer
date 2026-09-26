"""Recheck the six saved 15-crossing polygons and independent Wirt_Hm rows.

Run from scripts/ with PYTHONPATH=.. (or from the repository root).
Numerical isometry is not a rigorous knot-type certificate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import openpyxl
import snappy  # noqa: F401, required for spherogram exterior()
import spherogram
from equistick.geometry import mr_ratio
from equistick.certify import mr_certificate_mp
from equistick.invariants import pd_code
from equistick.interval_certificate import certify_file

SIX = ('K15n40184', 'K15n40185', 'K15n41189', 'K15n41193', 'K15n41235', 'K15n59007')
FIVE = set(SIX) - {'K15n59007'}
FOLDER = ROOT / 'results' / 'ten_new_stick_knots'
WORKBOOK = ROOT / 'data' / 'external' / 'Wirt_Hm' / 'all_data_A.xlsx'
EDDY = ROOT / 'stick-knot-gen' / 'stick_number' / 'mseq_knots'


def rows_for(names, workbook=WORKBOOK):
    names = set(names)
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = {}
        for row in wb.active.iter_rows(values_only=True):
            name = row[0]
            if name and 'K' + str(name) in names:
                rows['K' + str(name)] = row
                if len(rows) == len(names):
                    break
        if set(rows) != names:
            raise ValueError(f'Wirt_Hm rows missing: {sorted(names - set(rows))}')
        return rows
    finally:
        wb.close()


def gauss_link(text):
    """Wirt_Hm +/- over/under Gauss sequence to an oriented DT diagram.

    Odd visits receive signed even partners. This convention is checked
    against the table complement, not trusted on the basis of the label.
    """
    code = ast.literal_eval(text)
    n = len(code)
    if n < 6 or n % 2 or sorted(abs(x) for x in code) != [i for i in range(1, n // 2 + 1) for _ in (0, 1)]:
        raise ValueError('invalid Gauss code')
    if any(code.count(i) != 1 or code.count(-i) != 1 for i in range(1, n // 2 + 1)):
        raise ValueError('Gauss code must have one over and one under visit')
    where = {x: i for i, x in enumerate(code, 1)}
    dt = []
    for i in range(1, n, 2):
        x = code[i - 1]
        partner = where[-x]
        dt.append(partner if x > 0 else -partner)
    return spherogram.Link('DT: ' + str(dt))


def map_data(row):
    if str(row[2]) != '4':
        return {'wirtinger_four': False, 'rank_four_map_listed': False, 'map_type': None}
    for flag, value, kind in ((row[4], row[5], 'S5'), (row[6], row[7], 'D4'), (row[8], row[9], 'H4')):
        if str(flag) == '1' and isinstance(value, str):
            mapping = ast.literal_eval(value)
            if isinstance(mapping, dict) and len(mapping) == 4:
                return {'wirtinger_four': True, 'rank_four_map_listed': True, 'map_type': kind}
    return {'wirtinger_four': True, 'rank_four_map_listed': False, 'map_type': None}


def verify_one(name, row, folder=FOLDER):
    path = folder / f'{name}_equilateral_10sticks.txt'
    V = np.loadtxt(path)
    interval = certify_file(path)
    reference = spherogram.Link(name).exterior()
    projections = [bool(spherogram.Link(pd_code(V, rng=np.random.default_rng(seed))).exterior().is_isometric_to(reference))
                   for seed in (1, 2, 3)]
    gauss = bool(gauss_link(row[1]).exterior().is_isometric_to(reference))
    high_precision = bool(mr_certificate_mp(V)[3])
    result = {'knot': name, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'interval': interval['status'], 'mr_ratio_below_one': bool(mr_ratio(V)[0] < 1),
              'high_precision': high_precision, 'projections': projections,
              'gauss_table_isometry': gauss, **map_data(row)}
    result['passed'] = (V.shape == (10, 3) and interval['status'] == 'certified' and
                        result['mr_ratio_below_one'] and high_precision and all(projections) and gauss and
                        result['wirtinger_four'] and result['rank_four_map_listed'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=FOLDER)
    args = parser.parse_args()
    workbook_names = {p.stem for p in EDDY.glob('K15n*.txt') if len(np.loadtxt(p)) == 11} | set(SIX)
    rows = rows_for(workbook_names)
    pool = sorted(name for name in workbook_names - set(SIX) if str(rows[name][2]) == '4')
    report = {'source_workbook': str(WORKBOOK.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(),
              'source_commit': '74fe52e57de6f91988f63ac105cbc157534ad966',
              'pool': pool, 'pool_count': len(pool),
              'pool_without_coxeter_map': [n for n in pool if not map_data(rows[n])['rank_four_map_listed']],
              'six': [verify_one(name, rows[name], args.folder) for name in SIX]}
    args.folder.mkdir(parents=True, exist_ok=True)
    path = args.folder / 'verification.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    for row in report['six']:
        print(row['knot'], 'PASS' if row['passed'] else 'FAIL', row['map_type'], row['projections'])
    print('Pool:', len(pool), 'without map:', report['pool_without_coxeter_map'])
    return int(not all(row['passed'] for row in report['six']))


if __name__ == '__main__':
    raise SystemExit(main())
