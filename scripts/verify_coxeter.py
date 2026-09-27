"""Reconstruct every Wirt_Hm strand map and record exact Coxeter checks.

Run from repo root with the external workbook present, or pass --workbook.
The group proof is exact. It does not certify the numerical identification
of any geometric 10-gon with the named table knot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openpyxl
from equistick.coxeter import verify_map

DEFAULT_REPORT = ROOT / 'results' / 'ten_new_stick_knots' / 'pool_verification.json'
DEFAULT_WORKBOOK = ROOT / 'data' / 'external' / 'Wirt_Hm' / 'all_data_A.xlsx'


def generate_proofs(report, workbook):
    """Attach complete labels, exact pass/fail and diagnostics to every knot."""
    source_hash = hashlib.sha256(Path(workbook).read_bytes()).hexdigest()
    if report.get('source_sha256') and report['source_sha256'] != source_hash:
        raise ValueError('the Wirt_Hm workbook differs from the previously verified source')
    data = json.loads(json.dumps(report))
    records = data['existing_six'] + data['pool_results']
    names = {item['knot'] for item in records}
    if len(records) != len(names):
        raise ValueError('duplicate knot in report')
    rows = {}
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        for row in wb.active.iter_rows(values_only=True):
            name = 'K' + str(row[0])
            if name in names:
                if name in rows:
                    raise ValueError(f'duplicate Wirt_Hm row: {name}')
                rows[name] = row
    finally:
        wb.close()
    proofs = {}
    for name in sorted(names):
        row = rows.get(name)
        if row is None:
            proof = {'passed': False, 'status': 'row_missing', 'group': None}
        elif str(row[2]) != '4':
            proof = {'passed': False, 'status': 'wirtinger_not_four', 'group': None}
        else:
            options = [(kind, row[image]) for kind, flag, image in
                       (('S5', 4, 5), ('D4', 6, 7))
                       if str(row[flag]) == '1' and isinstance(row[image], str)]
            if len(options) != 1:
                proof = {'passed': False, 'status': 'map_missing' if not options else 'ambiguous_map',
                         'group': None, 'listed_map_count': len(options)}
            else:
                kind, seed_images = options[0]
                try:
                    proof = verify_map(row[1], seed_images, kind)
                except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                    proof = {'passed': False, 'status': 'invalid_map_data',
                             'group': kind, 'reason': str(exc)}
        proofs[name] = proof
    for item in records:
        proof = proofs[item['knot']]
        item['exact_coxeter_passed'] = proof['passed']
        item['exact_coxeter_status'] = proof['status']
        if 'checks' in item:
            item['checks']['exact_coxeter_passed'] = proof['passed']
    data['source_sha256'] = source_hash
    data['exact_coxeter_maps'] = proofs
    data['verification_note'] = ('The Wirt_Hm S5/D4 Coxeter maps were checked '
                                 'exactly for their listed diagrams; this check '
                                 'does not certify saved polygon geometry or '
                                 'establish formal named-knot identity.')
    saved = data['existing_six'] + [item for item in data['pool_results']
                                    if item.get('search_status') == 'certified']
    if saved and all(item['exact_coxeter_passed'] for item in saved):
        data['lower_bound_source'] = ('Wirt_Hm all_data_A.xlsx: rank-four S5/D4 '
                                      'maps independently checked by exact integer relations and '
                                      'generation for the listed diagrams; identifying the saved '
                                      'polygons with those diagrams remains numerical')
    else:
        data.pop('lower_bound_source', None)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path,
                        default=Path(os.environ.get('EQUISTICK_WIRT_HM', DEFAULT_WORKBOOK)))
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if len(report['existing_six']) != 6 or len(report['pool_results']) != 29:
        parser.error('expected six original knots and 29 pool knots')
    updated = generate_proofs(report, args.workbook)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=args.report.parent,
                                     prefix='.coxeter-', suffix='.json', delete=False) as handle:
        temp = Path(handle.name)
        json.dump(updated, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temp, args.report)
    proofs = updated['exact_coxeter_maps']
    passed = sum(item['passed'] for item in proofs.values())
    missing = sum(item['status'] == 'map_missing' for item in proofs.values())
    print(f'exact Coxeter maps: {passed} passed, {missing} without map, {len(proofs)} total')
    return int(passed != 31 or missing != 4 or len(proofs) != 35)


if __name__ == '__main__':
    raise SystemExit(main())
