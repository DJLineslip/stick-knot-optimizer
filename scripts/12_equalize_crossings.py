"""Checkpointed safe equalization of exact-rank-four crossing witnesses.

Reads scripts/11_crossing_changes.py's frozen scan manifest and per-source
scratch witnesses. It never modifies the source scan manifest; its own
per-target attempts and statuses are committed atomically.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = ROOT / 'scripts/11_crossing_changes.py'
spec = importlib.util.spec_from_file_location('frozen_crossing_scan', SCAN_SCRIPT)
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def _code_digest():
    code = [SCAN_SCRIPT, ROOT / 'equistick/crossing_change.py',
            ROOT / 'equistick/coxeter.py', ROOT / 'equistick/crss.py']
    return hashlib.sha256(b''.join(path.read_bytes() for path in code)).hexdigest()


def _source_output(scratch, manifest, source_id):
    token = hashlib.sha256(f'source:{source_id}'.encode()).hexdigest()[:20]
    return scratch / manifest['code_sha256'][:16] / f'{token}.output.json'


def _archived_witness(archive, code_sha, source_id, source_record, name):
    """Read a selected derived crossing polygon when raw scratch is absent."""
    if archive is None:
        return None
    if archive.get('code_sha256') != code_sha:
        raise ValueError('archived witness belongs to another scan revision')
    row = archive.get('witnesses', {}).get(name)
    if row is None or row.get('source_id') != source_id:
        return None
    if (row.get('source_sha256') != source_record['source_sha256'] or
            row.get('source_coordinates_sha256') != source_record['coordinates_sha256']):
        raise ValueError(f'archived source differs from manifest: {source_id}')
    coordinates = np.asarray(row['coordinates'], dtype=np.float64)
    if (coordinates.shape != (10, 3) or not np.all(np.isfinite(coordinates)) or
            hashlib.sha256(coordinates.tobytes()).hexdigest() !=
            row.get('candidate_coordinates_sha256')):
        raise ValueError(f'archived candidate coordinates differ: {name}')
    return {key: row[key] for key in ('move_index', 'move', 'coordinates', 'crossing')}


def _qualified(classification):
    if not classification or not classification.get('exact_rank_four_passed'):
        return False
    proof = classification.get('exact_map') or {}
    return (proof.get('passed') is True and proof.get('generation_passed') is True and
            proof.get('relations_checked') == proof.get('strand_count') and
            isinstance(proof.get('strand_count'), int) and proof['strand_count'] > 0 and
            (proof.get('group'), proof.get('group_order')) in (('S5', 120), ('D4', 192)))


def existing_certified(name, results_dir):
    """Only skip a polygon whose saved hash occurs in its interval report."""
    filename = f'{name}_equilateral_10sticks.txt'
    for folder in (Path(results_dir), Path(results_dir).parent):
        report = folder / 'interval_certificates.json'
        file = folder / filename
        if not file.is_file() or not report.is_file():
            continue
        digest = scan._digest(file)
        for row in json.loads(report.read_text()).get('files', []):
            if (row['file'] == filename and row['status'] == 'certified' and
                    row['sticks'] == 10 and row['sha256'] == digest):
                return file
    return None


def _target_state(entry, scan_done, incomplete_sources=False):
    if entry.get('certified'):
        return 'certified'
    if entry.get('existing_certified'):
        return 'certified_existing'
    if not entry['sources_reached']:
        if not scan_done:
            return 'pending'
        return 'not_reached_in_completed_sources' if incomplete_sources else 'not_reached'
    if not entry['exact_rank_four_passed']:
        return 'reached_without_verified_map'
    if entry['attempts']:
        return 'reached_not_certified_within_budget' if scan_done else 'reached_equalization_pending'
    return 'reached_equalization_pending'


def _setup(args, manifest):
    timeouts = [entry['name'] for entry in json.loads(args.pool_results.read_text())['results']
                if entry['status'] == 'timeout']
    if len(timeouts) != len(set(timeouts)):
        raise ValueError('duplicate timeout names in pool report')
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if (report['code_sha256'] != manifest['code_sha256'] or
                report['pool_timeouts'] != timeouts or
                report['candidate_budget_seconds'] != args.candidate_budget or
                report['max_attempts_per_target'] != args.max_attempts):
            raise ValueError('cannot resume a changed equalization campaign')
        return report
    return {'schema': 1, 'code_sha256': manifest['code_sha256'],
            'scan_manifest': str(args.manifest.resolve()),
            'pool_timeouts': timeouts, 'candidate_budget_seconds': args.candidate_budget,
            'max_attempts_per_target': args.max_attempts,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'targets': {name: {'status': 'pending', 'outside_pool': False,
                               'sources_reached': [], 'attempts': [],
                               'exact_rank_four_passed': False} for name in timeouts},
            'scanned_sources_seen': 0, 'scan_inventory_count': manifest['inventory_count']}


def run_once(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest['code_sha256'] != _code_digest():
        raise ValueError('scan code changed during campaign; refusing mixed revisions')
    report = _setup(args, manifest)
    scan_done = len(manifest['sources']) == manifest['inventory_count']
    report['incomplete_source_ids'] = sorted(
        source_id for source_id, record in manifest['sources'].items()
        if record['status'] != 'completed')
    incomplete_sources = bool(report['incomplete_source_ids'])
    report['all_sources_completed'] = scan_done and not incomplete_sources
    witness_path = args.witnesses or args.results / 'crossing_witnesses.json'
    archive = json.loads(witness_path.read_text()) if witness_path.is_file() else None
    for name, hits in sorted(manifest['reached_four_bridge'].items(),
                             key=lambda item: (item[0] not in report['pool_timeouts'], item[0])):
        classification = manifest['map_classifications'].get(name)
        exact = _qualified(classification)
        entry = report['targets'].setdefault(name, {'status': 'pending',
            'outside_pool': True, 'sources_reached': [], 'attempts': [],
            'exact_rank_four_passed': exact})
        entry['exact_rank_four_passed'] = exact
        for source_id in sorted(hits):
            if source_id not in entry['sources_reached']:
                entry['sources_reached'].append(source_id)
        entry['sources_reached'].sort()
        if not exact or entry.get('certified'):
            continue
        certified_path = existing_certified(name, args.results)
        if certified_path is not None:
            entry['existing_certified'] = scan._digest(certified_path)
            continue
        for source_id in sorted(hits, key=lambda sid: (manifest['sources'][sid]['source']['kind'] != 'ours', sid)):
            if len(entry['attempts']) >= args.max_attempts or entry.get('certified'):
                break
            source_record = manifest['sources'][source_id]
            if source_record['status'] != 'completed':
                continue
            source_file = _source_output(args.scratch, manifest, source_id)
            if source_file.is_file():
                full = json.loads(source_file.read_text())
                if (full['status'] != 'completed' or full['source']['id'] != source_id or
                        full['source_sha256'] != source_record['source_sha256'] or
                        full['coordinates_sha256'] != source_record['coordinates_sha256']):
                    raise ValueError(f'raw source witness differs from manifest: {source_id}')
                witnesses = full['scan']['candidates'].get(name, [])
            else:
                selected = _archived_witness(archive, manifest['code_sha256'],
                                             source_id, source_record, name)
                witnesses = [selected] if selected is not None else []
            if not witnesses:
                entry.setdefault('missing_witness_sources', []).append(source_id)
                continue
            for witness in witnesses:
                if len(entry['attempts']) >= args.max_attempts or entry.get('certified'):
                    break
                move_id = witness['move_index']
                if any(a['source_id'] == source_id and a['move_index'] == move_id
                       for a in entry['attempts']):
                    continue
                source = dict(source_record['source'])
                source['coordinates_sha256'] = source_record['coordinates_sha256']
                task = {'mode': 'equalize', 'source': source, 'witness': witness,
                        'name': name, 'results': str(args.results),
                        'budget_seconds': args.candidate_budget}
                result = scan._run_worker(task, args.scratch / manifest['code_sha256'][:16],
                                          f'equalize:{name}:{source_id}:{move_id}',
                                          args.candidate_budget)
                attempt = {'source_id': source_id, 'source_label': source['label'],
                           'move_index': move_id, 'budget_seconds': args.candidate_budget,
                           'status': result['status'], 'result': result}
                entry['attempts'].append(attempt)
                if result['status'] == 'certified':
                    file = args.results / result['file']
                    independently_checked = scan.validate_saved_candidate(file, name)
                    if independently_checked['sha256'] != result['checks']['sha256']:
                        raise ValueError(f'candidate changed after publication: {name}')
                    entry['certified'] = result
                report['scanned_sources_seen'] = manifest['completed_count']
                entry['status'] = _target_state(entry, scan_done, incomplete_sources)
                scan._atomic_json(args.output, report)
                print(f'{name} from {source_id}: {result["status"]}', flush=True)
    for entry in report['targets'].values():
        entry['status'] = _target_state(entry, scan_done, incomplete_sources)
    report['scanned_sources_seen'] = manifest['completed_count']
    report['scan_inventory_count'] = manifest['inventory_count']
    scan._atomic_json(args.output, report)
    print(f'equalization: scan={manifest["completed_count"]}/{manifest["inventory_count"]} '
          f'incomplete_sources={len(report["incomplete_source_ids"])} '
          f'timeouts={len(report["pool_timeouts"])} '
          f'certified={sum(bool(x.get("certified")) for x in report["targets"].values())}',
          flush=True)
    return scan_done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'results/ten_new_stick_knots/crossing_results.json')
    parser.add_argument('--scratch', type=Path,
                        default=ROOT / 'results/logs/crossing_changes')
    parser.add_argument('--witnesses', type=Path,
                        help='selected derived witnesses when raw scan scratch is unavailable')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/ten_new_stick_knots/crossing_equalization.json')
    parser.add_argument('--results', type=Path,
                        default=ROOT / 'results/ten_new_stick_knots')
    parser.add_argument('--pool-results', type=Path,
                        default=ROOT / 'results/ten_new_stick_knots/pool_results.json')
    parser.add_argument('--candidate-budget', type=float, default=180.)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--follow', action='store_true',
                        help='rerun after each completed source until scan finishes')
    args = parser.parse_args()
    if args.candidate_budget <= 0 or args.max_attempts <= 0:
        parser.error('budgets and max attempts must be positive')
    while True:
        if run_once(args) or not args.follow:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
