"""Finite crossing-change search from every available 10-stick source.

A validated single-vertex move is a transverse pierce of exactly one swept
triangle by one edge. The move grid is finite: not reached is not a
nonexistence result. The NetCDF dataset is user supplied, never downloaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from itertools import chain

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'BLIS_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import openpyxl
import snappy  # noqa: F401, required for spherogram's exterior()
import spherogram
from equistick import crss
from equistick.certify import mr_certificate_mp
from equistick.coxeter import verify_map
from equistick.crossing_change import find_single_crossing, propose_moves, propose_nearest_moves
from equistick.data import seed_for
from equistick.geometry import lengths, min_dist, mr_ratio, normalize
from equistick.invariants import pd_code
from equistick.interval_certificate import certify_file
from equistick.optimize import fatten, homotopy_equalize


def collect_sources(eddy_dir, results_dir):
    """Inventory every NetCDF 10-stick group, Eddy file and certified local file."""
    eddy_dir, results_dir = Path(eddy_dir), Path(results_dir)
    sources = [{'id': f'crss:{name}', 'label': name, 'kind': 'crss', 'path': None}
               for name, (_, sticks) in sorted(crss.crss_index().items()) if sticks == 10]
    for path in sorted(eddy_dir.glob('*.txt')):
        V = np.loadtxt(path)
        if V.shape == (10, 3):
            sources.append({'id': f'eddy:{path.stem}', 'label': path.stem,
                            'kind': 'eddy', 'path': str(path.resolve()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for folder in (results_dir, results_dir / 'ten_new_stick_knots'):
        report = folder / 'interval_certificates.json'
        if not report.is_file():
            continue
        for entry in json.loads(report.read_text())['files']:
            filename = entry['file']
            if (entry['status'] != 'certified' or entry['sticks'] != 10 or
                    not filename.endswith('_equilateral_10sticks.txt')):
                continue
            if Path(filename).name != filename:
                raise ValueError(f'unsafe certified file path: {filename}')
            path = folder / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry['sha256']:
                raise ValueError(f'certified source hash mismatch: {path}')
            label = filename.removesuffix('_equilateral_10sticks.txt')
            sources.append({'id': f'ours:{label}', 'label': label,
                            'kind': 'ours', 'path': str(path.resolve()), 'sha256': digest})
    ids = [source['id'] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source IDs')
    return sources


def select_table_name(names):
    """Choose an unambiguous Rolfsen or HT name, preserving other aliases."""
    short = {name for name in names if re.fullmatch(r'\d+_\d+', name)}
    htw = {name for name in names if re.fullmatch(r'K\d+[an]\d+', name)}
    if len(short) == 1:
        selected = next(iter(short))
        crossing = int(selected.split('_')[0])
        if all(int(re.match(r'K(\d+)', name).group(1)) == crossing for name in htw):
            return selected
    if not short and len(htw) == 1:
        return next(iter(htw))
    return None


def identify_polygon(V, needed=1, seeds=range(1, 13)):
    """Identify with meridian-preserving SnapPy census matches.

    Report uncertainty instead of silently treating numerical exceptions
    as nonmatches. The name itself remains a numerical identification.
    """
    if needed < 1:
        raise ValueError('needed must be positive')
    agreed = None
    used, errors, aliases = [], [], []
    for seed in seeds:
        try:
            pd = pd_code(V, rng=np.random.default_rng(seed))
            if not pd:
                name = 'unknot'
            else:
                link = spherogram.Link(pd)
                link.simplify('global')
                if not link.crossings:
                    name = 'unknot'
                else:
                    matches = link.exterior().identify(extends_to_link=True)
                    names = [str(m).split('(')[0] for m in matches]
                    name = select_table_name(names)
                    if name is None:
                        errors.append({'seed': seed, 'reason': 'ambiguous or no table match',
                                       'matches': names})
                        continue
                    aliases = names
        except (ValueError, RuntimeError, ZeroDivisionError, ArithmeticError) as exc:
            errors.append({'seed': seed, 'reason': str(exc)})
            continue
        if agreed is not None and name != agreed:
            return {'status': 'disagreement', 'name': None, 'projection_seeds': used,
                    'conflicting_name': name, 'errors': errors}
        agreed = name
        used.append(seed)
        if len(used) == needed:
            return {'status': 'identified', 'name': name, 'projection_seeds': used,
                    'aliases': aliases, 'errors': errors}
    return {'status': 'inconclusive', 'name': None, 'projection_seeds': used,
            'errors': errors}


def scan_polygon(V, fractions=(.25, .5, .75), weights=(.25, .5), times=(.5,),
                 seed=1, strategy='both', near_offsets=(-.15, 0., .15),
                 near_times=(.7, .9)):
    """Test the entire finite move grid and identify every valid resulting polygon."""
    V = np.asarray(V, dtype=np.float64)
    if min_dist(V) <= 1e-9 * lengths(V).mean():
        raise ValueError('source polygon is not safely embedded')
    source_identity = identify_polygon(V, needed=2, seeds=range(seed, seed + 12))
    if strategy not in ('uniform', 'nearest', 'both'):
        raise ValueError(f'unknown move strategy: {strategy}')
    nearest = propose_nearest_moves(V, offsets=near_offsets, times=near_times)
    uniform = propose_moves(V, fractions=fractions, weights=weights, times=times)
    moves = (uniform if strategy == 'uniform' else nearest if strategy == 'nearest'
             else chain(nearest, uniform))
    moves_tried = single_crossings = inconclusive = 0
    counts = Counter()
    candidates = {}
    examples = []
    for move in moves:
        moves_tried += 1
        crossing = find_single_crossing(V, move['vertex'], move['destination'])
        if crossing is None:
            continue
        single_crossings += 1
        W = V.copy()
        W[move['vertex']] = move['destination']
        identification = identify_polygon(W, seeds=range(seed, seed + 4))
        if identification['status'] != 'identified':
            inconclusive += 1
            if len(examples) < 10:
                examples.append({'move_index': moves_tried, 'identification': identification})
            continue
        name = identification['name']
        counts[name] += 1
        edge_lengths = lengths(W)
        defect = float(np.max(np.abs(edge_lengths / edge_lengths.mean() - 1)))
        score = defect / max(crossing['endpoint_clearance'] / edge_lengths.mean(), 1e-12)
        witness = {'move_index': moves_tried,
                   'move': {**move, 'destination': move['destination'].tolist()},
                   'crossing': crossing, 'identification': identification,
                   'score': score, 'coordinates': W.tolist()}
        saved = candidates.setdefault(name, [])
        saved.append(witness)
        saved.sort(key=lambda item: (item['score'], item['move_index']))
        del saved[3:]
    return {'moves_tried': moves_tried, 'single_crossings': single_crossings,
            'identified_counts': dict(sorted(counts.items())),
            'inconclusive_count': inconclusive, 'inconclusive_examples': examples,
            'candidates': candidates, 'source_identification': source_identity,
            'grid': {'strategy': strategy, 'fractions': list(fractions),
                     'weights': list(weights), 'times': list(times),
                     'near_offsets': list(near_offsets), 'near_times': list(near_times)},
            'seed': seed}


def build_wirt_index(workbook):
    """Index the 4-Wirtinger rows once, without claiming an untested map."""
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    rows = {}
    try:
        for number, row in enumerate(wb.active.iter_rows(values_only=True), start=1):
            if not row[0] or str(row[2]) != '4':
                continue
            name = 'K' + str(row[0])
            if name in rows:
                raise ValueError(f'duplicate Wirt_Hm row: {name}')
            maps = {kind: row[image] for kind, flag, image in
                    (('S5', 4, 5), ('D4', 6, 7))
                    if str(row[flag]) == '1' and isinstance(row[image], str)}
            rows[name] = {'row': number, 'gauss': row[1], 'maps': maps,
                          'h4_listed': str(row[8]) == '1'}
    finally:
        wb.close()
    return rows


def classify_reached(name, index, cache=None):
    """A 4-bridge target only if an actual S5/D4 map passes exact checks."""
    if cache is not None and name in cache:
        return cache[name]
    row = index.get(name)
    if row is None:
        result = {'wirtinger_four': False, 'exact_rank_four_passed': False,
                  'status': 'not_in_wirt_hm_four_index'}
    else:
        attempts = {}
        for kind, seeds in row['maps'].items():
            try:
                attempts[kind] = verify_map(row['gauss'], seeds, kind)
            except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                attempts[kind] = {'passed': False, 'status': 'invalid_map_data',
                                  'group': kind, 'reason': str(exc)}
        passed = next((proof for proof in attempts.values() if proof['passed']), None)
        result = {'wirtinger_four': True, 'exact_rank_four_passed': passed is not None,
                  'status': 'pass' if passed else 'no_supported_map_passed',
                  'wirt_row': row['row'], 'h4_listed': row['h4_listed'],
                  'map_attempts': attempts, 'exact_map': passed}
    if cache is not None:
        cache[name] = result
    return result


def process_source(source, **grid):
    """Read and scan exactly one source; exceptions denote incomplete work."""
    started = time.monotonic()
    if source['kind'] == 'crss':
        V = crss.load_crss(source['label'])
        source_hash = hashlib.sha256(V.tobytes()).hexdigest()
    else:
        path = Path(source['path'])
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash != source['sha256']:
            raise ValueError(f'source file changed since inventory: {path}')
        V = np.loadtxt(path)
    if V.shape != (10, 3) or not np.isfinite(V).all():
        raise ValueError(f'invalid ten-stick polygon: {source["id"]}')
    scan = scan_polygon(V, seed=seed_for(source['id']), **grid)
    return {'status': 'completed', 'source': source, 'source_sha256': source_hash,
            'coordinates_sha256': hashlib.sha256(V.tobytes()).hexdigest(),
            'elapsed_seconds': time.monotonic() - started, 'scan': scan}


def summarize_source(result, wirt_index, classifications=None):
    """Log all recognized four-Wirtinger hits; queue only exact-map targets."""
    if result['status'] != 'completed':
        raise ValueError('cannot summarize an incomplete source')
    scan = result['scan']
    source = result['source']
    identity = scan.get('source_identification', {})
    summary = {'status': 'completed', 'source': source, 'source_sha256': result['source_sha256'],
               'coordinates_sha256': result.get('coordinates_sha256'),
               'elapsed_seconds': result['elapsed_seconds'],
               'moves_tried': scan['moves_tried'], 'single_crossings': scan['single_crossings'],
               'identified_counts': scan['identified_counts'],
               'inconclusive_count': scan['inconclusive_count'],
               'inconclusive_examples': scan.get('inconclusive_examples', []),
               'source_identification': identity,
               'source_name_confirmed': identity.get('status') == 'identified' and
                                        identity.get('name') == source['label'],
               'grid': scan.get('grid'), 'seed': scan.get('seed'),
               'four_bridge_reached': {}}
    targets = {}
    for name, count in sorted(scan['identified_counts'].items()):
        if name not in wirt_index:
            continue
        checked = classify_reached(name, wirt_index, classifications)
        proof = checked.get('exact_map') or {}
        summary['four_bridge_reached'][name] = {
            'crossing_changes': count,
            'exact_rank_four_passed': checked['exact_rank_four_passed'],
            'wirt_row': checked['wirt_row'], 'status': checked['status'],
            'map_group': proof.get('group'), 'group_order': proof.get('group_order'),
            'relations_checked': proof.get('relations_checked'),
            'h4_listed': checked['h4_listed']}
        if checked['exact_rank_four_passed']:
            targets[name] = scan['candidates'].get(name, [])
    return summary, targets


def validate_saved_candidate(path, name, seed=1):
    """Reopen the decimal file and independently validate geometry and name."""
    path = Path(path)
    V = np.loadtxt(path)
    if V.shape != (10, 3) or not np.isfinite(V).all():
        raise ValueError('saved candidate has invalid coordinates')
    ratio = float(mr_ratio(V)[0])
    high_precision = bool(mr_certificate_mp(V)[3])
    interval = certify_file(path)
    if not ratio < 1 or not high_precision or interval['status'] != 'certified':
        raise ValueError('saved candidate failed a geometric certificate')
    identity = identify_polygon(V, needed=3, seeds=range(seed, seed + 12))
    if identity['status'] != 'identified' or identity['name'] != name:
        raise ValueError(f'saved candidate identity not confirmed: {identity}')
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'mr_ratio': ratio, 'high_precision': high_precision,
            'interval': interval, 'identity': identity}


def attempt_equalization(source, witness, name, out_dir, budget_seconds=180):
    """Replay a crossing witness, safely equalize, then publish only after checks.

    This function runs inside a subprocess under a parent-enforced hard wall.
    A timeout or unsuccessful homotopy is a search status, never a negative
    mathematical statement.
    """
    if budget_seconds <= 0 or not np.isfinite(budget_seconds):
        raise ValueError('equalization budget must be positive and finite')
    started = time.monotonic()
    if source['kind'] == 'crss':
        initial = crss.load_crss(source['label'])
    else:
        path = Path(source['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('crossing source changed since inventory')
        initial = np.loadtxt(path)
    if source.get('coordinates_sha256') and source['coordinates_sha256'] != hashlib.sha256(initial.tobytes()).hexdigest():
        raise ValueError('crossing source coordinates changed since scan')
    move = witness['move']
    i = move['vertex']
    W = initial.copy()
    W[i] = np.asarray(move['destination'], dtype=np.float64)
    if not np.array_equal(W, np.asarray(witness['coordinates'], dtype=np.float64)):
        return {'status': 'invalid_crossing', 'reason': 'witness does not replay'}
    event = find_single_crossing(initial, i, W[i])
    if event is None or event['edge'] != witness['crossing']['edge']:
        return {'status': 'invalid_crossing', 'reason': 'not one transverse edge event'}
    source_identity = identify_polygon(initial, needed=2, seeds=range(seed_for(source['id']),
                                                                   seed_for(source['id']) + 12))
    initial_identity = identify_polygon(W, needed=2, seeds=range(seed_for(name),
                                                                seed_for(name) + 12))
    if initial_identity['status'] != 'identified' or initial_identity['name'] != name:
        return {'status': 'identity_inconclusive', 'reason': initial_identity,
                'crossing': event}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f'{name}_equilateral_10sticks.txt'
    if final.exists():
        checks = validate_saved_candidate(final, name, seed=seed_for(name))
        return {'status': 'already_certified', 'file': final.name, 'checks': checks}
    rng = np.random.default_rng(seed_for(f'{source["id"]}:{name}:{witness.get("move_index", 0)}'))
    attempts = []
    starts = [W]
    for run in range(2):
        if time.monotonic() - started >= budget_seconds:
            break
        if run == 1:
            starts.append(fatten(W, rng, 200))
        for floor in (.9, .5, .2):
            remaining = budget_seconds - (time.monotonic() - started)
            if remaining <= 1:
                break
            if mr_ratio(starts[run])[0] < 1:
                E, done, reached = starts[run], True, 1.
            else:
                E, reached, _, done = homotopy_equalize(
                    starts[run], mu_floor=floor, tlimit=min(60, remaining))
            attempts.append({'fattened': bool(run), 'floor': floor,
                             'completed': bool(done), 'homotopy_fraction': float(reached)})
            if not done:
                continue
            with tempfile.NamedTemporaryFile('w', dir=out_dir, prefix=f'.{name}-',
                                             suffix='.txt', delete=False) as handle:
                temp = Path(handle.name)
                np.savetxt(handle, normalize(E), fmt='%.17g')
            try:
                checks = validate_saved_candidate(temp, name, seed=seed_for(name))
            except ValueError as exc:
                temp.unlink()
                attempts[-1]['validation_failure'] = str(exc)
                continue
            os.replace(temp, final)
            checks = validate_saved_candidate(final, name, seed=seed_for(name))
            return {'status': 'certified', 'route': 'crossing_change',
                    'source_knot': source_identity.get('name'),
                    'source_label': source['label'], 'source_identification': source_identity,
                    'source_id': source['id'], 'move_index': witness.get('move_index'),
                    'crossing': event, 'file': final.name, 'checks': checks,
                    'attempts': attempts, 'elapsed_seconds': time.monotonic() - started}
    return {'status': 'not_certified_within_budget', 'budget_seconds': budget_seconds,
            'attempts': attempts, 'elapsed_seconds': time.monotonic() - started}


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}-', delete=False) as handle:
        temp = Path(handle.name)
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _digest(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as source:
        while chunk := source.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def _revision():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                            capture_output=True, check=True)
    return result.stdout.strip()


def _metadata(args, inventory, grid):
    code = [ROOT / 'scripts/11_crossing_changes.py', ROOT / 'equistick/crossing_change.py',
            ROOT / 'equistick/coxeter.py', ROOT / 'equistick/crss.py']
    code_sha = hashlib.sha256(b''.join(path.read_bytes() for path in code)).hexdigest()
    try:
        eddy_rev = subprocess.run(['git', '-C', str(args.eddy.parent.parent), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        eddy_rev = None
    packages = {}
    for package in ('numpy', 'snappy', 'spherogram', 'netCDF4', 'openpyxl'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'command': sys.argv, 'git_commit': _revision(), 'code_sha256': code_sha,
            'workbook_sha256': _digest(args.workbook), 'crss_sha256': _digest(crss.crss_path()),
            'crss_path': str(crss.crss_path()), 'eddy_commit': eddy_rev,
            'packages': packages, 'source_budget_seconds': args.source_budget,
            'candidate_budget_seconds': args.candidate_budget,
            'grid': grid, 'inventory_count': len(inventory), 'inventory': inventory,
            'sources': {}, 'map_classifications': {}, 'reached_four_bridge': {},
            'targets': {}, 'completed_count': 0, 'pending_count': len(inventory)}


def _options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eddy', type=Path, default=Path(os.environ.get(
        'EQUISTICK_DATA', ROOT / 'stick-knot-gen')) / 'stick_number/mseq_knots')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--workbook', type=Path,
                        default=ROOT / 'data/external/Wirt_Hm/all_data_A.xlsx')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/ten_new_stick_knots/crossing_results.json')
    parser.add_argument('--scratch', type=Path, default=ROOT / 'results/logs/crossing_changes')
    parser.add_argument('--source-budget', type=float, default=60.)
    parser.add_argument('--candidate-budget', type=float, default=180.)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--strategy', choices=('nearest', 'uniform', 'both'), default='both')
    parser.add_argument('--fractions', nargs='+', type=float, default=[.25, .5, .75])
    parser.add_argument('--weights', nargs='+', type=float, default=[.25, .5])
    parser.add_argument('--times', nargs='+', type=float, default=[.5])
    parser.add_argument('--near-offsets', nargs='+', type=float, default=[-.15, 0., .15])
    parser.add_argument('--near-times', nargs='+', type=float, default=[.7, .9])
    parser.add_argument('--worker-json', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-out', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.source_budget <= 0 or args.candidate_budget <= 0 or
            not np.isfinite([args.source_budget, args.candidate_budget]).all() or
            args.limit is not None and args.limit <= 0):
        parser.error('positive, finite budgets and limit are required')
    return args


def _worker(args):
    task = json.loads(args.worker_json.read_text())
    if task['mode'] == 'source':
        result = process_source(task['source'], **task['grid'])
    elif task['mode'] == 'equalize':
        result = attempt_equalization(task['source'], task['witness'], task['name'],
                                      task['results'], task['budget_seconds'])
    else:
        raise ValueError(f'unknown worker mode: {task["mode"]}')
    _atomic_json(args.worker_out, result)


def _run_worker(task, scratch, tag, seconds):
    token = hashlib.sha256(tag.encode()).hexdigest()[:20]
    source_input = scratch / f'{token}.input.json'
    source_output = scratch / f'{token}.output.json'
    _atomic_json(source_input, task)
    if source_output.exists():
        return json.loads(source_output.read_text())
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               '--worker-json', str(source_input), '--worker-out', str(source_output)],
                              capture_output=True, text=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'budget_seconds': seconds}
    if proc.returncode:
        return {'status': 'error', 'return_code': proc.returncode,
                'stderr': proc.stderr[-4000:]}
    if not source_output.is_file():
        return {'status': 'error', 'reason': 'worker returned zero without output'}
    return json.loads(source_output.read_text())


def main():
    args = _options()
    if args.worker_json is not None:
        _worker(args)
        return
    grid = {'strategy': args.strategy, 'fractions': args.fractions,
            'weights': args.weights, 'times': args.times,
            'near_offsets': args.near_offsets, 'near_times': args.near_times}
    inventory = collect_sources(args.eddy, args.results)
    if not inventory:
        raise ValueError('no ten-stick sources found')
    metadata = _metadata(args, inventory, grid)
    if args.output.exists():
        manifest = json.loads(args.output.read_text())
        for key in ('code_sha256', 'workbook_sha256', 'crss_sha256', 'inventory',
                    'grid', 'source_budget_seconds', 'candidate_budget_seconds'):
            if manifest[key] != metadata[key]:
                raise ValueError(f'cannot resume a changed campaign: {key}')
    else:
        manifest = metadata
        _atomic_json(args.output, manifest)
    index = build_wirt_index(args.workbook)
    scratch = args.scratch / manifest['code_sha256'][:16]
    scratch.mkdir(parents=True, exist_ok=True)
    work = sorted(inventory, key=lambda item: (not item['label'].startswith('K15'),
                                               item['kind'] != 'ours', item['id']))
    if args.limit:
        work = work[:args.limit]
    skipped = 0
    for position, source in enumerate(work, start=1):
        source_id = source['id']
        if manifest['sources'].get(source_id, {}).get('status') == 'completed':
            skipped += 1
            continue
        result = _run_worker({'mode': 'source', 'source': source, 'grid': grid},
                             scratch, f'source:{source_id}', args.source_budget)
        if result['status'] == 'completed':
            summary, _ = summarize_source(result, index, manifest['map_classifications'])
            manifest['sources'][source_id] = summary
            for name, reached in summary['four_bridge_reached'].items():
                manifest['reached_four_bridge'].setdefault(name, {})[source_id] = reached
        else:
            manifest['sources'][source_id] = {'status': result['status'], 'source': source,
                                               'moves_tried': None, 'failure': result}
        manifest['completed_count'] = sum(record['status'] == 'completed'
                                           for record in manifest['sources'].values())
        manifest['pending_count'] = len(inventory) - manifest['completed_count']
        _atomic_json(args.output, manifest)
        print(f'{position}/{len(work)} {source_id}: {result["status"]}; '
              f'moves={manifest["sources"][source_id]["moves_tried"]}; '
              f'completed={manifest["completed_count"]}/{len(inventory)}', flush=True)
    print(f'completed={manifest["completed_count"]}/{len(inventory)} '
          f'skipped_completed={skipped}', flush=True)


if __name__ == '__main__':
    main()
