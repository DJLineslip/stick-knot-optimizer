"""Hard-deadline, logged T(8,9) and T(9,10) search supervisor.

Default is one worker so other worktrees can share the five-CPU quota.
No negative outcome is evidence of an equilateral stick obstruction.
"""
import os

# These must precede imports of numerical packages, including in spawn children.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
parallel = importlib.import_module('07_parallel')
from equistick.data import seed_for


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knots', default='T8_9,T9_10', help='subset of T8_9,T9_10')
    parser.add_argument('--budget', type=float, default=1800,
                        help='hard per-knot wall seconds (maximum 1800), including validation')
    parser.add_argument('--workers', type=int, default=1, help='one by default; at most two shared workers')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--scan-trials', type=int, default=1500)
    parser.add_argument('--run-index', type=int, default=0, help='nonnegative seed offset')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args(argv)
    args.knots = args.knots.split(',')
    if not args.knots or len(set(args.knots)) != len(args.knots) or any(
            n not in ('T8_9', 'T9_10') for n in args.knots):
        parser.error('knots must be distinct names from T8_9,T9_10')
    if not math.isfinite(args.budget) or not 0 < args.budget <= 1800:
        parser.error('budget must be positive and no more than 1800 seconds')
    if args.workers not in (1, 2) or args.trials < 1 or args.scan_trials < 1 or args.run_index < 0:
        parser.error('workers must be 1 or 2; trials positive; run-index nonnegative')
    return args


def torus_worker(job):
    import numpy as np
    torus = importlib.import_module('04_torus_family')
    name, p = job['name'], job['p']
    stage = Path(job['stage'])
    floor = torus.run(p, trials=job['trials'], scan_trials=job['scan_trials'],
                      run_index=job['run_index'], out=stage)
    if floor is None:
        result = dict(status='not_found_within_budget',
                      message=f'not found within budget ({job["budget"]:g}s); no obstruction inferred')
    else:
        filename = f'{name}_equilateral_{job["target"]}sticks.txt'
        path = stage / filename
        # Check the exact published bytes a second time, before the hard deadline.
        cert, hfk, crossings, ratio = torus.validate_candidate(np.loadtxt(path), p, p + 1)
        result = dict(status='certified', message=f'numerical certificate at floor {floor:g}',
                      floor=floor, ratio=ratio[0], genus=hfk['seifert_genus'],
                      tau=hfk['tau'], crossings=crossings, certificate=[str(x) for x in cert[:3]] + [bool(cert[3])],
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result['completed_monotonic'] = time.monotonic()
    parallel.atomic_json(stage / 'outcome.json', result)
    print('OUTCOME', json.dumps(result), flush=True)


def collect_outcome(job, exitcode, deadline, out):
    """Never publish an incomplete, late, altered, or conflicting staged file."""
    if time.monotonic() >= deadline:
        return dict(status='timeout', message='not found within budget; completion observed after deadline')
    try:
        result = json.loads((Path(job['stage']) / 'outcome.json').read_text())
        if exitcode != 0 or result['status'] not in ('certified', 'not_found_within_budget', 'error'):
            raise ValueError(f'worker exit {exitcode} or unknown status')
        if result.get('completed_monotonic', float('inf')) > deadline:
            return dict(status='timeout', message='not found within budget; late completion')
        if result['status'] == 'certified':
            filename = f"{job['name']}_equilateral_{job['target']}sticks.txt"
            stage = Path(job['stage']) / filename
            if hashlib.sha256(stage.read_bytes()).hexdigest() != result['sha256']:
                raise ValueError('staged coordinates changed after validation')
            if time.monotonic() >= deadline:
                return dict(status='timeout', message='not found within budget; publication deadline passed')
            target = Path(out) / filename
            # stage lives under out: hard-link publication is atomic and refuses
            # an existing name, unlike exists() followed by os.replace().
            created = False
            try:
                os.link(stage, target)
                created = True
            except FileExistsError:
                if target.read_bytes() != stage.read_bytes():
                    raise ValueError('conflicting existing coordinates')
            if time.monotonic() >= deadline:
                # Only roll back a name we created, and only while it still
                # refers to our validated inode.
                if created and target.exists() and os.path.samefile(stage, target):
                    target.unlink()
                return dict(status='timeout', message='not found within budget; publication exceeded deadline')
            if created and not os.path.samefile(stage, target):
                raise ValueError('published coordinates replaced by another writer')
            if not created and target.read_bytes() != stage.read_bytes():
                raise ValueError('existing coordinates changed by another writer')
            result['coordinates'] = str(target)
        return result
    except (OSError, ValueError, KeyError) as exc:
        return dict(status='error', message=f'worker exit {exitcode}: {exc}')


def run_batch(ps, budget, workers, out, worker=torus_worker, run_index=0, trials=8, scan_trials=1500):
    if (not math.isfinite(budget) or not 0 < budget <= 1800 or not ps or any(p not in (8, 9) for p in ps)
            or len(set(ps)) != len(ps) or workers not in (1, 2) or run_index < 0
            or trials < 1 or scan_trials < 1):
        raise ValueError('invalid knot list or search limits')
    workers = min(workers, parallel.effective_cpus(), 2)
    out = Path(out).resolve()
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    run_id = 'torus_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    names = [f'T{p}_{p+1}' for p in ps]
    manifest_path = logs / (run_id + '.json')
    manifest = parallel.provenance(names, budget, workers)
    manifest.pop('projection_seeds')  # Ten-stick runner metadata is not torus metadata.
    manifest.update(run_id=run_id, manifest=str(manifest_path),
                    seeds={n: seed_for(n) + run_index for n in names},
                    projection_seed=123, projection_count=4,
                    stick_number_source='Jin 1997: s(T(p,p+1)) = 2(p+1) for p > 2',
                    run_index=run_index, trials=trials, scan_trials=scan_trials, results=[])
    with (out / 'RUNLOG.md').open('a') as f:
        f.write('\n## ' + run_id + '\n\n```json\n' + json.dumps(manifest, indent=2) + '\n```\n')
    parallel.atomic_json(manifest_path, manifest)
    pending, active = list(ps), []
    context = mp.get_context('spawn')
    try:
        while pending or active:
            while pending and len(active) < workers:
                p = pending.pop(0)
                name = f'T{p}_{p+1}'
                stage = tempfile.mkdtemp(prefix='.torus_candidate_', dir=out)
                job = dict(p=p, name=name, target=2*p+2, budget=budget, stage=stage,
                           log=str(logs / f'{run_id}_{name}.log'),
                           trials=trials, scan_trials=scan_trials, run_index=run_index)
                Path(job['log']).touch()
                process = context.Process(target=parallel.logged_worker, args=(worker, job))
                started = time.monotonic()
                process.start()
                active.append((process, job, started))
            for process, job, started in active[:]:
                if process.is_alive() and time.monotonic() - started < budget:
                    continue
                if process.is_alive():
                    parallel.stop_process(process)
                    result = dict(status='timeout',
                                  message=f'not found within budget ({budget:g}s); hard wall deadline')
                else:
                    process.join()
                    result = collect_outcome(job, process.exitcode, started + budget, out)
                result.update(name=job['name'], elapsed_seconds=time.monotonic() - started, log=job['log'])
                with open(job['log'], 'a') as f:
                    f.write('END ' + json.dumps(result) + '\n')
                with (out / 'torus_search.log').open('a') as f:
                    f.write(f"{job['name']} [{result['status']}] {result['message']}\n")
                manifest['results'].append(result)
                parallel.atomic_json(manifest_path, manifest)
                with (out / 'RUNLOG.md').open('a') as f:
                    f.write(f"\n{job['name']} [{result['status']}] "
                            + json.dumps(result, sort_keys=True) + '\n')
                print(f"{job['name']} [{result['status']}] {result['message']}", flush=True)
                shutil.rmtree(job['stage'])
                process.close()
                active.remove((process, job, started))
            if active:
                time.sleep(0.01)
    finally:
        for process, job, _ in active:
            if process.is_alive():
                parallel.stop_process(process)
            process.close()
            shutil.rmtree(job['stage'], ignore_errors=True)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    workers = min(args.workers, parallel.effective_cpus())
    print(f'Torus search: {args.knots}, {workers} worker(s), {args.budget:g}s hard per knot', flush=True)
    parallel.warmup()
    result = run_batch([int(n.split('_')[0][1:]) for n in args.knots], args.budget,
                       workers, args.out, run_index=args.run_index,
                       trials=args.trials, scan_trials=args.scan_trials)
    print('Manifest:', result['manifest'], flush=True)
    return int(any(x['status'] == 'error' for x in result['results']))


if __name__ == '__main__':
    raise SystemExit(main())
