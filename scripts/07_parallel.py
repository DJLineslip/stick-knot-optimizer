"""Bounded, spawn-based ten-stick searches. Certificates are numerical only."""
import os

# Set before importing any scientific library, including in spawned interpreters.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import math
from pathlib import Path
import re
import sys

import contextlib
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
import zlib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_KNOTS = ['K13n285', 'K13n602', 'K13n608', 'K13n1192', 'K13n5018']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knots', default=','.join(DEFAULT_KNOTS), help='comma-separated canonical knot names')
    parser.add_argument('--budget', type=float, default=1800, help='hard wall seconds per knot, including worker startup and validation')
    parser.add_argument('--workers', type=int, help='maximum workers, capped by effective CPU quota and five')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args(argv)
    args.knots = args.knots.split(',')
    if any(not re.fullmatch(r'K[1-9][0-9]*[an][1-9][0-9]*', n) for n in args.knots):
        parser.error('malformed knot name (example: K13n285)')
    if len(set(args.knots)) != len(args.knots):
        parser.error('duplicate knot names')
    if not math.isfinite(args.budget) or args.budget <= 0:
        parser.error('budget must be finite and positive')
    if args.workers is not None and args.workers < 1:
        parser.error('workers must be positive')
    return args


def cpu_limit(affinity, quotas):
    limits = [affinity, 5]
    limits.extend(max(1, q // p) for q, p in quotas if q > 0 and p > 0)
    return max(1, min(limits))


def effective_cpus():
    """Respect affinity and visible cgroup v1/v2 quotas, rounding down."""
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    roots = {Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu,cpuacct')}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, path = line.split(':', 2)
            base = Path('/sys/fs/cgroup') if not controllers else Path('/sys/fs/cgroup') / controllers
            child = base / path.lstrip('/')
            roots.update([child, *[p for p in child.parents if p == base or base in p.parents]])
    except (OSError, ValueError):
        pass
    quotas = []
    for root in roots:
        try:
            q, p = (root / 'cpu.max').read_text().split()
            if q != 'max':
                quotas.append((int(q), int(p)))
        except (OSError, ValueError):
            pass
        try:
            quotas.append((int((root / 'cpu.cfs_quota_us').read_text()), int((root / 'cpu.cfs_period_us').read_text())))
        except (OSError, ValueError):
            pass
    return cpu_limit(affinity, quotas)



def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def git_commit(path):
    try:
        top = subprocess.check_output(['git', '-C', str(path), 'rev-parse', '--show-toplevel'], stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() != Path(path).resolve():
            return None
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(names, budget, workers):
    packages = {}
    for name in ('numpy', 'scipy', 'numba', 'mpmath', 'snappy', 'spherogram'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = Path(os.environ.get('EQUISTICK_DATA', ROOT / 'stick-knot-gen')).resolve()
    return dict(command=[sys.executable, *sys.argv], seeds={n: zlib.crc32(n.encode()) for n in names},
                projection_seeds=[1, 2, 3], budget_seconds=budget, workers=workers,
                git_commit=git_commit(ROOT), packages=packages, data_path=str(data),
                data_commit=git_commit(data), python=sys.version,
                stick_number_source='TEN_STICK_19: Cantarella et al., JKTR 2026; otherwise exact_values.csv')


def logged_worker(worker, job):
    # Redirect OS descriptors as well as Python streams to include native diagnostics.
    with open(job['log'], 'a', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print('START', json.dumps(job), flush=True)
            try:
                worker(job)
            except BaseException as exc:
                traceback.print_exc()
                atomic_json(Path(job['stage']) / 'outcome.json',
                            dict(status='error', message=f'{type(exc).__name__}: {exc}'))
            outcome = Path(job['stage']) / 'outcome.json'
            if outcome.exists():
                result = json.loads(outcome.read_text())
                result.setdefault('completed_monotonic', time.monotonic())
                atomic_json(outcome, result)


def stop_process(process):
    process.terminate()
    process.join(0.2)
    if process.is_alive():
        process.kill()
        process.join()


def record_result(out, manifest, job, result, elapsed, log_name='tenstick.log'):
    """Append only from the supervisor; update the per-run manifest atomically."""
    result.update(name=job['name'], elapsed_seconds=elapsed, log=job['log'])
    with open(job['log'], 'a') as f:
        f.write('END ' + json.dumps(result) + '\n')
    with (out / log_name).open('a') as f:
        f.write(f"{job['name']} [{result['status']}] {result['message']}\n")
    manifest['results'].append(result)
    atomic_json(Path(manifest['manifest']), manifest)
    with (out / 'RUNLOG.md').open('a') as f:
        f.write(f"\n{job['name']} [{result['status']}] "
                + json.dumps(result, sort_keys=True) + '\n')
    print(f"{job['name']} [{result['status']}] {result['message']}", flush=True)


def collect_outcome(job, exitcode, deadline, out):
    """Publish only a completed, in-budget validation of the staged file."""
    if time.monotonic() >= deadline:
        return dict(status='timeout', message='not found within budget; completion observed after deadline')
    try:
        result = json.loads((Path(job['stage']) / 'outcome.json').read_text())
        if exitcode != 0:
            raise RuntimeError(f'worker exit {exitcode}')
        if result['status'] not in ('certified', 'not_found_within_budget', 'error', 'missing_data'):
            raise ValueError('unknown worker status')
        if result['status'] not in ('error', 'missing_data') and result['completed_monotonic'] > deadline:
            return dict(status='timeout', message=f"not found within budget; completion exceeded hard wall deadline")
        if result['status'] == 'certified':
            filename = f"{job['name']}_equilateral_{job['target']}sticks.txt"
            stage = Path(job['stage']) / filename
            target = out / filename
            if time.monotonic() >= deadline:
                return dict(status='timeout', message='not found within budget; publication deadline passed')
            # stage is created beneath out, so an atomic hard link publishes
            # the exact validated inode without replacing an existing result.
            os.link(stage, target)
            if time.monotonic() >= deadline:
                if target.exists() and os.path.samefile(stage, target):
                    target.unlink()
                return dict(status='timeout', message='not found within budget; publication exceeded deadline')
            if not os.path.samefile(stage, target):
                raise ValueError('published coordinates replaced by another writer')
            result['coordinates'] = str(target)
        return result
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return dict(status='error', message=f'worker exit {exitcode}: {exc}')


def run_batch(names, budget, workers, out, worker, target=10,
              log_name='tenstick.log', extra_provenance=None):
    """Parent owns all public files. Each spawned knot has a hard deadline.

    Timeouts are search logs only. No file from an interrupted worker is published.
    """
    workers = min(workers or effective_cpus(), effective_cpus())
    out = Path(out).resolve()
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    manifest_path = logs / (run_id + '.json')
    manifest = dict(run_id=run_id, manifest=str(manifest_path), **provenance(names, budget, workers), results=[])
    if extra_provenance:
        manifest.update(extra_provenance)
    with (out / 'RUNLOG.md').open('a') as f:
        f.write('\n## ' + run_id + '\n\n```json\n' + json.dumps(manifest, indent=2) + '\n```\n')
    atomic_json(manifest_path, manifest)
    context = mp.get_context('spawn')
    pending = list(names)
    active = []
    try:
        while pending or active:
            while pending and len(active) < workers:
                name = pending.pop(0)
                stage = Path(tempfile.mkdtemp(prefix='.candidate_', dir=out))
                job = dict(name=name, budget=budget, target=target, stage=str(stage),
                           log=str(logs / f'{run_id}_{name}.log'))
                Path(job['log']).touch()
                if worker is search_worker:
                    source = Path(manifest['data_path']) / 'stick_number' / 'mseq_knots' / (name + '.txt')
                    if not source.is_file():
                        record_result(out, manifest, job,
                                      dict(status='missing_data', message='no starting data in Eddy repository'), 0., log_name)
                        shutil.rmtree(stage)
                        continue
                process = context.Process(target=logged_worker, args=(worker, job))
                start = time.monotonic()
                process.start()
                active.append((process, job, start))
            for process, job, start in active[:]:
                elapsed = time.monotonic() - start
                if process.is_alive() and elapsed < budget:
                    continue
                if process.is_alive():
                    stop_process(process)
                    result = dict(status='timeout', message=f'not found within budget ({budget:g}s); hard wall deadline')
                else:
                    process.join()
                    result = collect_outcome(job, process.exitcode, start + budget, out)
                record_result(out, manifest, job, result, time.monotonic() - start, log_name)
                shutil.rmtree(job['stage'])
                process.close()
                active.remove((process, job, start))
            if active:
                time.sleep(0.01)
    finally:
        for process, job, _ in active:
            if process.is_alive():
                stop_process(process)
            process.close()
            shutil.rmtree(job['stage'], ignore_errors=True)
    return manifest


def warmup():
    """Compile caches serially in the parent before any spawn."""
    import numpy as np
    from equistick.geometry import min_dist, safe_move
    from equistick.reduce import penalties
    from equistick.optimize import dist_jac
    polygon = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    min_dist(polygon)
    safe_move(polygon, 0, polygon[0] + np.array([0., 0., .01]), 1e-9)
    penalties(polygon)
    dist_jac(polygon)


def search_worker(job):
    import numpy as np
    ten = importlib.import_module('05_tenstick')
    name, stage, target = job['name'], Path(job['stage']), job['target']
    if not ten.eddy_available(name):
        result = dict(status='missing_data', message='no starting data in Eddy repository')
    else:
        message = ten.run(name, budget=job['budget'], target=target, out=str(stage))
        candidate = stage / f'{name}_equilateral_{target}sticks.txt'
        if candidate.exists():
            # Re-read the 17-digit file that will actually be published.
            coordinates = np.loadtxt(candidate)
            certificate = ten.validate_candidate(coordinates, name, target)
            result = dict(status='certified', message=message,
                          certificate=[str(x) for x in certificate[:3]] + [bool(certificate[3])],
                          projection_seeds=[1, 2, 3])
        elif message.startswith('not found:'):
            result = dict(status='not_found_within_budget', message=message)
        else:
            raise RuntimeError('search did not produce a valid outcome: ' + message)
    result['completed_monotonic'] = time.monotonic()
    print(json.dumps(result), flush=True)
    atomic_json(stage / 'outcome.json', result)


def main(argv=None):
    args = parse_args(argv)
    workers = min(args.workers or effective_cpus(), effective_cpus())
    print(f'Spawn workers: {workers}; per-knot hard wall budget: {args.budget:g}s', flush=True)
    warmup()
    manifest = run_batch(args.knots, args.budget, workers, args.out, worker=search_worker)
    print('Manifest:', manifest['manifest'], flush=True)
    return int(any(r['status'] in ('error', 'missing_data') for r in manifest['results']))


if __name__ == '__main__':
    raise SystemExit(main())
