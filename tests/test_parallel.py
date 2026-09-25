import importlib
import sys
from pathlib import Path
import unittest
import contextlib
import io
import tempfile
import time
import json
import os

def hanging_worker(job):
    print("worker started", flush=True)
    time.sleep(60)

def crashing_worker(job):
    os._exit(7)

def empty_worker(job):
    m = importlib.import_module("07_parallel")
    m.atomic_json(Path(job["stage"]) / "outcome.json", {"status": "not_found_within_budget", "message": "not found within budget"})

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

class ParallelTests(unittest.TestCase):
    def setUp(self):
        for redirect in (contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            redirect.__enter__()
            self.addCleanup(redirect.__exit__, None, None, None)

    def runner(self):
        self.assertTrue((Path(__file__).resolve().parents[1] / 'scripts/07_parallel.py').exists(), 'runner missing')
        return importlib.import_module('07_parallel')

    def test_cli_and_cpu_quota(self):
        m = self.runner()
        args = m.parse_args([])
        self.assertEqual(args.budget, 1800)
        self.assertEqual(len(args.knots), 5)
        for argv in [['--budget', 'nan'], ['--budget', '0'], ['--budget', '-1'],
                     ['--knots', 'K13n285,K13n285'], ['--knots', '../bad'],
                     ['--knots', 'K013n285'], ['--workers', '0']]:
            with self.assertRaises(SystemExit):
                m.parse_args(argv)
        self.assertEqual(m.cpu_limit(affinity=8, quotas=[(100000, 100000)]), 1)
        self.assertEqual(m.cpu_limit(affinity=8, quotas=[(550000, 100000)]), 5)
        self.assertLessEqual(m.effective_cpus(), 5)

    def test_hard_timeout_and_parent_logging(self):
        m = self.runner()
        self.assertTrue(hasattr(m, 'run_batch'), 'supervisor missing')
        with tempfile.TemporaryDirectory() as out:
            start = time.monotonic()
            manifest = m.run_batch(['K13n285'], 0.3, 5, Path(out), worker=hanging_worker)
            self.assertLess(time.monotonic() - start, 3)
            self.assertEqual(manifest['results'][0]['status'], 'timeout')
            self.assertEqual(manifest['workers'], m.effective_cpus())
            self.assertIn('not found within budget', (Path(out) / 'tenstick.log').read_text())
            self.assertEqual(len(list((Path(out) / 'logs').glob('*.log'))), 1)
            stored = json.loads(Path(manifest['manifest']).read_text())
            self.assertEqual(stored['results'], manifest['results'])
            runlog = (Path(out) / 'RUNLOG.md').read_text()
            for field in ['seeds', 'command', 'git_commit', 'packages', 'data_commit']:
                self.assertIn(field, runlog)

    def test_worker_crash_is_error_and_runlog_appends(self):
        m = self.runner()
        self.assertTrue(hasattr(m, 'run_batch'), 'supervisor missing')
        with tempfile.TemporaryDirectory() as out:
            a = m.run_batch(['K13n285'], 2, 1, Path(out), worker=crashing_worker)
            self.assertEqual(a['results'][0]['status'], 'error')
            before = (Path(out) / 'RUNLOG.md').read_text()
            b = m.run_batch(['K13n602'], 2, 1, Path(out), worker=empty_worker)
            self.assertEqual(b['results'][0]['status'], 'not_found_within_budget')
            self.assertTrue((Path(out) / 'RUNLOG.md').read_text().startswith(before))
            self.assertEqual(len((Path(out) / 'tenstick.log').read_text().splitlines()), 2)

    def test_search_worker_revalidates_saved_file_and_missing_data(self):
        from unittest.mock import patch
        import numpy as np
        m = self.runner()
        self.assertTrue(hasattr(m, 'search_worker'), 'search worker missing')
        ten = importlib.import_module('05_tenstick')
        with tempfile.TemporaryDirectory() as stage:
            job = dict(name='K13n285', budget=0.01, target=10, stage=stage)
            with patch.object(ten, 'eddy_available', return_value=False):
                m.search_worker(job)
            self.assertEqual(json.loads((Path(stage) / 'outcome.json').read_text())['status'], 'missing_data')
            def fake_run(name, budget, target, out):
                np.savetxt(Path(out) / f'{name}_equilateral_{target}sticks.txt', np.ones((9, 3)))
                return 'FOUND fake'
            with patch.object(ten, 'eddy_available', return_value=True), patch.object(ten, 'run', side_effect=fake_run):
                with self.assertRaises(ValueError):
                    m.search_worker(job)

    def test_warmup_compiles_all_four_kernels(self):
        m = self.runner()
        self.assertTrue(hasattr(m, 'warmup'), 'parent warmup missing')
        m.warmup()
        from equistick.geometry import min_dist, safe_move
        from equistick.reduce import penalties
        from equistick.optimize import dist_jac
        for kernel in [min_dist, safe_move, penalties, dist_jac]:
            self.assertTrue(kernel.signatures)

    def test_cli_help_and_spawn_missing_data(self):
        import subprocess
        with tempfile.TemporaryDirectory() as out:
            script = str(Path(__file__).resolve().parents[1] / 'scripts/07_parallel.py')
            help_result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
            self.assertIn('--budget', help_result.stdout)
            env = dict(os.environ, EQUISTICK_DATA=out)
            result = subprocess.run([sys.executable, script, '--knots', 'K13n285', '--budget', '0.001', '--out', out], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            manifest = json.loads(next((Path(out) / 'logs').glob('*.json')).read_text())
            self.assertEqual(manifest['results'][0]['status'], 'missing_data')

    def test_provenance_does_not_report_parent_repo_as_data_commit(self):
        m = self.runner()
        with tempfile.TemporaryDirectory(dir=m.ROOT) as data:
            self.assertIsNone(m.git_commit(data))

    def test_late_completion_never_published(self):
        m = self.runner()
        self.assertTrue(hasattr(m, 'collect_outcome'), 'deadline-aware collection missing')
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as stage:
            job = dict(name='K13n285', target=10, stage=stage)
            candidate = Path(stage) / 'K13n285_equilateral_10sticks.txt'
            candidate.write_text('not valid coordinates')
            m.atomic_json(Path(stage) / 'outcome.json', dict(status='certified', message='late', completed_monotonic=11))
            result = m.collect_outcome(job, 0, 10, Path(out))
            self.assertEqual(result['status'], 'timeout')
            self.assertFalse((Path(out) / candidate.name).exists())

    def test_existing_certificate_is_rechecked_and_atomically_published(self):
        from unittest.mock import patch
        import shutil
        m = self.runner()
        self.assertTrue(hasattr(m, 'collect_outcome'), 'collection missing')
        ten = importlib.import_module('05_tenstick')
        filename = 'K11n71_equilateral_10sticks.txt'
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as stage:
            job = dict(name='K11n71', budget=1, target=10, stage=stage)
            def existing_result(name, budget, target, out):
                shutil.copyfile(m.ROOT / 'results' / filename, Path(out) / filename)
                return 'FOUND existing test fixture, not a search'
            with patch.object(ten, 'eddy_available', return_value=True), patch.object(ten, 'run', side_effect=existing_result):
                m.search_worker(job)
            result = m.collect_outcome(job, 0, time.monotonic() + 1, Path(out))
            self.assertEqual(result['status'], 'certified')
            self.assertEqual((Path(out) / filename).read_bytes(), (m.ROOT / 'results' / filename).read_bytes())
            self.assertTrue((Path(stage) / filename).samefile(Path(out) / filename))
