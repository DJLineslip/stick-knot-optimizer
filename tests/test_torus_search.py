"""Search safety acceptance tests for T(8,9) and T(9,10)."""
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def hanging_worker(job):
    time.sleep(60)


def fake_certified_worker(job):
    # An invalid staged file may never be published, even if a worker lies.
    stage = Path(job['stage'])
    (stage / f"{job['name']}_equilateral_{job['target']}sticks.txt").write_text('invalid')
    (stage / 'outcome.json').write_text(json.dumps(dict(status='certified', message='fake',
                                                       completed_monotonic=time.monotonic())))


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TorusSearchTests(unittest.TestCase):
    def test_direct_publish_does_not_replace_an_existing_result(self):
        m = script('04_torus_family')
        with tempfile.TemporaryDirectory() as out:
            target = Path(out) / 'T8_9_equilateral_18sticks.txt'
            target.write_text('preexisting certified coordinates')
            with patch.object(m, 'validate_candidate', return_value=((0, 1, 1, True), {}, 70, (0, 0, 1))):
                with self.assertRaisesRegex(ValueError, 'existing'):
                    m.publish_candidate(np.ones((18, 3)), 8, out)
            self.assertEqual(target.read_text(), 'preexisting certified coordinates')

    def test_known_torus_file_passes_full_saved_coordinate_gate(self):
        m = script('04_torus_family')
        source = ROOT / 'results/T7_8_equilateral_16sticks.txt'
        V = np.loadtxt(source)
        with tempfile.TemporaryDirectory() as out:
            cert, h, _, ratio = m.publish_candidate(V, 7, out)
            self.assertTrue(cert[3])
            self.assertEqual(h['seifert_genus'], 21)
            self.assertLess(ratio[0], 1)
            self.assertEqual((Path(out) / source.name).read_bytes(), source.read_bytes())

    def test_documented_starts_are_embedded_torus_knots(self):
        from equistick.torus import STARTS, torus_poly
        from equistick.geometry import min_dist
        from equistick.invariants import is_torus
        for p in (8, 9):
            with self.subTest(p=p):
                V = torus_poly(p, *STARTS[p])
                self.assertGreater(min_dist(V), 1e-3)
                self.assertTrue(is_torus(V, p, p + 1, nproj=4, seed=123))

    def test_supervisor_rejects_invalid_budget_knots_and_workers(self):
        m = script('08_torus_batch')
        self.assertEqual(m.parse_args([]).budget, 1800)
        self.assertEqual(m.parse_args([]).workers, 1)
        for argv in (['--budget', '1801'], ['--budget', 'nan'], ['--budget', '0'],
                     ['--knots', 'T8_9,T8_9'], ['--knots', 'T8_10'],
                     ['--workers', '3'], ['--run-index', '-1']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                m.parse_args(argv)

    def test_supervisor_hard_timeout_and_provenance(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out:
            started = time.monotonic()
            result = m.run_batch([8], 0.2, 1, Path(out), hanging_worker, 2, 1)
            self.assertLess(time.monotonic() - started, 4)
            self.assertEqual(result['results'][0]['status'], 'timeout')
            self.assertEqual(result['seeds']['T8_9'], m.seed_for('T8_9') + 2)
            self.assertEqual(result['budget_seconds'], 0.2)
            self.assertEqual(result['projection_count'], 4)
            self.assertEqual(result['projection_seed'], 123)
            self.assertIn('Jin', result['stick_number_source'])
            self.assertIn('not found within budget', (Path(out) / 'torus_search.log').read_text())
            self.assertEqual(json.loads(Path(result['manifest']).read_text())['results'], result['results'])
            self.assertIn('packages', (Path(out) / 'RUNLOG.md').read_text())
            self.assertEqual(len(list((Path(out) / 'logs').glob('*.log'))), 1)

    def test_supervisor_refuses_unvalidated_stage_and_late_completion(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out:
            result = m.run_batch([9], 3, 1, Path(out), fake_certified_worker, 0, 1)
            self.assertEqual(result['results'][0]['status'], 'error')
            self.assertFalse(list(Path(out).glob('*equilateral*.txt')))
            with tempfile.TemporaryDirectory() as stage:
                job = dict(name='T9_10', target=20, stage=stage)
                staged = Path(stage) / 'T9_10_equilateral_20sticks.txt'
                staged.write_text('invalid')
                (Path(stage) / 'outcome.json').write_text(json.dumps(dict(status='certified',
                    message='late', completed_monotonic=11)))
                late = m.collect_outcome(job, 0, 10, Path(out))
                self.assertEqual(late['status'], 'timeout')
                self.assertFalse((Path(out) / staged.name).exists())

    def test_validation_rejects_every_incomplete_certificate(self):
        m = script('04_torus_family')
        V = np.ones((18, 3))
        h = dict(seifert_genus=28, tau=-28, fibered=True, L_space_knot=True)
        good = dict(mr_ratio=lambda x: (0.5, 0.01, 0.1),
                    mr_certificate_mp=lambda x: (0.01, 0.1, 0.02, True),
                    verify_torus=lambda x, p, q, nproj=4: (True, h, 70))
        with patch.multiple(m, **good):
            self.assertTrue(m.validate_candidate(V, 8, 9)[3])
            with self.assertRaises(ValueError):
                m.validate_candidate(V[:17], 8, 9)
            with self.assertRaises(ValueError):
                m.validate_candidate(np.full((18, 3), np.nan), 8, 9)
        for changed in (
            dict(mr_ratio=lambda x: (1.0, 0.01, 0.1)),
            dict(mr_certificate_mp=lambda x: (0.01, 0.1, 0.02, False)),
            dict(verify_torus=lambda x, p, q, nproj=4: (False, h, 70)),
            *[dict(verify_torus=lambda x, p, q, nproj=4, field=field: (True, {**h, field: value}, 70))
              for field, value in [('seifert_genus', 27), ('tau', 0),
                                   ('fibered', False), ('L_space_knot', False)]],
        ):
            with self.subTest(changed=changed), patch.multiple(m, **(good | changed)):
                with self.assertRaises(ValueError):
                    m.validate_candidate(V, 8, 9)

    def test_empty_scan_is_logged_without_index_error_or_saved_coordinates(self):
        m = script('04_torus_family')
        with tempfile.TemporaryDirectory() as out, patch.dict(m.STARTS, {}, clear=True), patch.object(m, 'symmetric_scan', return_value=[]) as scan:
            self.assertIsNone(m.run(8, out=out, scan_trials=2))
            self.assertFalse(list(Path(out).glob('*.txt')))
            self.assertEqual(scan.call_args.kwargs['seed'], m.seed_for('T8_9'))

    def test_saved_coordinates_are_revalidated_after_serialization(self):
        m = script('04_torus_family')
        V = np.ones((18, 3))
        with tempfile.TemporaryDirectory() as out, patch.multiple(m,
                symmetric_scan=lambda *args, **kwargs: [(1.1, 0.8, 0.5, 0.01)],
                torus_poly=lambda *args: V.copy(), agitate=lambda *args: None,
                min_dist=lambda x: 0.1, normalize=lambda x: x,
                clearance_floor_solve=lambda *args, **kwargs: V,
                mr_ratio=lambda x: (0.1, 0.01, 0.1), is_torus=lambda *args: True,
                polish=lambda x: x):
            with patch.object(m, 'validate_candidate', side_effect=ValueError('rejected saved coords')):
                self.assertIsNone(m.run(8, trials=1, out=out))
            self.assertFalse(list(Path(out).glob('*.txt')))

    def test_repeated_start_seed_is_stable_and_run_index_changes_it(self):
        m = script('04_torus_family')
        seeds = []
        with tempfile.TemporaryDirectory() as out, patch.dict(m.STARTS, {}, clear=True), patch.object(m, 'symmetric_scan', side_effect=lambda p, trials, seed: seeds.append(seed) or []):
            for i in (0, 0, 1):
                m.run(9, out=out, scan_trials=1, run_index=i)
        self.assertEqual(seeds[0], seeds[1])
        self.assertNotEqual(seeds[0], seeds[2])


if __name__ == '__main__':
    unittest.main()
