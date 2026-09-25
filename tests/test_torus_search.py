"""Search safety acceptance tests for T(8,9) and T(9,10)."""
import importlib.util
import hashlib
import json
import os
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
            self.assertIn('T8_9 [timeout]', (Path(out) / 'RUNLOG.md').read_text())
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

    def test_supervisor_rejects_completion_observed_after_deadline_despite_early_worker_stamp(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as stage:
            job = dict(name='T9_10', target=20, stage=stage)
            (Path(stage) / 'outcome.json').write_text(json.dumps(dict(
                status='not_found_within_budget', message='early stamp', completed_monotonic=9)))
            with patch.object(m.time, 'monotonic', return_value=10.1):
                observed = m.collect_outcome(job, 0, 10, Path(out))
            self.assertEqual(observed['status'], 'timeout')

    def test_supervisor_rolls_back_publication_that_crosses_deadline(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as stage:
            job = dict(name='T8_9', target=18, stage=stage)
            staged = Path(stage) / 'T8_9_equilateral_18sticks.txt'
            staged.write_text('validated bytes')
            (Path(stage) / 'outcome.json').write_text(json.dumps(dict(
                status='certified', message='early', completed_monotonic=9,
                sha256=hashlib.sha256(staged.read_bytes()).hexdigest())))
            with patch.object(m.time, 'monotonic', side_effect=(9, 9, 10.1)):
                result = m.collect_outcome(job, 0, 10, Path(out))
            self.assertEqual(result['status'], 'timeout')
            self.assertFalse((Path(out) / staged.name).exists())

    def test_supervisor_never_overwrites_peer_created_at_publication(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory(dir=out) as stage:
            job = dict(name='T8_9', target=18, stage=stage)
            staged = Path(stage) / 'T8_9_equilateral_18sticks.txt'
            staged.write_text('our validated bytes')
            (Path(stage) / 'outcome.json').write_text(json.dumps(dict(
                status='certified', message='early', completed_monotonic=time.monotonic(),
                sha256=hashlib.sha256(staged.read_bytes()).hexdigest())))
            target = Path(out) / staged.name
            real_link = os.link
            def peer_first(source, destination):
                target.write_text('peer result')
                return real_link(source, destination)
            with patch.object(m.os, 'link', side_effect=peer_first):
                result = m.collect_outcome(job, 0, time.monotonic() + 10, Path(out))
            self.assertEqual(result['status'], 'error')
            self.assertEqual(target.read_text(), 'peer result')

    def test_supervisor_accepts_identical_existing_coordinates_without_replacing_them(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory(dir=out) as stage:
            job = dict(name='T8_9', target=18, stage=stage)
            staged = Path(stage) / 'T8_9_equilateral_18sticks.txt'
            staged.write_text('validated bytes')
            target = Path(out) / staged.name
            target.write_bytes(staged.read_bytes())
            original_inode = target.stat().st_ino
            (Path(stage) / 'outcome.json').write_text(json.dumps(dict(
                status='certified', message='early', completed_monotonic=time.monotonic(),
                sha256=hashlib.sha256(staged.read_bytes()).hexdigest())))
            result = m.collect_outcome(job, 0, time.monotonic() + 10, Path(out))
            self.assertEqual(result['status'], 'certified')
            self.assertEqual(result['coordinates'], str(target))
            self.assertEqual(target.stat().st_ino, original_inode)

    def test_supervisor_never_removes_peer_replacement_on_timeout(self):
        m = script('08_torus_batch')
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory(dir=out) as stage:
            job = dict(name='T8_9', target=18, stage=stage)
            staged = Path(stage) / 'T8_9_equilateral_18sticks.txt'
            staged.write_text('our validated bytes')
            (Path(stage) / 'outcome.json').write_text(json.dumps(dict(
                status='certified', message='early', completed_monotonic=9,
                sha256=hashlib.sha256(staged.read_bytes()).hexdigest())))
            target = Path(out) / staged.name
            peer = Path(out) / 'peer.txt'
            peer.write_text('peer result')
            calls = iter((9, 9, 10.1))
            def tick():
                now = next(calls)
                if now > 10:
                    os.replace(peer, target)
                return now
            with patch.object(m.time, 'monotonic', side_effect=tick):
                result = m.collect_outcome(job, 0, 10, Path(out))
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(target.read_text(), 'peer result')

    def test_direct_batch_rejects_nonfinite_deadline(self):
        m = script('08_torus_batch')
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError), patch.object(
                    m.parallel, 'effective_cpus', side_effect=AssertionError('reached workers')):
                m.run_batch([8], value, 1, Path(tempfile.gettempdir()))

    def test_validation_rejects_every_incomplete_certificate(self):
        m = script('04_torus_family')
        V = np.ones((18, 3))
        h = dict(seifert_genus=28, tau=-28, fibered=True, L_space_knot=True, total_rank=15)
        good = dict(mr_ratio=lambda x: (0.5, 0.01, 0.1),
                    mr_certificate_mp=lambda x: (0.01, 0.1, 0.02, True),
                    verify_torus=lambda x, p, q, nproj=4: (True, h, 63))
        with patch.multiple(m, **good):
            self.assertTrue(m.validate_candidate(V, 8, 9)[3])
            with self.assertRaises(ValueError):
                m.validate_candidate(V[:17], 8, 9)
            with self.assertRaises(ValueError):
                m.validate_candidate(np.full((18, 3), np.nan), 8, 9)
        for changed in (
            dict(mr_ratio=lambda x: (1.0, 0.01, 0.1)),
            dict(mr_certificate_mp=lambda x: (0.01, 0.1, 0.02, False)),
            dict(verify_torus=lambda x, p, q, nproj=4: (False, h, 63)),
            dict(verify_torus=lambda x, p, q, nproj=4: (True, h, 62)),
            *[dict(verify_torus=lambda x, p, q, nproj=4, field=field, value=value: (True, {**h, field: value}, 63))
              for field, value in [('seifert_genus', 27), ('tau', 0),
                                   ('fibered', False), ('L_space_knot', False), ('total_rank', 14)]],
        ):
            with self.subTest(changed=changed), patch.multiple(m, **(good | changed)):
                with self.assertRaises(ValueError):
                    m.validate_candidate(V, 8, 9)

    def test_alexander_rank_formula_matches_known_torus_cases(self):
        m = script('04_torus_family')
        for p, q, rank in ((3, 7, 9), (3, 8, 11), (8, 9, 15), (9, 10, 17)):
            with self.subTest(p=p, q=q):
                self.assertEqual(m.torus_alexander_rank(p, q), rank)

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
