import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import shutil
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class Torus37Tests(unittest.TestCase):
    def runner(self):
        return importlib.import_module('08_torus37')

    def test_explicit_smooth_samples_identify_both_torus_knots(self):
        from equistick.geometry import min_dist
        from equistick.invariants import is_torus
        m = self.runner()
        for q in (7, 8):
            V = m.torus_start(q)
            self.assertEqual(V.shape, (24, 3))
            self.assertGreater(min_dist(V), 0)
            self.assertTrue(is_torus(V, 3, q, nproj=4, seed=123))
            self.assertFalse(is_torus(V, 3, 15 - q, nproj=4, seed=123))
            self.assertTrue(m.validate_start(V, q))

    def test_start_rejects_wrong_type_or_zero_clearance(self):
        m = self.runner()
        V = m.torus_start(7)
        self.assertRaises(m.CandidateRejected, m.validate_start, V, 8)
        V[0] = V[4]
        self.assertRaises(m.CandidateRejected, m.validate_start, V, 7)

    def test_candidate_rejects_uncertified_and_wrong_type(self):
        m = self.runner()
        V = np.loadtxt(ROOT / 'results/T5_6_equilateral_12sticks.txt')
        self.assertRaises(m.CandidateRejected, m.validate_candidate, V, 7)
        with patch.object(m, 'verify_torus', return_value=(True, dict(seifert_genus=6, fibered=True,
                         L_space_knot=True, tau=6, total_rank=11), 14)):
            self.assertRaises(m.CandidateRejected, m.validate_candidate, V, 7)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertRaises(m.CandidateRejected, m.validate_candidate, np.ones((12, 3)), 7)
        self.assertRaises(m.CandidateRejected, m.validate_candidate, np.ones((11, 3)), 7)

    def test_cli_rejects_invalid_names_and_budget(self):
        m = self.runner()
        args = m.parse_args([])
        self.assertEqual(args.knots, ['T3_7', 'T3_8'])
        self.assertEqual(args.budget, 1800)
        for argv in (['--knots', 'T3_9'], ['--knots', '../T3_7'],
                     ['--knots', 'T3_7,T3_7'], ['--budget', 'nan'],
                     ['--budget', '0'], ['--workers', '0']):
            with self.assertRaises(SystemExit):
                m.parse_args(argv)

    def test_worker_does_not_publish_unvalidated_saved_coordinates(self):
        m = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            job = dict(name='T3_7', budget=1., target=12, stage=directory)
            def bad_run(q, budget, stage):
                np.savetxt(Path(stage) / 'T3_7_equilateral_12sticks.txt', np.ones((12, 3)))
                return 'candidate'
            with patch.object(m, 'run_search', side_effect=bad_run):
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    with self.assertRaises(m.CandidateRejected):
                        m.search_worker(job)
            self.assertFalse((Path(directory) / 'outcome.json').exists())

    def test_committed_coordinates_pass_saved_file_validation(self):
        m = self.runner()
        for q in (7, 8):
            path = ROOT / 'results' / f'T3_{q}_equilateral_12sticks.txt'
            V = np.loadtxt(path)
            result = m.validate_candidate(V, q)
            self.assertTrue(result['mp_certified'])
            self.assertEqual(result['alexander_projections'], 4)
            self.assertEqual(result['hfk']['seifert_genus'], q - 1)

    def test_worker_rechecks_bytes_written_to_stage(self):
        m = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            job = dict(name='T3_8', budget=1., target=12, stage=directory)
            filename = 'T3_8_equilateral_12sticks.txt'
            def copy_fixture(q, budget, stage):
                shutil.copyfile(ROOT / 'results' / filename, Path(stage) / filename)
                return 'candidate'
            with patch.object(m, 'run_search', side_effect=copy_fixture):
                m.search_worker(job)
            outcome = json.loads((Path(directory) / 'outcome.json').read_text())
            self.assertEqual(outcome['status'], 'certified')
            self.assertTrue(outcome['certificate']['mp_certified'])

    def test_supervisor_supports_twelve_sticks_and_separate_log(self):
        m = importlib.import_module('07_parallel')
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as staged:
            job = dict(name='T3_7', target=12, stage=staged)
            candidate = Path(staged) / 'T3_7_equilateral_12sticks.txt'
            candidate.write_text('staged')
            m.atomic_json(Path(staged) / 'outcome.json', dict(status='certified',
                          message='checked by worker', completed_monotonic=10))
            with patch.object(m.time, 'monotonic', return_value=10.5):
                result = m.collect_outcome(job, 0, 11, Path(directory))
            self.assertEqual(result['status'], 'certified')
            self.assertEqual((Path(directory) / candidate.name).read_text(), 'staged')
            self.assertFalse(candidate.exists())
            candidate.write_text('conflicting saved coordinates')
            with patch.object(m.time, 'monotonic', return_value=10.5):
                conflict = m.collect_outcome(job, 0, 11, Path(directory))
            self.assertEqual(conflict['status'], 'error')
            self.assertEqual((Path(directory) / candidate.name).read_text(), 'staged')
            # A timeout must never publish a staged file.
            candidate.write_text('late')
            with patch.object(m.time, 'monotonic', return_value=10.5):
                result = m.collect_outcome(job, 0, 9, Path(directory))
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual((Path(directory) / candidate.name).read_text(), 'staged')
            candidate.write_text('early stamp, late observation')
            with patch.object(m.time, 'monotonic', return_value=11.5):
                result = m.collect_outcome(job, 0, 11, Path(directory))
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual((Path(directory) / candidate.name).read_text(), 'staged')
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as staged:
            job = dict(name='T3_7', target=12, stage=staged)
            candidate = Path(staged) / 'T3_7_equilateral_12sticks.txt'
            candidate.write_text('new candidate')
            m.atomic_json(Path(staged) / 'outcome.json', dict(status='certified',
                          message='in time before publication', completed_monotonic=9))
            with patch.object(m.time, 'monotonic', side_effect=(9, 9, 10.1)):
                result = m.collect_outcome(job, 0, 10, Path(directory))
            self.assertEqual(result['status'], 'timeout')
            self.assertFalse((Path(directory) / candidate.name).exists())
        with tempfile.TemporaryDirectory() as directory:
            manifest = m.run_batch(['T3_7'], 0.15, 1, Path(directory),
                                   worker=importlib.import_module('test_parallel').hanging_worker,
                                   target=12, log_name='torus37.log')
            self.assertEqual(manifest['results'][0]['status'], 'timeout')
            self.assertIn('T3_7', (Path(directory) / 'torus37.log').read_text())
            self.assertIn('T3_7 [timeout]', (Path(directory) / 'RUNLOG.md').read_text())
            self.assertFalse((Path(directory) / 'tenstick.log').exists())


if __name__ == '__main__':
    unittest.main()
