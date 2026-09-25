import importlib.util
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def load_script(name):
    spec = importlib.util.spec_from_file_location('script_' + name, ROOT / 'scripts' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_run_logs_reduction_and_homotopy_progress(self):
        m = load_script('05_tenstick')
        v = np.arange(30, dtype=float).reshape(10, 3)
        with tempfile.TemporaryDirectory() as out, patch.multiple(m,
                eddy_available=lambda n: True, load_eddy=lambda n: v,
                reduce_to=lambda *a: v, identify=lambda *a, **k: 'K13n285',
                fatten=lambda *a: v, homotopy_equalize=lambda *a, **k: (v, 1, 1, True),
                mr_ratio=lambda v: (0.1,), mr_certificate_mp=lambda v: (0, 1, 1, True)):
            self.assertIn('FOUND', m.run('K13n285', budget=1, out=out))
            for label in ['seed=', 'reduction attempt', 'homotopy floor=', 'final validation']:
                self.assertIn(label, self.output.getvalue())

    def test_final_acceptance_requires_shape_known_sticks_and_three_seeds(self):
        m = load_script('05_tenstick')
        self.assertTrue(hasattr(m, 'validate_candidate'), 'final coordinate validator missing')
        v = np.arange(30, dtype=float).reshape(10, 3)
        calls = []
        def identify(v, seed):
            calls.append(seed)
            return 'wrong' if seed == 3 else 'K13n285'
        with patch.multiple(m, mr_ratio=lambda v: (0.1,),
                            mr_certificate_mp=lambda v: (0, 1, 1, True), identify=identify):
            with self.assertRaises(ValueError):
                m.validate_candidate(v, 'K13n285', 10)
            self.assertEqual(calls, [1, 2, 3])
            with self.assertRaises(ValueError):
                m.validate_candidate(v[:9], 'K13n285', 10)
            with self.assertRaises(ValueError):
                m.validate_candidate(v, 'K13n285', 9)
        with patch.multiple(m, mr_ratio=lambda v: (0.1,),
                            mr_certificate_mp=lambda v: (0, 1, 1, True),
                            identify=lambda v, seed: 'K13n285'):
            self.assertEqual(m.validate_candidate(v, 'K13n285', 10), (0, 1, 1, True))

    def test_run_checks_normalized_final_coordinates(self):
        m = load_script('05_tenstick')
        v = np.arange(30, dtype=float).reshape(10, 3)
        normalized = v + 100
        with tempfile.TemporaryDirectory() as out, patch.multiple(m,
                eddy_available=lambda n: True, load_eddy=lambda n: v,
                reduce_to=lambda *a: v, identify=lambda *a, **k: 'K13n285',
                fatten=lambda *a: v, homotopy_equalize=lambda *a, **k: (v, 1, 1, True),
                normalize=lambda v: normalized,
                mr_ratio=lambda v: (2 if np.array_equal(v, normalized) else 0.1,),
                mr_certificate_mp=lambda v: (0, 1, 1, True)):
            result = m.run('K13n285', budget=0.01, out=out)
            self.assertNotIn('FOUND', result)
            self.assertEqual(list(Path(out).iterdir()), [])

    def test_identification_computation_error_is_not_search_failure(self):
        m = load_script('05_tenstick')
        v = np.arange(30, dtype=float).reshape(10, 3)
        def identify(v, seed=None):
            if seed is not None:
                raise ValueError('identification engine error')
            return 'K13n285'
        with tempfile.TemporaryDirectory() as out, patch.multiple(m,
                eddy_available=lambda n: True, load_eddy=lambda n: v,
                reduce_to=lambda *a: v, identify=identify, fatten=lambda *a: v,
                homotopy_equalize=lambda *a, **k: (v, 1, 1, True),
                mr_ratio=lambda v: (0.1,), mr_certificate_mp=lambda v: (0, 1, 1, True)):
            with self.assertRaisesRegex(ValueError, 'engine error'):
                m.run('K13n285', budget=0.001, out=out)

    def test_run_rejects_failed_mp_certificate(self):
        m = load_script('05_tenstick')
        v = np.arange(30, dtype=float).reshape(10, 3)
        with tempfile.TemporaryDirectory() as out, patch.multiple(m,
                eddy_available=lambda n: True, load_eddy=lambda n: v,
                reduce_to=lambda *a: v, identify=lambda *a, **k: 'K13n285',
                fatten=lambda *a: v, homotopy_equalize=lambda *a, **k: (v, 1, 1, True),
                mr_ratio=lambda v: (0.1,), mr_certificate_mp=lambda v: (0, 1, 1, False)):
            result = m.run('K13n285', budget=0.01, out=out)
            self.assertNotIn('FOUND', result)
            self.assertEqual(list(Path(out).iterdir()), [])

if __name__ == '__main__':
    unittest.main()
