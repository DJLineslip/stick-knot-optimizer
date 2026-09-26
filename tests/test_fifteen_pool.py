"""Regression for undecidable numerical SnapPy isometries in pool searches."""
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
pool = importlib.import_module('10_fifteen_pool')


class PoolIsometryTests(unittest.TestCase):
    def test_indeterminate_projection_is_retried_until_three_pass(self):
        attempts = []

        class Exterior:
            def is_isometric_to(self, other):
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError('The SnapPea kernel was not able to determine if the manifolds are isometric.')
                return True

        class Link:
            def exterior(self):
                return Exterior()

        with patch.object(pool.spherogram, 'Link', return_value=Link()), patch.object(pool, 'pd_code', return_value=[]):
            matches = pool.table_isometries(np.zeros((10, 3)), 'K15n124836', seeds=(1, 2, 3, 4), needed=3)
        self.assertEqual(matches, [True, True, True])
        self.assertEqual(len(attempts), 4)

    def test_negative_worker_count_is_rejected_before_starting_search(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / '10_fifteen_pool.py'
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run([sys.executable, str(script), '--workers', '-1',
                                     '--knots', 'K15n124836', '--budget', '0.01', '--out', out],
                                    capture_output=True, text=True, timeout=8)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('workers must be positive', result.stderr)


if __name__ == '__main__':
    unittest.main()
