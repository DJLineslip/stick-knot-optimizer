"""Task-2 candidate gate, using the installed CRSS file only for integration."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from equistick.crss import load_crss
from equistick.geometry import normalize

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'crss_close_known.py'
spec = importlib.util.spec_from_file_location('crss_close_known', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CloseKnownTests(unittest.TestCase):
    def test_equilateral_unknot_is_rejected_as_nine_29(self):
        angles = 2 * np.pi * np.arange(9) / 9
        polygon = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(9)))
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / '9_29_equilateral_9sticks.txt'
            np.savetxt(file, polygon, fmt='%.17g')
            with self.assertRaisesRegex(module.CandidateRejected, 'identity'):
                module.validate_candidate(file, '9_29', 9)

    def test_real_source_nine_29_passes_saved_decimal_gate(self):
        source = ROOT / 'data/external/crss/stick-number-bounds.nc'
        if not source.is_file():
            self.skipTest('user-provided source not installed')
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / '9_29_equilateral_9sticks.txt'
            np.savetxt(file, normalize(load_crss('9_29')), fmt='%.17g')
            record = module.validate_candidate(file, '9_29', 9)
            self.assertEqual(record['interval']['status'], 'certified')
            self.assertEqual(len(record['coordinate_ids']), 3)


if __name__ == '__main__':
    unittest.main()
