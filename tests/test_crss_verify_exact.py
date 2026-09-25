"""Exact-stick coverage tests with an external data integration fixture."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from equistick.data import MSEQ

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'crss_verify_exact.py'
spec = importlib.util.spec_from_file_location('crss_verify_exact', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactCoverageTests(unittest.TestCase):
    def test_source_mapping_handles_named_torus_and_external_file(self):
        with tempfile.TemporaryDirectory() as folder:
            eddy = Path(folder)
            (eddy / '3_1.txt').write_text('0 0 0\n')
            self.assertEqual(module.select_source('K14n21881', 12,
                                                  {'T(3,7)': 12}, eddy),
                             ('in_repo', 'T(3,7)'))
            self.assertEqual(module.select_source('3_1', 6, {}, eddy),
                             ('eddy', '3_1'))
            self.assertEqual(module.select_source('K13n586', 10, {}, eddy),
                             ('missing', None))

    def test_equilateral_unknot_does_not_verify_as_trefoil(self):
        angles = 2 * np.pi * np.arange(6) / 6
        coords = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / '3_1.txt'
            np.savetxt(file, coords, fmt='%.17g')
            self.assertNotEqual(module.verify_eddy('3_1', 6, file)['status'], 'certified')

    def test_15_crossing_external_name_uses_published_table(self):
        file = Path(MSEQ) / 'K15n41127.txt'
        if not file.is_file():
            self.skipTest('Eddy data not installed')
        result = module.verify_eddy('K15n41127', 10, file)
        self.assertEqual(result['status'], 'certified', result)
        self.assertEqual(result['interval']['status'], 'certified')
        from equistick.invariants import matches_census_name
        self.assertTrue(all(matches_census_name(name, 'K15n41127')
                            for name in result['coordinate_ids']))


if __name__ == '__main__':
    unittest.main()
