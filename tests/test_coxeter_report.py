"""Exact-map report generation from an immutable workbook fixture."""
import hashlib
import importlib.util
import tempfile
from pathlib import Path
import unittest

from openpyxl import Workbook

from test_coxeter import S5_GAUSS, S5_SEEDS

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_coxeter.py'


class CoxeterReportTests(unittest.TestCase):
    def test_map_report_does_not_assert_untested_geometry_from_forged_metadata(self):
        spec = importlib.util.spec_from_file_location('verify_coxeter', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            workbook = Path(folder) / 'Wirt_Hm_fixture.xlsx'
            wb = Workbook()
            wb.active.append(['15n41189', S5_GAUSS, 4, None, 1, S5_SEEDS, 0, None, 0, None])
            wb.save(workbook)
            report = {'source_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(),
                      'existing_six': [{'knot': 'K15n41189',
                                        'file': 'nonexistent_equilateral_10sticks.txt',
                                        'checks': {'interval_certified': False}}],
                      'pool_results': []}
            enriched = module.generate_proofs(report, workbook)
            self.assertTrue(enriched['existing_six'][0]['exact_coxeter_passed'])
            self.assertIn('does not certify saved polygon geometry',
                          enriched['verification_note'])

    def test_per_knot_pass_and_missing_map_failure_are_recorded(self):
        spec = importlib.util.spec_from_file_location('verify_coxeter', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Wirt_Hm_fixture.xlsx'
            wb = Workbook()
            wb.active.append(['15n41189', S5_GAUSS, 4, None, 1, S5_SEEDS, 0, None, 0, None])
            wb.active.append(['15n59060', S5_GAUSS, 4, None, 0, None, 0, None, 0, None])
            wb.save(path)
            report = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'existing_six': [{'knot': 'K15n41189'}],
                      'pool_results': [{'knot': 'K15n59060'}]}
            enriched = module.generate_proofs(report, path)
            self.assertEqual(set(enriched['exact_coxeter_maps']), {'K15n41189', 'K15n59060'})
            self.assertTrue(enriched['existing_six'][0]['exact_coxeter_passed'])
            self.assertFalse(enriched['pool_results'][0]['exact_coxeter_passed'])
            self.assertEqual(enriched['exact_coxeter_maps']['K15n59060']['status'], 'map_missing')
            self.assertEqual(enriched['exact_coxeter_maps']['K15n41189']['relations_checked'], 15)
            self.assertEqual(len(enriched['exact_coxeter_maps']['K15n41189']['strand_images']), 15)


if __name__ == '__main__':
    unittest.main()
