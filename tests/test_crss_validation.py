"""Tests for the bounded, stratified CRSS source audit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'crss_validate.py'
spec = importlib.util.spec_from_file_location('crss_validate', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CRSSValidationTests(unittest.TestCase):
    def test_sample_is_deterministic_stratified_and_unique(self):
        index = {'3_1': (3, 6), '4_1': (4, 7),
                 **{f'K13n{i}': (13, 10) for i in range(20)}}
        sample = module.stratified_sample(index, 7, 1234)
        self.assertEqual(len(sample), 7)
        self.assertEqual(len(set(sample)), 7)
        self.assertIn('3_1', sample)
        self.assertIn('4_1', sample)
        self.assertEqual(sample, module.stratified_sample(dict(reversed(list(index.items()))), 7, 1234))

    def test_exact_table_names_above_dataset_range_are_not_missing(self):
        index = {'3_1': (3, 6), 'K13n593': (13, 10)}
        exact = {'0_1': 3, '3_1': 6, 'K14n21881': 9, 'K15n41185': 10}
        self.assertEqual(module.missing_in_range_exact(index, exact), [])
        exact['K13n586'] = 10
        self.assertEqual(module.missing_in_range_exact(index, exact), ['K13n586'])

    def test_paper_table_comparison_detects_a_changed_stick_count(self):
        expected = {(c, sticks): number
                    for c, distribution in module.PAPER_TABLE.items()
                    for sticks, number in distribution.items()}
        self.assertEqual(module.table_discrepancies(expected), {})
        changed = dict(expected)
        changed[13, 12] -= 1
        self.assertIn((13, 12), module.table_discrepancies(changed))
    def test_written_audit_csv_uses_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'index.csv'
            module.write_csv(path, ('knot', 'sticks'),
                             [{'knot': '3_1', 'sticks': 6}])
            self.assertEqual(path.read_bytes(), b'knot,sticks\n3_1,6\n')


if __name__ == '__main__':
    unittest.main()
