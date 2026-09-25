"""End-to-end tests of interval geometric certificates from decimal files."""
import tempfile
import unittest
import json
import subprocess
import sys
import hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np

from equistick.interval_certificate import certify_file, squared_segment_distance


class IntervalCertificateTests(unittest.TestCase):
    def test_exact_decimal_square_is_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'square.txt'
            path.write_text('0.1 0.1 0\n1.1 0.1 0\n1.1 1.1 0\n0.1 1.1 0\n')
            record = certify_file(path, precision_bits=160)
            self.assertEqual(record['status'], 'certified')
            self.assertEqual(record['pair_count'], 2)
            self.assertEqual(record['precision_bits'], 160)
            self.assertEqual(Fraction(record['mu_lower']), 1)
            self.assertEqual(Fraction(record['defect_upper']), 0)
            self.assertEqual(Fraction(record['threshold_lower']), Fraction(1, 4))
            self.assertEqual(len(record['sha256']), 64)

    def test_exact_segment_minimum_is_intersection(self):
        a = ((Fraction(0), Fraction(0), Fraction(0)), (Fraction(1), Fraction(1), Fraction(0)))
        b = ((Fraction(0), Fraction(1), Fraction(0)), (Fraction(1), Fraction(0), Fraction(0)))
        self.assertEqual(squared_segment_distance(*a, *b), 0)

    def test_nonunit_scale_normalizes_clearance_by_interval_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scaled.txt'
            path.write_text('0 0 0\n2 0 0\n2 2 0\n0 2 0\n')
            record = certify_file(path)
            self.assertEqual(record['status'], 'certified')
            self.assertEqual(Fraction(record['mu_lower']), 1)
            self.assertEqual(Fraction(record['threshold_lower']), Fraction(1, 4))

    def test_crossing_edges_are_not_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'crossing.txt'
            path.write_text('0 0 0\n1 1 0\n0 1 0\n1 0 0\n')
            record = certify_file(path)
            self.assertEqual(record['status'], 'inconclusive')
            self.assertEqual(Fraction(record['mu_lower']), 0)
            self.assertEqual(Fraction(record['threshold_lower']), 0)

    def test_zero_length_edge_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'degenerate.txt'
            path.write_text('0 0 0\n0 0 0\n1 0 0\n0 1 0\n')
            record = certify_file(path)
            self.assertEqual(record['status'], 'inconclusive')
            self.assertEqual(record['reason'], 'zero-length edge')

    def test_interval_contains_exact_nonbinary_rational_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rectangle.txt'
            path.write_text('0 0 0\n2 0 0\n2 1 0\n0 1 0\n')
            record = certify_file(path, precision_bits=53)
            self.assertEqual(record['status'], 'inconclusive')
            self.assertLessEqual(Fraction(record['mu_lower']), Fraction(2, 3))
            self.assertGreaterEqual(Fraction(record['mu_upper']), Fraction(2, 3))
            self.assertLessEqual(Fraction(record['defect_lower']), Fraction(1, 3))
            self.assertGreaterEqual(Fraction(record['defect_upper']), Fraction(1, 3))

    def test_extreme_exponent_is_rejected_before_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'malicious.txt'
            path.write_text('1e100000000 0 0\n1 0 0\n1 1 0\n0 1 0\n')
            with self.assertRaisesRegex(ValueError, 'exponent'):
                certify_file(path)

    def test_budget_exhaustion_does_not_claim_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'square.txt'
            path.write_text('0 0 0\n1 0 0\n1 1 0\n0 1 0\n')
            record = certify_file(path, max_pairs=1)
            self.assertEqual(record['status'], 'inconclusive')
            self.assertEqual(record['pair_count'], 1)
            self.assertIn('budget', record['reason'])

    def test_exact_boundary_and_parallel_segment_minima(self):
        F = Fraction
        self.assertEqual(squared_segment_distance((F(0),F(0),F(0)), (F(2),F(0),F(0)),
                                                  (F(3),F(1),F(0)), (F(3),F(2),F(0))), 2)
        self.assertEqual(squared_segment_distance((F(0),F(0),F(0)), (F(2),F(0),F(0)),
                                                  (F(1),F(1),F(0)), (F(3),F(1),F(0))), 1)

    def test_existing_polygon_interval_encloses_numerical_crosscheck(self):
        from equistick.geometry import min_dist
        path = Path(__file__).resolve().parents[1] / 'results' / 'T4_5_equilateral_10sticks.txt'
        record = certify_file(path)
        self.assertEqual(record['status'], 'certified')
        vertices = np.loadtxt(path)
        mean = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1).mean()
        numeric_mu = min_dist(vertices) / mean
        # This is only a sanity check, never the certificate's decision path.
        self.assertLessEqual(float(Fraction(record['mu_lower'])) - 1e-12, numeric_mu)
        self.assertGreaterEqual(float(Fraction(record['mu_upper'])) + 1e-12, numeric_mu)

    def test_batch_writes_machine_readable_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw = b'0 0 0\n1 0 0\n1 1 0\n0 1 0\n'
            (folder / 'sample_equilateral_4sticks.txt').write_bytes(raw)
            output = folder / 'certificates.json'
            script = Path(__file__).resolve().parents[1] / 'scripts' / '08_interval_certificates.py'
            subprocess.run([sys.executable, str(script), '--results-dir', str(folder),
                            '--output', str(output), '--precision-bits', '160'],
                           check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertEqual(report['certificate_scope'], 'geometric MR inequality only; not knot identity')
            self.assertEqual(report['code_revision'].__len__(), 40)
            self.assertEqual(report['certified_count'], 1)
            self.assertEqual(report['files'][0]['sha256'], hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    unittest.main()
