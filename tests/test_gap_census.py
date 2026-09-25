"""Small synthetic census, never the external NetCDF source."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / '09_gap_census.py'
FIELDS = ['knot', 'crossings', 's_ub', 'exact_s', 'e_ub',
          'e_ub_minus_s_ub', 'source_sticks', 'input_status', 'e_status']


def polygon(n):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack((np.cos(angles), np.sin(angles), np.zeros(n)))


class GapCensusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'eddy' / 'stick_number'
        (self.data / 'mseq_knots').mkdir(parents=True)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.netcdf = self.root / 'tiny.nc'
        # Deliberately insert groups out of order and use PD crossings unlike
        # the table crossing count, to test the name-derived value.
        with Dataset(self.netcdf, 'w') as dataset:
            for name, sticks in [('K13n593', 10), ('9_29', 9),
                                 ('K13n586', 10), ('4_1', 8)]:
                group = dataset.createGroup(name)
                group.createDimension('vertices', sticks)
                group.createDimension('xyz', 3)
                group.createDimension('projected', 4)
                group.createDimension('four', 4)
                group.createVariable('coords', 'f8', ('vertices', 'xyz'))[:] = polygon(sticks)
                group.createVariable('pdcode', 'i4', ('projected', 'four'))[:] = 0
                group.createVariable('sticks', 'i4').assignValue(sticks)
                group.createVariable('crossings', 'i4').assignValue(4)
        self.validation = self.root / 'validation.json'
        self.validation.write_text(json.dumps(dict(
            groups=4, source_sha256=hashlib.sha256(self.netcdf.read_bytes()).hexdigest(),
            identity_total=4, identity_matched=4, identity_mismatches=0,
            identity_pending=0,
            identity_details=[dict(knot=name, status='matched') for name in
                              ('4_1', '9_29', 'K13n586', 'K13n593')])))
        (self.data / 'exact_values.csv').write_text(
            'knot, stick number\n4_1, 7\n9_29, 9\nK13n586, 10\n')
        np.savetxt(self.data / 'mseq_knots' / '4_1.txt', polygon(8), fmt='%.17g')
        np.savetxt(self.data / 'mseq_knots' / 'K13n593.txt', polygon(11), fmt='%.17g')
        # An Eddy file outside the NetCDF index must not create an extra row.
        np.savetxt(self.data / 'mseq_knots' / '3_1.txt', polygon(6), fmt='%.17g')
        certified = []
        with (self.results / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['knot', 'sticks', 'certified', 'type_confirmed'])
            writer.writeheader()
            for name, sticks in [('4_1', 7), ('K13n586', 10)]:
                file = self.results / f'{name}_equilateral_{sticks}sticks.txt'
                np.savetxt(file, polygon(sticks), fmt='%.17g')
                writer.writerow(dict(knot=name, sticks=sticks,
                                     certified='True', type_confirmed='True'))
                certified.append(dict(file=file.name, sticks=sticks, status='certified',
                                      sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        (self.results / 'interval_certificates.json').write_text(
            json.dumps(dict(files=certified)))
        self.output = self.root / 'census.csv'

    def run_census(self, expected_count='4', extra=()):
        env = os.environ.copy()
        env.update(EQUISTICK_CRSS=str(self.netcdf), EQUISTICK_DATA=str(self.data.parent),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                   PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, str(SCRIPT), '--output', str(self.output),
                               '--results-dir', str(self.results), '--expected-count', expected_count,
                               '--validation-report', str(self.validation),
                               *extra],
                              cwd=ROOT / 'scripts', env=env, capture_output=True, text=True)

    def test_csv_schema_join_and_sorted_crossings(self):
        proc = self.run_census()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn(b'\r\n', self.output.read_bytes())
        with self.output.open(newline='') as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, FIELDS)
            rows = list(reader)
        self.assertEqual(rows, [
            dict(knot='4_1', crossings='4', s_ub='8', exact_s='7', e_ub='7', e_ub_minus_s_ub='-1', source_sticks='8', input_status='matched', e_status='checked'),
            dict(knot='9_29', crossings='9', s_ub='9', exact_s='9', e_ub='', e_ub_minus_s_ub='', source_sticks='9', input_status='matched', e_status='none'),
            dict(knot='K13n586', crossings='13', s_ub='10', exact_s='10', e_ub='10', e_ub_minus_s_ub='0', source_sticks='10', input_status='matched', e_status='checked'),
            dict(knot='K13n593', crossings='13', s_ub='10', exact_s='', e_ub='11', e_ub_minus_s_ub='1', source_sticks='10', input_status='matched', e_status='reported_only'),
        ])
        self.assertIn('4 1 1 1 1 0 0\n', proc.stdout)
        self.assertIn('9 1 1 0 0 0 1\n', proc.stdout)
        self.assertIn('13 2 1 2 1 1 0\n', proc.stdout)
        self.assertIn('Wrote 4 unique, sorted knots', proc.stdout)

    def test_eddy_only_equilateral_count_is_reported_only(self):
        proc = self.run_census()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with self.output.open(newline='') as f:
            rows = {row['knot']: row for row in csv.DictReader(f)}
        self.assertEqual(rows['K13n593']['e_status'], 'reported_only')

    def test_saved_interval_count_is_checked(self):
        proc = self.run_census()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with self.output.open(newline='') as f:
            rows = {row['knot']: row for row in csv.DictReader(f)}
        self.assertEqual(rows['K13n586']['e_status'], 'checked')
        self.assertEqual(rows['9_29']['e_status'], 'none')

    def test_mismatched_source_name_has_no_knot_upper_bound(self):
        report = json.loads(self.validation.read_text())
        report['identity_matched'] = 3
        report['identity_mismatches'] = 1
        next(item for item in report['identity_details'] if item['knot'] == 'K13n593')['status'] = 'mismatch'
        self.validation.write_text(json.dumps(report))
        proc = self.run_census()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with self.output.open(newline='') as f:
            rows = {row['knot']: row for row in csv.DictReader(f)}
        self.assertEqual(rows['K13n593']['source_sticks'], '10')
        self.assertEqual(rows['K13n593']['s_ub'], '')
        self.assertEqual(rows['K13n593']['e_ub_minus_s_ub'], '')
        self.assertEqual(rows['K13n593']['input_status'], 'mismatch')
        self.assertEqual(rows['9_29']['input_status'], 'matched')
        self.assertEqual(rows['4_1']['input_status'], 'matched')

    def test_pending_identity_keeps_provisional_source_out_of_bounds(self):
        report = json.loads(self.validation.read_text())
        report['identity_matched'] = 3
        report['identity_pending'] = 1
        next(item for item in report['identity_details'] if item['knot'] == 'K13n593')['status'] = 'pending_nonhyperbolic'
        self.validation.write_text(json.dumps(report))
        proc = self.run_census()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with self.output.open(newline='') as f:
            rows = {row['knot']: row for row in csv.DictReader(f)}
        self.assertEqual(rows['K13n593']['input_status'], 'pending_nonhyperbolic')
        self.assertEqual(rows['K13n593']['source_sticks'], '10')
        self.assertEqual(rows['K13n593']['s_ub'], '')
        self.assertEqual(rows['K13n593']['e_ub_minus_s_ub'], '')
        self.assertEqual(rows['K13n593']['e_ub'], '11')

    def test_refuses_wrong_dataset_size_without_writing_csv(self):
        proc = self.run_census(expected_count='12965')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('expected 12965, got 4', proc.stderr)
        self.assertFalse(self.output.exists())

    def test_production_index_missing_special_name_rejects_complete_audit(self):
        # Exercise the production-size name set without building HDF5 groups.
        spec = importlib.util.spec_from_file_location('gap_census_test', SCRIPT)
        census = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, 'path', [str(SCRIPT.parent), *sys.path]):
            spec.loader.exec_module(census)
        missing_name = 'K11n71'
        extras = set(census.TEN_STICK_19) | {'9_29'} | set(census.TORUS)
        self.assertIn(missing_name, extras)
        present = extras - {missing_name}
        index = {name: (13, 10) for name in present}
        index.update({f'K13n_fake_{n}': (13, 10)
                      for n in range(12965 - len(index))})
        self.assertEqual(len(index), 12965)
        base = census.stratified_sample(index, 300, census.seed_for('crss-validation'))
        audited = set(base) | present
        report = json.loads(self.validation.read_text())
        report.update(groups=len(index), identity_total=len(audited),
                      identity_matched=len(audited), identity_mismatches=0,
                      identity_pending=0,
                      identity_details=[dict(knot=name, status='matched')
                                        for name in sorted(audited)])
        self.validation.write_text(json.dumps(report))
        with mock.patch.object(census, 'crss_path', return_value=self.netcdf):
            with self.assertRaisesRegex(ValueError, r'Required identity knots absent:.*K11n71'):
                census.validation_statuses(index, self.validation)

    def test_rejects_self_consistent_audit_missing_required_mismatch(self):
        report = json.loads(self.validation.read_text())
        report['identity_details'][-1]['status'] = 'mismatch'
        report['identity_matched'] -= 1
        report['identity_mismatches'] += 1
        # Hide the mismatch, also adjusting every total so it is self-consistent.
        report['identity_details'].pop()
        report['identity_total'] -= 1
        report['identity_mismatches'] -= 1
        self.validation.write_text(json.dumps(report))
        proc = self.run_census()
        self.assertFalse(self.output.exists(), proc.stdout + proc.stderr)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('audited', proc.stderr)

    def test_rejects_empty_zero_audit(self):
        report = json.loads(self.validation.read_text())
        report.update(identity_total=0, identity_matched=0, identity_mismatches=0,
                      identity_pending=0, identity_details=[])
        self.validation.write_text(json.dumps(report))
        proc = self.run_census()
        self.assertFalse(self.output.exists(), proc.stdout + proc.stderr)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('audited', proc.stderr)

    def test_rejects_omitted_identity_detail_before_writing(self):
        report = json.loads(self.validation.read_text())
        report['identity_details'] = report['identity_details'][1:]
        self.validation.write_text(json.dumps(report))
        proc = self.run_census()
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn('identity_total', proc.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_false_identity_status_totals(self):
        report = json.loads(self.validation.read_text())
        report['identity_mismatches'] = 1
        self.validation.write_text(json.dumps(report))
        proc = self.run_census()
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn('identity_mismatches', proc.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_modified_saved_polygon_after_certificate(self):
        file = self.results / 'K13n586_equilateral_10sticks.txt'
        file.write_bytes(file.read_bytes() + b'\n')
        proc = self.run_census()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('SHA256', proc.stderr)

    def test_rejects_non_equilateral_eddy_polygon(self):
        file = self.data / 'mseq_knots' / '4_1.txt'
        coordinates = polygon(8)
        coordinates[0, 0] += 0.3
        np.savetxt(file, coordinates, fmt='%.17g')
        proc = self.run_census()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('equilateral', proc.stderr)


if __name__ == '__main__':
    unittest.main()
