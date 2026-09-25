"""Synthetic NetCDF fixtures keep the external 140 MB file out of unit tests."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spherogram
from netCDF4 import Dataset

from equistick import crss
from equistick.invariants import identify_pd


PD_FIGURE_EIGHT = spherogram.Link('4_1').PD_code()
COORDS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                   [0, 1, 0], [0.2, 0.3, 0.5], [0.6, 0.2, 0.6]], dtype=float)


def fixture(path):
    with Dataset(path, 'w', format='NETCDF4') as dataset:
        for name, attrs in (('4_1', False), ('K13n586', True)):
            group = dataset.createGroup(name)
            group.createDimension('vertices', len(COORDS))
            group.createDimension('xyz', 3)
            group.createDimension('diagram', len(PD_FIGURE_EIGHT))
            group.createDimension('four', 4)
            coordinates = group.createVariable('coords', 'f8', ('vertices', 'xyz'),
                                               fill_value=-999.0)
            coordinates[:] = COORDS
            group.createVariable('pdcode', 'i8', ('diagram', 'four'))[:] = PD_FIGURE_EIGHT
            if attrs:
                group.setncattr('sticks', len(COORDS))
                group.setncattr('crossings', len(PD_FIGURE_EIGHT))
            else:
                group.createVariable('sticks', 'i8').assignValue(len(COORDS))
                group.createVariable('crossings', 'i8').assignValue(len(PD_FIGURE_EIGHT))


class UnreadablePD:
    """Large chunked dataset metadata without ever allocating its contents."""
    dtype = np.dtype('int64')
    chunks = (1024, 4)
    attrs = {}

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        raise AssertionError('pdcode was materialized')


class PDGroup(dict):
    name = '/4_1'

    def __init__(self, shape, crossings):
        super().__init__(pdcode=UnreadablePD(shape))
        self.attrs = {'crossings': crossings}


class UnreadableCoords:
    """Coordinate metadata proxy that fails instead of materializing data."""
    attrs = {}

    def __init__(self, shape, dtype='float64'):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def __getitem__(self, key):
        raise AssertionError('coords was materialized')


class CoordGroup(dict):
    name = '/4_1'

    def __init__(self, shape, sticks, dtype='float64'):
        super().__init__(coords=UnreadableCoords(shape, dtype))
        self.attrs = {'sticks': sticks}


class CRSSReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'fixture.nc'
        fixture(self.path)
        self.env = patch.dict(os.environ, {'EQUISTICK_CRSS': str(self.path)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_index_uses_name_crossing_number_not_diagram_crossings(self):
        self.assertEqual(crss.crss_index(), {'4_1': (4, 6), 'K13n586': (13, 6)})
        self.assertEqual(crss.crss_path(), self.path)

    def test_shared_readonly_handle_supports_index_and_coordinates(self):
        with crss.open_crss() as source:
            self.assertEqual(crss.crss_index(dataset=source)['K13n586'], (13, 6))
            np.testing.assert_array_equal(crss.load_crss('4_1', dataset=source), COORDS)

    def test_load_coordinates_and_pd_without_masks(self):
        V = crss.load_crss('4_1')
        self.assertIs(type(V), np.ndarray)
        self.assertEqual(V.dtype, np.dtype('float64'))
        np.testing.assert_array_equal(V, COORDS)
        self.assertEqual(crss.load_crss_pd('4_1'), [tuple(row) for row in PD_FIGURE_EIGHT])
        with self.assertRaises(KeyError):
            crss.load_crss('absent')

    def test_coords_rejects_huge_mismatched_shape_before_read(self):
        group = CoordGroup((10**12, 3), sticks=6)
        with patch.object(crss, '_group', return_value=group):
            with self.assertRaisesRegex(ValueError, 'coordinate shape'):
                crss.load_crss('4_1', dataset=object())

    def test_coords_rejects_matching_excessive_sticks_before_read(self):
        group = CoordGroup((100_001, 3), sticks=100_001)
        with patch.object(crss, '_group', return_value=group):
            with self.assertRaisesRegex(ValueError, 'coordinates exceed maximum'):
                crss.load_crss('4_1', dataset=object())

    def test_coords_rejects_unsafe_dtype_before_read(self):
        for dtype in ('complex128', 'float128', 'object'):
            with self.subTest(dtype=dtype):
                group = CoordGroup((6, 3), sticks=6, dtype=dtype)
                with patch.object(crss, '_group', return_value=group):
                    with self.assertRaisesRegex(ValueError, 'coordinate dtype'):
                        crss.load_crss('4_1', dataset=object())

    def test_pd_rejects_out_of_range_labels_before_counting(self):
        count = len(PD_FIGURE_EIGHT)
        real_bincount = np.bincount

        def safe_bincount(labels, **kwargs):
            if np.any((labels < 0) | (labels >= 2 * count)):
                raise AssertionError('out-of-range label reached np.bincount')
            return real_bincount(labels, **kwargs)

        for bad_label in (-1, 1_000_000_000):
            with self.subTest(bad_label=bad_label):
                with Dataset(self.path, 'a') as source:
                    source.groups['4_1'].variables['pdcode'][0, 0] = bad_label
                with patch.object(crss.np, 'bincount', side_effect=safe_bincount):
                    with self.assertRaisesRegex(ValueError, 'PD labels must be 0-based and occur twice'):
                        crss.load_crss_pd('4_1')

    def test_pd_rejects_huge_chunked_shape_before_read(self):
        group = PDGroup((10**12, 4), crossings=4)
        with patch.object(crss, '_group', return_value=group):
            with self.assertRaisesRegex(ValueError, 'PD shape disagrees with projected crossings'):
                crss.load_crss_pd('4_1', dataset=object())

    def test_pd_rejects_matching_excessive_count_before_read(self):
        group = PDGroup((100_001, 4), crossings=100_001)
        with patch.object(crss, '_group', return_value=group):
            with self.assertRaisesRegex(ValueError, 'PD code exceeds maximum'):
                crss.load_crss_pd('4_1', dataset=object())

    def test_pd_rejects_unsafe_dtype_before_read(self):
        for dtype in ('float64', 'uint64'):
            with self.subTest(dtype=dtype):
                group = PDGroup((4, 4), crossings=4)
                group['pdcode'].dtype = np.dtype(dtype)
                with patch.object(crss, '_group', return_value=group):
                    with self.assertRaisesRegex(ValueError, 'PD code is not integral'):
                        crss.load_crss_pd('4_1', dataset=object())

    def test_masked_coordinate_is_rejected_but_index_still_works(self):
        with Dataset(self.path, 'a') as d:
            d.groups['4_1'].variables['coords'][0, 0] = -999.0
        self.assertEqual(crss.crss_index()['4_1'], (4, 6))
        with self.assertRaisesRegex(ValueError, 'mask|missing'):
            crss.load_crss('4_1')

    def test_path_requires_exactly_one_file_unless_overridden(self):
        with patch.dict(os.environ, {'EQUISTICK_CRSS': ''}):
            with patch.object(crss, 'DEFAULT_DIR', Path(self.tmp.name)):
                self.assertEqual(crss.crss_path(), self.path)
                (Path(self.tmp.name) / 'extra.nc').touch()
                with self.assertRaisesRegex(ValueError, '[Mm]ultiple'):
                    crss.crss_path()
                self.path.unlink()
                (Path(self.tmp.name) / 'extra.nc').unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'NetCDF'):
                    crss.crss_path()
        with patch.dict(os.environ, {'EQUISTICK_CRSS': str(self.path)}):
            with self.assertRaises(FileNotFoundError):
                crss.crss_path()

    def test_census_aliases_are_checked_against_named_complement(self):
        from equistick.invariants import matches_census_name
        self.assertTrue(matches_census_name('K9a31', '9_29'))
        self.assertFalse(matches_census_name('K9a31', '9_28'))
        self.assertTrue(matches_census_name('K13n586', 'K13n586'))
        self.assertFalse(matches_census_name(None, '9_29'))

    def test_pd_identification_uses_rolfsen_alias(self):
        self.assertEqual(identify_pd([tuple(row) for row in PD_FIGURE_EIGHT]), '4_1')
        self.assertEqual(identify_pd([]), 'unknot')

    @unittest.skipUnless(sys.platform == 'linux', 'memory limit requires Linux')
    def test_real_source_index_stays_under_memory_budget(self):
        source = crss.DEFAULT_DIR / 'stick-number-bounds.nc'
        if not source.is_file():
            self.skipTest('external CRSS file is not installed')
        code = ('import resource; resource.setrlimit(resource.RLIMIT_AS, '
                '(850 * 1024**2, 850 * 1024**2)); '
                'from equistick.crss import crss_index; '
                'assert len(crss_index()) == 12965')
        env = {key: os.environ[key] for key in ('PATH', 'HOME') if key in os.environ}
        env.update(EQUISTICK_CRSS=str(source), PYTHONPATH=str(source.parents[3]),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
        process = subprocess.run([sys.executable, '-c', code], env=env,
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stderr[-1000:])


if __name__ == '__main__':
    unittest.main()
