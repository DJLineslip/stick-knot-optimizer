"""Synthetic NetCDF fixtures, never the external Cantarella dataset."""
import os
import resource
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import netCDF4
import numpy as np

from equistick import crss


class CrssReaderTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('EQUISTICK_CRSS'), 'large external dataset absent')
    def test_large_index_does_not_retain_gigabytes_of_hdf5_cache_in_parent(self):
        before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        entries = crss.crss_index()
        retained_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before_kib
        self.assertGreater(len(entries), 1000)
        self.assertLess(retained_kib, 500 * 1024)

    def test_index_reads_scalar_attributes_and_variables_without_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.nc'
            with netCDF4.Dataset(path, 'w') as ds:
                a = ds.createGroup('K13n586')
                a.setncattr('crossings', 13)
                a.setncattr('sticks', 10)
                a.createDimension('vertex', 10)
                a.createDimension('axis', 3)
                a.createVariable('coords', 'f8', ('vertex', 'axis'))[:] = np.arange(30).reshape(10, 3)
                b = ds.createGroup('8_19')
                b.createVariable('crossings', 'i4').assignValue(8)
                b.createVariable('sticks', 'i4').assignValue(8)
            with patch.dict('os.environ', {'EQUISTICK_CRSS': str(path)}):
                self.assertEqual(crss.crss_path(), path)
                self.assertEqual(crss.crss_index(), {'K13n586': (13, 10), '8_19': (8, 8)})

    def test_coordinates_and_pd_are_loaded_without_masked_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.nc'
            with netCDF4.Dataset(path, 'w') as ds:
                group = ds.createGroup('K13n586')
                group.setncattr('sticks', 10)
                group.setncattr('crossings', 13)
                group.createDimension('vertex', 10)
                group.createDimension('axis', 3)
                group.createDimension('crossing', 2)
                group.createDimension('leg', 4)
                vertices = np.arange(30, dtype=np.float64).reshape(10, 3)
                group.createVariable('coords', 'f8', ('vertex', 'axis'), fill_value=-9999)[:] = vertices
                group.createVariable('pdcode', 'i4', ('crossing', 'leg'))[:] = [[0, 1, 2, 3], [4, 5, 6, 7]]
            with patch.dict('os.environ', {'EQUISTICK_CRSS': str(path)}):
                loaded = crss.load_crss('K13n586')
                self.assertIs(type(loaded), np.ndarray)
                self.assertEqual(loaded.dtype, np.float64)
                np.testing.assert_array_equal(loaded, vertices)
                self.assertEqual(crss.load_crss_pd('K13n586'), [(0, 1, 2, 3), (4, 5, 6, 7)])
                with netCDF4.Dataset(path, 'a') as ds:
                    ds.groups['K13n586'].variables['coords'][0, 0] = -9999
                with self.assertRaisesRegex(ValueError, 'masked'):
                    crss.load_crss('K13n586')


if __name__ == '__main__':
    unittest.main()
