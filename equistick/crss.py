"""Read the user-supplied Cantarella NetCDF dataset (doi:10.7910/DVN/NFJIII).

The original dataset is external and never committed. Source inventory reads
scalar metadata without loading its coordinate arrays.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

CRSS_DIR = Path(__file__).resolve().parents[1] / 'data' / 'external' / 'crss'


def crss_path():
    """Return the one external .nc file, or the EQUISTICK_CRSS override."""
    override = os.environ.get('EQUISTICK_CRSS')
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'EQUISTICK_CRSS is not a file: {path}')
        return path
    files = sorted(CRSS_DIR.glob('*.nc'))
    if len(files) != 1:
        raise FileNotFoundError(f'expected one .nc file in {CRSS_DIR}, found {len(files)}')
    return files[0]


def _scalar(group, field):
    if field in group.ncattrs():
        value = group.getncattr(field)
    elif field in group.variables:
        value = group.variables[field][...]
    else:
        raise ValueError(f'{group.path}: missing {field}')
    return int(value)


def _direct_index(path):
    with Dataset(path, 'r') as ds:
        return {name: (_scalar(group, 'crossings'), _scalar(group, 'sticks'))
                for name, group in ds.groups.items()}


def crss_index():
    """Index metadata in a short-lived process, releasing HDF5's large cache.

    The 13-crossing dataset's group enumeration retains about 3 GiB even
    after closing Dataset. Keeping that cache in the search supervisor while
    a source worker opens the same file exceeds the 5 GiB container limit.
    """
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             '--index', str(crss_path())], text=True,
                            capture_output=True, check=True, timeout=120)
    return {name: tuple(map(int, values))
            for name, values in json.loads(result.stdout).items()}


def load_crss(name):
    """Return a plain float64 (sticks, 3) polygon, rejecting masked input."""
    with Dataset(crss_path(), 'r') as ds:
        group = ds.groups[name]
        raw = group.variables['coords'][:]
        if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
            raise ValueError(f'{name}: masked coordinates')
        coords = np.asarray(raw, dtype=np.float64)
        if coords.shape != (_scalar(group, 'sticks'), 3) or not np.isfinite(coords).all():
            raise ValueError(f'{name}: invalid coordinates')
        return coords


def load_crss_pd(name):
    """Return the stored zero-indexed planar diagram as four-tuples."""
    with Dataset(crss_path(), 'r') as ds:
        raw = ds.groups[name].variables['pdcode'][:]
        if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
            raise ValueError(f'{name}: masked planar diagram')
        pd = np.asarray(raw)
        if pd.ndim != 2 or pd.shape[1] != 4:
            raise ValueError(f'{name}: invalid planar diagram shape')
        return [tuple(map(int, crossing)) for crossing in pd]


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--index':
        raise SystemExit('usage: crss.py --index path-to-source.nc')
    json.dump(_direct_index(Path(sys.argv[2])), sys.stdout)
