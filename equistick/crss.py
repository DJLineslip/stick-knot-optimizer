"""Read the user-provided Cantarella et al. NetCDF4 dataset (doi:10.7910/DVN/NFJIII).

The source file is kept outside Git. Its ``crossings`` scalar counts crossings in
its projected PD diagram, not the minimal crossing number encoded by the knot
name; see ``data/README.md``. Callers may reuse one read-only Dataset via the
optional ``dataset`` argument when processing many groups.
"""
from contextlib import nullcontext
from pathlib import Path
import os
import re

import h5py
import numpy as np


DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'external' / 'crss'
_NAME = re.compile(r'(?:(\d+)_\d+|K(\d+)[an]\d+)\Z')
_MAX_PD_ROWS = 100_000  # 400,000 entries / 3.2 MB as int64; source maximum is 53 rows.
_MAX_COORD_ROWS = 100_000  # At most 2.4 MB as float64; source maximum is 13 sticks.


def crss_path():
    """Find exactly one NetCDF source, unless EQUISTICK_CRSS names one."""
    override = os.environ.get('EQUISTICK_CRSS')
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f'NetCDF source not found: {path}')
        return path
    paths = sorted(path for path in DEFAULT_DIR.glob('*.nc') if path.is_file())
    if not paths:
        raise FileNotFoundError(f'No NetCDF source in {DEFAULT_DIR}; set EQUISTICK_CRSS')
    if len(paths) != 1:
        raise ValueError(f'Multiple NetCDF sources in {DEFAULT_DIR}; set EQUISTICK_CRSS')
    return paths[0]


def open_crss():
    """Open a lazy, read-only HDF5 handle to the NetCDF4 source."""
    return h5py.File(crss_path(), 'r')


def _dataset(dataset):
    return nullcontext(dataset) if dataset is not None else open_crss()


def _group(dataset, name):
    if name not in dataset or '/' in name or not isinstance(dataset[name], h5py.Group):
        raise KeyError(f'No CRSS knot group: {name}') from None
    return dataset[name]


def _scalar(group, field):
    if field in group:
        variable = group[field]
        if variable.shape != ():
            raise ValueError(f'{group.name}/{field} is not scalar')
        raw = variable[()]
    elif field in group.attrs:
        raw = group.attrs[field]
    else:
        raise ValueError(f'{group.name} lacks {field}')
    values = np.asarray(raw)
    if values.size != 1:
        raise ValueError(f'{group.name}/{field} is not scalar')
    raw = values.reshape(-1)[0]
    value = int(raw)
    if value != raw or value < 0:
        raise ValueError(f'{group.name}/{field} must be a nonnegative integer')
    return value


def knot_crossings(name):
    """Minimal crossing count encoded by the table name, not a PD projection."""
    match = _NAME.fullmatch(name)
    if match is None:
        raise ValueError(f'Unrecognised knot name: {name}')
    return int(match.group(1) or match.group(2))


def crss_index(*, dataset=None):
    """Return {knot: (table crossing number, sticks)} without reading coords."""
    with _dataset(dataset) as source:
        index = {}
        for name, group in source.items():
            sticks = _scalar(group, 'sticks')
            projected = _scalar(group, 'crossings')
            if not sticks or not projected:
                raise ValueError(f'{group.name}: zero sticks or PD crossings')
            coords = group.get('coords')
            pdcode = group.get('pdcode')
            if coords is None or coords.shape != (sticks, 3):
                raise ValueError(f'{group.name}: wrong coordinate shape')
            if pdcode is None or pdcode.shape != (projected, 4):
                raise ValueError(f'{group.name}: wrong PD code shape')
            index[name] = knot_crossings(name), sticks
        return index


def load_crss(name, *, dataset=None):
    """Return an ordinary finite float64 (sticks, 3) polygon, never masked."""
    with _dataset(dataset) as source:
        group = _group(source, name)
        var = group['coords']
        sticks = _scalar(group, 'sticks')
        if var.shape != (sticks, 3):
            raise ValueError(f'{name}: invalid coordinate shape')
        if sticks > _MAX_COORD_ROWS:
            raise ValueError(f'{name}: coordinates exceed maximum of {_MAX_COORD_ROWS} rows')
        if not np.issubdtype(var.dtype, np.number) or not np.can_cast(
                var.dtype, np.float64, casting='safe'):
            raise ValueError(f'{name}: coordinate dtype is not safely representable')
        coords = var[:]
        missing = [var.attrs[k] for k in ('_FillValue', 'missing_value')
                   if k in var.attrs]
        if (np.ma.isMaskedArray(coords) and np.ma.getmaskarray(coords).any()) or any(
                np.any(coords == value) for value in missing):
            raise ValueError(f'{name}: masked or missing coordinates')
        V = np.asarray(coords, dtype=np.float64)
        if V.shape != (sticks, 3) or not np.isfinite(V).all():
            raise ValueError(f'{name}: invalid coordinate shape or non-finite value')
        if np.any(np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1) == 0):
            raise ValueError(f'{name}: zero-length edge')
        return V


def load_crss_pd(name, *, dataset=None):
    """Return the stored 0-based PD code as a list of four-int tuples."""
    with _dataset(dataset) as source:
        group = _group(source, name)
        var = group['pdcode']
        count = _scalar(group, 'crossings')
        if var.shape != (count, 4):
            raise ValueError(f'{name}: PD shape disagrees with projected crossings')
        if count > _MAX_PD_ROWS:
            raise ValueError(f'{name}: PD code exceeds maximum of {_MAX_PD_ROWS} rows')
        if not np.issubdtype(var.dtype, np.integer) or not np.can_cast(
                var.dtype, np.int64, casting='safe'):
            raise ValueError(f'{name}: PD code is not integral or safely representable')
        array = var[:]
        if (np.ma.isMaskedArray(array) and np.ma.getmaskarray(array).any()) or any(
                np.any(array == var.attrs[k]) for k in ('_FillValue', 'missing_value')
                if k in var.attrs):
            raise ValueError(f'{name}: masked PD code')
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f'{name}: PD code is not integral')
        pd = np.asarray(array, dtype=np.int64)
        if pd.shape != (count, 4):
            raise ValueError(f'{name}: PD shape disagrees with projected crossings')
        if np.any((pd < 0) | (pd >= 2 * count)):
            raise ValueError(f'{name}: PD labels must be 0-based and occur twice')
        if not np.array_equal(np.bincount(pd.ravel(), minlength=2 * count),
                              np.full(2 * count, 2, dtype=np.int64)):
            raise ValueError(f'{name}: PD labels must be 0-based and occur twice')
        return [tuple(map(int, row)) for row in pd]
