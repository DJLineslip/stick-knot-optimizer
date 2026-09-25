"""
equistick.data
==============

Access to Eddy's stick-knot-gen repository (Eddy and Shonkwiler 2022;
Blair, Eddy, Morrison and Shonkwiler 2020), which holds equilateral
coordinates for every knot type seen in 220 billion random confined
polygons, plus a table of knots whose exact stick number is known.

    git clone https://github.com/thomaseddy/stick-knot-gen

Set EQUISTICK_DATA to the clone's location (default: stick-knot-gen/ next to
the equistick/ package directory).
"""
import os
import csv
import zlib
import numpy as np

ROOT = os.environ.get('EQUISTICK_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stick-knot-gen'))
MSEQ = os.path.join(ROOT, 'stick_number', 'mseq_knots')


def load_eddy(name):
    """Equilateral coordinates for knot `name` ('8_19', 'K11n71', ...)."""
    return np.loadtxt(os.path.join(MSEQ, name + '.txt'))


def eddy_available(name):
    return os.path.exists(os.path.join(MSEQ, name + '.txt'))


def exact_stick_numbers():
    """{knot: exact stick number} from exact_values.csv."""
    out = {}
    with open(os.path.join(ROOT, 'stick_number', 'exact_values.csv')) as f:
        for row in csv.reader(f):
            if row[0].strip() == 'knot':
                continue
            out[row[0].strip()] = int(row[1].strip())
    return out


def seed_for(name):
    """Deterministic RNG seed from a knot name (Python's hash() is salted
    per process, so the legacy batch scripts were not reproducible)."""
    return zlib.crc32(name.encode())


# The 19 knots whose stick number Cantarella, Rechnitzer, Schumacher and
# Shonkwiler (JKTR 2026) proved to be exactly 10 (bridge index 4).
TEN_STICK_19 = ['K11n71', 'K11n75', 'K11n76', 'K11n78', 'K13n225', 'K13n230',
                'K13n285', 'K13n288', 'K13n307', 'K13n584', 'K13n586', 'K13n593',
                'K13n602', 'K13n603', 'K13n604', 'K13n607', 'K13n608', 'K13n1192',
                'K13n5018']
