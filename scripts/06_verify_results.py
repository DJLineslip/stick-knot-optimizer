"""
06_verify_results.py: independently re-verify every polygon in results/.

For each file <knot>_equilateral_<n>sticks.txt:
  * edge-length spread and the Millett-Rawdon certificate recomputed in
    40-digit arithmetic (certify.mr_certificate_mp);
  * knot type re-identified from scratch: torus knots T(p,q) by the
    Alexander polynomial in 4 random projections plus knot Floer homology
    (genus, fibredness, L-space property, tau, total rank) and the crossing
    number of the simplified diagram; hyperbolic knots by SnapPy census
    identification in 3 random projections;
  * ladder statistics: sum of interior angles (bound 2 pi for bridge-tight
    polygons).
Writes results/summary.csv and prints a table.
"""
import glob, os, re, csv
import numpy as np
from equistick.geometry import lengths, angle_sum
from equistick.invariants import identify
from equistick.certify import mr_certificate_mp, verify_torus

if __name__ == '__main__':
    rows = []
    for f in sorted(glob.glob('../results/*_equilateral_*sticks.txt')):
        base = os.path.basename(f)
        knot = base.split('_equilateral_')[0]
        V = np.loadtxt(f)
        L = lengths(V)
        d, mu, b, ok = mr_certificate_mp(V)
        m = re.fullmatch(r'T(\d+)_(\d+)', knot)
        if m:
            p, q = int(m.group(1)), int(m.group(2))
            alex, h, c = verify_torus(V, p, q)
            g = (p - 1) * (q - 1) // 2
            ident = (alex and h['seifert_genus'] == g and h['fibered'] and h['L_space_knot']
                     and abs(h['tau']) == g)
            how = 'Alexander x4, HFK g=%d L-space fibred tau=%d, %d crossings' % (h['seifert_genus'], h['tau'], c)
            name = 'T(%d,%d)' % (p, q)
        else:
            ids = [identify(V, seed=s) for s in (1, 2, 3)]
            ident = all(x == knot for x in ids)
            how = 'SnapPy identify x3: %s' % ','.join(sorted(set(str(x) for x in ids)))
            name = knot
        rows.append(dict(knot=name, sticks=len(V), length_spread='%.1e' % (L.max() - L.min()),
                         defect='%.1e' % float(d), mu='%.4g' % float(mu), MR_bound='%.2e' % float(b),
                         certified=ok, type_confirmed=ident, angle_sum='%.3f' % angle_sum(V),
                         identification=how))
        r = rows[-1]
        print('%-9s %2d  defect %s  mu %-7s bound %s  cert %s  type %s  angles %s  [%s]' % (
            r['knot'], r['sticks'], r['defect'], r['mu'], r['MR_bound'], r['certified'],
            r['type_confirmed'], r['angle_sum'], r['identification']), flush=True)
    with open('../results/summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
