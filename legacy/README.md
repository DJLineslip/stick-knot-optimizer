# legacy/: the exact scripts that produced the first results

These are the original research scripts, copied verbatim. They are kept for
provenance: every number reported before the `equistick` package existed came
from one of them. Use `equistick/` and `scripts/` for anything new; the logic
is the same, but documented, tidied and tested.

| Legacy file | What it did | Superseded by |
|---|---|---|
| `ladder.py` | Downloaded Eddy's equilateral 3₁, 8₁₉, 8₂₀, 5₁, 5₂, 10₁₂₄ and printed interior-angle sums against the ladder bound n*pi - 2*pi*b. | `scripts/01_ladder_check.py` |
| `sym.py` | Built the symmetric equilateral "star" polygons (2m vertices on two rings, heights ±H) and a pyknotid planar-diagram wrapper. Needs `pyknotid` and the `np.float` alias patch at the top. | `equistick/torus.py` (`star_poly`) |
| `scan.py` | Grid scan of the star family over (k, r, H) for 6 to 12 sticks, recording distinct Alexander polynomials (pyknotid planar diagrams, spherogram `alexander_polynomial`). Found only unknots and singular polygons. | `scripts/02_symmetric_nogo.py` |
| `sk.py` | Core toolkit: numba segment distances, safety tests, PD codes, Alexander polynomial, SnapPy/HFK wrappers. | `equistick/geometry.py`, `equistick/invariants.py` |
| `load.py` | Loads Eddy's coordinates from a hard-coded path. | `equistick/data.py` |
| `torus.py` | T(p, p+1) construction and random parameter scan. | `equistick/torus.py` |
| `eq.py` | Equalizer #1 (path lifting + fibre ascent). Produced the false collapse signal on T(4,5). | `equistick/flows.py` |
| `nlp.py` | Equalizer #2 (clearance-floor SLSQP) and Newton polishing. Certified T(4,5) to T(7,8). | `equistick/optimize.py`, `equistick/certify.py` |
| `verify.py` | 40-digit Millett-Rawdon certificate and torus verification. | `equistick/certify.py` |
| `run_torus.py` | Clearance-floor scans for T(5,6), T(6,7), T(7,8). | `scripts/04_torus_family.py` |
| `reduce.py` | Annealer that lowers the stick count by one. | `equistick/reduce.py` |
| `batch19.py` | First ten-stick batch. **Buggy in effect**: it sent lopsided 10-gons straight to SLSQP, which jumped to equilateral unknots, so every knot "failed". Kept as a record of that mistake. | `scripts/05_tenstick.py` |
| `homo.py` | Equalizer #3 (fatten + safe staged homotopy), the fix for the bug above. | `equistick/optimize.py` |
| `batch19b.py` | Corrected ten-stick batch (reduce, fatten, homotopy, certify). Its log is `results/legacy_tenstick_batch.log`. Seeds came from Python's salted `hash()`, so reruns differ; `equistick.data.seed_for` fixes this. | `scripts/05_tenstick.py` |

To run a legacy script, run it from a directory where the other legacy files
and a `stick-knot-gen` clone sit side by side. `load.py` hard-codes
`/home/claude/stick-knot-gen`; edit that path.
