# equistick

A numerical search for a knot whose **equilateral stick number** e(K) is larger than its **stick number** s(K). Whether such a knot exists is an open problem.

## Bottom line

**No counterexample was found.** The repository now contains 46 stored decimal polygons at the reported minimal stick counts: the earlier 25 (eight torus knots and 17 ten-stick knots) and 21 further ten-stick polygons. Exact-decimal interval checks establish the geometric Millett and Rawdon inequality for each set. Knot identification for the stored coordinates remains numerical or invariant-based, not a rigorous proof of named-knot identity. The new exact-stick conclusions below therefore remain conditional on that identification. See [Results](#results) and the [ten-stick campaign](results/ten_new_stick_knots/README.md) for the different lower-bound routes and limitations.

- the torus knots T(3,7), T(3,8), T(4,5), T(5,6), T(6,7), T(7,8), T(8,9) and T(9,10), with 12, 12, 10, 12, 14, 16, 18 and 20 sticks;
- 17 of the 19 four-bridge knots whose stick number Cantarella, Rechnitzer, Schumacher and Shonkwiler proved to be exactly 10, plus 21 other stored ten-stick polygons (38 ten-stick files in total). Of those 21, the exact stick and equilateral stick numbers of K13n592 and K15n41127 were already published; K13n593 has a previously published exact stick number but an additional equal-stick candidate here.

For the earlier 21 cases, we did not find previously published equilateral minimal polygons in the sources checked. No publication-priority claim is made for the four newer torus-knot coordinate sets. Of the 21 later ten-stick polygons, 12 came from crossing changes and nine from the earlier 15-crossing pool. Eighteen candidate named exact-stick conclusions were **not found in KnotInfo or the literature checked**, which is not a claim of novelty or an exhaustive review. K13n586, the one remaining member of the Cantarella 19 without a saved equilateral 10-gon here, has a ten-stick source in the locally supplied Cantarella data; it is not a certified equal-stick result in this repository.

Along the way the code confirmed the "ladder" constraint on bridge-tight polygons, confirmed a symmetry no-go lemma, and produced one instructive false alarm. A flow appeared to show an obstruction for T(4,5), which turned out to be an artifact.

---

## Contents

```
equistick/                 the package (documented, tested)
    geometry.py            distances, knot-type-safe moves, Millett-Rawdon ratio
    interval_certificate.py exact-decimal and outward-rounded geometric checks
    invariants.py          projections, PD codes, Alexander polynomial, SnapPy, HFK
    certify.py             40-digit certificates, polishing, torus verification
    torus.py               T(p, p+1) constructions, symmetric no-go scan
    flows.py               equalizer #1: path lifting (the false-alarm flow)
    optimize.py            equalizers #2 (clearance floor) and #3 (safe homotopy)
    reduce.py              lower the stick count by annealing
    data.py                access to Eddy's stick-knot-gen data
    crss.py                local Cantarella NetCDF indexing and polygon access
    coxeter.py             exact full-strand S5/D4 Wirtinger-map checks
    crossing_change.py     single-edge crossing proposals and witnesses
scripts/                   experiments and verification (01 to 12)
results/                   earlier coordinates and summary.csv; ten_new_stick_knots/ holds the later campaign
legacy/                    the original research scripts, verbatim (see legacy/README.md)
requirements.txt
```

## Installation

Python 3.10 or later.

```bash
pip install numpy scipy numba mpmath snappy
git clone https://github.com/thomaseddy/stick-knot-gen
```

`snappy` brings `spherogram` (link diagrams) and the knot Floer homology calculator. Put the `stick-knot-gen` clone next to the `equistick/` directory, or set `EQUISTICK_DATA=/path/to/stick-knot-gen`. Run the scripts from `scripts/` with the package on the path:

```bash
cd scripts
export PYTHONPATH=..
python 01_ladder_check.py
```

The first run of each module compiles its numba kernels, which takes a few seconds.

---

## Background

### The question

The **stick number** s(K) is the fewest straight segments needed to build a polygon of knot type K. The **equilateral stick number** e(K) is the same count when every segment must have the same length. Clearly e(K) ≥ s(K). Whether equality always holds is open. Every known lower bound on s (crossing number, bridge and superbridge index) ignores edge lengths, so none of them can separate the two.

### Knot-type-safe moves

Sliding one vertex v_i in a straight line from P to P2 sweeps two triangles, (v_{i-1}, P, P2) and (P, P2, v_{i+1}). If no other edge meets either triangle, the polygon stays embedded throughout, so the knot type cannot change. This is Reidemeister's triangle move. Deleting v_i is the special case where the vertex collapses onto the segment v_{i-1}v_{i+1}. It is safe exactly when the triangle (v_{i-1}, v_i, v_{i+1}) is not pierced. The functions `geometry.safe_move` and `geometry.deletable` implement these tests conservatively: coplanar or near-degenerate cases count as unsafe.

### The Millett and Rawdon criterion (the certificate)

Take a polygon with mean edge length 1, and let μ be the minimum distance between non-adjacent edges. If every edge length is within min(μ/n, μ²/4) of 1, then an exactly equilateral polygon of the same knot type exists nearby. So an equal-stick polygon is *certified* by showing

```
defect = max_i |L_i - 1|  <  min(mu/n, mu^2/4)
```

and `geometry.mr_ratio` returns defect divided by that bound. Values below 1 certify. `certify.mr_certificate_mp` recomputes everything in 40-digit arithmetic.

`equistick.interval_certificate` instead reads the stored decimal strings exactly as rational coordinates. For each nonadjacent edge pair it minimizes squared separation over both segment parameters using an exact rational convex quadratic: an interior stationary point, if feasible, and all four clamped boundary projections. This also handles parallel pairs. Outward-rounded `mpmath.iv` then bounds every edge length, the interval mean, normalized defect, normalized clearance and theorem threshold. It accepts only when the upper defect endpoint is strictly below the lower threshold endpoint. Pair-count and time budgets yield `inconclusive`, never a negative theorem claim. The generated JSON contains exact binary-rational interval endpoints (as fraction strings), file hashes and the code revision. This is a geometric existence certificate only: it does not rigorously identify the knot or prove its stick number.

### The length map

The map from polygons to their edge-length vectors is a submersion except at collinear polygons. (The only length change that cannot be achieved to first order is one that requires all edges to be parallel.) So one can ask for any small change of lengths and realize it with a minimal-norm vertex displacement. This leaves a fibre of dimension 2n − 6 (modulo rigid motions) that can be used for other goals, such as pushing edges apart. `certify.length_jacobian` supplies the operators J, Jᵀ and the Gram matrix M = JJᵀ.

### The ladder lemma (our own observation)

Call a knot **bridge-tight** if s(K) = 2b(K) + 2, where b is the bridge index. This is the smallest value the superbridge inequality b < sb ≤ s/2 allows. Examples include the trefoil, 8₁₉, 8₂₀, T(p, p+1) and the 19 ten-stick knots.

Take any (2b + 2)-gon of such a knot. Every generic direction sees at least b local maxima. Milnor's formula says the total curvature is 2π times the average number of maxima, so the turning angles sum to at least 2πb. Therefore the interior angles satisfy Σβᵢ ≤ 2π.

If every edge also has length 1, then |v_{i+1} − v_{i−1}| = 2 sin(βᵢ/2) ≤ βᵢ. So the closed "rails" through the even and through the odd vertices have total length at most 2π, while 2b + 2 unit "rungs" zigzag between them. The polygon must look like a thin ladder.

`geometry.angle_sum` and `geometry.rail_lengths` measure these quantities.

### The symmetry no-go (our own observation)

The classical torus-knot constructions put the vertices on two rings, with a rotation that advances every vertex two steps. In that family, equal lengths force each odd vertex into the vertical plane bisecting its neighbours. Reflection in that plane then swaps each odd edge with an even edge at the same heights. So every even/odd crossing seen down the axis is an actual intersection, and no such polygon is knotted.

More generally, if a polygon's rigid symmetries act transitively on its edges, orientation-preserving ones force it to be planar. So an edge-transitive knotted polygon must be amphichiral. Symmetry cannot hand you an equal-stick chiral knot for free.

---

## How the search works

```
starting polygon
  torus knots:      symmetric two-ring construction (torus.torus_poly)
  ten-stick knots:  Eddy's equilateral 11- or 12-stick polygon
        |
        v
reduce stick count if needed          reduce.reduce_to (annealing + deletable)
        |
        v
equalize edge lengths                 optimize.clearance_floor_solve   (#2)
                                      optimize.fatten + homotopy_equalize (#3)
        |
        v
certify                               geometry.mr_ratio, certify.mr_certificate_mp
        |
        v
re-identify knot type                 invariants.identify (SnapPy), or
                                      certify.verify_torus (Alexander + HFK)
```

The later ten-stick campaign also starts from any inventoried 10-gon, proposes a vertex passage through exactly one edge, numerically identifies the resulting knot, checks the target Wirt_Hm diagram's rank-four S5/D4 map exactly, then safely equalizes and interval-certifies a selected candidate. The finite-group check establishes a bridge lower bound for the diagram; numerical diagram-to-table and polygon-to-table comparisons remain a distinct, non-formal identity step. The final 12 crossing-derived polygons and selected replayable witnesses are archived under `results/ten_new_stick_knots/`.

### The three equalizers, and why there are three

| # | Where | Method | Preserves type by construction? | Outcome |
|---|---|---|---|---|
| 1 | `flows.py` | Path lifting through the length map with safe moves; in-fibre repulsion; fibre ascent | Yes | Recovered an equilateral 8₁₉. On T(4,5) it produced a **false obstruction signal** (see Results). |
| 2 | `optimize.clearance_floor_solve` | SLSQP: minimize Σ(Lᵢ − 1)² subject to every non-adjacent distance ≥ μ₀ | **No**: iterates can jump through edges | Certified T(4,5) through T(7,8), starting near equilateral. From lopsided starts it jumped to equilateral unknots. |
| 3 | `optimize.homotopy_equalize` | Move target lengths to all-ones in stages; solve each stage with #2's floor; accept a stage only if the vertex-by-vertex transition passes `safe_move` | Yes | Certified 12 of the ten-stick knots within seconds each. |

Scanning the clearance floor μ₀ in #2 is also a diagnostic. If the best reachable length defect only goes to zero as μ₀ → 0, the polygon is being squeezed toward a self-intersection: the "collapse signature" of a possible obstruction. If the defect reaches about 10⁻¹⁰ at a positive μ₀, the knot has an equal-stick version with room to spare.

### Stick reduction

`reduce.reduce_once` anneals with random safe vertex moves on the energy E = minᵢ Pᵢ. Here Pᵢ adds up, over every edge piercing triangle i, the barycentric weight of the apex at the piercing point. A piercing edge can only leave through the base v_{i−1}v_{i+1}, where that weight is 0. Every 50 steps any vertex with Pᵢ = 0 is tested with `deletable` and removed if possible.

---

## Module reference

| Module | Key functions | Notes |
|---|---|---|
| `geometry` | `seg_seg`, `min_dist`, `pair_dists`, `clearance_grad`, `seg_tri`, `safe_move`, `deletable`, `lengths`, `normalize`, `mr_ratio`, `interior_angles`, `angle_sum`, `rail_lengths` | numba-compiled kernels; all safety tests conservative |
| `invariants` | `pd_code`, `alexander_abs`, `torus_alexander_abs`, `is_torus`, `identify`, `hfk` | Compares \|Δ\| on the unit circle, which ignores units and chirality; `identify` prefers Hoste-Thistlethwaite-Weeks names |
| `certify` | `mr_certificate_mp`, `length_jacobian`, `polish`, `verify_torus` | `polish` is not path-safe; always re-identify afterwards |
| `torus` | `torus_poly`, `STARTS`, `symmetric_scan`, `star_poly`, `symmetric_equilateral_check` | `STARTS` holds the parameter sets used for every run |
| `flows` | `agitate`, `step`, `equalize`, `fiber_ascent`, `equalize2` | Equalizer #1 |
| `optimize` | `dist_jac`, `clearance_floor_solve`, `stage_solve`, `safe_path`, `fatten`, `homotopy_equalize` | Equalizers #2 and #3 |
| `reduce` | `tri_pierce_weight`, `penalties`, `reduce_once`, `reduce_to` | Stick-count reduction |
| `data` | `load_eddy`, `eddy_available`, `exact_stick_numbers`, `seed_for`, `TEN_STICK_19` | `seed_for` gives reproducible per-knot seeds |
| `interval_certificate` | exact-decimal segment distance and outward interval MR checks | Separate geometry proof for each stored decimal polygon; no named-knot identification |
| `crss`, `coxeter`, `crossing_change` | NetCDF source indexing, exact S5/D4 diagram checks, crossing witnesses | Index construction runs in a short-lived process to avoid retaining the large NetCDF allocation |

### Using it on a new knot

```python
import numpy as np
from equistick.data import load_eddy, seed_for
from equistick.reduce import reduce_to
from equistick.optimize import fatten, homotopy_equalize
from equistick.geometry import mr_ratio
from equistick.invariants import identify

name = 'K13n1192'
rng = np.random.default_rng(seed_for(name))
V = reduce_to(load_eddy(name), 10, rng)           # None if the annealer gives up
if V is not None and identify(V) == name:
    E, t, mu0, done = homotopy_equalize(fatten(V, rng), mu_floor=0.9)
    print(done, mr_ratio(E)[0] < 1, identify(E))
```

---

## Scripts and reproduction

Run each from `scripts/` with `PYTHONPATH=..`. Times are for one CPU core.

| Script | What it does | Time |
|---|---|---|
| `01_ladder_check.py` | Angle sums and rail lengths for every bridge-tight equilateral polygon in Eddy's data | seconds |
| `02_symmetric_nogo.py` | Samples the symmetric equilateral family at 6 to 12 sticks over every angular step; counts knotted samples | ~1 minute |
| `03_false_collapse.py` | Reproduces the false obstruction trace for T(4,5) | seconds |
| `04_torus_family.py p [trials]` | Unbounded single-knot clearance-floor search; use `08_torus_batch.py` for a hard deadline | seconds to minutes |
| `05_tenstick.py K11n71,... [seconds]` | Reduce, fatten, safe homotopy, certify, for each named knot | seconds to minutes per knot |
| `06_verify_results.py` | Re-verifies the earlier top-level coordinate files in `results/` from scratch and writes `results/summary.csv`; does not traverse `ten_new_stick_knots/` | ~1 minute |
| `07_parallel.py --budget 1800 --workers 5` | Ran the earlier five pending knots concurrently, subject to the effective CPU quota; enforces a hard wall-clock budget per knot | up to 30 minutes with five workers, plus cache warmup |
| `08_torus37.py --budget 1800 --workers 2` | Samples explicit T(3,7) and T(3,8), checks their invariants, reduces safely to 12 sticks, homotopy equalizes, validates saved coordinates; hard wall budget per knot | up to 30 minutes with two workers, plus cache warmup |
| `08_interval_certificates.py --expected-count 25` | Writes `results/interval_certificates.json` with exact-decimal and interval geometric certificates for the earlier 25 stored polygons, without knot identification; the later 21 have a separate report | seconds |
| `08_torus_batch.py --budget 1800 --workers 2` | T(8,9) and T(9,10) preparation with deterministic starts, per-knot hard deadline and final-coordinate validation | both completed; up to 30 minutes per knot |
| `09_verify_six.py` and `verify_coxeter.py` | Check the saved 15-crossing candidates and propagate the listed S5/D4 seed images through the full Wirt_Hm Wirtinger diagrams, checking every relation and generation exactly | report in `results/ten_new_stick_knots/pool_verification.json` |
| `10_fifteen_pool.py --budget 1200 --workers 1` | Bounded 29-knot 11-stick pool reduction and safe equalization, with per-knot outcomes and deterministic seeds | completed run: three additional saved 10-gons; 26 reduction timeouts |
| `11_crossing_changes.py --source-budget 60 --candidate-budget 180` | Inventory Eddy, locally provided Cantarella NetCDF and our 10-gons; try single-vertex, single-edge crossings; checkpoint per-source outcomes | 870 source grids completed; 12 timed out, not retried |
| `12_equalize_crossings.py --candidate-budget 180 --max-attempts 1` | Safely equalize selected map-verified crossing candidates and save interval-certified decimal 10-gons | 12 additional saved 10-gons |

From the repository root, run:

```bash
.venv/bin/python scripts/07_parallel.py --knots K13n285,K13n602,K13n608,K13n1192,K13n5018 --budget 1800 --workers 5
```

The runner uses spawned processes and one numerical-library thread per worker. The per-knot budget includes worker startup and final validation. It publishes coordinates only after checking the saved file's Millett-Rawdon ratio, 40-digit certificate, known stick number, and knot identification in three projections. Per-knot logs and a machine-readable run manifest go to `results/logs/`; outcomes are appended to `results/tenstick.log` and provenance to `results/RUNLOG.md`. Timeouts are search logs, not evidence of an obstruction. The later scripts run from the repository root; their exact execution settings, dependencies and saved manifests are in [the campaign run log](results/ten_new_stick_knots/RUNLOG.md) and [results documentation](results/ten_new_stick_knots/README.md). The NetCDF file is a separate user-provided input, not part of Git. Do not interpret these script-table examples as an instruction to rerun timed-out source grids.

For the two superbridge-tight torus knots, run from the repository root:

```bash
.venv/bin/python scripts/08_torus37.py --knots T3_7,T3_8 --budget 1800 --workers 2
```

The torus runner uses 24 vertices sampled from `(R + r cos(qt)) (cos(3t), sin(3t)), r sin(qt)` with R = 2.5, r = 1 and q = 7 or 8. It verifies the polygonal start using four Alexander projections and HFK (not just the smooth curve), then calls `reduce_to` for knot-type-safe vertex deletion. It rechecks type after reduction and after equalization/polishing. Only 17-digit saved coordinates passing the float MR ratio, 40-digit MR check, four projections and HFK are published. Logs are in `results/torus37.log` and `results/logs/`. The separate interval checker addresses the geometric inequality, not knot identity.

For the next torus pair, from the repository root (using the local dependencies):

```bash
EQUISTICK_DATA=./stick-knot-gen \
  .venv/bin/python scripts/08_torus_batch.py \
  --knots T8_9,T9_10 --budget 1800 --workers 2 --trials 1000
```

`--workers 2` ran on the shared five-CPU quota; the optional default of one leaves more capacity for other work. The per-knot deadline includes worker startup, all trials and final validation as observed by the supervisor; the operating-system `process.start()` call itself is not interruptible by this Python supervisor. Seeds are `seed_for('T8_9')` and `seed_for('T9_10')`, plus `--run-index` (default 0). Starts in `torus.STARTS` were found by a short deterministic 250-trial `symmetric_scan`, not by an equal-stick certification. If a start is absent, an empty scan is logged without an exception. Each trial records floor, defect, clearance, ratio and eligibility (float MR plus preliminary Alexander match) in `results/logs/torus_*.log`. The manifest, append-only `RUNLOG.md` and `torus_search.log` record outcomes and provenance. Only 17-digit coordinates that pass a float MR ratio below 1, the 40-digit MR check and `verify_torus` (four Alexander projections, correct genus, absolute tau, exact Alexander-coefficient rank, fibredness, L-space property and crossing lower bound) after serialization may be published. Both searches succeeded within their 1800-second-per-knot limits. A timeout or unsuccessful trial would be only "not found within budget," never an obstruction.

---

## Results

### 1. Geometrically certified equal-stick polygons

The earlier 25 were re-verified from scratch by `06_verify_results.py` and the exact-decimal interval checker. The later 21 have separate interval certificates in `results/ten_new_stick_knots/interval_certificates.json`; the nine earlier campaign rows were carried into that report without a new check. All 46 have a saved polygon, geometric interval pass and numerical final-coordinate identification, but *named-knot* exact stick and equilateral stick conclusions still require a rigorous identification. For the later 21, the exact S5 or D4 diagram checks establish a four-bridge lower bound for the checked diagram, not a formal identification of that diagram with the stored polygon.

"Defect" is the largest deviation of an edge length from the mean (mean scaled to 1). The certificate requires defect < bound. "Angles" is Σβᵢ; the ladder bound of 2π ≈ 6.283 applies only to bridge-tight polygons, not to T(3,7) or T(3,8). In the later rows, defect is the rounded *interval upper endpoint*, μ the rounded *interval lower endpoint*, and bound the rounded *threshold lower endpoint*. They are display approximations, not substitute certificate bounds. An angle marked `n/r` was not recorded in the archived campaign table. All later rows have three numerical SnapPy polygon-to-table matches, an exact full-strand rank-four map and a separate numerical Gauss-diagram comparison. `R` means reduction or an existing Eddy 10-gon; `C` means crossing change followed by safe equalization. Sources and detailed checks are in the [campaign results](results/ten_new_stick_knots/README.md).

| Knot | Sticks (s if identified) | Defect | μ | Bound | Angles | Identification |
|---|---|---|---|---|---|---|
| T(3,7) | 12 | 9.8e-17 | 0.0319 | 2.55e-04 | 8.837 | Alexander ×4; HFK genus 6, L-space, fibred, τ = −6, rank 9; 14 crossings |
| T(3,8) | 12 | 8.3e-17 | 0.0295 | 2.18e-04 | 6.212 | Alexander ×4; HFK genus 7, L-space, fibred, τ = −7, rank 11; 16 crossings |
| T(4,5) | 10 | 1.2e-16 | 0.0100 | 2.50e-05 | 4.154 | Alexander ×4; HFK genus 6, L-space, fibred, τ = −6; 15 crossings |
| T(5,6) | 12 | 1.0e-16 | 0.0050 | 6.25e-06 | 3.820 | Alexander ×4; HFK genus 10, L-space, fibred, τ = 10; 24 crossings |
| T(6,7) | 14 | 1.3e-16 | 0.0025 | 1.56e-06 | 2.646 | Alexander ×4; HFK genus 15, L-space, fibred, τ = −15; 35 crossings |
| T(7,8) | 16 | 1.3e-16 | 0.0010 | 2.50e-07 | 3.053 | Alexander ×4; HFK genus 21, L-space, fibred, τ = 21; 48 crossings |
| T(8,9) | 18 | 1.6e-16 | 0.001001 | 2.50e-07 | 3.663 | Alexander ×4; HFK genus 28, rank 15, L-space, fibred, τ = 28; 63 crossings |
| T(9,10) | 20 | 1.4e-16 | 0.0005 | 6.25e-08 | 2.889 | Alexander ×4; HFK genus 36, rank 17, L-space, fibred, τ = 36; 80 crossings |
| K11n71 | 10 | 4.5e-14 | 0.0153 | 5.85e-05 | 4.992 | SnapPy ×3 |
| K11n75 | 10 | 1.5e-13 | 0.0134 | 4.50e-05 | 4.707 | SnapPy ×3 |
| K11n76 | 10 | 6.5e-11 | 0.0113 | 3.20e-05 | 4.769 | SnapPy ×3 |
| K11n78 | 10 | 1.3e-10 | 0.0312 | 2.43e-04 | 4.907 | SnapPy ×3 |
| K13n1192 | 10 | 8.3e-11 | 0.0163 | 6.62e-05 | 4.695 | SnapPy ×3 |
| K13n225 | 10 | 2.0e-16 | 0.0189 | 8.97e-05 | 4.772 | SnapPy ×3 |
| K13n230 | 10 | 2.2e-10 | 0.0219 | 1.20e-04 | 5.267 | SnapPy ×3 |
| K13n285 | 10 | 1.1e-11 | 0.0100 | 2.52e-05 | 5.210 | SnapPy ×3 |
| K13n288 | 10 | 6.3e-16 | 0.0055 | 7.68e-06 | 5.000 | SnapPy ×3 |
| K13n307 | 10 | 2.6e-15 | 0.0232 | 1.34e-04 | 4.883 | SnapPy ×3 |
| K13n5018 | 10 | 2.0e-16 | 0.0187 | 8.71e-05 | 4.617 | SnapPy ×3 |
| K13n584 | 10 | 1.1e-11 | 0.0275 | 1.89e-04 | 4.986 | SnapPy ×3 |
| K13n602 | 10 | 4.3e-10 | 0.0108 | 2.91e-05 | 3.901 | SnapPy ×3 |
| K13n603 | 10 | 1.5e-14 | 0.0262 | 1.71e-04 | 4.876 | SnapPy ×3 |
| K13n604 | 10 | 1.7e-16 | 0.0062 | 9.63e-06 | 3.676 | SnapPy ×3 |
| K13n607 | 10 | 9.4e-13 | 0.0007 | 1.21e-07 | 5.702 | SnapPy ×3 |
| K13n608 | 10 | 1.6e-12 | 0.0242 | 1.46e-04 | 5.333 | SnapPy ×3 |
| **Published exact s=e=10, additional saved polygons** | | | | | | |
| K13n592 | 10 | 7.84e-17 | 1.56e-02 | 6.07e-05 | n/r | C; S5; published by Blair et al. (2020) |
| K15n41127 | 10 | 2.30e-16 | 8.88e-03 | 1.97e-05 | n/r | C; S5; published by Blair et al. (2020) |
| **Known s=10, additional equilateral finding** | | | | | | |
| K13n593 | 10 | 6.10e-11 | 6.71e-03 | 1.12e-05 | n/r | C; S5; published s=10; e=10 conditional here |
| **Named s=e=10 findings not found in KnotInfo or the literature checked** | | | | | | |
| K13n501 | 10 | 6.07e-13 | 6.72e-04 | 1.13e-07 | n/r | C from K15n41235; S5; conditional identity |
| K13n585 | 10 | 5.88e-11 | 1.41e-03 | 4.94e-07 | n/r | C from 11_6; S5; conditional identity |
| K15n40180 | 10 | 4.26e-11 | 8.46e-05 | 1.79e-09 | n/r | C from K13n586; D4; conditional identity |
| K15n40184 | 10 | 2.39e-16 | 3.06e-02 | 2.34e-04 | n/r | R from Eddy 11-gon; D4; conditional identity |
| K15n40185 | 10 | 7.13e-13 | 1.20e-02 | 3.58e-05 | n/r | R; D4; conditional identity |
| K15n41189 | 10 | 1.77e-16 | 1.74e-02 | 7.54e-05 | n/r | R; S5; conditional identity |
| K15n41193 | 10 | 2.25e-11 | 6.36e-04 | 1.01e-07 | n/r | R; S5; conditional identity |
| K15n41235 | 10 | 2.33e-15 | 1.07e-03 | 2.85e-07 | n/r | R; S5; conditional identity |
| K15n43517 | 10 | 7.58e-12 | 9.98e-03 | 2.49e-05 | n/r | C from K13n608; S5; conditional identity |
| K15n45460 | 10 | 3.13e-16 | 1.18e-02 | 3.46e-05 | n/r | R; S5; conditional identity |
| K15n46935 | 10 | 1.89e-11 | 7.45e-03 | 1.39e-05 | n/r | C from K13n225; S5; conditional identity |
| K15n47800 | 10 | 1.93e-12 | 3.60e-03 | 3.24e-06 | n/r | R; S5; conditional identity |
| K15n48957 | 10 | 1.54e-10 | 7.38e-03 | 1.36e-05 | n/r | C from K13n225; S5; conditional identity |
| K15n49035 | 10 | 1.19e-11 | 3.42e-04 | 2.93e-08 | n/r | C from K13n288; S5; conditional identity |
| K15n51709 | 10 | 1.15e-11 | 3.48e-03 | 3.02e-06 | n/r | C from K13n3969; D4; conditional identity |
| K15n52941 | 10 | 6.42e-15 | 1.73e-04 | 7.46e-09 | n/r | R; S5; conditional identity |
| K15n52944 | 10 | 5.25e-14 | 8.84e-03 | 1.95e-05 | n/r | C from K13n1192; S5; conditional identity |
| K15n59007 | 10 | 4.02e-11 | 2.94e-03 | 2.16e-06 | n/r | R, Eddy 10-gon copied unchanged; D4; conditional identity |

The T(p,p+1) stick numbers come from Jin's theorem, s(T(p,q)) = 2q for p < q < 2p. T(3,7) and T(3,8) have known stick number 12 as listed in `AGENTS.md`; this specific value is not an instance of that p < q < 2p formula. The earlier 17 ten-stick knots' stick numbers come from the four-bridge lower bound together with the Cantarella group's 10-stick examples. For the later 21, two named s=e results were published by Blair et al.; K13n593 has published s=10, while its equal-stick 10-gon is an additional conditional finding here. The other 18 have checked diagrammatic lower bounds and stored upper-bound polygons, but their named exact s=e=10 conclusions depend on the numerical knot identification. **“Not found in KnotInfo or the literature checked” is a bounded literature status, not a novelty or publication-priority claim.** The sources checked for these exact-stick claims were KnotInfo's stick-number description, Blair et al. (2020) and Cantarella et al. (2026); this was not an exhaustive literature search. The earlier 25 have the margins shown in their own rows, not the later campaign's values.

### 2. Validation and later search results

The pipeline recovered an equilateral 8-stick 8₁₉ = T(3,4), which Millett found first; it had defeated Rawdon and Scharein's 2002 search. The earlier reduction annealer produced 10-stick versions of 17 of the Cantarella group's 19 knots. The latest five of those each needed one successful reduction. The K13n602 first run hit a numerical division error; a regression-tested conservative guard was added, and a separate full-budget retry succeeded. Logs for both attempts are retained.

The later Wirt_Hm pool contains nine saved 15-crossing 10-gons and 25 pool knots with listed rank-four maps, including three overlaps. **31 distinct maps passed exact strand propagation, every Wirtinger relation, and group generation.** Four additional pool rows had no listed map, which is `map_missing`, not a failed map. A bounded 29-knot reduction search saved three of the nine and timed out on 26 after 1,200 seconds per candidate. This did not establish that any target lacks a 10-gon.

The crossing-change pass inventoried 882 ten-stick sources from Eddy, locally supplied Cantarella data and our saved polygons, irrespective of source knot type. **870 source grids completed, 12 timed out at 60 seconds and none were left unattempted**; the completed grids tried 2,923,200 proposed moves, recorded 164,022 single-edge crossings and reached 33 distinct four-Wirtinger knots. Of those 33, 32 had passing exact rank-four maps; K14n22583 lacked a passing listed map, so no exact-stick conclusion is asserted for it. One selected map-verified candidate per reached target was eligible for at most 180 seconds of safe equalization. **Twelve additional 10-gons** were saved and interval-certified, including two of the earlier 26 reduction timeouts, K15n43517 (from K13n608) and K15n51709 (from K13n3969). The other 24 priority knots were not reached in the 870 *completed* source grids. No timed-out source was retried, so this is not a complete-grid negative result. See [the archived manifest, selected crossing witnesses and interval reports](results/ten_new_stick_knots/README.md).

The large NetCDF source index initially caused memory exits. A short-lived indexing subprocess releases that memory before source scanning; the run resumed from a preserved checkpoint rather than repeating completed sources. Exact finite-group checking, bounded per-source work, a single results writer, replayable selected-witness coordinates and qualified timeout statuses improve reproducibility. None of these optimizations turns numerical SnapPy matches into formal knot-identity proofs. The initial equalization launcher failed before computation because its log directory did not exist; the corrected launch completed. Both outcomes are recorded in the campaign run log.

### 3. The torus family: clearance shrinks but stays positive

This table shows the highest clearance floor at which equalizer #2 certified during each recorded search. The earlier family runs used 6 to 12 random starts; the newer bounded runs used the trial counts in `results/RUNLOG.md`. These are floors this optimizer reached, not true maxima.

| Knot | Sticks | Certified at μ₀ | Failed at μ₀ |
|---|---|---|---|
| T(3,4) | 8 | 0.02 | 0.04 |
| T(4,5) | 10 | 0.01 | 0.02 |
| T(5,6) | 12 | 0.005 | 0.01 |
| T(6,7) | 14 | 0.0025 | 0.005 |
| T(7,8) | 16 | 0.001 | 0.0025 |
| T(8,9) | 18 | 0.001 | not tested |
| T(9,10) | 20 | 0.0005 | not tested |

The ladder is real (angle sums 2.6 to 4.2, all under 2π) and gets thinner with p, but it does not obstruct up to p = 9. If the decay continues, the family may become too thin to certify numerically before it becomes provably impossible, so settling it needs a proof, not just more computation.

### 4. Ladder lemma on published data (`01`)

Angle sums for bridge-tight equilateral polygons in Eddy's data:

| Knot | n | Angle sum |
|---|---|---|
| 3₁ | 6 | 4.827 |
| 8₁₉ | 8 | 5.435 |
| 8₂₀ | 8 | 5.027 |
| K13n592 | 10 | 4.739 |
| K15n41127 | 10 | 4.942 |

All are below 2π, as the lemma requires for bridge-tight polygons. The new T(3,7) polygon is not bridge-tight and has a larger angle sum.

### 5. Symmetry no-go (`02`)

At 6, 8, 10 and 12 sticks, 816, 808, 796 and 791 embedded equilateral symmetric polygons were sampled over every angular step. **Zero were knotted.** The legacy grid scan found the same.

### 6. The false alarm (`03`)

Equalizer #1, started near the symmetric T(4,5), drove the length defect from 2.6 × 10⁻² to 1.8 × 10⁻⁵ over 40 steps, while μ fell in lockstep: μ/defect stayed between 1.5 and 2. That is exactly the "collapse signature" of an obstruction, and the same thing Rawdon and Scharein saw for 8₁₉. But the flow was heading for the symmetric equal-length configuration, which the no-go lemma says is always singular. Equalizer #2 then certified T(4,5) with μ = 0.01.

Lesson: a numerical collapse is weak evidence. Only certified positive results carry weight.

### 7. Unfinished

| Knot | Status |
|---|---|
| K13n586 | No stored equilateral 10-gon here; the locally supplied Cantarella data did provide a ten-stick source, but it was not saved here as an equilateral certificate |
| K13n593 | Now has a saved, interval-certified equal-stick 10-gon from a crossing change; named identity remains numerical |

The 24 other priority reduction timeouts and the 12 incomplete crossing-source grids are detailed in the campaign README. Failing to reduce or to reach a knot within the recorded budgets says nothing about nonexistence.

---

## Limitations

1. **The geometric inequality is checked rigorously, but knot identity is not.** The legacy 40-digit checker still uses floats on input and is not a rigorous certificate. The separate interval checker uses the stored decimal strings exactly, exact rational segment minimization and outward-rounded arithmetic for the remaining bounds. It establishes the Millett and Rawdon inequality for those exact decimal polygons. It does not prove that they have the knot names in the results table, nor does it independently verify the theorem hypotheses or knot identification. Replacing the interval results with the legacy checker would lose this guarantee.

2. **Knot identification relies on invariants.**
   - Hyperbolic knots are identified by SnapPy's `identify()`. It matches the complement against census manifolds using numerically computed hyperbolic structures, without SnapPy's rigorous `verified=True` mode. It also ignores chirality, which is harmless here because e and s are mirror invariant.
   - Torus knots are matched on every invariant checked (Alexander polynomial in four projections; knot Floer homology genus, fibredness, L-space property, absolute τ, total rank; crossing number after simplification). The matching ignores chirality, so torus labels are up to mirror image. Knot Floer homology is not known to detect T(p, p+1) in general, so this is strong evidence rather than proof.
   - Two routes would make it rigorous: exhibit a knot-type-safe path back to the symmetric construction, whose type follows from Jin's work, or use a rigorous recognition tool.
   - SnapPy sometimes reports a non-table census name for a table knot (K15n41127 comes back as K6_37). Scripts that compare names can therefore report false mismatches. For the later ten-stick results, exact S5/D4 checks establish the listed diagram's four-bridge lower bound, and numerical SnapPy comparisons connect diagrams and polygons to named table knots. The checks do not replace a formal proof of those identifications.

3. **Safety tests are floating point.** `safe_move` and `deletable` are conservative, with tolerances around 10⁻⁹ to 10⁻¹², but they are not exact predicates. The final re-identification is the backstop.

4. **Equalizer #2 does not preserve knot type along its path.** Its results are trusted only because every final polygon was re-identified. Equalizer #3 and the reducer do preserve type (up to point 3).

5. **Negative results mean nothing.** A flow that collapses, an optimizer that stalls, or a reducer that times out is not evidence that e(K) > s(K); the false alarm in section 6 of the Results shows how misleading such signals are. Only the certified polygons are results.

6. **Coverage is narrow and incomplete.** The earlier search covered 25 knots and only the listed torus families. The later ten-stick campaign attempted 882 polygon sources, but 12 timed out and were not retried. A knot type can occupy disconnected regions of polygon space; these runs did not exhaust any such region. A source timeout does not mean a target was unreachable. The 24 unreached priority knots were absent from completed grids only.

7. **Literature status is bounded.** The earlier 21 candidate equilateral minimal polygons were not found in the sources checked at the time. For the later 21 polygons, the published exact s=e=10 results for K13n592 and K15n41127 are explicitly separated above; K13n593's s=10 is published. The 18 other candidate named exact-stick conclusions were not found in KnotInfo's stick-number description or the papers checked, not proven novel. Some polygons were copied from or derived from public data, including Eddy's 10-gon for K15n59007. The Cantarella coordinates were unavailable to the earlier search but a locally supplied NetCDF file was used in the later crossing campaign; the external dataset is not committed here.

8. **Reproducibility.**
   - Package scripts use fixed seeds (`data.seed_for`).
   - The legacy batch seeded from Python's salted `hash()`, so it cannot be rerun exactly.
   - SLSQP outcomes can vary with the SciPy version.
   - Results files from legacy runs are included as produced. One exception: K11n76's file was overwritten by a later package run of the same pipeline.

9. **Theory is unreviewed.** The ladder lemma, the symmetry no-go and the length-map reformulation are our own arguments, checked by hand and numerically, not peer-reviewed. The no-go scan tests knottedness through a nontrivial Alexander polynomial, so it would miss knots with trivial Alexander polynomial, and it samples randomly rather than exhaustively.

10. **Compute and publication concurrency.** Early exploratory runs used one CPU core under a 300-second limit per command. The later bounded torus search used two workers and a separate 1800-second limit per knot; it does not imply exhaustive coverage of polygon space. The supervisors use exclusive hard links to avoid replacing existing result files and accept identical existing bytes on rerun. This requires hard-link support on the output filesystem; staging directories are created beneath the output directory. An unrelated writer that disregards this publication protocol and mutates a result in place remains outside its protection.

## Next steps

1. Find and certify an equilateral 10-gon for K13n586, the remaining member of the Cantarella 19 without such a saved polygon here. K13n593 is now represented by an interval-certified crossing-change 10-gon, subject to formal knot identification.
2. Extend the explicit torus-sampling approach beyond the now certified T(3,7) and T(3,8) to other superbridge-tight knots; numerical certificates are not formal proofs.
3. Push T(p, p+1) to p = 10 and beyond and fit the clearance decay. Better still, find an explicit equal-stick construction for all p, which would settle that family.
4. For publication: independent audit of the interval-arithmetic geometric certificates, rigorous knot identification, and a more complete literature and data-overlap check. The later source scan left 12 timeouts incomplete; no retry is claimed or implied.

## References

- J. A. Calvo, Geometric knot spaces and polygonal isotopy, *JKTR* 10 (2001).
- J. Cantarella, A. Rechnitzer, H. Schumacher, C. Shonkwiler, New upper bounds for stick numbers, *JKTR* 35 (2026), arXiv:2508.18263.
- T. D. Eddy, C. Shonkwiler, New stick number bounds from random sampling of confined polygons, *Exp. Math.* 31 (2022); data at github.com/thomaseddy/stick-knot-gen.
- R. Blair, T. D. Eddy, N. Morrison, C. Shonkwiler, Knots with exactly 10 sticks, *JKTR* 29 (2020).
- KnotInfo, "Stick Number" description and cited exact-stick sources (checked 2026-09-27).
- G. T. Jin, Polygon indices and superbridge indices of torus knots and links, *JKTR* 6 (1997).
- K. C. Millett, E. J. Rawdon, Energy, ropelength, and other physical aspects of equilateral knots, *J. Comput. Phys.* 186 (2003).
- E. J. Rawdon, R. G. Scharein, Upper bounds for equilateral stick numbers, *Contemp. Math.* 304 (2002).
