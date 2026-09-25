# equistick

A numerical search for a knot whose **equilateral stick number** e(K) is larger than its **stick number** s(K). Whether such a knot exists is an open problem.

## Bottom line

**No counterexample was found.** For 16 knots, the code produced an equal-stick polygon with exactly the knot's stick number, certified numerically by the Millett and Rawdon criterion and re-identified from scratch. So for these 16 knots, e(K) = s(K):

- the torus knots T(4,5), T(5,6), T(6,7) and T(7,8), with 10, 12, 14 and 16 sticks;
- 12 of the 19 four-bridge knots whose stick number Cantarella, Rechnitzer, Schumacher and Shonkwiler proved to be exactly 10.

As far as we could find, none of these equilateral minimal polygons was previously published. Eddy's public data had only 11- or 12-stick equilateral versions of the ten-stick knots, and nothing for T(4,5). The two candidate families that the original research plan singled out therefore give way, as far as the search reached. Seven of the 19 ten-stick knots remain untested or unresolved (see [Results](#results)).

Along the way the code confirmed the "ladder" constraint on bridge-tight polygons, confirmed a symmetry no-go lemma, and produced one instructive false alarm. A flow appeared to show an obstruction for T(4,5), which turned out to be an artifact.

---

## Contents

```
equistick/                 the package (documented, tested)
    geometry.py            distances, knot-type-safe moves, Millett-Rawdon ratio
    invariants.py          projections, PD codes, Alexander polynomial, SnapPy, HFK
    certify.py             40-digit certificates, polishing, torus verification
    torus.py               T(p, p+1) constructions, symmetric no-go scan
    flows.py               equalizer #1: path lifting (the false-alarm flow)
    optimize.py            equalizers #2 (clearance floor) and #3 (safe homotopy)
    reduce.py              lower the stick count by annealing
    data.py                access to Eddy's stick-knot-gen data
scripts/                   one script per experiment (01 to 06)
results/                   coordinates of every certified polygon, summary.csv, logs
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
| `04_torus_family.py p [trials]` | Clearance-floor search for T(p, p+1), then polishes, verifies and saves | seconds to minutes |
| `05_tenstick.py K11n71,... [seconds]` | Reduce, fatten, safe homotopy, certify, for each named knot | seconds to minutes per knot |
| `06_verify_results.py` | Re-verifies every file in `results/` from scratch and writes `results/summary.csv` | ~1 minute |

---

## Results

### 1. Certified equal-stick minimal polygons

All 16 were re-verified from scratch by `06_verify_results.py`. "Defect" is the largest deviation of an edge length from the mean (mean scaled to 1). The certificate requires defect < bound. "Angles" is Σβᵢ; the ladder bound is 2π ≈ 6.283.

| Knot | Sticks (= s) | Defect | μ | Bound | Angles | Identification |
|---|---|---|---|---|---|---|
| T(4,5) | 10 | 1.2e-16 | 0.0100 | 2.50e-05 | 4.154 | Alexander ×4; HFK genus 6, L-space, fibred, τ = −6; 15 crossings |
| T(5,6) | 12 | 1.0e-16 | 0.0050 | 6.25e-06 | 3.820 | Alexander ×4; HFK genus 10, L-space, fibred, τ = 10; 24 crossings |
| T(6,7) | 14 | 1.3e-16 | 0.0025 | 1.56e-06 | 2.646 | Alexander ×4; HFK genus 15, L-space, fibred, τ = −15; 35 crossings |
| T(7,8) | 16 | 1.3e-16 | 0.0010 | 2.50e-07 | 3.053 | Alexander ×4; HFK genus 21, L-space, fibred, τ = 21; 48 crossings |
| K11n71 | 10 | 4.5e-14 | 0.0153 | 5.85e-05 | 4.992 | SnapPy ×3 |
| K11n75 | 10 | 1.5e-13 | 0.0134 | 4.50e-05 | 4.707 | SnapPy ×3 |
| K11n76 | 10 | 6.5e-11 | 0.0113 | 3.20e-05 | 4.769 | SnapPy ×3 |
| K11n78 | 10 | 1.3e-10 | 0.0312 | 2.43e-04 | 4.907 | SnapPy ×3 |
| K13n225 | 10 | 2.0e-16 | 0.0189 | 8.97e-05 | 4.772 | SnapPy ×3 |
| K13n230 | 10 | 2.2e-10 | 0.0219 | 1.20e-04 | 5.267 | SnapPy ×3 |
| K13n288 | 10 | 6.3e-16 | 0.0055 | 7.68e-06 | 5.000 | SnapPy ×3 |
| K13n307 | 10 | 2.6e-15 | 0.0232 | 1.34e-04 | 4.883 | SnapPy ×3 |
| K13n584 | 10 | 1.1e-11 | 0.0275 | 1.89e-04 | 4.986 | SnapPy ×3 |
| K13n603 | 10 | 1.5e-14 | 0.0262 | 1.71e-04 | 4.876 | SnapPy ×3 |
| K13n604 | 10 | 1.7e-16 | 0.0062 | 9.63e-06 | 3.676 | SnapPy ×3 |
| K13n607 | 10 | 9.4e-13 | 0.0007 | 1.21e-07 | 5.702 | SnapPy ×3 |

The torus stick numbers come from Jin's theorem, s(T(p,q)) = 2q for p < q < 2p. The ten-stick knots' stick numbers come from the four-bridge lower bound together with the Cantarella group's 10-stick examples. The certified margins are large: the defect sits at least five orders of magnitude below the bound in every case.

### 2. Validation

The pipeline recovered an equilateral 8-stick 8₁₉ = T(3,4), which Millett found first; it had defeated Rawdon and Scharein's 2002 search. The reduction annealer independently reproduced the Cantarella group's 10-stick versions of 12 of the 19 knots (their own coordinates were not accessible; see Limitations). It usually needed only one reduction and a few seconds.

### 3. The torus family: clearance shrinks but stays positive

This table shows the highest clearance floor at which equalizer #2 certified, from 6 to 12 random starts. These are floors this optimizer reached, not true maxima.

| Knot | Sticks | Certified at μ₀ | Failed at μ₀ |
|---|---|---|---|
| T(3,4) | 8 | 0.02 | 0.04 |
| T(4,5) | 10 | 0.01 | 0.02 |
| T(5,6) | 12 | 0.005 | 0.01 |
| T(6,7) | 14 | 0.0025 | 0.005 |
| T(7,8) | 16 | 0.001 | 0.0025 |

The ladder is real (angle sums 2.6 to 4.2, all under 2π) and gets thinner with p, but it does not obstruct up to p = 7. If the decay continues, the family will become too thin to certify numerically before it becomes provably impossible, so settling it needs a proof, not more computation.

### 4. Ladder lemma on published data (`01`)

Angle sums for bridge-tight equilateral polygons in Eddy's data:

| Knot | n | Angle sum |
|---|---|---|
| 3₁ | 6 | 4.827 |
| 8₁₉ | 8 | 5.435 |
| 8₂₀ | 8 | 5.027 |
| K13n592 | 10 | 4.739 |
| K15n41127 | 10 | 4.942 |

All are below 2π, as the lemma requires, and so are all 16 new polygons.

### 5. Symmetry no-go (`02`)

At 6, 8, 10 and 12 sticks, 816, 808, 796 and 791 embedded equilateral symmetric polygons were sampled over every angular step. **Zero were knotted.** The legacy grid scan found the same.

### 6. The false alarm (`03`)

Equalizer #1, started near the symmetric T(4,5), drove the length defect from 2.6 × 10⁻² to 1.8 × 10⁻⁵ over 40 steps, while μ fell in lockstep: μ/defect stayed between 1.5 and 2. That is exactly the "collapse signature" of an obstruction, and the same thing Rawdon and Scharein saw for 8₁₉. But the flow was heading for the symmetric equal-length configuration, which the no-go lemma says is always singular. Equalizer #2 then certified T(4,5) with μ = 0.01.

Lesson: a numerical collapse is weak evidence. Only certified positive results carry weight.

### 7. Unfinished

| Knot | Status |
|---|---|
| K13n285 | Annealer found no 10-stick version in 240 s |
| K13n602 | Annealer found no 10-stick version in 60 s |
| K13n608, K13n5018 | Not attempted (Eddy's data starts at 12 sticks, so two reductions are needed) |
| K13n1192 | Not attempted |
| K13n586, K13n593 | No starting data in Eddy's repository |

Failing to reduce within a time budget says nothing about these knots. It is a search limitation, not evidence.

---

## Limitations

1. **Certificates are numerical, not formal proofs.** Coordinates are 64-bit floats. The Millett and Rawdon test is recomputed in 40-digit arithmetic from those floats, but not with interval arithmetic, and the high-precision distance routine follows the same branch logic as the float one. The margins are at least five orders of magnitude, so rounding is not a plausible failure mode, but a publishable proof should use interval arithmetic throughout.

2. **Knot identification relies on invariants.**
   - Hyperbolic knots are identified by SnapPy's `identify()`. It matches the complement against census manifolds using numerically computed hyperbolic structures, without SnapPy's rigorous `verified=True` mode. It also ignores chirality, which is harmless here because e and s are mirror invariant.
   - Torus knots are matched on every invariant checked (Alexander polynomial in four projections; knot Floer homology genus, fibredness, L-space property, τ, total rank; crossing number after simplification). Knot Floer homology is not known to detect T(p, p+1) in general, so this is overwhelming evidence rather than proof.
   - Two routes would make it rigorous: exhibit a knot-type-safe path back to the symmetric construction, whose type follows from Jin's work, or use a rigorous recognition tool.
   - SnapPy sometimes reports a non-table census name for a table knot (K15n41127 comes back as K6_37). Scripts that compare names can therefore report false mismatches, but not false matches.

3. **Safety tests are floating point.** `safe_move` and `deletable` are conservative, with tolerances around 10⁻⁹ to 10⁻¹², but they are not exact predicates. The final re-identification is the backstop.

4. **Equalizer #2 does not preserve knot type along its path.** Its results are trusted only because every final polygon was re-identified. Equalizer #3 and the reducer do preserve type (up to point 3).

5. **Negative results mean nothing.** A flow that collapses, an optimizer that stalls, or a reducer that times out is not evidence that e(K) > s(K); section 6 of the Results shows how misleading such signals are. Only the certified polygons are results.

6. **Coverage is narrow.** The search covered 16 knots, torus knots only up to T(7,8), and for each knot only the regions of polygon space reachable from the starting data. A knot type can occupy several disconnected regions at the minimal stick count, and we sampled at most a few.

7. **"New" is as far as we could find.** Eddy's repository had no equal-stick minimal versions of these knots. The Cantarella group's coordinates (the non-equilateral 10-stick polygons for the 19 knots and their torus-knot data) sit on Harvard Dataverse, which blocked automated access, and we did not survey every other source.

8. **Reproducibility.**
   - Package scripts use fixed seeds (`data.seed_for`).
   - The legacy batch seeded from Python's salted `hash()`, so it cannot be rerun exactly.
   - SLSQP outcomes can vary with the SciPy version.
   - Results files from legacy runs are included as produced. One exception: K11n76's file was overwritten by a later package run of the same pipeline.

9. **Theory is unreviewed.** The ladder lemma, the symmetry no-go and the length-map reformulation are our own arguments, checked by hand and numerically, not peer-reviewed. The no-go scan tests knottedness through a nontrivial Alexander polynomial, so it would miss knots with trivial Alexander polynomial, and it samples randomly rather than exhaustively.

10. **Compute.** Everything ran on one CPU core under a 300-second limit per command, and background jobs did not survive between commands. That is why the ten-stick batch is incomplete.

## Next steps

1. Finish the seven remaining ten-stick knots: longer reduction budgets, two-stage reduction for K13n608 and K13n5018, and starting polygons built from diagrams for K13n586 and K13n593.
2. Move to *superbridge-tight* knots such as T(3,7) and T(3,8), which need exactly 12 sticks. Their minimal polygons are not thin ladders, so the geometry differs from everything tested here.
3. Push T(p, p+1) to p = 8, 9, 10 and fit the clearance decay. Better still, find an explicit equal-stick construction for all p, which would settle that family.
4. For publication: interval-arithmetic certificates, rigorous identification, and a check with the Cantarella group for overlap with their data.

## References

- J. A. Calvo, Geometric knot spaces and polygonal isotopy, *JKTR* 10 (2001).
- J. Cantarella, A. Rechnitzer, H. Schumacher, C. Shonkwiler, New upper bounds for stick numbers, *JKTR* 35 (2026), arXiv:2508.18263.
- T. D. Eddy, C. Shonkwiler, New stick number bounds from random sampling of confined polygons, *Exp. Math.* 31 (2022); data at github.com/thomaseddy/stick-knot-gen.
- R. Blair, T. D. Eddy, N. Morrison, C. Shonkwiler, Knots with exactly 10 sticks, *JKTR* 29 (2020).
- G. T. Jin, Polygon indices and superbridge indices of torus knots and links, *JKTR* 6 (1997).
- K. C. Millett, E. J. Rawdon, Energy, ropelength, and other physical aspects of equilateral knots, *J. Comput. Phys.* 186 (2003).
- E. J. Rawdon, R. G. Scharein, Upper bounds for equilateral stick numbers, *Contemp. Math.* 304 (2002).
