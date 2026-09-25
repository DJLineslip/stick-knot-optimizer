# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor and others) working in this repository. Read this file fully before doing anything. Then read `README.md` (mathematics, results, limitations) and `legacy/README.md` (provenance).

## Mission

Find a knot K whose **equilateral stick number e(K) is larger than its stick number s(K)**, or accumulate strong evidence about where such knots cannot be. The question is open.

The only thing that counts as a result is a **certified polygon**: an equal-stick polygon with n sticks whose knot type has been re-identified from scratch. Everything else (stalled runs, collapsing flows, timeouts) is a search log, not evidence.

## Current status

16 knots have certified equal-stick polygons at exactly their stick number (see `results/summary.csv`):

- **Torus knots:** T(4,5), T(5,6), T(6,7), T(7,8), with 10, 12, 14 and 16 sticks.
- **Ten-stick knots:** K11n71, K11n75, K11n76, K11n78, K13n225, K13n230, K13n288, K13n307, K13n584, K13n603, K13n604, K13n607. These are 12 of the 19 four-bridge knots whose stick number is proven to be exactly 10.

Unfinished ten-stick knots:

| Knot | Status |
|---|---|
| K13n285 | Reducer found no 10-stick polygon in 240 s |
| K13n602 | Reducer found no 10-stick polygon in 60 s |
| K13n608, K13n1192, K13n5018 | Not attempted |
| K13n586, K13n593 | No starting data in Eddy's repository |

## Setup

```bash
git clone https://github.com/thomaseddy/stick-knot-gen     # data; never commit it
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd scripts && PYTHONPATH=.. python 06_verify_results.py   # must print 16 lines, all cert True, type True
```

If `.gitignore` does not exist yet, create it with at least:

```
stick-knot-gen/
.venv/
__pycache__/
data/external/
results/logs/
```

All scripts run from `scripts/` with `PYTHONPATH=..`. Output paths such as `../results` are relative to `scripts/`. The first import of each module compiles numba kernels (a few seconds).

## Repository map

| Path | Contents |
|---|---|
| `equistick/geometry.py` | Segment distances, knot-type-safe moves (`safe_move`, `deletable`), Millett and Rawdon ratio (`mr_ratio`), angle sums |
| `equistick/invariants.py` | Projections to PD codes, Alexander polynomial on the unit circle, SnapPy `identify`, knot Floer homology `hfk` |
| `equistick/certify.py` | 40-digit certificate `mr_certificate_mp`, `polish`, `verify_torus`, `length_jacobian` |
| `equistick/torus.py` | T(p, p+1) constructions (`torus_poly`, `STARTS`), symmetric no-go scan |
| `equistick/flows.py` | Equalizer #1, path lifting. It produced a false obstruction signal; see Pitfalls. |
| `equistick/optimize.py` | Equalizer #2 `clearance_floor_solve` (not path-safe) and #3 `homotopy_equalize` (path-safe), plus `fatten` |
| `equistick/reduce.py` | `reduce_once` and `reduce_to`: lower the stick count by annealing toward a deletable vertex |
| `equistick/data.py` | `load_eddy`, `exact_stick_numbers`, `seed_for`, `TEN_STICK_19` |
| `scripts/01` to `06` | One script per experiment; `06_verify_results.py` re-verifies everything in `results/` |
| `results/` | Certified coordinates (`<knot>_equilateral_<n>sticks.txt`), `summary.csv`, logs |
| `legacy/` | Original research scripts. **Read-only provenance. Never edit.** |

## Ground rules (non-negotiable)

1. **Definition of a result.** A polygon counts only if all of the following hold:
   - (a) every edge length is equal to within the Millett and Rawdon bound, that is `mr_ratio(V)[0] < 1` **and** `mr_certificate_mp(V)[3]` is True;
   - (b) its knot type is re-identified from the final coordinates, not inferred from the path: `identify(V)` for hyperbolic knots, checked in at least 3 projections, or `verify_torus` for torus knots;
   - (c) the stick number s(K) is known exactly, from `data.exact_stick_numbers()` or a cited theorem.

   If (c) fails, report "equilateral realization with n sticks", never "e(K) = s(K)".
2. **Never trust equalizer #2 without re-identification.** `clearance_floor_solve` uses SLSQP, whose iterates can pass edges through each other. From lopsided starts it jumps to equilateral unknots. Use `homotopy_equalize` whenever the start is far from equilateral.
3. **Failure is not evidence.** Do not write "obstruction", "counterexample" or "e > s" on the strength of a run that stalls, times out, or shows μ shrinking with the defect. Log it as "not found within budget", with the budget.
4. **A candidate counterexample needs an evidence dossier, not a claim.** If a knot with known s(K) resists equalization, escalate to the user with:
   - at least 3 independent starting polygons, from different reductions or different sources;
   - all three equalizers tried;
   - a clearance-floor scan logging (μ₀, best defect) for each start;
   - the μ-versus-defect traces.

   Proving e > s requires a mathematical proof; your job is to supply the evidence.
5. **Do not weaken safety tests.** The tolerances in `geometry.safe_move`, `deletable` and `seg_tri` are deliberately conservative. Change them only with tests showing no false "safe" answers.
6. **Reproducibility.**
   - Seed every run with `data.seed_for(knot_name)`, plus an explicit run index if you need several seeds.
   - Never use Python's `hash()` for seeding; it is salted per process.
   - Record the command, seed, budget, git commit and package versions for every run in `results/RUNLOG.md` (append-only).
7. **Result files.**
   - Save coordinates as `results/<knot>_equilateral_<n>sticks.txt` with `np.savetxt(..., fmt='%.17g')`.
   - Torus knots are named `T<p>_<q>`.
   - After adding files, rerun `06_verify_results.py`, update the Results tables in `README.md`, and commit the coordinates, `summary.csv` and README together.
8. **Style.** No em dashes or en dashes anywhere in prose: docs, comments, commit messages. Use commas, colons, parentheses or "to" for ranges.
9. **Data sources.**
   - The Cantarella group's datasets on Harvard Dataverse disallow automated access. Do not scrape them. Ask the user to download them manually into `data/external/crss/`: doi:10.7910/DVN/NFJIII (knots through 13 crossings) and doi:10.7910/DVN/GCNJLI (torus knots).
   - Eddy's data comes from the `stick-knot-gen` clone.

## Compute practice

- Runs longer than a few minutes go in the background (`nohup` or `tmux`), log to `results/logs/<script>_<timestamp>.log`, and write each certified polygon to disk as soon as it is found. Never hold results only in memory.
- Parallelize across knots, one process per core, rather than within a knot.
- Before spawning workers, warm the numba cache once in the parent (import `equistick` and call `min_dist`, `safe_move`, `penalties` and `dist_jac` on a small polygon). Otherwise concurrent compiles can race on the cache files. If SnapPy misbehaves under `fork`, use the `spawn` start method.
- For runs expected to exceed one hour, state the plan (knots, budgets, cores, estimated time) and confirm with the user first.

## Task queue (in priority order)

Each task lists its definition of done.

1. **Parallel runner and the unfinished ten-stick knots.**
   - Write `scripts/07_parallel.py`, which runs `05_tenstick.run()` over a knot list with `multiprocessing`, one knot per core, appending to `results/tenstick.log`.
   - Run it on K13n285, K13n602, K13n608, K13n1192 and K13n5018 with a 30-minute budget each. `reduce_to` already handles the 12-stick starts.
   - *Done when:* every knot is either certified or logged "not found in 30 min", and 06 and the README have been updated.
2. **Cantarella group data as starting polygons.**
   - Once the user has placed the files in `data/external/crss/`, add a loader `data.load_crss(name)` and let `05_tenstick.py` start from those 10-stick polygons, skipping reduction.
   - Run all remaining ten-stick knots, including K13n586 and K13n593.
   - *Done when:* all 19 are certified or have an evidence dossier (rule 4).
3. **Extend the torus family.**
   - Run `04_torus_family.py` for p = 8 and p = 9. `STARTS` lacks these, so the script falls back to `symmetric_scan`; add the parameters it finds to `STARTS`.
   - Record the highest certified floor μ₀ for each p and fit how it decays with p.
   - *Done when:* results are certified and the decay table in the README is extended.
4. **Superbridge-tight torus knots.**
   - T(3,7) (K14n21881) and T(3,8) (K16n783154) need exactly 12 sticks. Check `stick-knot-gen/stick_number/mseq_knots/` for existing equilateral data and its stick count.
   - If there is none at 12 sticks, build a starting polygon from a finely sampled smooth parametrization, confirm its type, and reduce it with `reduce_to`.
   - *Done when:* certified at 12 sticks, or an evidence dossier exists.
5. **Systematic candidate list.**
   - Write `scripts/08_gap_candidates.py`, which lists every knot in `exact_stick_numbers()` whose best equal-stick polygon in Eddy's data (or ours) uses more sticks than s(K). Once the Cantarella group's tables are available, extend it to their upper bounds through 13 crossings.
   - *Done when:* the table has been generated and committed.
6. **Rigor for publication.**
   - Interval-arithmetic certificates (`python-flint` arb or `mpmath.iv`).
   - Rigorous identification: SnapPy verified computations for hyperbolic knots; for torus knots, a knot-type-safe path back to the symmetric construction.
   - *Done when:* every file in `results/` carries a rigorous certificate.
7. **Tests.**
   - Add a `tests/` directory with pytest covering: `safe_move` on hand-built blocked and unblocked moves; `pd_code` plus `alexander_abs` on Eddy's 3_1, 8_19 and 10_124 against the known polynomials; `mr_certificate_mp` on a certified file; and `reduce_once` on a padded trefoil.
   - *Done when:* `pytest` passes in CI or locally.

## Pitfalls already hit

- **False obstruction.** Equalizer #1, started near the symmetric T(4,5), showed μ falling in lockstep with the length defect (the classic collapse signature). It was heading for the symmetric equal-length point, which is always singular. T(4,5) certifies easily with equalizer #2. Reproduce with `scripts/03_false_collapse.py`.
- **Knot-type jumps.** SLSQP from lopsided starts returns equilateral unknots. Use the safe homotopy (rule 2).
- **Alternate census names.** `identify` can return a non-table name for a table knot (K15n41127 comes back as K6_37). A mismatch may therefore be a naming issue; a match is reliable.
- **Polishing.** `certify.polish` is not path-safe. Always re-identify after polishing.
- **Salted hashes.** The legacy batch seeded with `hash()`; use `seed_for`.

## Communicating with the user

Report in plain prose with a short results table. Separate certified results from search logs. Say explicitly what was not done and why. Never overstate: a numerical certificate is strong evidence, not a formal proof, until task 6 is complete.
