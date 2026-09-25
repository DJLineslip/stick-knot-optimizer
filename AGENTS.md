# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor and others) working in this repository. Read this file fully before doing anything. Then read `README.md` (mathematics, results, limitations) and `legacy/README.md` (provenance).

## Mission

Find a knot K whose **equilateral stick number e(K) is larger than its stick number s(K)**, or accumulate strong evidence about where such knots cannot be. The question is open.

A counterexample does **not** require knowing s(K) exactly. Let s_ub(K) be the stick count of any verified polygon of K. If no equal-stick polygon of K exists with s_ub(K) or fewer sticks, then e(K) > s_ub(K) ≥ s(K). The Cantarella group dataset offers a provisional low-stick input for every named prime knot through 13 crossings, but a group's polygon must be re-identified before its stick count is used as a bound for that name. Two sampled groups currently have recorded naming mismatches.

The only thing that counts as a result is a **certified polygon**: an equal-stick polygon whose knot type has been re-identified from scratch. Everything else (stalled runs, collapsing flows, timeouts) is a search log, not evidence.

## Current status

- **Certified results.** 28 in-repository polygons have exact-decimal interval certificates for the geometric inequality and numerical knot re-identification (`results/summary.csv`, `results/interval_certificates.json`).
  - Torus knots: T(3,7), T(3,8), T(4,5), T(5,6), T(6,7), T(7,8), T(8,9), T(9,10), with 12, 12, 10, 12, 14, 16, 18 and 20 sticks. Names are up to mirror image.
  - Ten-stick knots: all 19 four-bridge knots with stick number exactly 10, plus the nine-stick `9_29` polygon from the supplied data.
- **Known stick numbers.** All 59 nontrivial knots in Eddy's 60-entry exact table have an equal-stick polygon at s(K): 23 in this repository and 36 hashed Eddy files independently checked in place (`data/exact_stick_coverage.json`). The unknot is trivial. Numerical knot identification remains short of a proof.
- **New data.** The user has downloaded the Cantarella group dataset doi:10.7910/DVN/NFJIII, "Low stick number polygons representing all knot types through 13 crossings", as a verified NetCDF4 file.
  - It holds 12,965 named groups, one purported low-stick polygon per prime knot type through 13 crossings. Two sampled groups have independently recorded naming mismatches (`10_86`, `10_162`).
  - Each group (`10_37`, `K11a367`, `K12n242`) contains `sticks` (number of edges), `crossings` (projected diagram crossings, not minimal table crossings), `coords` (an N x 3 array of vertices) and `pdcode` (a 0-indexed planar-diagram code).
  - The dataset's per-knot `.tab` text exports are not needed; the same data is in the NetCDF file.
  - Eddy's equilateral data covers only 1,937 of these knots (15% overall, 7% at 13 crossings).
- **Open.** No counterexample. Rigorous knot-type identification is still missing.

## Setup

```bash
git clone https://github.com/thomaseddy/stick-knot-gen     # data; never commit it
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd scripts && PYTHONPATH=.. python 06_verify_results.py   # every file in results/ must show cert True, type True
```

Place the verified NetCDF4 file (`*.nc`), unchanged, in `data/external/crss/`, and record its SHA256 in `data/README.md`.
- That directory is gitignored; the file is never committed. Commit only derived metadata and our own polygons.
- `equistick.crss` uses the single `.nc` file found there, or the path in the environment variable `EQUISTICK_CRSS`.
- Check the dataset's license on its Dataverse page, and cite doi:10.7910/DVN/NFJIII in the README.

`.gitignore` must contain at least:

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
| `equistick/interval_certificate.py` | Exact-decimal segment distances and outward interval geometric certificates |
| `equistick/torus.py` | T(p, p+1) constructions (`torus_poly`, `STARTS`), symmetric no-go scan |
| `equistick/flows.py` | Equalizer #1, path lifting. It produced a false obstruction signal; see Pitfalls. |
| `equistick/optimize.py` | Equalizer #2 `clearance_floor_solve` (not path-safe) and #3 `homotopy_equalize` (path-safe), plus `fatten` |
| `equistick/reduce.py` | `reduce_once` and `reduce_to`: lower the stick count by annealing toward a deletable vertex |
| `equistick/data.py` | `load_eddy`, `exact_stick_numbers`, `seed_for`, `TEN_STICK_19` |
| `equistick/crss.py` | Lazy reader for the supplied NetCDF4 file: index, coordinates and PD code per knot group |
| `scripts/01` to `09` and `scripts/crss_*.py` | Experiments, source checks and census; `06_verify_results.py` re-verifies saved polygons |
| `results/` | Tier A results (see rule 1): `<knot>_equilateral_<n>sticks.txt`, `summary.csv`, interval report, logs |
| `results/sweep/` | **To be created (task 4):** tier B results and `sweep.csv` |
| `legacy/` | Original research scripts. **Read-only provenance. Never edit.** |

## Ground rules (non-negotiable)

1. **Result tiers.** Every certified polygon must satisfy:
   - (a) `mr_ratio(V)[0] < 1` **and** `mr_certificate_mp(V)[3]` is True, and later the interval checker;
   - (b) its knot type is re-identified from the final coordinates, not inferred from the path: `identify(V)` for hyperbolic knots in at least 3 projections, or `verify_torus` for torus knots.

   Then report it in exactly one tier:
   - **Tier A, "e(K) = s(K)":** s(K) is known exactly, from `data.exact_stick_numbers()` or a cited theorem, and the polygon has s(K) sticks. Files go in `results/`.
   - **Tier B, "equilateral realization at the best known stick bound", meaning e(K) ≤ s_ub(K):** s(K) is not known exactly. Files go in `results/sweep/`. Never write "e(K) = s(K)" for tier B.
   - **Tier C, "gap candidate":** no certified polygon at s_ub(K) after the full escalation protocol (task 6). It needs a dossier (rule 4). Never write "counterexample", "obstruction" or "e > s".
2. **Never trust equalizer #2 without re-identification.** `clearance_floor_solve` uses SLSQP, whose iterates can pass edges through each other. From lopsided starts it jumps to equilateral unknots. Use `homotopy_equalize` whenever the start is far from equilateral.
3. **Failure is not evidence.** Do not write "obstruction", "counterexample" or "e > s" on the strength of a run that stalls, times out, or shows μ shrinking with the defect. Log it as "not found within budget", with the budget.
4. **Gap candidates need an evidence dossier, not a claim.** For any tier C knot, write `results/dossiers/<knot>/` containing:
   - at least 3 independent starting polygons (dataset polygon, Eddy's polygon reduced to s_ub if one exists, and a re-reduction from s_ub + 1 sticks);
   - all three equalizers tried from each start;
   - a clearance-floor scan logging (μ₀, best defect) per start;
   - the μ-versus-defect traces.

   Then report to the user. Proving e > s requires a mathematical proof; your job is to supply the evidence.
5. **Do not weaken safety tests.** The tolerances in `geometry.safe_move`, `deletable` and `seg_tri` are deliberately conservative. Change them only with tests showing no false "safe" answers.
6. **Reproducibility.**
   - Seed every run with `data.seed_for(knot_name)`, plus an explicit run index if you need several seeds.
   - Never use Python's `hash()` for seeding; it is salted per process.
   - Record the command, seed, budget, git commit and package versions for every run in `results/RUNLOG.md` (append-only).
7. **Result files.**
   - **Tier A:** save as `results/<knot>_equilateral_<n>sticks.txt` with `np.savetxt(..., fmt='%.17g')`; torus knots are named `T<p>_<q>`. After adding files, rerun `06_verify_results.py` and the interval checker with the expected file count, update the README tables, and commit coordinates, `summary.csv`, interval report and README together.
   - **Tier B:** store polygons as one JSON Lines file per crossing number, `results/sweep/polygons_<c>.jsonl`, one object per knot, with coordinates as **decimal strings** in `%.17g` format so the interval checker reads the exact values. Keep one row per knot in `results/sweep/sweep.csv`. Do not create thousands of separate files.
8. **Style.** No em dashes or en dashes anywhere in prose: docs, comments, commit messages. Use commas, colons, parentheses or "to" for ranges.
9. **Data sources.**
   - Cantarella group data: the NetCDF4 file in `data/external/crss/`, placed by the user and never committed, cited as doi:10.7910/DVN/NFJIII.
     - Read it only through `equistick.crss`.
     - Do not ask for the `.tab` exports; they duplicate the NetCDF content.
     - The torus-knot dataset doi:10.7910/DVN/GCNJLI is optional; ask the user if task 9 needs it.
   - Dataverse disallows automated access. Never scrape it.
   - Eddy's data comes from the `stick-knot-gen` clone.
10. **Dataset polygons are unverified input.**
    - Re-identify every input polygon before relying on its name. Use the stored `pdcode` as an independent cross-check.
    - Record mismatches in `results/sweep/input_mismatches.csv`; never "fix" a name by assumption.
    - A dataset polygon is a starting point, not a result; only our certified equal-stick polygons are results.

## Compute practice

- Runs longer than a few minutes go in the background (`nohup` or `tmux`), log to `results/logs/<script>_<timestamp>.log`, and write each certified polygon to disk as soon as it is found. Never hold results only in memory.
- Parallelize across knots, one process per core, using the process model of `07_parallel.py`.
- Before spawning workers, warm the numba cache once in the parent (import `equistick` and call `min_dist`, `safe_move`, `penalties` and `dist_jac` on a small polygon). Otherwise concurrent compiles can race on the cache files. If SnapPy misbehaves under `fork`, use the `spawn` start method.
- **Sweeps must be resumable.** Skip knots that already have a verified row in `sweep.csv`, enforce a hard per-knot deadline, and append results incrementally.
- For runs expected to exceed one hour, state the plan (knots, per-knot budget, cores, estimated wall time) and confirm with the user first. This applies to every stage of task 5.

## Task queue (in priority order)

Each task lists its definition of done. Completed work is summarized at the end.

1. **Read and validate the Cantarella NetCDF file.**
   - **Inspect the header first** (`ncdump -h`, or the `groups` of `netCDF4.Dataset(path)`). Document in `data/README.md`:
     - the file name, its SHA256, and the number of groups;
     - whether `sticks` and `crossings` are stored as variables or as attributes;
     - the dtype and scale of `coords`;
     - the exact layout of `pdcode`: its shape, and whether its rows are operationally compatible with spherogram's convention (incoming under-strand first, counterclockwise); do not infer a proof of geometric row orientation from parsing alone;
     - any global attributes.
   - **Write `equistick/crss.py`** with:
     - `crss_path()`: the single `.nc` file in `data/external/crss/`, or `EQUISTICK_CRSS`; raise a clear error if there are zero or several;
     - `crss_index()`: {knot: (crossings, sticks)} for every group, read without loading coordinates;
     - `load_crss(name)`: `coords` as a plain float64 (N, 3) array with no masked values (disable auto-masking or check the mask);
     - `load_crss_pd(name)`: `pdcode` as a list of 4-tuples, 0-indexed as stored;
     - `export_cache()`: an optional derived cache, for example one `.npz` per crossing number in `data/external/crss/cache/`, so parallel workers need not open the HDF5 file;
     - `identify_pd(pd)`, added to `invariants.py`: a PD-code version of `identify`.
   - **Group names should already match our conventions**: Rolfsen style (`10_37`) through 10 crossings, HTW with a `K` prefix (`K13n586`) above. Confirm this against `data.exact_stick_numbers()` and Eddy's file names rather than assuming it.
   - **Validate every group, without SnapPy:**
     - `sticks` equals the number of rows of `coords`;
     - `crossings` matches the stored PD row count; derive the table crossing number from the group name separately;
     - all coordinates are finite, with no zero-length edges;
     - `min_dist` > 0.

     Tabulate sticks by crossing number and compare with Table 1 of the Cantarella group's paper.
   - **Validate identity** on a stratified sample of 300 knots, plus all 19 ten-stick knots, 9_29 and every torus knot in the file. Compare `identify(coords)` (3 projections) and `identify_pd(pdcode)` with the group name via confirmed SnapPy aliases; torus knots go through `verify_torus`. Log real mismatches per rule 10 and do not silently correct them.
   - **Commit:**
     - `data/crss_index.csv` (knot, crossings, sticks, input `min_dist`);
     - `data/README.md`;
     - the validation report;
     - tests that build a tiny synthetic NetCDF file in a temporary directory (never commit the real file);
     - `netCDF4` added to `requirements.txt`.
   - *Done when:* all of the above is committed and the group count is reported to the user.
2. **Close the known stick numbers.**
   - Load K13n586, K13n593 and 9_29 with `load_crss`, and check that their `sticks` values are 10, 10 and 9. Use `fatten` plus `homotopy_equalize` on unequal inputs; no reduction is needed. The 9_29 source was already equilateral and is independently checked without optimisation.
   - Run full tier A verification, including interval certificates.
   - *Done when:* every knot in `exact_stick_numbers()` except the unknot has a certified tier A polygon (or a dossier), and the README states this.
3. **Gap census.**
   - Write `scripts/09_gap_census.py`, producing `results/gap_census.csv` for all 12,965 knots with these columns: crossing number, provisional s_ub (the group's `sticks`, unless its name is a documented mismatch), exact s if known, best **reported equilateral candidate count** in the legacy `e_ub` field (Eddy's data plus ours; blank if none), and e_ub minus s_ub as a candidate-count difference. Preserve `source_sticks`, `input_status` and `e_status` so unchecked input or a merely reported Eddy file is not presented as a validated named-knot bound.
   - Add a summary by crossing number to the README.
   - *Done when:* committed.
4. **Pilot sweep through 10 crossings (249 knots).**
   - Write `scripts/10_sweep.py`. For each knot, start from `load_crss(name)` at s_ub sticks and run pass 1: `fatten`, then `homotopy_equalize` at floors 0.9, 0.5 and 0.2. Certify and re-identify (rule 1).
   - Save polygons per rule 7. Log one `sweep.csv` row per knot: knot, crossings, sticks, input μ (of the dataset polygon), status, floor used, final μ, defect, seconds, seed, commit, and `first_attempt_success` (whether the straight-line lift at floor 0.9 succeeded with no retries).
   - Workers read coordinates from the task 1 cache, not from the NetCDF file directly.
   - Make it resumable and parallel, with hard per-knot deadlines.
   - *Done when:* pilot results are committed, with the success rate, the distribution of time per knot, and a proposed pass-1 budget for task 5.
5. **Full sweep, 11 to 13 crossings (12,716 knots).**
   - Run in stages: 11 crossings (552), 12 (2,176), 13 (9,988). Before each stage, give the user the estimated wall time from the pilot's time distribution and confirm.
   - Interval-certify every success; generalize the checker to read the JSONL files, and run it in parallel.
   - *Done when:* every knot has status `certified` or `not_found_pass1` in `sweep.csv`, and the README reports counts by crossing number.
6. **Escalate the residue.**
   - **Pass 2**, up to 15 minutes per knot, using three independent starts:
     - the dataset polygon, with 10 agitations;
     - Eddy's polygon reduced to s_ub with `reduce_to`, where available;
     - the dataset polygon subdivided to s_ub + 1 sticks and re-reduced to a different s_ub-gon.

     Try all three equalizers from each start.
   - **Pass 3**, up to 2 hours per knot: clearance-floor scans and μ-versus-defect traces, written as a dossier (rule 4).
   - *Done when:* every residue knot is certified or has a dossier; send the user the tier C list.
7. **Difficulty map.**
   - From `sweep.csv`, show the distribution of μ at certification by stick count and crossing number.
   - For the lowest 1% and a random 1% control, estimate μ_max: bisect on the floor over 5 starts, then apply length-preserving clearance ascent. Include K13n607 (μ = 0.0007) as a known outlier.
   - Correlate with crossing number, bridge index (KnotInfo, via `database_knotinfo`), hyperbolic volume, and the gap between s_ub and the best stick lower bound.
   - Report whether `first_attempt_success` ever fails. If it never does, that is evidence for a star-shapedness conjecture that would imply e = s.
   - *Done when:* `results/difficulty.csv` and a README section are committed.
8. **Rigorous identification for tier A.**
   - For each tier A file, compute the projection in exact rational arithmetic from the stored decimals, check genericity exactly (no concurrent crossings, no vertex over another edge, no equal heights at crossings), and build the PD code.
   - Prove equivalence to a reference diagram: the HTW or Rolfsen table diagram for named knots (for example `spherogram.Link('K13n586')`), or the braid closure for torus knots. Record the Reidemeister moves, and write an independent checker that replays them and tests diagram isomorphism up to mirror.
   - The dataset's `pdcode` is a useful cross-check but not a reference: it is derived data, so the reference must be the knot-table diagram.
   - *Done when:* every tier A file carries a type certificate that the checker verifies.
9. **Torus families.**
   - Measure μ_max properly for T(p, p+1), p = 3 to 10, by bisection. The current decay table is capped by the floor schedule: T(8,9) and T(9,10) certified at the first floor tried. Fit the decay.
   - Test the constrained families with known s:
     - T(4,7), T(5,9) and T(6,11), where s = 2q;
     - T(4,9), T(4,11), T(5,11), T(5,12), T(5,13) and T(5,14), where s = 4p.

     Use `08_torus37.py`-style starts, or the GCNJLI dataset if the user provides it.
   - Look for shared structure in the certified T(p, p+1) polygons that could become an explicit family valid for all p.
10. **Write-up.**
    - Draft `docs/results_note.md` covering: tier A theorem list (final only after task 8), tier B summary table, the tier C list, methods, certificates, limitations and data citations. It is for the user to share with the Cantarella group.
11. **Hygiene.**
    - CI on every pull request: pytest, `06_verify_results.py`, the tier A interval report, and verification of any changed sweep rows.
    - Renumber the duplicate `08_` scripts and update every reference to them.

### Completed

- Parallel runner `07_parallel.py`, with K13n285, K13n602, K13n608, K13n1192 and K13n5018 certified at 10 sticks (the K13n602 reducer crash was fixed with a regression test).
- T(8,9) and T(9,10) certified numerically with `08_torus_batch.py`.
- T(3,7) and T(3,8) certified at 12 sticks with `08_torus37.py`.
- Exact-decimal interval checker certifying the geometric inequality for all 28 in-repository tier A files.
- Tasks 1 to 3: 12,965-group audit with 319 of 321 sampled identities matched and two mismatches logged; three new in-repository polygons; 59 of 59 nontrivial exact-stick entries geometrically interval-certified with numerical type checks across in-repository and external Eddy files; 12,965-row provisional gap census. See `data/README.md` and the main README. These outputs are being shared with an unresolved independent security-review failure in the NetCDF reader, not as a hardened parser or formal knot-type proof.

### Required handoff before task 4

- Fix the two documented reader memory hazards: unbounded `_FillValue` / `missing_value` attribute shapes can broadcast coordinate or PD comparisons; `_scalar` can read oversized or unsafe scalar variable and attribute payloads before validating dtype and size. Until then, use only the SHA256-pinned external `stick-number-bounds.nc`, not arbitrary `EQUISTICK_CRSS` inputs.
- Add RED tests with unreadable metadata proxies and a nonscalar broadcast-shaped sentinel, then validate scalar metadata before reading and comparing. Re-run the full suite (93 tests at this handoff), 12,965-group audit, 12,965-row census, 28 saved interval geometries and 59 exact-stick coverage records. Obtain an independent fail-closed reader review before running task 4 or calling the parser safe for external files.
- Independently resolve `10_86` and `10_162` source-name mismatches; 12,644 other source names remain unchecked. The 1,919 Eddy-only equilateral counts are labelled reported candidates, not verified named-knot bounds in the census; 744 positive candidate differences are not proofs. Geometric interval certificates do not prove knot identity.

## Pitfalls already hit

- **False obstruction.** Equalizer #1, started near the symmetric T(4,5), showed μ falling in lockstep with the length defect (the classic collapse signature). It was heading for the symmetric equal-length point, which is always singular. T(4,5) certifies easily with equalizer #2. Reproduce with `scripts/03_false_collapse.py`.
- **Knot-type jumps.** SLSQP from lopsided starts returns equilateral unknots. Use the safe homotopy (rule 2).
- **Alternate census names.** `identify` can return a non-table name for a table knot (K15n41127 comes back as K6_37). A mismatch may therefore be a naming issue; a match is reliable.
- **Polishing.** `certify.polish` is not path-safe. Always re-identify after polishing.
- **Salted hashes.** The legacy batch seeded with `hash()`; use `seed_for`.

## Pitfalls to expect with the dataset

- **Masked arrays.** `netCDF4` returns masked arrays by default. Disable auto-masking or check the mask, and convert to plain float64 before numba sees the data.
- **Scalars.** `sticks` and `crossings` may be stored as variables or as attributes. Read whichever the file uses, and cast to int.
- **PD codes.** `pdcode` is 0-indexed. spherogram accepts 0-based labels (our own `pd_code` output is 0-based), but tools expecting KnotTheory's 1-based labels need +1. Verify its crossing convention on known knots before trusting it.
- **Parallel reads.** HDF5 file locking can make many processes opening one file fail on some filesystems. Use the task 1 cache, or set `HDF5_USE_FILE_LOCKING=FALSE` for read-only access.
- **Names.** Through 10 crossings the names are Rolfsen style. Check 10_161 to 10_165 by identification, because knot tables number the Perko pair differently. Above 10 crossings the names are HTW with a `K` prefix, matching Eddy's files.
- **Chirality.** Polygons may represent either mirror image. Our identification ignores chirality, and so do e and s, but never mix a knot's polygon with its mirror's metadata.
- **Scale.** Coordinates may be at arbitrary scale. Always `normalize` before optimizing. Store our outputs as `%.17g` strings so the interval checker reads exact values.
- **Thin inputs.** The paper reports that its polygons are several orders of magnitude from singular, but some look almost singular to the eye. Let `fatten` raise the clearance before the homotopy.
- **Non-hyperbolic knots.** `identify` returns None for torus and satellite knots. Route those through `verify_torus`, or mark them "identification pending"; never count them as mismatches.

## Communicating with the user

Report in plain prose with a short results table. State the tier of every result. Separate certified results from search logs. Say explicitly what was not done and why. Never overstate: a numerical certificate is strong evidence, not a formal proof, until task 8 is complete.
