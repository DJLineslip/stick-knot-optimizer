# Nine saved 15-crossing 10-gons: conditional exact stick-number conclusions

The first six were found on 2026-09-25; the bounded search completed on 2026-09-27 and added three. The method combines three public sources and follows Blair, Eddy, Morrison and Shonkwiler, "Knots with exactly 10 sticks" (JKTR 2020, arXiv:1909.06947):

1. **Lower bound for the listed diagrams.** The Wirt_Hm Wirtinger number is 4 and the listed S5 or D4 seed images extend to every strand, satisfy every Wirtinger relation exactly, and generate the full group. These exact finite checks establish bridge index 4 for those diagrams. Kuiper's b < sb and Randell's sb <= stick/2 then give stick >= 10 for the diagram's knot. Connecting the saved polygon to that named knot remains a numerical identification.
2. **Upper bound.** An equal-stick 10-gon, certified by the exact-decimal interval Millett and Rawdon checker (`equistick.interval_certificate`).
3. **Identity.** The polygon's knot complement is isometric (SnapPy, with `snappy_15_knots`) to the table knot in 3 random projections. The Wirt_Hm Gauss code, converted to a DT code, is also isometric to the table knot, so the bridge data refers to the same knot.

The nine saved decimal coordinates and Wirt_Hm rows were checked in the prior pool campaign. `3/3` means three independently seeded numerical SnapPy complement isometries to the table knot. The Gauss-code column is a separate DT-diagram comparison. `Rank-4 map` below means an **exact** propagation, relation, and generation pass, not merely a listed map. The nine saved-knot checks and 25 listed-map pool checks overlap on three saved pool knots, giving **31 distinct exact passes** (nine saved plus 22 other pool knots). Four additional pool rows have no listed map and are recorded as `map_missing`, not failed relations. See `pool_verification.json` for complete strand images and per-knot exact pass/fail, and `interval_certificates.json` for geometric certificates.

| Knot | 10-gon source | Interval | MR / 40-digit | Table isometry | Wirt no. | Rank-4 map | Gauss isometry |
|---|---|---|---|---|---|---|---|
| K15n59007 | Eddy, copied unchanged | pass | pass / pass | 3/3 | 4 | D4, exact pass | pass |
| K15n40184 | Eddy 11-gon, reduced and safely equalized | pass | pass / pass | 3/3 | 4 | D4, exact pass | pass |
| K15n40185 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | D4, exact pass | pass |
| K15n41189 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, exact pass | pass |
| K15n41193 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, exact pass | pass |
| K15n41235 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, exact pass | pass |
| K15n45460 | bounded 11-gon reduction, safely equalized | pass | pass / pass | 3/3 | 4 | S5, exact pass | pass |
| K15n47800 | bounded 11-gon reduction, safely equalized | pass | pass / pass | 3/3 | 4 | S5, exact pass | pass |
| K15n52941 | bounded 11-gon reduction, safely equalized | pass | pass / pass | 3/3 | 4 | S5, exact pass | pass |

All angle sums are below 2*pi, as the ladder lemma requires.

**Status.** Tier A by the repo's rules, with the same caveats as the existing results. The rank-four Coxeter maps now pass exact relation and generation checks; if the numerical complement identifications are correct, the nine named knots have stick number and equilateral stick number exactly 10:
- identification is numerical (SnapPy isometry checks, not a rigorous proof; Knoodle covers only knots through 13 crossings);
- the lower bounds use the Wirt_Hm diagrams and exact checks in `pool_verification.json`; neither numerical complement matching nor exact Coxeter relations alone prove a formal identification of a decimal polygon with a named table knot.

**Literature check.** These nine candidate exact stick-number claims were not found in KnotInfo's stick-number description or in the 2019 Blair et al. paper checked here (checked 2026-09-27). This is not a novelty claim or an exhaustive literature review.

**Eleven-stick pool.** A fresh join of Eddy's 15-crossing equal-stick 11-gons with Wirt_Hm sheet A finds **34** rows with Wirtinger number 4, not 33. After removing the five reductions above, **29** remain, not 28. Four of the 29 (K15n59060, K15n67540, K15n84457, K15n94888) have no rank-4 Coxeter map listed in that row. Any successful 10-gon for one of those four would give an upper bound only, not exact stick number 10 on this evidence. The user authorised searching all 29 at 20 minutes per knot. Search logs are not evidence of a lower bound.

A further 51 knots reportedly have equal-stick 12-gons and Wirtinger number 4; that separate pool was not checked here.

## Completed 11-stick pool (29 candidates)

Command: `.venv/bin/python scripts/10_fifteen_pool.py --budget 1200 --workers 1`. One worker was used because this container had a one-core quota; each knot had an independent 1,200-second hard wall deadline. The runner recorded deterministic per-knot seeds and source/code revisions in `pool_results.json`. The search code revision was `7470f58ec410aeb729b51a33bb217f2a19339cd8`; Wirt_Hm was at `74fe52e57de6f91988f63ac105cbc157534ad966`. The raw manifest's `stick_number_source: TEN_STICK_19 ... otherwise exact_values.csv` is inherited generic metadata from `scripts/07_parallel.py`; it is **not** the lower-bound source for these 15-crossing knots. Their listed bridge-index evidence comes from the Wirt_Hm workbook, as checked in `pool_verification.json`.

**Outcome of the reduction run: three saved, checked equal-stick 10-gons; 26 searches timed out.** The original six plus these three have the stated *conditional* evidence for exact stick number and equilateral stick number 10. For the 26 reduction timeouts, the existence of a 10-stick representation remained open after that run. Four timeout rows lack a listed rank-4 map, so their bridge-index lower bound also requires additional evidence. The later crossing-change search is reported separately below.

| Knot | 20-minute search outcome | Listed rank-4 map |
|---|---|---|
| K15n124836 | No 10-stick polygon found in 20 min | listed |
| K15n124999 | No 10-stick polygon found in 20 min | listed |
| K15n131344 | No 10-stick polygon found in 20 min | listed |
| K15n40214 | No 10-stick polygon found in 20 min | listed |
| K15n41126 | No 10-stick polygon found in 20 min | listed |
| K15n41131 | No 10-stick polygon found in 20 min | listed |
| K15n41142 | No 10-stick polygon found in 20 min | listed |
| K15n41183 | No 10-stick polygon found in 20 min | listed |
| K15n41188 | No 10-stick polygon found in 20 min | listed |
| K15n41202 | No 10-stick polygon found in 20 min | listed |
| K15n41213 | No 10-stick polygon found in 20 min | listed |
| K15n41237 | No 10-stick polygon found in 20 min | listed |
| K15n41238 | No 10-stick polygon found in 20 min | listed |
| K15n43517 | No 10-stick polygon found in 20 min | listed |
| K15n45460 | Certified 10-gon | listed |
| K15n45603 | No 10-stick polygon found in 20 min | listed |
| K15n46532 | No 10-stick polygon found in 20 min | listed |
| K15n47800 | Certified 10-gon | listed |
| K15n49036 | No 10-stick polygon found in 20 min | listed |
| K15n51709 | No 10-stick polygon found in 20 min | listed |
| K15n51757 | No 10-stick polygon found in 20 min | listed |
| K15n52940 | No 10-stick polygon found in 20 min | listed |
| K15n52941 | Certified 10-gon | listed |
| K15n56079 | No 10-stick polygon found in 20 min | listed |
| K15n56089 | No 10-stick polygon found in 20 min | listed |
| K15n59060 | No 10-stick polygon found in 20 min | none listed |
| K15n67540 | No 10-stick polygon found in 20 min | none listed |
| K15n84457 | No 10-stick polygon found in 20 min | none listed |
| K15n94888 | No 10-stick polygon found in 20 min | none listed |

Saved polygons: `K15n45460_equilateral_10sticks.txt`, `K15n47800_equilateral_10sticks.txt`, `K15n52941_equilateral_10sticks.txt`. The 29 machine-readable search outcomes (with elapsed times and seeds) are in `pool_results.json`; `pool_verification.json` independently checks all nine saved polygons against Wirt_Hm and records each pool outcome. `interval_certificates.json` certifies the geometry of all nine files. `RUNLOG.md` retains *launch-time* run metadata (its embedded `results: []` is not the final outcome); the final manifest is `pool_results.json` (run `20260926T163138_60a140ef`). `tenstick.log` starts with one error from an earlier abandoned attempt, then records the 29 outcomes of the completed run. Per-knot scratch logs and the external Wirt_Hm checkout remain gitignored.

## Crossing-change campaign (2026-09-27)

The first pass inventoried every available 10-stick polygon, **of any knot type**, including Cantarella group data (doi:10.7910/DVN/NFJIII): 534 crss, 321 eddy, 27 ours (882 total). The completed source grids tried **2,923,200 moves** and recorded **164,022 one-edge crossings**. The scan reached **33 distinct four-Wirtinger knots**. The full per-source inventory, measured durations, exact moves tried, seed, identifiers and reached names are in `crossing_results.json`; the full run log is `crossing_scan.log`.

**Source status:** 870 completed, 12 timed out at 60 seconds, zero never attempted. The timed-out records have `moves_tried: null`, not zero: their incomplete grids cannot be counted as fully processed. **The strict every-source completion criterion is unmet.** No source retries were run, at the user’s instruction. “Not reached” below means *not reached in the 870 completed source grids*, not unreachable.

Timed-out sources: `crss:10_124`, `crss:8_8`, `crss:9_1`, `crss:9_14`, `crss:K11n133`, `crss:K11n21`, `crss:K11n79`, `crss:K11n90`, `crss:K11n91`, `crss:K11n92`, `crss:K11n93`, `crss:K12n242`.

Scan settings: 60 seconds per source, 180 seconds reserved per candidate, deterministic grid and seeds recorded in the manifest. An initial launch lacked a log directory; subsequent attempts exhausted memory (exit 137). NetCDF indexing was moved to a short-lived subprocess, preserving 30 checkpointed sources and rescuing one complete source. The resumed first pass finished with `completed=870/882 skipped_completed=30`; no further OOM kill was recorded. The checkpoint archive hash, exact scan command, input hashes, code hash, source revisions and package versions are in `crossing_results.json`. The external NetCDF dataset and full scratch worker outputs remain outside Git; selected crossing moves, source-coordinate hashes and output hashes are in `crossing_witnesses.json`. Raw external coordinate arrays are not redistributed in this repo.

After the scan, `scripts/12_equalize_crossings.py` gave **one** selected witness per reached map-verified target at most **180 seconds** for safe equalization and certification. This is not a retry of timed-out source grids. The actual attempt durations, move indices, source names and hashes are in `crossing_equalization.json`; stage stdout is in `crossing_equalization.log`. Twelve previously unsaved equal-stick 10-gons were saved and interval-certified. These are new *saved polygons in this repo*, not an assertion of literature novelty.

| Newly saved knot | Crossing source knot (source ID) | Map | Geometry | Diagram-to-table | Named-knot conclusion |
|---|---|---|---|---|---|
| K13n501 | K15n41235 (`ours:K15n41235`) | S5, 13/13 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K13n585 | 11_6 (`crss:K11n22`) | S5, 13/13 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K13n592 | 11_196 (`ours:K11n78`) | S5, 13/13 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K13n593 | 11_194 (`ours:K11n75`) | S5, 13/13 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n40180 | K13n586 (`crss:K13n586`) | D4, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n41127 | K13n307 (`ours:K13n307`) | S5, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n43517 | K13n608 (`ours:K13n608`) | S5, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n46935 | K13n225 (`ours:K13n225`) | S5, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n48957 | K13n225 (`ours:K13n225`) | S5, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n49035 | K13n288 (`ours:K13n288`) | S5, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n51709 | K13n3969 (`crss:K13n3969`) | D4, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |
| K15n52944 | K13n1192 (`ours:K13n1192`) | S5, 15/15 exact relations, generates | 10 equal sticks, interval pass | Gauss pass; polygon 3/3 numerical | Stick and equilateral stick = 10, conditional on numerical identity |

Each of these twelve has an exact S5 or D4 full-strand Wirtinger pass and an independently checked Wirt_Hm Gauss diagram-to-table numerical isometry in `crossing_diagram_checks.json`. The saved decimal files and their 160-bit interval certificates are in this directory and `interval_certificates.json` (the nine earlier certificate rows were copied, **not rerun**). The saved polygon-to-table checks used three SnapPy projections; these and the Gauss comparison are **numerical**, not formal knot-identity proofs. Thus the exact stick-number claims remain **conditional on correct identification**. No claim of novelty relative to KnotInfo or the literature is made for the twelve. The exact lower bound follows from the checked bridge-index-four diagram, while the certified equal-stick 10-gon gives the upper bound.

### Outcomes for the 26 earlier reduction timeouts

All 26 still have their earlier 1,200-second reduction outcomes recorded in `pool_results.json`. Two were reached from another knot and gained crossing-change polygons; the other 24 were not reached within the completed source grids. A listed rank-four map alone is not a proof; exact map outcomes are in `pool_verification.json` and `crossing_results.json`.

| Knot | Crossing outcome | Exact rank-four map | Stick-number implication |
|---|---|---|---|
| K15n124836 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n124999 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n131344 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n40214 | Not reached in 870 completed source grids; 12 source timeouts | D4 exact pass | No new upper bound; exact 10 not established |
| K15n41126 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41131 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41142 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41183 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41188 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41202 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41213 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41237 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n41238 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n43517 | Certified from K13n608 (`ours:K13n608`), crossing change | S5 exact pass | Conditional exact 10 |
| K15n45603 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n46532 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n49036 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n51709 | Certified from K13n3969 (`crss:K13n3969`), crossing change | D4 exact pass | Conditional exact 10 |
| K15n51757 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n52940 | Not reached in 870 completed source grids; 12 source timeouts | S5 exact pass | No new upper bound; exact 10 not established |
| K15n56079 | Not reached in 870 completed source grids; 12 source timeouts | D4 exact pass | No new upper bound; exact 10 not established |
| K15n56089 | Not reached in 870 completed source grids; 12 source timeouts | D4 exact pass | No new upper bound; exact 10 not established |
| K15n59060 | Not reached in 870 completed source grids; 12 source timeouts | none verified (no listed map) | No new upper bound; exact 10 not established |
| K15n67540 | Not reached in 870 completed source grids; 12 source timeouts | none verified (no listed map) | No new upper bound; exact 10 not established |
| K15n84457 | Not reached in 870 completed source grids; 12 source timeouts | none verified (no listed map) | No new upper bound; exact 10 not established |
| K15n94888 | Not reached in 870 completed source grids; 12 source timeouts | none verified (no listed map) | No new upper bound; exact 10 not established |

Other reached names: **20** already had certified polygons in the repository and were not rechecked; `K13n1192`, `K13n225`, `K13n230`, `K13n285`, `K13n288`, `K13n307`, `K13n5018`, `K13n584`, `K13n602`, `K13n603`, `K13n604`, `K13n607`, `K13n608`, `K15n40184`, `K15n40185`, `K15n41189`, `K15n41193`, `K15n47800`, `K15n52941`, `K15n59007`. One additional reached knot, `K14n22583`, lacked a passing listed rank-four map. Its crossing-change hits are logged, but **no exact stick-number conclusion** is asserted.

**Acceptance boundary:** The 12 source timeouts must remain marked incomplete. Retrying them could change the “not reached” target statuses, but none were retried here. `crossing_results.json` reports each completed source’s move count; for timeouts a full-grid count cannot honestly be supplied. Numerical SnapPy matching and exact finite-group relations serve different purposes; neither is a formal certificate of knot identity.
