# Nine 15-crossing 10-gons: exact stick number 10 conditional on knot identity and Coxeter maps

The first six were found on 2026-09-25; the bounded search completed on 2026-09-27 and added three. The method combines three public sources and follows Blair, Eddy, Morrison and Shonkwiler, "Knots with exactly 10 sticks" (JKTR 2020, arXiv:1909.06947):

1. **Lower bound.** Bridge index 4 is supported by the Blair, Kjuchukova and Morrison data (github.com/ThisSentenceIsALie/Wirt_Hm): the Wirtinger number is 4 (upper bound) and a map to a rank-4 Coxeter group is listed (lower bound). Kuiper's b < sb and Randell's sb <= stick/2 then give stick >= 10. This verification checks the listed four generator images and diagram identity, but does not independently prove the Coxeter homomorphism relations.
2. **Upper bound.** An equal-stick 10-gon, certified by the exact-decimal interval Millett and Rawdon checker (`equistick.interval_certificate`).
3. **Identity.** The polygon's knot complement is isometric (SnapPy, with `snappy_15_knots`) to the table knot in 3 random projections. The Wirt_Hm Gauss code, converted to a DT code, is also isometric to the table knot, so the bridge data refers to the same knot.

The checks below were independently rerun for all nine saved decimal coordinate files and their Wirt_Hm `all_data_A.xlsx` rows. `3/3` means three independently seeded SnapPy complement isometries to the table knot. The Gauss-code column is a separate DT-diagram complement comparison. `Rank-4 map` means that the row lists four generator images in the indicated Coxeter group; the relations of the listed homomorphism were not independently proved here. See `verification.json` (original six), `pool_verification.json` (all nine), and `interval_certificates.json` (all nine) for per-file outcomes and hashes.

| Knot | 10-gon source | Interval | MR / 40-digit | Table isometry | Wirt no. | Rank-4 map | Gauss isometry |
|---|---|---|---|---|---|---|---|
| K15n59007 | Eddy, copied unchanged | pass | pass / pass | 3/3 | 4 | D4, listed | pass |
| K15n40184 | Eddy 11-gon, reduced and safely equalized | pass | pass / pass | 3/3 | 4 | D4, listed | pass |
| K15n40185 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | D4, listed | pass |
| K15n41189 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n41193 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n41235 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n45460 | bounded 11-gon reduction, safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n47800 | bounded 11-gon reduction, safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n52941 | bounded 11-gon reduction, safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |

All angle sums are below 2*pi, as the ladder lemma requires.

**Status.** Tier A by the repo's rules, with the same caveats as the existing results. If the numerical complement identifications are correct and the listed rank-4 Coxeter maps satisfy their relations, the nine named knots have stick number and equilateral stick number exactly 10:
- identification is numerical (SnapPy isometry checks, not a rigorous proof; Knoodle covers only knots through 13 crossings);
- the bridge-index lower bounds come from the published Wirt_Hm data. The Coxeter maps listed there can be re-checked independently against the Wirtinger relations, which is an exact finite computation.

**Literature check.** These nine candidate exact stick-number claims were not found in KnotInfo's stick-number description or in the 2019 Blair et al. paper checked here (checked 2026-09-27). This is not a novelty claim or an exhaustive literature review.

**Eleven-stick pool.** A fresh join of Eddy's 15-crossing equal-stick 11-gons with Wirt_Hm sheet A finds **34** rows with Wirtinger number 4, not 33. After removing the five reductions above, **29** remain, not 28. Four of the 29 (K15n59060, K15n67540, K15n84457, K15n94888) have no rank-4 Coxeter map listed in that row. Any successful 10-gon for one of those four would give an upper bound only, not exact stick number 10 on this evidence. The user authorised searching all 29 at 20 minutes per knot. Search logs are not evidence of a lower bound.

A further 51 knots reportedly have equal-stick 12-gons and Wirtinger number 4; that separate pool was not checked here.

## Completed 11-stick pool (29 candidates)

Command: `.venv/bin/python scripts/10_fifteen_pool.py --budget 1200 --workers 1`. One worker was used because this container had a one-core quota; each knot had an independent 1,200-second hard wall deadline. The runner recorded deterministic per-knot seeds and source/code revisions in `pool_results.json`. The search code revision was `7470f58ec410aeb729b51a33bb217f2a19339cd8`; Wirt_Hm was at `74fe52e57de6f91988f63ac105cbc157534ad966`. The raw manifest's `stick_number_source: TEN_STICK_19 ... otherwise exact_values.csv` is inherited generic metadata from `scripts/07_parallel.py`; it is **not** the lower-bound source for these 15-crossing knots. Their listed bridge-index evidence comes from the Wirt_Hm workbook, as checked in `pool_verification.json`.

**Outcome: three saved, independently checked equal-stick 10-gons; 26 searches timed out.** The original six plus these three have the stated *conditional* evidence for exact stick number and equilateral stick number 10, subject to the identity and Coxeter-map caveats above. For the 26 timeouts, the existence of a 10-stick representation remains open: "no 10-stick polygon found in 20 min" reports this run only, not a lower bound or a nonexistence proof. Four timeout rows lack a listed rank-4 map, so their bridge-index lower bound would also require additional evidence.

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
