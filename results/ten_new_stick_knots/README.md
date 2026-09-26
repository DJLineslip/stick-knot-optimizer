# Six 15-crossing knots with stick number and equilateral stick number exactly 10

Found on 2026-09-25 by combining three public sources, following the method of Blair, Eddy, Morrison and Shonkwiler, "Knots with exactly 10 sticks" (JKTR 2020, arXiv:1909.06947):

1. **Lower bound.** Bridge index 4 is supported by the Blair, Kjuchukova and Morrison data (github.com/ThisSentenceIsALie/Wirt_Hm): the Wirtinger number is 4 (upper bound) and a map to a rank-4 Coxeter group is listed (lower bound). Kuiper's b < sb and Randell's sb <= stick/2 then give stick >= 10. This verification checks the listed four generator images and diagram identity, but does not independently prove the Coxeter homomorphism relations.
2. **Upper bound.** An equal-stick 10-gon, certified by the exact-decimal interval Millett and Rawdon checker (`equistick.interval_certificate`).
3. **Identity.** The polygon's knot complement is isometric (SnapPy, with `snappy_15_knots`) to the table knot in 3 random projections. The Wirt_Hm Gauss code, converted to a DT code, is also isometric to the table knot, so the bridge data refers to the same knot.

The checks below were rerun from the saved decimal coordinates and the Wirt_Hm `all_data_A.xlsx` rows. `3/3` means three independently seeded SnapPy complement isometries to the table knot. The Gauss-code column is a separate DT-diagram complement comparison. `Rank-4 map` means that the row lists four generator images in the indicated Coxeter group; the relations of the listed homomorphism were not independently proved here. See `verification.json` and `interval_certificates.json` for per-file outcomes and hashes.

| Knot | 10-gon source | Interval | MR / 40-digit | Table isometry | Wirt no. | Rank-4 map | Gauss isometry |
|---|---|---|---|---|---|---|---|
| K15n59007 | Eddy, copied unchanged | pass | pass / pass | 3/3 | 4 | D4, listed | pass |
| K15n40184 | Eddy 11-gon, reduced and safely equalized | pass | pass / pass | 3/3 | 4 | D4, listed | pass |
| K15n40185 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | D4, listed | pass |
| K15n41189 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n41193 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |
| K15n41235 | reduced and safely equalized | pass | pass / pass | 3/3 | 4 | S5, listed | pass |

All angle sums are below 2*pi, as the ladder lemma requires.

**Status.** Tier A by the repo's rules, with the same caveats as the existing results:
- identification is numerical (SnapPy isometry checks, not a rigorous proof; Knoodle covers only knots through 13 crossings);
- the bridge-index lower bounds come from the published Wirt_Hm data. The Coxeter maps listed there can be re-checked independently against the Wirtinger relations, which is an exact finite computation.

**Literature check.** These six exact stick-number claims were not found in KnotInfo's stick-number description or in the 2019 Blair et al. paper checked here. This is not a novelty claim or an exhaustive literature review.

**Eleven-stick pool.** A fresh join of Eddy's 15-crossing equal-stick 11-gons with Wirt_Hm sheet A finds **34** rows with Wirtinger number 4, not 33. After removing the five reductions above, **29** remain, not 28. Four of the 29 (K15n59060, K15n67540, K15n84457, K15n94888) have no rank-4 Coxeter map listed in that row. Any successful 10-gon for one of those four would give an upper bound only, not exact stick number 10 on this evidence. The user authorised searching all 29 at 20 minutes per knot. Search logs are not evidence of a lower bound.

A further 51 knots reportedly have equal-stick 12-gons and Wirtinger number 4; that separate pool was not checked here.
