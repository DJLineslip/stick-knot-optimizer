
## 20260925T114837_bc118efe

```json
{
  "run_id": "20260925T114837_bc118efe",
  "manifest": "/workspace/repos/stick-knot-optimizer/results/logs/20260925T114837_bc118efe.json",
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "scripts/07_parallel.py",
    "--knots",
    "K13n285,K13n602,K13n608,K13n1192,K13n5018",
    "--budget",
    "1800",
    "--workers",
    "5"
  ],
  "seeds": {
    "K13n285": 1341989301,
    "K13n602": 508139202,
    "K13n608": 4271669212,
    "K13n1192": 272686914,
    "K13n5018": 3068936500
  },
  "projection_seeds": [
    1,
    2,
    3
  ],
  "budget_seconds": 1800.0,
  "workers": 5,
  "git_commit": "bbb6f8d017d56bef5da917212c65818f18228ec5",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "numba": "0.67.0",
    "mpmath": "1.4.1",
    "snappy": "3.3.2",
    "spherogram": "2.4.1"
  },
  "data_path": "/workspace/repos/stick-knot-optimizer/stick-knot-gen",
  "data_commit": "9f05018917d3cd568867412a0c39ec843aa8744e",
  "python": "3.11.2 (main, May 12 2026, 05:17:27) [GCC 12.2.0]",
  "stick_number_source": "TEN_STICK_19: Cantarella et al., JKTR 2026; otherwise exact_values.csv",
  "results": []
}
```

## 20260925T120002_5419dd09

```json
{
  "run_id": "20260925T120002_5419dd09",
  "manifest": "/workspace/repos/stick-knot-optimizer/results/logs/20260925T120002_5419dd09.json",
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "scripts/07_parallel.py",
    "--knots",
    "K13n602",
    "--budget",
    "1800",
    "--workers",
    "1",
    "--out",
    "/workspace/repos/stick-knot-optimizer/results"
  ],
  "seeds": {
    "K13n602": 508139202
  },
  "projection_seeds": [
    1,
    2,
    3
  ],
  "budget_seconds": 1800.0,
  "workers": 1,
  "git_commit": "4e5630ed47c06671655344468600275569ee1cc3",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "numba": "0.67.0",
    "mpmath": "1.4.1",
    "snappy": "3.3.2",
    "spherogram": "2.4.1"
  },
  "data_path": "/workspace/repos/stick-knot-optimizer/stick-knot-gen",
  "data_commit": "9f05018917d3cd568867412a0c39ec843aa8744e",
  "python": "3.11.2 (main, May 12 2026, 05:17:27) [GCC 12.2.0]",
  "stick_number_source": "TEN_STICK_19: Cantarella et al., JKTR 2026; otherwise exact_values.csv",
  "results": []
}
```

## 20260925T124648_6ea32812: T(3,7) and T(3,8) short probe

Command from `/tmp/equistick-torus37`:
`PYTHONPATH=. /workspace/repos/stick-knot-optimizer/.venv/bin/python scripts/08_torus37.py --knots T3_7,T3_8 --budget 35 --workers 2 --out /tmp/equistick-torus37-probe`.
Hard wall budget was 35 seconds per knot, including worker startup and validation. Both started from explicit 24-vertex samples of T(3,q) with R=2.5, r=1, phase=0.013, and were reduced with `reduce_to` using 40,000 annealing steps per deletion. Seeds from `seed_for`: T3_7=3331913908, T3_8=1445455141. Base git commit at invocation: `987ab18cd1e4117e0c2fe3602141d588be94eaf9` (the new script was uncommitted during the probe). Python 3.11.2; numpy 2.4.6, scipy 1.17.1, numba 0.67.0, mpmath 1.4.1, snappy 3.3.2, spherogram 2.4.1. Eddy data was not used.

| Knot | Reductions | Elapsed | Saved float MR | 40-digit defect | 40-digit mu | 40-digit bound | Alexander | HFK |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T(3,7) | 1 | 8.49 s | 8.70e-13 | 9.83e-17 | 0.03194 | 2.55e-04 | 4 projections | genus 6, rank 9, fibred, L-space, tau -6 |
| T(3,8) | 1 | 8.52 s | 5.09e-13 | 8.32e-17 | 0.02953 | 2.18e-04 | 4 projections | genus 7, rank 11, fibred, L-space, tau -7 |

Both saved 17-digit coordinate files were reloaded and validated in the worker before publication, then copied to `results/` and independently reverified by `scripts/06_verify_results.py` (all 23 rows `certified=True`, `type_confirmed=True`). This probe yielded numerical evidence; interval checks were run later (below). Knot identification is not a formal proof. Full per-knot logs and the run manifest were written under `/tmp/equistick-torus37-probe/logs/` (temporary, not committed).

## Exact-decimal interval geometry, 23 stored polygons

Ran `scripts/08_interval_certificates.py --expected-count 23 --timeout-s 30` on this integration branch after re-running `scripts/06_verify_results.py`. All 23 saved coordinate files satisfied the strict geometric Millett-Rawdon inequality. `results/interval_certificates.json` records the exact-decimal file hashes, the verifier source hash and revision, and enclosing interval bounds per file. These certificates establish a geometric inequality for the literal saved decimal vertices, **not** a formal identification of the knot types. The earlier 40-digit checks and four-projection/HFK identifications remain numerical.

## torus_20260925T130320_ab7ca8ff

```json
{
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "scripts/08_torus_batch.py",
    "--knots",
    "T8_9,T9_10",
    "--budget",
    "1800",
    "--workers",
    "2",
    "--trials",
    "1000",
    "--scan-trials",
    "1500",
    "--out",
    "/tmp/equistick-integration/results"
  ],
  "seeds": {
    "T8_9": 762657362,
    "T9_10": 32929710
  },
  "budget_seconds": 1800.0,
  "workers": 2,
  "git_commit": "40f4e9aec7b9a11f7eccc2e59ad1f1a3c988af92",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "numba": "0.67.0",
    "mpmath": "1.4.1",
    "snappy": "3.3.2",
    "spherogram": "2.4.1"
  },
  "data_path": "/workspace/repos/stick-knot-optimizer/stick-knot-gen",
  "data_commit": "9f05018917d3cd568867412a0c39ec843aa8744e",
  "python": "3.11.2 (main, May 12 2026, 05:17:27) [GCC 12.2.0]",
  "stick_number_source": "Jin 1997: s(T(p,p+1)) = 2(p+1) for p > 2",
  "run_id": "torus_20260925T130320_ab7ca8ff",
  "manifest": "/tmp/equistick-integration/results/logs/torus_20260925T130320_ab7ca8ff.json",
  "projection_seed": 123,
  "projection_count": 4,
  "run_index": 0,
  "trials": 1000,
  "scan_trials": 1500,
  "results": []
}
```

The JSON above is the start-of-run snapshot. The completed manifest and detailed per-knot logs are in ignored `results/logs/` (run ID `torus_20260925T130320_ab7ca8ff`); the outcome is recorded here for durable provenance. Both jobs used deterministic `torus.STARTS` parameters, not Dataverse polygons. The run used at most two search workers within the five-CPU quota, with a separate 1800-second hard wall deadline for each knot including final validation.

| Knot | Sticks | Certified floor | Elapsed | SHA256 of published file | Saved-coordinate identification |
|---|---:|---:|---:|---|---|
| T(8,9) | 18 | 0.001 | 42.46 s | `f48a9480bca2c72e556c489fd27433314eba17554d6d8cc43ad93e1c20d3267e` | Alexander in four projections; HFK genus 28, fibred, L-space, absolute tau 28; 63 crossings |
| T(9,10) | 20 | 0.0005 | 418.69 s | `550c32c4b141633c0707f19ce3f9a648f2674127f5bef9e2360c15fa67e361f8` | Alexander in four projections; HFK genus 36, fibred, L-space, absolute tau 36; 80 crossings |

The independent `scripts/06_verify_results.py` run rechecked all 25 saved files with `certified=True` and `type_confirmed=True`. `scripts/08_interval_certificates.py --expected-count 25 --timeout-s 30` established strict exact-decimal geometric inequalities for 25/25 files. `results/interval_certificates.json` binds each coordinate file and the checker source to SHA256 hashes. Knot identity is still numerical, and the torus labels ignore chirality.

A post-run code review found missing HFK-rank and crossing-count gates in the T(8,9)/T(9,10) search and failure-path gaps in the interval batch and supervisors. The saved T(8,9) and T(9,10) files were **revalidated** under the stricter gate without rerunning the searches: Alexander ×4, HFK rank 15 and 17, respectively, genus 28 and 36, and crossing counts 63 and 80. The certificate checker now rejects empty inputs and nonfinite timeouts and publishes JSON atomically; the supervisors reject late observed completions, use atomic no-replace hard links for publication, guard late cleanup by inode ownership, and append outcomes to `RUNLOG.md`. All 54 unit tests pass. The interval report binds the geometric checker commit and exact source hashes; these changes do not make knot identity rigorous.

## 20260925T203435_0ece8cc7

```json
{
  "run_id": "20260925T203435_0ece8cc7",
  "manifest": "/tmp/equistick-crss/results/logs/20260925T203435_0ece8cc7.json",
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "/tmp/equistick-crss/scripts/crss_close_known.py",
    "--knots",
    "K13n586,K13n593,9_29",
    "--budget",
    "300",
    "--workers",
    "2"
  ],
  "seeds": {
    "9_29": 2045959619
  },
  "projection_seeds": [
    1,
    2,
    3
  ],
  "budget_seconds": 300.0,
  "workers": 2,
  "git_commit": "65cf4ad721feae577b70853dbde921f712d1d4af",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "numba": "0.67.0",
    "mpmath": "1.4.1",
    "snappy": "3.3.2",
    "spherogram": "2.4.1"
  },
  "data_path": "/workspace/repos/stick-knot-optimizer/stick-knot-gen",
  "data_commit": "9f05018917d3cd568867412a0c39ec843aa8744e",
  "python": "3.11.2 (main, May 12 2026, 05:17:27) [GCC 12.2.0]",
  "stick_number_source": "TEN_STICK_19: Cantarella et al., JKTR 2026; otherwise exact_values.csv",
  "results": [],
  "crss_sha256": "a7e8bfc961bc2156d373ea206fc52beb3fbb122931f0743a979e5823de07cb22",
  "input_doi": "10.7910/DVN/NFJIII",
  "optimizer": "fatten plus homotopy_equalize",
  "interval_checker": "exact decimal outward geometric bounds"
}
```

9_29 [certified] {"attempts": [], "certificate": {"coordinate_ids": ["K9a31", "K9a31", "K9a31"], "defect": 2.220446049250313e-16, "interval": {"backend": "mpmath.iv", "defect_lower": "28942428095329972412888687025755/91343852333181432387730302044767688728495783936", "defect_upper": "463078849525279558606218992412095/1461501637330902918203684832716283019655932542976", "file": "9_29_equilateral_9sticks.txt", "mpmath_version": "1.4.1", "mu_lower": "1104692712620786525809946810050397694618862845051/5986310706507378352962293074805895248510699696029696", "mu_upper": "138086589077598315726243351256299711827357855633/748288838313422294120286634350736906063837462003712", "pair_count": 27, "precision_bits": 160, "sha256": "2462e3430b3d440a5198660bbef9b8786f270223afae59b443a80d106ba555d1", "status": "certified", "sticks": 9, "threshold_lower": "417497304876836580947885294456073636706481072187/49039857307708443467467104868809893875799651909875269632", "threshold_upper": "834994609753673161895770588912147273412962144395/98079714615416886934934209737619787751599303819750539264"}, "mp_certificate": ["3.203086506533322006991294537504743941951e-16", "0.0001845364811118047421431045133154390633704", "0.000000008513428215281867090225527742645906462412", true], "mu": 0.00018453648111178275, "ratio": 2.608169110141873e-08}, "completed_monotonic": 36370.497727702, "coordinates": "/tmp/equistick-crss/results/9_29_equilateral_9sticks.txt", "elapsed_seconds": 5.25087370199617, "input_identity": {"coordinate_ids": ["K9a31", "K9a31", "K9a31"], "pd_id": "9_29"}, "log": "/tmp/equistick-crss/results/logs/20260925T203435_0ece8cc7_9_29.log", "message": "certified unchanged equal-stick source", "method": "unchanged equal-stick source", "name": "9_29", "seed": 2045959619, "status": "certified"}

## 20260925T203440_c2a98843

```json
{
  "run_id": "20260925T203440_c2a98843",
  "manifest": "/tmp/equistick-crss/results/logs/20260925T203440_c2a98843.json",
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "/tmp/equistick-crss/scripts/crss_close_known.py",
    "--knots",
    "K13n586,K13n593,9_29",
    "--budget",
    "300",
    "--workers",
    "2"
  ],
  "seeds": {
    "K13n586": 3552274058,
    "K13n593": 3133842244
  },
  "projection_seeds": [
    1,
    2,
    3
  ],
  "budget_seconds": 300.0,
  "workers": 2,
  "git_commit": "65cf4ad721feae577b70853dbde921f712d1d4af",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "numba": "0.67.0",
    "mpmath": "1.4.1",
    "snappy": "3.3.2",
    "spherogram": "2.4.1"
  },
  "data_path": "/workspace/repos/stick-knot-optimizer/stick-knot-gen",
  "data_commit": "9f05018917d3cd568867412a0c39ec843aa8744e",
  "python": "3.11.2 (main, May 12 2026, 05:17:27) [GCC 12.2.0]",
  "stick_number_source": "TEN_STICK_19: Cantarella et al., JKTR 2026; otherwise exact_values.csv",
  "results": [],
  "crss_sha256": "a7e8bfc961bc2156d373ea206fc52beb3fbb122931f0743a979e5823de07cb22",
  "input_doi": "10.7910/DVN/NFJIII",
  "optimizer": "fatten plus homotopy_equalize",
  "interval_checker": "exact decimal outward geometric bounds"
}
```

K13n593 [certified] {"attempts": [{"method": "unchanged equal-stick source", "rejected": "Millett-Rawdon ratio is not below one"}, {"completed": true, "elapsed_seconds": 8.89, "floor": 0.9, "method": "fatten plus safe homotopy", "mu_floor": 0.01662570698306124, "reached": 1.0}], "certificate": {"coordinate_ids": ["K13n593", "K13n593", "K13n593"], "defect": 5.024192173408437e-11, "interval": {"backend": "mpmath.iv", "defect_lower": "73428678002010591166075826221852526251/1461501637330902918203684832716283019655932542976", "defect_upper": "73428678002010591166075826221852526263/1461501637330902918203684832716283019655932542976", "file": "K13n593_equilateral_10sticks.txt", "mpmath_version": "1.4.1", "mu_lower": "777559468672155786575564720518998858401858177083/46768052394588893382517914646921056628989841375232", "mu_upper": "777559468672155786575564720518998858401858177091/46768052394588893382517914646921056628989841375232", "pair_count": 35, "precision_bits": 160, "sha256": "1503f4a70271e84cc1ce423b5cdbfe95af55dda79a62188e5e6a8a3b78e2da09", "status": "certified", "sticks": 10, "threshold_lower": "827366472781906608242433949818949087501322058031/11972621413014756705924586149611790497021399392059392", "threshold_upper": "827366472781906608242433949818949087501322058049/11972621413014756705924586149611790497021399392059392"}, "mp_certificate": ["0.00000000005024194039701341765021689807372772924901", "0.01662586806291981000661903876034949645561", "0.00006910487221140422886728925517548651157157", true], "mu": 0.01662586806291981, "ratio": 7.270387763743387e-07}, "completed_monotonic": 36383.261939749, "coordinates": "/tmp/equistick-crss/results/K13n593_equilateral_10sticks.txt", "elapsed_seconds": 13.08771803700074, "input_identity": {"coordinate_ids": ["K13n593", "K13n593", "K13n593"], "pd_id": "K13n593"}, "log": "/tmp/equistick-crss/results/logs/20260925T203440_c2a98843_K13n593.log", "message": "certified fatten plus homotopy at floor 0.9", "method": "fatten plus homotopy at floor 0.9", "name": "K13n593", "seed": 3133842244, "status": "certified"}

K13n586 [certified] {"attempts": [{"method": "unchanged equal-stick source", "rejected": "Millett-Rawdon ratio is not below one"}, {"completed": true, "elapsed_seconds": 8.86, "floor": 0.9, "method": "fatten plus safe homotopy", "mu_floor": 0.019392046592406228, "reached": 1.0}], "certificate": {"coordinate_ids": ["K13n586", "K13n586", "K13n586"], "defect": 2.4424906541753444e-15, "interval": {"backend": "mpmath.iv", "defect_lower": "429417874015125663229091036795531/182687704666362864775460604089535377456991567872", "defect_upper": "1717671496060502652916364147182133/730750818665451459101842416358141509827966271488", "file": "K13n586_equilateral_10sticks.txt", "mpmath_version": "1.4.1", "mu_lower": "906928251072015971159055141893007060351909381181/46768052394588893382517914646921056628989841375232", "mu_upper": "113366031384001996394881892736625882543988672649/5846006549323611672814739330865132078623730171904", "pair_count": 35, "precision_bits": 160, "sha256": "c6e1cec1ca096a563eb35e132a970cf19ae60359665961baf3ffce15f3c0c8d1", "status": "certified", "sticks": 10, "threshold_lower": "1125580473648578892201655028550426537012899506375/11972621413014756705924586149611790497021399392059392", "threshold_upper": "281395118412144723050413757137606634253224876601/2993155353253689176481146537402947624255349848014848"}, "mp_certificate": ["2.350027414299950654926698747543473144249e-15", "0.01939204659240735502469936704334409358097", "0.00009401286776052442742526920277775592858854", true], "mu": 0.01939204659240737, "ratio": 2.5980386646613195e-11}, "completed_monotonic": 36383.267092092, "coordinates": "/tmp/equistick-crss/results/K13n586_equilateral_10sticks.txt", "elapsed_seconds": 13.120902274997206, "input_identity": {"coordinate_ids": ["K13n586", "K13n586", "K13n586"], "pd_id": "K13n586"}, "log": "/tmp/equistick-crss/results/logs/20260925T203440_c2a98843_K13n586.log", "message": "certified fatten plus homotopy at floor 0.9", "method": "fatten plus homotopy at floor 0.9", "name": "K13n586", "seed": 3552274058, "status": "certified"}

## CRSS steps 1-3 handoff with known limitations

The supplied NetCDF SHA256 is `a7e8bfc961bc2156d373ea206fc52beb3fbb122931f0743a979e5823de07cb22`; its bytes and Eddy's external clone were not committed. Final local checks: 93 unittest tests passed, `pip check` passed, 12,965 groups indexed, Table 1 frequencies matched, 319 of 321 sampled knot names matched, and `10_86` / `10_162` were recorded as mismatches. The validator intentionally writes a complete report before exiting 1 on those two findings. All 28 in-repository saved polygons passed the saved-file numerical checks and exact-decimal geometric interval checks. The exact-stick coverage report records 59 nontrivial matches, 23 in-repository and 36 checked in the Eddy clone. `gap_census.csv` has 12,965 unique sorted rows, including 1,919 Eddy-only reported counts and 744 positive candidate-count differences; these are not proved gaps.

The independent reader security review did **not** pass. `_FillValue` and `missing_value` attributes can be nonscalar and broadcast coordinate or PD comparisons into large temporaries. `_scalar` reads a scalar variable before dtype or payload-size checks and reads an attribute fallback before checking its shape. The fixed source above was audited, but the reader is not safe for arbitrary `EQUISTICK_CRSS` files. Next session: add RED proxy tests for unsafe scalar metadata and nonscalar fill sentinels, validate metadata before reading or comparing, rerun all reports and tests, then obtain independent approval before task 4. Knot identity is numerical throughout; interval certificates cover geometric inequalities only.
