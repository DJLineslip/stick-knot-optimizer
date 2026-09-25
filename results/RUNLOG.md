
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
