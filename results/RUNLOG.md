
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

Both saved 17-digit coordinate files were reloaded and validated in the worker before publication, then copied to `results/` and independently reverified by `scripts/06_verify_results.py` (all 23 rows `certified=True`, `type_confirmed=True`). This is numerical evidence, not interval arithmetic or a formal knot-type proof. Full per-knot logs and the run manifest were written under `/tmp/equistick-torus37-probe/logs/` (temporary, not committed).
