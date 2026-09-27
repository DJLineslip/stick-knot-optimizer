
## 20260926T162136_75af5c17

```json
{
  "run_id": "20260926T162136_75af5c17",
  "manifest": "/workspace/repos/stick-knot-optimizer/results/ten_new_stick_knots/logs/20260926T162136_75af5c17.json",
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "scripts/10_fifteen_pool.py",
    "--budget",
    "1200",
    "--workers",
    "1"
  ],
  "seeds": {
    "K15n124836": 3491315336,
    "K15n124999": 3146490276,
    "K15n131344": 2008940288,
    "K15n40214": 1602230870,
    "K15n41126": 542765445,
    "K15n41131": 2804307303,
    "K15n41142": 1903071770,
    "K15n41183": 2866605440,
    "K15n41188": 1024341000,
    "K15n41202": 390392135,
    "K15n41213": 2035869840,
    "K15n41237": 1275285003,
    "K15n41238": 3703329690,
    "K15n43517": 3514036871,
    "K15n45460": 620741640,
    "K15n45603": 3894994522,
    "K15n46532": 2767295672,
    "K15n47800": 3518386785,
    "K15n49036": 4248076828,
    "K15n51709": 3141393300,
    "K15n51757": 569508566,
    "K15n52940": 3198559440,
    "K15n52941": 3382776902,
    "K15n56079": 1826906991,
    "K15n56089": 3950804896,
    "K15n59060": 1413544668,
    "K15n67540": 3351402326,
    "K15n84457": 3975841804,
    "K15n94888": 4254153476
  },
  "projection_seeds": [
    1,
    2,
    3
  ],
  "budget_seconds": 1200.0,
  "workers": 1,
  "git_commit": "b8b2539a43673c6355e924111384d2720ce936bd",
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

## 20260926T163138_60a140ef

```json
{
  "run_id": "20260926T163138_60a140ef",
  "manifest": "/workspace/repos/stick-knot-optimizer/results/ten_new_stick_knots/logs/20260926T163138_60a140ef.json",
  "command": [
    "/workspace/repos/stick-knot-optimizer/.venv/bin/python",
    "scripts/10_fifteen_pool.py",
    "--budget",
    "1200",
    "--workers",
    "1"
  ],
  "seeds": {
    "K15n124836": 3491315336,
    "K15n124999": 3146490276,
    "K15n131344": 2008940288,
    "K15n40214": 1602230870,
    "K15n41126": 542765445,
    "K15n41131": 2804307303,
    "K15n41142": 1903071770,
    "K15n41183": 2866605440,
    "K15n41188": 1024341000,
    "K15n41202": 390392135,
    "K15n41213": 2035869840,
    "K15n41237": 1275285003,
    "K15n41238": 3703329690,
    "K15n43517": 3514036871,
    "K15n45460": 620741640,
    "K15n45603": 3894994522,
    "K15n46532": 2767295672,
    "K15n47800": 3518386785,
    "K15n49036": 4248076828,
    "K15n51709": 3141393300,
    "K15n51757": 569508566,
    "K15n52940": 3198559440,
    "K15n52941": 3382776902,
    "K15n56079": 1826906991,
    "K15n56089": 3950804896,
    "K15n59060": 1413544668,
    "K15n67540": 3351402326,
    "K15n84457": 3975841804,
    "K15n94888": 4254153476
  },
  "projection_seeds": [
    1,
    2,
    3
  ],
  "budget_seconds": 1200.0,
  "workers": 1,
  "git_commit": "7470f58ec410aeb729b51a33bb217f2a19339cd8",
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

## 2026-09-27 crossing-change first pass and equalization

- Source inventory: 882 known ten-stick polygons, 870 completed, 12 timed out, zero unattempted. Do not treat the timeout records as full grids. No retry of those sources was run.
- Completed grids: 2,923,200 proposed moves, 164,022 single-edge crossings, 33 distinct four-Wirtinger names reached.
- Initial memory failures: exit 137 during NetCDF indexing. The indexing fix released the group enumeration memory in a subprocess. A preserved checkpoint resumed 30 completed sources; checkpoint archive SHA256 `510143ce01558123e1ba2ef0e06529c5b17345e5b73e5be51519470833bee5fe`.
- Code base at execution: `9f7c93e98342617a6f4ed6cf059dec508ca4524b`; scan code SHA256 `75b8d941ededd2827e40c2bf4525b3b6adf5957c2682304f1b5298a6858be329`. Eddy commit `9f05018917d3cd568867412a0c39ec843aa8744e`. NetCDF SHA256 `a7e8bfc961bc2156d373ea206fc52beb3fbb122931f0743a979e5823de07cb22`. Wirt_Hm SHA256 `514b12e118fa681520da5702a5ed4307ce7cf93b6b9f7b5e53f265ca7b01647b`. Full package versions, deterministic source seeds, grid, file hashes and times are in `ten_new_stick_knots/crossing_results.json`.
- Scan command (from the manifest): `scripts/11_crossing_changes.py --eddy /workspace/repos/stick-knot-optimizer/stick-knot-gen/stick_number/mseq_knots --workbook /workspace/repos/stick-knot-optimizer/data/external/Wirt_Hm/all_data_A.xlsx --results results --output results/ten_new_stick_knots/crossing_results.json --scratch results/logs/crossing_changes --source-budget 60 --candidate-budget 180`. Budget: 60 seconds per source, 180 seconds per candidate. The original execution wrote into `/tmp/equistick-sync-b3iRpz`; the manifest and `crossing_scan.log` were copied byte-for-byte into `results/ten_new_stick_knots/`.
- Equalization command from the isolated worktree (same flags shown as repo-relative paths): `/workspace/repos/stick-knot-optimizer/.venv/bin/python scripts/12_equalize_crossings.py --manifest results/ten_new_stick_knots/crossing_results.json --scratch results/logs/crossing_changes --pool-results results/ten_new_stick_knots/pool_results.json --results results/ten_new_stick_knots --output results/ten_new_stick_knots/crossing_equalization.json --candidate-budget 180 --max-attempts 1`. One candidate per reached, map-verified target, at most 180 seconds. No source timeout was retried. Twelve new equal-stick ten-gons were saved and interval-certified; two are among the 26 prior reduction timeouts. Twenty-four targets were not reached in completed source grids. Twenty reached knots had prior certified polygons and were not rechecked; one reached knot lacks a passing map.
- Equalization details: `ten_new_stick_knots/crossing_equalization.json` and `.log`; successful move witnesses: `crossing_witnesses.json`; new Wirt_Hm Gauss comparisons: `crossing_diagram_checks.json`; decimal geometry: twelve `*_equilateral_10sticks.txt` files; augmented interval records: `interval_certificates.json`. The README lists all outcomes and limitations. Raw NetCDF coordinates and scratch worker outputs are not committed.
- Mathematical limit: interval checks certify embedded decimal polygon geometry; S5/D4 checks certify finite-group relations and generation on Wirt_Hm diagrams. Named-knot identities depend on numerical complement comparisons, so exact stick-number conclusions remain conditional on those identities.
