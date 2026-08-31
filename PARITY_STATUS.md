# gp3mlpy parity status

- Python package: `gp3mlpy` 0.1.0
- Frozen R reference: `gp3ml` 0.3.0
- Frozen R release tag: `v0.3.0`
- Frozen R release commit: `0e23684f32a44417827aa0e7fee9fefc9e6c35d3`
- Frozen R archive SHA-256: `04c5fdb017c63ecbf6f5e85c29fff6492880201a2f2aa5c3c1d3fe998e60bae6`
- Frozen API inventory: 127 exports
- Stable exports: 71
- Experimental exports: 56
- Stable public classes: 38
- Included Python tests: 125 passing
- Statement coverage: 4,020 / 4,020 executable statements (100%)
- Branch coverage: 1,700 / 1,700 branches with 0 partial branches (100%)
- Coverage enforcement: permanent `--cov-fail-under=100` CI gate with branch coverage enabled
- Runtime matrix: Ubuntu, Windows, and macOS on Python 3.11, 3.12, and 3.13
- Package gates: Ruff, public-stub mypy, strict MkDocs, build, Twine, and installed-wheel API smoke
- Behavioral parity campaign: **completed and frozen on `parity/r-0.3.0-behavioral-freeze`**
- R-runtime behavioral fixture parity: **completed for all 71 stable exports**
- Stable matrix: **67 PASS / 4 EXPECTED-DIFFERENCE / 0 PENDING / 0 FAIL**
- `r_parity_tested`: `true`

The behavioral campaign uses the exact upstream `gp3ml_0.3.0.tar.gz` release archive as the R oracle. GitHub Actions verifies its SHA-256 before installation, executes shared fixtures through both runtimes, compares normalized outputs under declared rules, and emits the machine-readable stable-API parity matrix.

The four `EXPECTED-DIFFERENCE` functions represent explicit safety/reference-defect boundaries rather than unresolved parity failures:

1. `assess_gazepoint_calibration` — frozen R 0.3.0 recycles unequal truth/probability vectors with warnings; gp3mlpy rejects unequal lengths.
2. `gazepoint_performance_metrics` — frozen R 0.3.0 recycles a shortened classification-probability vector; gp3mlpy rejects row misalignment.
3. `summarize_gazepoint_resample_uncertainty` — frozen R 0.3.0 fails for repeat-level uncertainty because of its reserved `repeat` formula path; gp3mlpy returns the intended repeat-level summary.
4. `write_gazepoint_release_model_card` — frozen R 0.3.0 has a NULL-selection partial-matching defect in Markdown writing; gp3mlpy retains the functioning writer.

No stable export remains `PENDING` or `FAIL`. The 56 experimental exports are represented by the compatibility layer but are outside this 71-export stable behavioral freeze.

API, semantic, numerical, and algorithmic parity remain separate claims. Python-native backend adapters must not be described as bitwise or algorithmically identical to the R implementation when their underlying engines differ. The completed behavioral freeze establishes the declared stable-API contract while preserving documented safer or functioning Python behavior at the four reference-defect boundaries.
