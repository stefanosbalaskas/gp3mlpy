# gp3mlpy parity status

- Python package: `gp3mlpy` 0.1.0.dev0
- Frozen R reference: `gp3ml` 0.3.0
- Frozen R release tag: `v0.3.0`
- Frozen R release commit: `0e23684f32a44417827aa0e7fee9fefc9e6c35d3`
- Frozen R archive SHA-256: `04c5fdb017c63ecbf6f5e85c29fff6492880201a2f2aa5c3c1d3fe998e60bae6`
- Frozen API inventory: 127 exports
- Stable exports: 71
- Experimental exports: 56
- Stable public classes: 38
- Included Python tests: 122 passing in the finalized coverage-hardening suite
- Statement coverage: 4,004 / 4,004 executable statements (100%)
- Branch coverage: 1,690 / 1,690 branches with 0 partial branches (100%)
- Coverage enforcement: permanent `--cov-fail-under=100` CI gate with branch coverage enabled
- Runtime matrix: Ubuntu, Windows, and macOS on Python 3.11, 3.12, and 3.13
- Package gates: Ruff, public-stub mypy, strict MkDocs, build, Twine, and installed-wheel API smoke
- Behavioral parity campaign: **in progress on `parity/r-0.3.0-behavioral-freeze`**
- First executable parity tranche: prohibited-use registry plus deterministic classification/regression metrics
- R-runtime behavioural fixture parity: **not yet completed**
- `r_parity_tested`: `false`

The behavioral campaign uses the exact upstream `gp3ml_0.3.0.tar.gz` release archive as the R oracle. GitHub Actions verifies its SHA-256 before installation, executes shared fixtures through both runtimes, compares normalized outputs under declared rules, and emits a machine-readable stable-API parity matrix.

The current candidate has comprehensive Python-side behavioral/contract coverage and a frozen governance/API/documentation compatibility layer. **100% Python line and branch coverage is not equivalent to completed R-versus-Python numerical or algorithmic parity.**

API, semantic, numerical, and algorithmic parity remain separate claims. Python-native backend adapters must not be described as bitwise or algorithmically identical to the R implementation when their underlying engines differ. `r_parity_tested` will remain `false` until the defined stable-API behavioral acceptance criteria are satisfied and the resulting evidence is frozen.
