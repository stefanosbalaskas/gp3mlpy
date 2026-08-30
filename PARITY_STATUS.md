# gp3mlpy parity status

- Python package: `gp3mlpy` 0.1.0.dev0
- Frozen R reference: `gp3ml` 0.3.0
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
- R-runtime behavioural fixture parity: **not yet completed**
- `r_parity_tested`: `false`

The current candidate has comprehensive Python-side behavioral/contract coverage and a frozen governance/API/documentation compatibility layer. **100% Python line and branch coverage is not equivalent to completed R-versus-Python numerical or algorithmic parity.**

API, semantic, numerical, and algorithmic parity remain separate claims. Python-native backend adapters must not be described as bitwise or algorithmically identical to the R implementation when their underlying engines differ. Real cross-language fixtures must be executed and frozen before `r_parity_tested` can become `true`.
