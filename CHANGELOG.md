# Changelog

## 0.1.1 - 2026-09-21

- Documentation and release-engineering maintenance release; scientific algorithms, stable API contracts, and frozen R/Python behavioral parity remain unchanged.
- Includes the post-0.1.0 documentation, navigation, accessibility, visual-design, citation, and ecosystem-documentation improvements already merged to `main`.
- Revalidated 125/125 Python tests with 4,020/4,020 executable statements and 1,700/1,700 measured branches covered, with zero partial branches.
- Retains 127 compatibility exports: 71 stable and 56 experimental.
- Retains the frozen behavioral matrix of 67 PASS / 4 EXPECTED-DIFFERENCE / 0 PENDING / 0 FAIL against `gp3ml 0.3.0`.
- Hardens release governance so version tags run the complete CI suite and PyPI publication uses exact, checksum-verified stable GitHub release assets.

## 0.1.0 - 2026-08-31

- Initial formal Python release targeting frozen gp3ml 0.3.0.
- 127/127 R exports represented: 71 stable and 56 experimental.
- 38 stable public object classes represented.
- Governance, provenance, leakage, splitting, resampling, modelling, uncertainty, calibration, external validation, decision governance, conformal prediction, shift auditing, analysis plans, model artifacts, robustness, reproducibility, interoperability, RO-Crate, and API contracts implemented.
- 16 R plot contracts mapped to Matplotlib.
- Python-native backend deviations are recorded rather than described as exact algorithmic parity.
- Completed the stable-API R/Python behavioral freeze for all 71 stable exports against the SHA-256-verified gp3ml 0.3.0 release archive.
- Final stable matrix: 67 PASS, 4 EXPECTED-DIFFERENCE, 0 PENDING, and 0 FAIL.
- Documented expected differences cover unequal calibration-vector recycling, shortened classification-probability recycling, the frozen-R repeat-level uncertainty defect, and the frozen-R release-model-card Markdown writer defect; gp3mlpy retains the safer or functioning behavior.
- Coverage-hardening suite expanded to 125 passing tests.
- Enforced 100% statement and branch coverage across 4,020 executable statements and 1,700 branches with zero partial branches.
- Added a permanent CI coverage floor using `--cov-branch --cov-fail-under=100`.
- Retained the full Ubuntu/Windows/macOS × Python 3.11/3.12/3.13 runtime matrix alongside Ruff, mypy, strict documentation, package build, Twine, and installed-wheel API gates.
- Fixed Keras backend-readiness validation so confirmed NumPy/pandas boolean scalars are accepted correctly.
- Refreshed public documentation, parity status, citation metadata, Zenodo metadata, and release automation for the 0.1.0 release.
