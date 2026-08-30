# Changelog

## 0.1.0.dev0

- Initial Python compatibility implementation targeting frozen gp3ml 0.3.0.
- 127/127 R exports represented: 71 stable and 56 experimental.
- 38 stable public object classes represented.
- Governance, provenance, leakage, splitting, resampling, modelling, uncertainty, calibration, external validation, decision governance, conformal prediction, shift auditing, analysis plans, model artifacts, robustness, reproducibility, interoperability, RO-Crate, and API contracts implemented.
- 16 R plot contracts mapped to Matplotlib.
- Python-native backend deviations are recorded rather than described as exact algorithmic parity.
- Coverage-hardening suite expanded to 122 passing tests.
- Enforced 100% statement and branch coverage across 4,004 executable statements and 1,690 branches with zero partial branches.
- Added a permanent CI coverage floor using `--cov-branch --cov-fail-under=100`.
- Retained the full Ubuntu/Windows/macOS × Python 3.11/3.12/3.13 runtime matrix alongside Ruff, mypy, strict documentation, package build, Twine, and installed-wheel API gates.
- Fixed Keras backend-readiness validation so confirmed NumPy/pandas boolean scalars are accepted correctly.
- Refreshed public documentation, parity status, validation evidence, and release-readiness messaging for the finalized development baseline.
