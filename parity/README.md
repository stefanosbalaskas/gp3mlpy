# gp3mlpy R/Python behavioral parity

This directory contains the executable cross-language parity campaign for the
Python port of `gp3ml`.

## Frozen upstream reference

The parity target is **gp3ml 0.3.0**, pinned to the formal upstream release:

- tag: `v0.3.0`
- annotated-tag target commit: `0e23684f32a44417827aa0e7fee9fefc9e6c35d3`
- release archive: `gp3ml_0.3.0.tar.gz`
- archive SHA-256: `04c5fdb017c63ecbf6f5e85c29fff6492880201a2f2aa5c3c1d3fe998e60bae6`

The parity workflow downloads that exact archive and verifies its digest before
installing it. A moving R branch is never used as the behavioral oracle.

## Parity states

Each comparison is classified as one of:

- `PASS`: the R and Python behavior satisfies the declared comparison rule;
- `EXPECTED-DIFFERENCE`: a documented backend/language difference is accepted;
- `NOT-APPLICABLE`: cross-language equality is not meaningful for that contract;
- `PENDING`: the case has not yet been executed against the frozen R runtime.

`r_parity_tested` remains `false` until the stable-API behavioral campaign and
its acceptance criteria are complete. Python-side 100% line/branch coverage is
a separate quality property.

## Tranche 1: deterministic core behavior

The first executable tranche covers deterministic functions that do not depend
on backend-specific model fitting or random-number-stream equivalence:

- `gp3ml_prohibited_uses()` — exact ordered string-vector equality;
- `gazepoint_classification_metrics()` — exact schema plus numeric tolerance;
- `gazepoint_regression_metrics()` — exact schema plus numeric tolerance.

Fixtures are defined once in `fixtures/core_metrics.json` and consumed by both
R and Python runners. The comparison uses `1e-12` absolute and relative
numeric tolerances, with missing numeric results normalized to JSON `null`.

## Files

- `fixtures/core_metrics.json`: shared deterministic inputs and tolerance policy;
- `core_cases.csv`: machine-readable tranche registry;
- `run_r_core.R`: executes the frozen R package against the shared fixtures;
- `run_python_core.py`: executes gp3mlpy against the same fixtures;
- `compare_core.py`: recursively compares the normalized results and exits
  nonzero on an unexplained mismatch;
- `.github/workflows/parity.yml`: installs the exact R release archive and runs
  the cross-language comparison in GitHub Actions.

Generated runtime outputs are CI artifacts rather than committed reference
truth. This avoids silently blessing stale results when either runner changes.

## Acceptance principle

Cross-language parity means matching the scientifically relevant contract, not
forcing false bitwise equality between different numerical/model backends.
Deterministic pure functions should generally match exactly within declared
floating-point tolerance. Randomized and backend-dependent workflows will use
explicit distributional, structural, or expected-difference rules in later
tranches.
