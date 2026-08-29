# gp3mlpy manual-upload validation report

Prepared: 2026-08-30

## Frozen R reference

- R reference package: `gp3ml` 0.3.0
- exports: 127
- stable exports: 71
- experimental exports: 56
- stable public classes: 38
- R source files: 38
- Rd files: 154
- source vignettes: 20
- R test files: 32
- print contracts: 49
- plot contracts: 16
- exported Rd topics with examples: 49
- explicit `expect_error()` contracts: 44

The packaged `reference/`, `docs/`, and `examples/` directories were regenerated directly from the frozen gp3ml 0.3.0 source and compared recursively with the packaged copies; no differences were found.

## Python validation performed locally

```text
python -m compileall -q src tests examples scripts
python -m pytest -q
```

Result:

```text
40 passed, 2 warnings
```

The two warnings are `statsmodels` `PerfectSeparationWarning` messages from the deliberately tiny calibration smoke fixture; they are warnings, not test failures.

`tests/test_examples.py` executes every one of the 20 Python article companion scripts and passed.

## Files included

- validated `src/gp3mlpy/` runtime from the validation wheel
- `py.typed` and public `__init__.pyi`
- 127 generated API reference pages
- 20 generated source-derived article pages
- 20 runnable Python article companion scripts
- frozen R inventories under `reference/`
- repository tests
- governance, prohibited-use, security, contribution, and citation files
- `mkdocs.yml`
- cross-platform GitHub Actions CI
- deterministic `scripts/generate_reference_layer.py`

## Gates not executed in this offline sandbox

The local environment does not currently provide `ruff`, `mypy`, or `mkdocs`, so the corresponding formatting/lint, static typing, and strict documentation-build gates were not executed here. The included GitHub Actions workflow is configured to run those gates after upload.

A canonical Hatch-built sdist/wheel and `twine check` should also be treated as GitHub/online CI gates. The existing validation wheel was used as the runtime source for this manual-upload bundle.

## Release status

This is a parity-candidate repository snapshot, not a claim of completed numerical/algorithmic parity with every R backend. R `ranger`/`nnet` versus Python backend differences remain explicitly documented, and R-runtime fixture parity remains a separate gate.
