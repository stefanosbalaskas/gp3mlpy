# gp3mlpy validation report

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

## Finalized Python quality baseline

The coverage-hardening tranche merged to `main` as commit `782768d20bb11bed11e4db53118fb49735340a81`.

The finalized Python-side validation state is:

```text
pytest: 122 passed
statements: 4,004 / 4,004 covered
branches: 1,690 / 1,690 covered
partial branches: 0
total line + branch coverage: 100.00%
coverage floor: --cov-fail-under=100
```

The coverage gate runs with branch coverage enabled and therefore fails if either executable statements or decision arcs fall below the enforced 100% floor.

## Runtime and packaging gates

GitHub Actions validates the package on:

- Ubuntu: Python 3.11, 3.12, 3.13
- Windows: Python 3.11, 3.12, 3.13
- macOS: Python 3.11, 3.12, 3.13

Additional gates cover:

- source/test compilation;
- Ruff semantic lint;
- public-stub mypy validation;
- strict MkDocs documentation build;
- deterministic documentation plot generation;
- sdist and wheel build;
- Twine package checks; and
- fresh installed-wheel frozen-API smoke validation.

The documentation deployment for the finalized coverage-hardening commit also completed successfully.

## Independent Windows validation

A fresh local Windows validation on Python 3.11 reproduced the finalized result:

```text
122 passed
4,004 statements / 0 missed
1,690 branches / 0 partial
Required test coverage of 100% reached.
Total coverage: 100.00%
```

The suite currently emits non-fatal warnings from deliberately small/stress test fixtures, including `statsmodels` perfect-separation warnings, short neural-network convergence warnings, one NumPy overflow stress warning, and pandas chained-assignment warnings in test-only mutation fixtures. These do not change the pass/fail or coverage result and are candidates for later warning cleanup.

## Production defect fixed during hardening

Coverage hardening exposed a real optional-backend guard defect in the Keras capability path: identity comparison against Python `True` could reject a valid NumPy/pandas boolean scalar. The guard now uses truth-value semantics while preserving the intended backend-readiness contract.

## Release status

The package remains a **development candidate** at `0.1.0.dev0`. No GitHub release or PyPI release has been declared yet.

The repository and documentation are suitable for public development use, but the first PyPI release should be made deliberately as a versioned release event rather than treating this development snapshot as already released.

## Parity status

`r_parity_tested` remains `false`.

The 100% Python statement/branch coverage result demonstrates comprehensive execution of the current Python package contracts. It does **not** establish completed R-versus-Python numerical or algorithmic parity. Cross-language behavioral fixtures remain a separate release-hardening milestone.
