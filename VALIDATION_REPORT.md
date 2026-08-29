# gp3mlpy quality-fix validation report

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

## Windows quality-gate evidence supplied by the user

Before this correction:
- `pytest`: 40 passed, 2 expected `PerfectSeparationWarning` messages.
- MkDocs strict build: completed.
- Hatch sdist/wheel build: completed.
- `twine check`: passed for wheel and sdist.
- Ruff: failed, dominated by the public stub wildcard-import/name-resolution problem plus 12 unused imports and one dynamic-base static-analysis finding.
- Mypy: failed with 135 undefined-name errors in `src/gp3mlpy/__init__.pyi`.

## Corrections in this bundle

- Replaced the `from .objects import *` stub dependency with private type aliases for the dynamic R-S3-compatible object types.
- Added `TypeVar("T")` explicitly in the stub.
- Preserved all 127 public function signatures.
- Removed the 12 unused imports reported by Ruff.
- Rewrote the release-model-card base reference so static analyzers no longer see `GP3MLModelCard` as undefined while preserving the runtime inheritance contract.
- Expanded `mkdocs.yml` navigation to include all generated article/reference pages.
- Updated the Windows quality runner to execute all gates through `uv`.

## Validation executed in the sandbox after correction

```text
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/validate_release.py
python -m compileall -q src/gp3mlpy tests examples scripts
```

Result:

```text
40 passed, 2 warnings
release validation: PASS
127 exports / 71 stable / 56 experimental / R reference 0.3.0
compileall: PASS
```

The two warnings are the expected `statsmodels` `PerfectSeparationWarning` messages from the tiny calibration fixture.

A static annotation-resolution audit found no undefined names in the corrected public stub.

## Gates requiring the user's installed development environment

Run these in the Positron checkout after applying the bundle:

```powershell
uv lock
uv sync --extra dev --extra docs
uv run python -m ruff check src tests examples
uv run python -m mypy --follow-imports=skip src\gp3mlpy\__init__.pyi
uv run python -m mkdocs build --strict
uv run python -m build
uv run python -m twine check dist\*
```

Do not merge the pull request until these gates and GitHub Actions are green.

## Parity status

`r_parity_tested` remains `false`. The bundle fixes static quality and packaging infrastructure; it does not claim completed numerical/algorithmic R/Python parity.
