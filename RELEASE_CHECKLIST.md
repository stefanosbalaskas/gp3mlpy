# gp3mlpy release checklist

This checklist governs the first formal PyPI release and subsequent versioned releases.

## 1. Freeze the intended release scope

- Confirm the frozen R reference version remains `gp3ml 0.3.0` for the release.
- Confirm the public API inventory remains intentional: 127 exports, 71 stable, 56 experimental.
- Decide explicitly whether any remaining cross-language parity gaps are acceptable for the planned release status.
- Keep `r_parity_tested: false` unless real R-versus-Python behavioral fixtures have been executed and frozen.

## 2. Complete Python quality gates

The release commit must pass:

```bash
python -m pytest --cov=gp3mlpy --cov-branch --cov-report=term-missing --cov-fail-under=100 -q
ruff check src tests examples scripts/build_doc_assets.py
mypy --follow-imports=skip src/gp3mlpy/__init__.pyi
python scripts/build_doc_assets.py
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Required baseline:

- 100% statement coverage;
- 100% branch coverage;
- zero partial branches;
- Ubuntu/Windows/macOS CI green on Python 3.11, 3.12, and 3.13;
- package, documentation, lint, typing, and installed-wheel gates green.

## 3. Validate a fresh built artifact

Do not release from an editable checkout alone.

- Build a fresh sdist and wheel.
- Create a clean virtual environment.
- Install the exact wheel that will be published.
- Import `gp3mlpy` outside the repository tree.
- Confirm the frozen API counts and `r_reference_version`.
- Run representative governed workflows from the installed artifact.

## 4. Decide release level

### Development snapshot

`0.1.0.dev0` is for repository development and should not be treated as the first stable PyPI release.

### Pre-release

Use a version such as `0.1.0a1` or `0.1.0rc1` if public installation/testing is desired before the cross-language parity freeze is complete. The documentation must continue to state the exact parity limitations.

### First normal PyPI release

Use `0.1.0` only after the intended first-release parity boundary has been frozen and documented. For gp3mlpy, the recommended threshold is completion of behavioral R/Python fixture validation for the stable API and the principal numerical/modeling workflows, not merely Python-side code coverage.

## 5. Update release metadata

Before tagging a formal release:

- change `project.version` in `pyproject.toml`;
- update `CITATION.cff` to the same version and add `date-released` for the real release date;
- add a versioned section to `CHANGELOG.md`;
- update `PARITY_STATUS.md` and `VALIDATION_REPORT.md` with the release evidence;
- remove development-only wording from README/docs if releasing `0.1.0`;
- verify all project URLs and installation instructions.

## 6. Protect `main`

Recommended GitHub repository rules for `main`:

- require pull requests before merging;
- require successful CI checks;
- block force pushes;
- block branch deletion;
- require the coverage, package, docs, quality, and supported-runtime checks used by the project.

Repository rules are configured in GitHub Settings and should be enabled before the first formal release.

## 7. Configure PyPI Trusted Publishing

Use GitHub OIDC Trusted Publishing rather than a long-lived PyPI API token.

The repository contains `.github/workflows/publish.yml`, which publishes only after a GitHub Release is published and only after the release commit re-passes the 100% coverage, Ruff, mypy, strict documentation, build, and Twine gates.

Configure the PyPI project/pending publisher for:

- owner: `stefanosbalaskas`
- repository: `gp3mlpy`
- workflow: `publish.yml`
- environment: `pypi`

## 8. Tag and publish

- Ensure `main` is clean and the release commit is final.
- Create an annotated/version tag matching `pyproject.toml`, preferably `v0.1.0` for the first normal release.
- Create and publish the corresponding GitHub Release.
- The PyPI workflow verifies that the release tag and package version match exactly before uploading.
- Verify the resulting PyPI project page and install from PyPI into a fresh environment.

## 9. Post-release smoke test

```bash
python -m venv /tmp/gp3mlpy-release-smoke
/tmp/gp3mlpy-release-smoke/bin/python -m pip install --upgrade pip
/tmp/gp3mlpy-release-smoke/bin/python -m pip install gp3mlpy==<released-version>
```

Then confirm imports, frozen API counts, a representative synthetic workflow, and documentation links.

## Current recommendation

The current `0.1.0.dev0` development baseline is ready for **release engineering and pre-release testing**, including TestPyPI or a PyPI pre-release if desired. The recommended first normal PyPI version `0.1.0` should wait until the intended R/Python behavioral parity boundary for the stable API has been completed and frozen.
