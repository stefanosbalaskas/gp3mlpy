# Manual upload instructions

This directory is a clean repository snapshot for `stefanosbalaskas/gp3mlpy`.

## Recommended upload

1. Back up the current remote branch if desired.
2. Replace the repository working tree with the contents of this directory.
3. Preserve the `.github/` directory and `src/gp3mlpy/py.typed`.
4. Commit all files in one source-tree commit.
5. Push to a staging branch first and let GitHub Actions run before merging to `main`.

Suggested commit message:

```text
feat: publish complete gp3mlpy parity candidate
```

## Validation performed before packaging

The bundle is assembled from the validated `gp3mlpy` wheel runtime and the frozen `gp3ml` 0.3.0 source-derived reference layer. Run `python -m pytest -q` after upload. The CI workflow also exercises Python 3.11–3.13 on Linux, Windows, and macOS, plus lint/type/docs/build jobs.

Do not add transient `__pycache__`, `.pytest_cache`, build, wheel, or bootstrap payload files.
