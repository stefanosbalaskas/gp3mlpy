# Manual upload checklist

1. Extract the ZIP to a clean directory.
2. Confirm `src/gp3mlpy/__init__.py`, `src/gp3mlpy/__init__.pyi`, and `src/gp3mlpy/py.typed` are present.
3. Confirm `.github/workflows/ci.yml` is preserved.
4. Do not copy any old `bootstrap/`, `.bootstrap`, `.runtime_payload/`, `__pycache__/`, or `.pytest_cache/` directories into the repository.
5. Copy this snapshot over the target repository working tree.
6. Run `python -m pytest -q` locally if desired.
7. Commit the complete tree, e.g. `feat: publish complete gp3mlpy parity candidate`.
8. Push to a staging branch first.
9. Wait for GitHub Actions: core OS/Python matrix, Ruff, Mypy, MkDocs, package build, Twine, and installed-wheel smoke.
10. Only merge/promote to `main` after the required CI jobs are green.

Use `FILE_MANIFEST.csv` to audit the uploaded file set and SHA-256 values.
