# gp3mlpy manual quality-fix checklist

1. Apply the correction bundle to `manual-parity-candidate`.
2. Run `uv lock` so `uv.lock` is regenerated from the current `pyproject.toml`.
3. Run `uv sync --extra dev --extra docs`.
4. Run `powershell -ExecutionPolicy Bypass -File scripts\run_quality_gates.ps1`.
5. Confirm:
   - 40 tests pass;
   - release validation reports 127 / 71 / 56 and R 0.3.0;
   - Ruff reports no errors;
   - mypy reports success for `src/gp3mlpy/__init__.pyi`;
   - MkDocs strict build completes;
   - wheel + sdist build;
   - Twine checks both artifacts successfully.
6. Run `git add -A` again after `uv lock` and all fixes.
7. Review `git status --short`.
8. Commit:
   `fix: close parity candidate quality gates`
9. Push:
   `git push origin manual-parity-candidate`
10. Wait for PR #1 GitHub Actions. Do not merge until all required checks are green.

`r_parity_tested` remains false until genuine cross-language fixtures are run.
