# Manual upload: clean parity candidate

This repository snapshot is intended to replace the working tree on the `manual-parity-candidate` branch of `stefanosbalaskas/gp3mlpy`.

## Why a full replacement is used

A previous pull-request quality job reported five Ruff `F841` findings in `src/gp3mlpy/evaluation.py` and `src/gp3mlpy/selection.py`. Those two modules are not part of the validated parity-candidate source tree or its public module layout. This bundle therefore uses a clean-tree replacement rather than inventing code for obsolete/synthetic CI files.

The parity-freeze Ruff policy blocks on semantic Pyflakes (`F`) diagnostics only. Broader formatting/import-order/modernization changes are deliberately deferred so they cannot alter validated runtime behavior during parity freeze.

## Recommended Windows/Positron procedure

1. Check out `manual-parity-candidate` in your local clone.
2. Keep a backup or create a safety branch.
3. Run the outer bundle script `apply_to_clone.ps1 -RepositoryPath <your clone>`; it removes tracked repository content and replaces it with this clean snapshot while preserving `.git`.
4. Inspect `git status` carefully.
5. Install development and documentation dependencies if necessary: `python -m pip install -e ".[dev,docs]"`.
6. Run `powershell -ExecutionPolicy Bypass -File scripts/run_quality_gates.ps1`.
7. Commit and push only after those checks are satisfactory.
8. Let PR #1 run completely before merging to `main`.

Suggested commit message:

```text
fix: replace parity candidate with clean validated source tree
```

## Important parity status

This snapshot freezes the gp3ml 0.3.0 public inventory (127 exports; 71 stable; 56 experimental), but it does **not** claim that every R/Python numerical backend has been cross-runtime tested. Keep `r_parity_tested=false` until genuine pinned R-vs-Python fixtures have run.
