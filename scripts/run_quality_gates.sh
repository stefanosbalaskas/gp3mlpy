#!/usr/bin/env bash
set -euo pipefail
python --version
python -m compileall -q src/gp3mlpy tests
python -m ruff check src tests examples
python -m mypy --follow-imports=skip src/gp3mlpy/__init__.pyi
MPLBACKEND=Agg python -m pytest -q
python scripts/validate_release.py
python -m mkdocs build --strict
rm -rf dist build
python -m build
python -m twine check dist/*
echo "All local quality commands completed successfully."
