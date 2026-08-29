from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_twenty_article_companions_execute():
    scripts = sorted((ROOT / "examples").glob("*.py"))
    assert len(scripts) == 20
    for script in scripts:
        runpy.run_path(str(script), run_name="__main__")
