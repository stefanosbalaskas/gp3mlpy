from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import gp3mlpy as gp

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"


def test_reference_summary_matches_frozen_r_030():
    summary = json.loads((REF / "reference_summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "r_reference_version": "0.3.0",
        "exports": 127,
        "stable_exports": 71,
        "experimental_exports": 56,
        "stable_public_classes": 38,
        "r_source_files": 38,
        "rd_files": 154,
        "vignettes": 20,
        "r_test_files": 32,
        "print_methods": 49,
        "plot_methods": 16,
        "rd_example_topics": 49,
        "explicit_expect_error_contracts": 44,
    }


def test_python_registry_matches_r_export_inventory():
    source = pd.read_csv(REF / "r_api_inventory.csv")
    py = gp.gp3ml_api_contracts().exports
    assert set(source["name"]) == set(py["name"])
    assert len(py) == 127
    stability = dict(zip(py["name"], py["stability"]))
    for row in source.itertuples(index=False):
        assert stability[row.name] == row.stability


def test_every_frozen_export_is_callable():
    source = pd.read_csv(REF / "r_api_inventory.csv")
    missing = [name for name in source["name"] if not callable(getattr(gp, name, None))]
    assert missing == []


def test_reference_inventory_counts_are_materialized():
    assert len(json.loads((REF / "r_public_classes.json").read_text())) == 38
    assert len(json.loads((REF / "r_print_methods.json").read_text())) == 49
    assert len(json.loads((REF / "r_plot_methods.json").read_text())) == 16
    assert len(json.loads((REF / "r_articles.json").read_text())) == 20
    assert len(json.loads((REF / "r_tests_inventory.json").read_text())) == 32
    assert len(json.loads((REF / "r_example_inventory.json").read_text())) == 49
    assert len(json.loads((REF / "r_failure_contracts.json").read_text())) == 44


def test_reference_pages_cover_all_exports():
    source = pd.read_csv(REF / "r_api_inventory.csv")
    pages = {p.stem for p in (ROOT / "docs" / "reference").glob("*.md") if p.name != "index.md"}
    assert set(source["name"]) == pages


def test_article_pages_cover_all_r_vignettes():
    articles = json.loads((REF / "r_articles.json").read_text())
    pages = {p.stem for p in (ROOT / "docs" / "articles").glob("*.md") if p.name != "index.md"}
    stems = {Path(x).stem for x in articles}
    assert stems == pages


def test_typing_marker_and_public_stub_present():
    pkg = Path(gp.__file__).resolve().parent
    assert (pkg / "py.typed").exists()
    assert (pkg / "__init__.pyi").exists()
