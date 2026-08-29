"""Validate the frozen gp3ml 0.3.0 reference layer and gp3mlpy public contract."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import gp3mlpy

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
EXPECTED = {
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


def load_json(name: str):
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"release validation FAILED: {message}")


def main() -> None:
    summary = load_json("reference_summary.json")
    if summary != EXPECTED:
        fail(f"reference_summary.json differs from frozen counts: {summary!r}")

    if gp3mlpy.r_reference_version != "0.3.0":
        fail(f"r_reference_version={gp3mlpy.r_reference_version!r}")

    registry = gp3mlpy.gp3ml_api_contracts()
    exports = registry.exports.copy()
    if len(exports) != 127:
        fail(f"public export count is {len(exports)}, expected 127")
    stable = int((exports["stability"] == "stable").sum())
    experimental = int((exports["stability"] == "experimental").sum())
    if (stable, experimental) != (71, 56):
        fail(f"stability counts are {(stable, experimental)}, expected (71, 56)")
    if not bool(exports["present"].all()):
        missing = exports.loc[~exports["present"], "name"].tolist()
        fail(f"contract exports missing from package namespace: {missing}")

    with (REFERENCE / "r_api_inventory.csv").open(encoding="utf-8", newline="") as handle:
        frozen_rows = list(csv.DictReader(handle))
    frozen = {(row["name"], row["stability"]) for row in frozen_rows}
    actual = set(zip(exports["name"], exports["stability"], strict=True))
    if actual != frozen:
        missing = sorted(frozen - actual)
        extra = sorted(actual - frozen)
        fail(f"API inventory mismatch; missing={missing}, extra={extra}")

    classes = registry.classes
    frozen_classes = set(load_json("r_public_classes.json"))
    actual_classes = set(classes["class"].tolist())
    if actual_classes != frozen_classes:
        fail(
            "stable class inventory mismatch; "
            f"missing={sorted(frozen_classes - actual_classes)}, "
            f"extra={sorted(actual_classes - frozen_classes)}"
        )

    inventory_lengths = {
        "r_print_methods.json": 49,
        "r_plot_methods.json": 16,
        "r_articles.json": 20,
        "r_tests_inventory.json": 32,
        "r_example_inventory.json": 49,
        "r_failure_contracts.json": 44,
    }
    for filename, expected_count in inventory_lengths.items():
        count = len(load_json(filename))
        if count != expected_count:
            fail(f"{filename} has {count} entries, expected {expected_count}")

    reference_pages = [p for p in (ROOT / "docs" / "reference").glob("*.md") if p.name != "index.md"]
    article_pages = [p for p in (ROOT / "docs" / "articles").glob("*.md") if p.name != "index.md"]
    article_examples = list((ROOT / "examples").glob("*.py"))
    if len(reference_pages) != 127:
        fail(f"docs/reference has {len(reference_pages)} topic pages, expected 127")
    if len(article_pages) != 20:
        fail(f"docs/articles has {len(article_pages)} article pages, expected 20")
    if len(article_examples) != 20:
        fail(f"examples has {len(article_examples)} article companions, expected 20")

    required = [
        ROOT / "src" / "gp3mlpy" / "__init__.pyi",
        ROOT / "src" / "gp3mlpy" / "py.typed",
        ROOT / "PROHIBITED-USE.md",
        ROOT / "GOVERNANCE.md",
        ROOT / ".github" / "workflows" / "ci.yml",
    ]
    for path in required:
        if not path.is_file():
            fail(f"required release file missing: {path.relative_to(ROOT)}")

    # These names belonged to an obsolete/synthetic CI workspace, not this
    # clean parity-candidate source layout. Their presence would reintroduce
    # the exact stale Ruff failures that triggered this manual replacement.
    stale = [
        ROOT / "src" / "gp3mlpy" / "evaluation.py",
        ROOT / "src" / "gp3mlpy" / "selection.py",
        ROOT / "bootstrap",
        ROOT / "bootstrap-fix",
        ROOT / "bootstrap-parts",
        ROOT / "payload",
        ROOT / "payload2",
        ROOT / ".runtime_payload",
    ]
    present_stale = [str(p.relative_to(ROOT)) for p in stale if p.exists()]
    if present_stale:
        fail(f"stale bootstrap/generated paths are present: {present_stale}")

    print("release validation: PASS")
    for key, value in EXPECTED.items():
        print(f"{key}: {value}")
    print("python_article_companions: 20")
    print("r_parity_tested: false")


if __name__ == "__main__":
    main()
