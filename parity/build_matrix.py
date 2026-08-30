from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/build_matrix.py <r-api.csv> <case-registry.csv> <core-report.json> <output.csv>"
        )

    api_path, cases_path, report_path, output_path = map(Path, sys.argv[1:])

    with api_path.open(newline="", encoding="utf-8") as handle:
        api_rows = list(csv.DictReader(handle))
    stable = [row for row in api_rows if row["stability"] == "stable"]
    if len(stable) != 71:
        raise RuntimeError(f"Expected 71 frozen stable exports, found {len(stable)}.")
    if {row["r_reference_version"] for row in stable} != {"0.3.0"}:
        raise RuntimeError("Stable API inventory is not uniformly pinned to R 0.3.0.")

    with cases_path.open(newline="", encoding="utf-8") as handle:
        case_rows = list(csv.DictReader(handle))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    result_by_case = {
        (row["function"], row["case_id"]): row["status"]
        for row in report["cases"]
    }
    cases_by_function: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        cases_by_function[row["function"]].append(row)

    output_rows: list[dict[str, str]] = []
    for api in stable:
        name = api["name"]
        registered = cases_by_function.get(name, [])
        if not registered:
            status = "PENDING"
            evidence = ""
            comparison = ""
        else:
            statuses = [result_by_case.get((name, row["case_id"]), "PENDING") for row in registered]
            if any(value == "FAIL" for value in statuses):
                status = "FAIL"
            elif statuses and all(value == "PASS" for value in statuses):
                status = "PASS"
            else:
                status = "PENDING"
            evidence = ";".join(row["case_id"] for row in registered)
            comparison = ";".join(sorted({row["comparison"] for row in registered}))
        output_rows.append(
            {
                "function": name,
                "stability": api["stability"],
                "r_reference_version": api["r_reference_version"],
                "status": status,
                "comparison": comparison,
                "evidence": evidence,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "function",
                "stability",
                "r_reference_version",
                "status",
                "comparison",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    counts = defaultdict(int)
    for row in output_rows:
        counts[row["status"]] += 1
    print(
        "stable API parity matrix: "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
