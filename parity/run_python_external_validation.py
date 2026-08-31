from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

import gp3mlpy as gp

PREDICTORS = ["fixation_duration", "pupil_change"]
QUALITY = [
    "pass","review","pass","review","review","pass","review","pass",
    "pass","review","review","pass","review","pass","review","pass",
    "pass","review","pass","review","review","pass","pass","review",
]


def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index, pd.Categorical)):
        return [_jsonable(v) for v in list(x)]
    return x if isinstance(x, str) else str(x)


def _capture(fn):
    try:
        return {"status": "success", "value": _jsonable(fn())}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}


def _training():
    n = np.arange(1, 25, dtype=float)
    return pd.DataFrame({
        "participant_id": [f"P{i:02d}" for i in range(1,13) for _ in range(2)],
        "trial_id": [f"T{i:02d}" for i in range(1,25)],
        "stimulus_id": ["S01","S02"] * 12,
        "fixation_duration": 180.0 + n,
        "pupil_change": np.sin(n / 3.0),
        "quality_status": pd.Categorical(QUALITY, categories=["pass","review"]),
    })


def _external(training, overlap=False):
    out = training.copy(deep=True)
    out["participant_id"] = [f"E{i:02d}" for i in range(1,13) for _ in range(2)]
    out["trial_id"] = [f"ET{i:02d}" for i in range(1,25)]
    out["stimulus_id"] = (["S01","S02"] if overlap else ["ES01","ES02"]) * 12
    out["fixation_duration"] = pd.to_numeric(out["fixation_duration"]) + 4.0
    out["pupil_change"] = np.cos(np.arange(1,25,dtype=float) / 4.0)
    out["quality_status"] = pd.Categorical(out["quality_status"].astype(object), categories=["pass","review"])
    return out


def _model(data, seed):
    task = gp.declare_gazepoint_task(
        data=data, outcome="quality_status",
        purpose="Predict predefined recording-quality review status",
        task_type="classification", unit_id="trial_id",
        participant_id="participant_id", stimulus_id="stimulus_id",
        generalization_target="new_participants", positive="review",
    )
    return gp.train_gazepoint_classifier(data, task, PREDICTORS, engine="glm", seed=seed)


def _metrics(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    out = {}
    row = frame.iloc[0]
    for name in frame.columns:
        value = row[name]
        if isinstance(value, (np.integer, int)):
            out[str(name)] = int(value)
        elif isinstance(value, (np.floating, float)) and math.isfinite(float(value)):
            out[str(name)] = float(value)
    return dict(sorted(out.items()))


def _decl(x):
    return {
        "class": x.r_class, "label": str(x.label), "independent": bool(x.independent),
        "origin": str(x.origin), "collection_period": _jsonable(x.collection_period), "participant_id": _jsonable(x.participant_id),
        "stimulus_id": _jsonable(x.stimulus_id), "n_rows": int(x.n_rows),
        "n_participants": None if pd.isna(x.n_participants) else int(x.n_participants),
        "n_stimuli": None if pd.isna(x.n_stimuli) else int(x.n_stimuli),
        "notes": [str(v) for v in x.notes], "hash_recorded": bool(str(x.data_hash)),
    }


def _direct(x):
    cal = x.calibration
    smd = {
        str(row.feature): None if pd.isna(row.standardized_mean_difference) else float(row.standardized_mean_difference)
        for row in x.shift.itertuples(index=False)
    }
    return {
        "class": x.r_class, "label": str(x.label), "model_engine": str(x.model_engine),
        "n_predictions": len(x.predictions),
        "prediction_columns": sorted(map(str, x.predictions.columns)),
        "metric_columns": sorted(map(str, x.metrics.columns)), "metric_values": _metrics(x.metrics),
        "calibration_present": cal is not None,
        "calibration_rows": 0 if cal is None else len(cal.summary),
        "calibration_columns": [] if cal is None else sorted(map(str, cal.summary.columns)),
        "shift_rows": len(x.shift),
        "shift_features": sorted(map(str, x.shift["feature"])), "shift_smd": dict(sorted(smd.items())),
        "hash_recorded": bool(str(x.external_hash)),
    }


def _report(x):
    dev = x.development_metrics
    return {
        "class": x.r_class, "validation_label": str(x.validation.label),
        "development_rows": 0 if dev is None else len(dev),
        "development_columns": [] if dev is None else sorted(map(str, dev.columns)),
        "limitations": [str(v) for v in x.limitations],
        "prohibited_count": len(x.prohibited_uses),
    }


def _vsum(x):
    return {
        "class": x.r_class, "status": str(x.status), "n_checks": len(x.checks),
        "check_ids": [str(v) for v in x.checks["check_id"]],
        "check_statuses": [str(v) for v in x.checks["status"]],
        "issues": len(x.issues),
    }


def _transport(x):
    schema, groups = x.schema, x.group_coverage
    statuses, overlaps = {}, {}
    if isinstance(groups, pd.DataFrame) and len(groups):
        for row in groups.itertuples(index=False):
            statuses[str(row.unit)] = str(row.status)
            overlaps[str(row.unit)] = None if pd.isna(row.overlapping_groups) else int(row.overlapping_groups)
    hash_match = None
    if x.declaration is not None and not pd.isna(x.declaration_hash_matches):
        hash_match = bool(x.declaration_hash_matches)
    return {
        "class": x.r_class, "status": str(x.status), "reason": str(x.reason),
        "declaration_attached": x.declaration is not None,
        "declaration_independent": None if x.declaration is None else bool(x.declaration.independent),
        "declaration_hash_matches": hash_match,
        "schema_rows": len(schema),
        "schema_predictor_rows": int(schema["predictor"].sum()) if len(schema) else 0,
        "missing_predictors": int((schema["predictor"] & ~schema["external_present"]).sum()) if len(schema) else 0,
        "type_mismatches": int((schema["predictor"] & schema["external_present"] & ~schema["class_match"]).sum()) if len(schema) else 0,
        "group_statuses": dict(sorted(statuses.items())), "group_overlaps": dict(sorted(overlaps.items())),
        "metrics_rows": len(x.metrics), "performance_rows": len(x.performance_comparison),
        "prevalence_rows": len(x.prevalence_shift), "predictor_shift_rows": len(x.predictor_shift),
        "validation_attached": x.validation is not None, "validation_summary": _vsum(x.validation_summary),
        "limitations_count": len(x.limitations),
    }


def _written(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "basename": path.name, "exists": path.is_file(), "line_count": len(lines),
        "headings": [v for v in lines if v.startswith("## ")],
        "header": lines[0] if lines else "",
    }


def _write_direct(report):
    with TemporaryDirectory() as d:
        path = Path(d) / "external_validation.md"
        returned = gp.write_external_validation_report(report, path)
        out = _written(path); out["return_basename"] = Path(returned).name
        return out


def _write_transport(report, basename):
    with TemporaryDirectory() as d:
        path = Path(d) / basename
        returned = gp.write_gazepoint_transportability_report(report, path)
        out = _written(path); out["return_basename"] = Path(returned).name
        return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_external_validation.py <fixture.json> <output.json>")
    fixture_path, output_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    seed, bootstrap = int(fixture["seed"]), int(fixture["bootstrap"])

    training = _training()
    model = _model(training, seed)
    external, overlap = _external(training), _external(training, overlap=True)
    dev = pd.DataFrame({"accuracy":[0.75], "brier":[0.20]})

    declaration = gp.declare_gazepoint_external_dataset(
        external, "independent_site", True, "Independent deterministic synthetic site",
        notes=["Synthetic parity fixture."],
    )
    nonind = gp.declare_gazepoint_external_dataset(
        external, "non_independent_site", False, "Development-linked synthetic site"
    )
    overlap_decl = gp.declare_gazepoint_external_dataset(
        overlap, "overlap_site", True, "Synthetic site with overlapping stimulus identifiers"
    )
    incompatible = external.drop(columns=["pupil_change"]).copy()
    incompatible_decl = gp.declare_gazepoint_external_dataset(
        incompatible, "schema_mismatch_site", True, "Synthetic schema mismatch site"
    )

    direct = gp.evaluate_external_validation(model, external, label="independent_site", bootstrap=bootstrap, seed=seed)
    report = gp.create_external_validation_report(
        direct, development_metrics=dev, limitations="Synthetic external-validation parity fixture."
    )

    noext = gp.evaluate_gazepoint_external_transportability(
        model, training, development_evaluation=dev, bootstrap=bootstrap, seed=seed
    )
    independent = gp.evaluate_gazepoint_external_transportability(
        model, training, external, declaration, dev, bootstrap=bootstrap, seed=seed
    )
    overlap_report = gp.evaluate_gazepoint_external_transportability(
        model, training, overlap, overlap_decl, dev, bootstrap=bootstrap, seed=seed
    )
    nonind_report = gp.evaluate_gazepoint_external_transportability(
        model, training, external, nonind, dev, bootstrap=bootstrap, seed=seed
    )
    mismatch_data = external.copy(deep=True)
    mismatch_data.loc[mismatch_data.index[0], "fixation_duration"] += 1.0
    mismatch = gp.evaluate_gazepoint_external_transportability(
        model, training, mismatch_data, declaration, dev, bootstrap=bootstrap, seed=seed
    )
    incompatible_report = gp.evaluate_gazepoint_external_transportability(
        model, training, incompatible, incompatible_decl, dev, bootstrap=bootstrap, seed=seed
    )

    cases = {}
    cases["declare_gazepoint_external_dataset::successful_independent"] = _capture(lambda: _decl(declaration))
    cases["declare_gazepoint_external_dataset::non_independent"] = _capture(lambda: _decl(nonind))
    cases["declare_gazepoint_external_dataset::invalid_label"] = _capture(
        lambda: gp.declare_gazepoint_external_dataset(external, "", True, "Synthetic site")
    )
    cases["declare_gazepoint_external_dataset::invalid_independent"] = _capture(
        lambda: gp.declare_gazepoint_external_dataset(external, "bad", "yes", "Synthetic site")
    )
    cases["declare_gazepoint_external_dataset::missing_identifier"] = _capture(
        lambda: gp.declare_gazepoint_external_dataset(external, "bad", True, "Synthetic site", participant_id="missing_id")
    )
    cases["evaluate_external_validation::classification_success"] = _capture(lambda: _direct(direct))
    cases["create_external_validation_report::successful_report"] = _capture(lambda: _report(report))
    cases["create_external_validation_report::invalid_validation"] = _capture(
        lambda: gp.create_external_validation_report("not validation")
    )
    cases["write_external_validation_report::successful_write"] = _capture(lambda: _write_direct(report))

    def overwrite_direct():
        with TemporaryDirectory() as d:
            p = Path(d) / "external_validation.md"
            gp.write_external_validation_report(report, p)
            return gp.write_external_validation_report(report, p)
    cases["write_external_validation_report::overwrite_rejected"] = _capture(overwrite_direct)

    cases["evaluate_gazepoint_external_transportability::no_external"] = _capture(lambda: _transport(noext))
    cases["evaluate_gazepoint_external_transportability::independent_external"] = _capture(lambda: _transport(independent))
    cases["evaluate_gazepoint_external_transportability::overlap_requires_review"] = _capture(lambda: _transport(overlap_report))
    cases["evaluate_gazepoint_external_transportability::non_independent_external"] = _capture(lambda: _transport(nonind_report))
    cases["evaluate_gazepoint_external_transportability::declaration_mismatch"] = _capture(lambda: _transport(mismatch))
    cases["evaluate_gazepoint_external_transportability::incompatible_schema"] = _capture(lambda: _transport(incompatible_report))
    cases["evaluate_gazepoint_external_transportability::missing_declaration"] = _capture(
        lambda: gp.evaluate_gazepoint_external_transportability(model, training, external, None, dev, bootstrap=bootstrap, seed=seed)
    )
    cases["evaluate_gazepoint_external_transportability::invalid_model"] = _capture(
        lambda: gp.evaluate_gazepoint_external_transportability("not model", training)
    )
    cases["validate_gazepoint_transportability::externally_validated"] = _capture(
        lambda: _vsum(gp.validate_gazepoint_transportability(independent))
    )
    cases["validate_gazepoint_transportability::not_externally_validated"] = _capture(
        lambda: _vsum(gp.validate_gazepoint_transportability(noext))
    )
    cases["validate_gazepoint_transportability::invalid_object"] = _capture(
        lambda: gp.validate_gazepoint_transportability("not report")
    )
    cases["write_gazepoint_transportability_report::successful_write"] = _capture(
        lambda: _write_transport(independent, "transportability.md")
    )
    cases["write_gazepoint_transportability_report::no_external_write"] = _capture(
        lambda: _write_transport(noext, "transportability_no_external.md")
    )

    def overwrite_transport():
        with TemporaryDirectory() as d:
            p = Path(d) / "transportability.md"
            gp.write_gazepoint_transportability_report(independent, p)
            return gp.write_gazepoint_transportability_report(independent, p)
    cases["write_gazepoint_transportability_report::overwrite_rejected"] = _capture(overwrite_transport)
    cases["write_gazepoint_transportability_report::invalid_object"] = _capture(
        lambda: gp.write_gazepoint_transportability_report("not report", "transportability.md")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "runtime":"python", "r_reference_version":fixture["r_reference"]["version"], "cases":cases
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
