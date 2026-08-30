from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import external_validation as ev
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import GP3MLObject, GP3MLResampleEvaluation


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]


def _fixture():
    data = gp.simulate_gazepoint_governed_data(
        n_participants=10,
        n_stimuli=4,
        trials_per_cell=1,
        seed=91,
    )
    task = gp.create_gazepoint_synthetic_task(
        data,
        workflow="assigned_condition",
        generalization_target="new_participants",
    )
    model = gp.fit_gazepoint_model(data, task, PREDICTORS, engine="glm", seed=91)
    return data, task, model


def _independent_copy(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["participant_id"] = "EXT_" + out["participant_id"].astype(str)
    out["stimulus_id"] = "EXT_" + out["stimulus_id"].astype(str)
    out["trial_id"] = "EXT_" + out["trial_id"].astype(str)
    return out


def test_external_dataset_declaration_helpers_and_schema_classes(tmp_path: Path):
    data, _, _ = _fixture()
    declaration = ev.declare_gazepoint_external_dataset(
        data,
        label=" external cohort ",
        independent=True,
        origin=" new site ",
        collection_period="2026",
        notes="single note",
    )
    assert declaration.label == "external cohort"
    assert declaration.origin == "new site"
    assert declaration.notes == ["single note"]
    assert declaration.n_rows == len(data)
    assert declaration.n_participants == data.participant_id.nunique()
    assert declaration.n_stimuli == data.stimulus_id.nunique()
    assert len(declaration.data_hash) == 32

    no_ids = ev.declare_gazepoint_external_dataset(
        data,
        "no ids",
        np.bool_(True),
        "site",
        participant_id=None,
        stimulus_id=None,
        notes=None,
    )
    assert np.isnan(no_ids.n_participants) and np.isnan(no_ids.n_stimuli)
    assert no_ids.notes == []
    many_notes = ev.declare_gazepoint_external_dataset(
        data, "notes", False, "site", notes=["a", 2]
    )
    assert many_notes.notes == ["a", "2"]

    invalid = [
        ({"label": ""}, "label"),
        ({"independent": 1}, "independent"),
        ({"origin": ""}, "origin"),
        ({"participant_id": "missing"}, "identifier"),
        ({"stimulus_id": "missing"}, "identifier"),
    ]
    base = dict(data=data, label="x", independent=True, origin="site")
    for changes, match in invalid:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            ev.declare_gazepoint_external_dataset(**args)

    classes = pd.DataFrame(
        {
            "factor": pd.Categorical(["a", "b"]),
            "logical": [True, False],
            "integer": pd.Series([1, 2], dtype="int64"),
            "numeric": [1.0, 2.0],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "character": pd.Series(["a", "b"], dtype=object),
        }
    )
    assert ev._column_class(classes.factor) == "factor"
    assert ev._column_class(classes.logical) == "logical"
    assert ev._column_class(classes.integer) == "integer"
    assert ev._column_class(classes.numeric) == "numeric"
    assert ev._column_class(classes.date) == "POSIXct/POSIXt"
    assert ev._column_class(classes.character) == "character"

    external = classes.drop(columns=["integer"]).copy()
    external["extra"] = [1, 2]
    schema = ev._schema_comparison(classes, external, ["integer", "numeric"])
    integer = schema.loc[schema.variable == "integer"].iloc[0]
    extra = schema.loc[schema.variable == "extra"].iloc[0]
    assert integer.predictor and not integer.external_present and not integer.class_match
    assert not extra.development_present and extra.external_present

    target = tmp_path / "nested" / "report.md"
    assert ev._safe_path(target, overwrite=False) == target
    target.write_text("x", encoding="utf-8")
    with pytest.raises(GP3MLError, match="File exists"):
        ev._safe_path(target, overwrite=False)
    assert ev._safe_path(target, overwrite=True) == target
    assert ev._data_hash(data) == ev._data_hash(data.copy())


def test_group_transportability_prevalence_metric_summary_and_long_helpers():
    development = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "stimulus_id": ["s1", "s2"],
            "outcome": ["control", "control"],
        }
    )
    external = pd.DataFrame(
        {
            "participant_id": ["p2", "p3"],
            "stimulus_id": ["s3", "s4"],
            "outcome": ["target", "target"],
        }
    )
    unavailable = ev._group_transportability(development, external, None, "participant")
    assert unavailable.loc[0, "status"] == "not_available"
    missing = ev._group_transportability(development, external, "absent", "participant")
    assert missing.loc[0, "status"] == "not_available"
    overlap = ev._group_transportability(
        development, external, "participant_id", "participant"
    )
    assert overlap.loc[0, "status"] == "review"
    assert overlap.loc[0, "overlapping_groups"] == 1
    novel = ev._group_transportability(development, external, "stimulus_id", "stimulus")
    assert novel.loc[0, "status"] == "pass"
    assert novel.loc[0, "external_coverage_prop"] == 1.0
    empty_external = external.iloc[:0]
    empty_groups = ev._group_transportability(
        development, empty_external, "stimulus_id", "stimulus"
    )
    assert np.isnan(empty_groups.loc[0, "external_coverage_prop"])

    task = GP3MLObject(
        task_type="classification", outcome="outcome", positive="target"
    )
    shift = ev._prevalence_shift(task, development, external)
    assert shift.loc[0, "development_prevalence"] == 0
    assert np.isnan(shift.loc[0, "relative_shift"])
    regression_task = GP3MLObject(task_type="regression", outcome="outcome", positive=None)
    assert ev._prevalence_shift(regression_task, development, external).empty

    empty_summary = ev._development_metric_summary(None)
    assert list(empty_summary.columns) == ["metric", "development_estimate"]
    resample = GP3MLResampleEvaluation(
        metrics=pd.DataFrame(
            {"metric": ["roc_auc", "roc_auc", "brier"], "value": [0.7, 0.9, 0.2]}
        )
    )
    summary = ev._development_metric_summary(resample)
    assert summary.loc[summary.metric == "roc_auc", "development_estimate"].iloc[0] == 0.8
    resample_empty = GP3MLResampleEvaluation(metrics=pd.DataFrame())
    assert ev._development_metric_summary(resample_empty).empty
    frame_summary = ev._development_metric_summary(
        pd.DataFrame({"roc_auc": [0.7, 0.9], "label": ["a", "b"]})
    )
    assert frame_summary.metric.tolist() == ["roc_auc"]
    with pytest.raises(GP3MLError, match="development_evaluation"):
        ev._development_metric_summary(object())

    assert ev._metric_long(pd.DataFrame()).empty
    one = ev._metric_long(pd.DataFrame({"n": [2], "roc_auc": [0.8], "threshold": [0.5]}))
    assert one.to_dict("records") == [{"metric": "roc_auc", "value": 0.8}]
    already_long = pd.DataFrame(
        {"metric": ["a", "b"], "value": [1.0, 2.0], "other": [3, 4]}
    )
    assert list(ev._metric_long(already_long).columns) == ["metric", "value"]
    assert ev._metric_long(pd.DataFrame({"a": [1, 2]})).empty


def test_transportability_no_external_declaration_mismatch_nonindependent_and_schema_failures():
    development, _, model = _fixture()
    no_external = ev.evaluate_gazepoint_external_transportability(model, development)
    assert no_external.status == "not_externally_validated"
    assert no_external.validation_summary.status == "review"
    assert no_external.declaration is None

    external = _independent_copy(development)
    with pytest.raises(GP3MLError, match="fitted"):
        ev.evaluate_gazepoint_external_transportability(object(), development, external)
    with pytest.raises(GP3MLError, match="declaration"):
        ev.evaluate_gazepoint_external_transportability(model, development, external)

    declaration = ev.declare_gazepoint_external_dataset(
        external, "external", True, "new site"
    )
    changed = external.copy()
    changed.loc[0, PREDICTORS[0]] += 99
    mismatch = ev.evaluate_gazepoint_external_transportability(
        model, development, changed, declaration=declaration, bootstrap=1
    )
    assert mismatch.status == "external_declaration_mismatch"
    assert mismatch.validation_summary.status == "fail"

    nonindependent_decl = ev.declare_gazepoint_external_dataset(
        external, "external", False, "new site"
    )
    nonindependent = ev.evaluate_gazepoint_external_transportability(
        model, development, external, declaration=nonindependent_decl, bootstrap=1
    )
    assert nonindependent.status == "not_externally_validated"
    assert nonindependent.validation_summary.status == "fail"

    missing_outcome = external.drop(columns=[model.task.outcome])
    missing_outcome_decl = ev.declare_gazepoint_external_dataset(
        missing_outcome, "missing outcome", True, "new site"
    )
    report = ev.evaluate_gazepoint_external_transportability(
        model, development, missing_outcome, declaration=missing_outcome_decl, bootstrap=1
    )
    assert report.status == "incompatible_external_schema"
    assert "Missing outcome" in report.reason

    wrong_outcome = external.copy()
    wrong_outcome[model.task.outcome] = np.arange(len(wrong_outcome), dtype=float)
    wrong_outcome_decl = ev.declare_gazepoint_external_dataset(
        wrong_outcome, "wrong outcome", True, "new site"
    )
    report = ev.evaluate_gazepoint_external_transportability(
        model, development, wrong_outcome, declaration=wrong_outcome_decl, bootstrap=1
    )
    assert "Outcome type mismatch" in report.reason

    missing_predictor = external.drop(columns=[PREDICTORS[0]])
    missing_predictor_decl = ev.declare_gazepoint_external_dataset(
        missing_predictor, "missing predictor", True, "new site"
    )
    report = ev.evaluate_gazepoint_external_transportability(
        model, development, missing_predictor, declaration=missing_predictor_decl, bootstrap=1
    )
    assert "Missing predictors" in report.reason

    wrong_predictor = external.copy()
    wrong_predictor[PREDICTORS[0]] = wrong_predictor[PREDICTORS[0]].map(str)
    wrong_predictor_decl = ev.declare_gazepoint_external_dataset(
        wrong_predictor, "wrong predictor", True, "new site"
    )
    report = ev.evaluate_gazepoint_external_transportability(
        model, development, wrong_predictor, declaration=wrong_predictor_decl, bootstrap=1
    )
    assert "Predictor type mismatches" in report.reason


def test_transportability_overlap_and_success_with_dataframe_development_metrics():
    development, _, model = _fixture()
    overlapping = development.copy()
    overlap_decl = ev.declare_gazepoint_external_dataset(
        overlapping, "overlap", True, "same identifiers"
    )
    overlap = ev.evaluate_gazepoint_external_transportability(
        model,
        development,
        overlapping,
        declaration=overlap_decl,
        development_evaluation=pd.DataFrame({"roc_auc": [0.7, 0.8], "brier": [0.2, 0.15]}),
        bootstrap=2,
        seed=3,
    )
    assert overlap.status == "external_independence_requires_review"
    assert overlap.validation is not None
    assert len(overlap.performance_comparison) > 0
    assert overlap.validation_summary.status == "review"

    external = _independent_copy(development)
    declaration = ev.declare_gazepoint_external_dataset(
        external, "independent", True, "new site"
    )
    success = ev.evaluate_gazepoint_external_transportability(
        model,
        development,
        external,
        declaration=declaration,
        development_evaluation=pd.DataFrame({"roc_auc": [0.75], "brier": [0.2]}),
        threshold=None,
        bootstrap=2,
        seed=4,
    )
    assert success.status == "externally_validated"
    assert success.declaration_hash_matches is True
    assert success.validation_summary.status == "pass"
    assert len(success.metrics) > 0
    assert len(success.performance_comparison) > 0
    assert "difference" in success.performance_comparison
    assert len(success.calibration_drift) == 1
    assert len(success.prevalence_shift) == 1
    assert len(success.predictor_shift) > 0


def test_transportability_with_grouped_development_calibration_drift():
    development, task, model = _fixture()
    manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    folds = gp.create_gazepoint_group_folds(
        development,
        task.outcome,
        PREDICTORS,
        manifest,
        "new_participants",
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        v=2,
        repeats=1,
        seed=7,
    )
    evaluation = gp.evaluate_gazepoint_group_folds(
        folds,
        task,
        PREDICTORS,
        engine="glm",
        assess_calibration=True,
        calibration_bins=3,
        calibration_bootstrap=0,
        keep_models=False,
        seed=8,
    )
    external = _independent_copy(development)
    declaration = ev.declare_gazepoint_external_dataset(
        external, "independent", True, "new site"
    )
    report = ev.evaluate_gazepoint_external_transportability(
        model,
        development,
        external,
        declaration=declaration,
        development_evaluation=evaluation,
        bootstrap=1,
        seed=9,
    )
    assert report.status == "externally_validated"
    assert any(name.startswith("development_") for name in report.calibration_drift.columns)
    assert any(name.startswith("drift_") for name in report.calibration_drift.columns)

    without_calibration = evaluation.__deepcopy__({})
    for result in without_calibration.fold_results:
        result["calibration"] = None
    report_no_dev_cal = ev.evaluate_gazepoint_external_transportability(
        model,
        development,
        external,
        declaration=declaration,
        development_evaluation=without_calibration,
        bootstrap=1,
    )
    assert report_no_dev_cal.status == "externally_validated"


def test_regression_transportability_validation_and_writer(tmp_path: Path):
    development = gp.simulate_gazepoint_governed_data(
        n_participants=8, n_stimuli=4, trials_per_cell=1, seed=101
    )
    task = gp.create_gazepoint_synthetic_task(
        development,
        workflow="observed_duration",
        generalization_target="new_participants",
    )
    model = gp.fit_gazepoint_model(development, task, PREDICTORS, engine="lm")
    external = _independent_copy(development)
    declaration = ev.declare_gazepoint_external_dataset(
        external, "regression external", True, "new site"
    )
    report = ev.evaluate_gazepoint_external_transportability(
        model,
        development,
        external,
        declaration=declaration,
        bootstrap=1,
    )
    assert report.status == "externally_validated"
    assert report.prevalence_shift.empty
    assert report.calibration_drift.empty

    validation = ev.validate_gazepoint_transportability(report)
    assert validation.status == "pass"
    path = Path(ev.write_gazepoint_transportability_report(report, tmp_path / "report.md"))
    text = path.read_text(encoding="utf-8")
    assert "externally_validated" in text
    assert "Regression" not in text or isinstance(text, str)
    with pytest.raises(GP3MLError, match="transportability_report"):
        ev.validate_gazepoint_transportability(object())
    with pytest.raises(GP3MLError, match="transportability_report"):
        ev.write_gazepoint_transportability_report(object(), tmp_path / "bad.md")
    with pytest.raises(GP3MLError, match="File exists"):
        ev.write_gazepoint_transportability_report(report, path)
    assert ev.write_gazepoint_transportability_report(report, path, overwrite=True) == str(path)

    no_external = ev.evaluate_gazepoint_external_transportability(model, development)
    no_path = Path(
        ev.write_gazepoint_transportability_report(no_external, tmp_path / "none.md")
    )
    assert "No independent external dataset was declared" in no_path.read_text(encoding="utf-8")


def test_transportability_validator_review_and_explicit_status_failure_branches():
    development, _, model = _fixture()
    no_external = ev.evaluate_gazepoint_external_transportability(model, development)
    assert ev.validate_gazepoint_transportability(no_external).status == "review"

    external = _independent_copy(development)
    declaration = ev.declare_gazepoint_external_dataset(external, "x", True, "site")
    report = ev.evaluate_gazepoint_external_transportability(
        model, development, external, declaration=declaration, bootstrap=1
    )
    malformed = report.__deepcopy__({})
    malformed.status = ""
    assert ev.validate_gazepoint_transportability(malformed).status == "fail"
    missing_schema = report.__deepcopy__({})
    missing_schema.schema = pd.DataFrame()
    missing_schema.group_coverage = pd.DataFrame()
    missing_schema.declaration = None
    assert ev.validate_gazepoint_transportability(missing_schema).status == "review"
