from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gp3mlpy import conformal as cf
from gp3mlpy import dataset_shift as ds
from gp3mlpy import decision_governance as dg
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import GP3MLConformal, GP3MLDecisionRule, GP3MLThresholdEvaluation


def test_probability_and_decision_rule_validation_paths():
    assert np.allclose(dg._assert_probability([0.0, np.nan, 1.0])[[0, 2]], [0.0, 1.0])
    with pytest.raises(GP3MLError, match="probabilities"):
        dg._assert_probability(["bad"])
    with pytest.raises(GP3MLError, match="probabilities"):
        dg._assert_probability([np.inf])
    with pytest.raises(GP3MLError, match="probabilities"):
        dg._assert_probability([-0.1])

    rule = dg.create_gazepoint_decision_rule(
        metric="balanced_accuracy",
        threshold=0.5,
        generalization_target="new_participants",
        scientific_justification="prespecified scientific threshold",
    )
    assert dg.validate_gazepoint_decision_rule(rule, require_threshold=True).status == "pass"
    assert "0.5" in repr(rule)

    unselected = dg.create_gazepoint_decision_rule(
        metric="balanced_accuracy",
        threshold=None,
        generalization_target="new_participants",
        scientific_justification="threshold will be estimated inside training data",
    )
    assert "<not selected>" in repr(unselected)
    assert dg.validate_gazepoint_decision_rule(unselected).status == "pass"
    assert dg.validate_gazepoint_decision_rule(unselected, require_threshold=True).status == "fail"

    invalid_creations = [
        ({"direction": "sideways"}, "direction"),
        ({"threshold_origin": "assessment"}, "threshold_origin"),
        ({"metric": ""}, "metric"),
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": 1.0}, "threshold"),
        ({"threshold": "x"}, "threshold"),
        ({"cost_false_positive": -1}, "cost"),
        ({"cost_false_negative": np.inf}, "cost"),
        ({"scientific_justification": ""}, "scientific_justification"),
        ({"abstention_allowed": True}, "Abstention"),
        ({"abstention_allowed": True, "abstention_interval": [0.5]}, "Abstention"),
        ({"abstention_allowed": True, "abstention_interval": [0.7, 0.6]}, "Abstention"),
        ({"abstention_allowed": True, "abstention_interval": [-0.1, 0.6]}, "Abstention"),
    ]
    base = dict(
        metric="balanced_accuracy",
        threshold=0.5,
        generalization_target="new_participants",
        scientific_justification="declared",
    )
    for changes, match in invalid_creations:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            dg.create_gazepoint_decision_rule(**args)

    bad = GP3MLDecisionRule(
        metric="",
        direction="bad",
        threshold=2.0,
        threshold_origin="bad",
        training_partition="assessment",
        generalization_target="",
        scientific_justification="",
        abstention_allowed=True,
        abstention_interval=[0.8, 0.2],
    )
    validation = dg.validate_gazepoint_decision_rule(bad, require_threshold=True)
    assert validation.status == "fail"
    assert (validation.checks.status == "fail").sum() >= 7
    external = bad.__deepcopy__({})
    external.training_partition = "external_validation"
    assert dg.validate_gazepoint_decision_rule(external).status == "fail"
    assert dg.validate_gazepoint_decision_rule(object()).status == "fail"


def test_threshold_evaluation_selection_application_and_abstention():
    truth = ["control", "control", "target", "target"]
    probability = [0.1, 0.6, 0.4, 0.9]
    evaluation = dg.evaluate_gazepoint_thresholds(
        truth,
        probability,
        positive="target",
        thresholds=[0.7, 0.3, 0.5, 0.5],
        cost_false_positive=2,
        cost_false_negative=3,
    )
    assert evaluation.thresholds.threshold.tolist() == [0.3, 0.5, 0.7]
    assert set(["precision", "f1", "balanced_accuracy", "expected_cost"]).issubset(
        evaluation.thresholds.columns
    )

    selected = dg.select_gazepoint_threshold(
        evaluation,
        metric="balanced_accuracy",
        direction="maximize",
        threshold_origin="training",
        training_partition="analysis",
        generalization_target="new_participants",
        scientific_justification="selected only on analysis data",
    )
    assert selected.threshold in evaluation.thresholds.threshold.tolist()
    minimized = dg.select_gazepoint_threshold(
        evaluation,
        metric="expected_cost",
        direction="minimize",
        generalization_target="new_participants",
        scientific_justification="minimum declared cost",
    )
    assert minimized.direction == "minimize"

    abstain_rule = dg.create_gazepoint_decision_rule(
        metric="balanced_accuracy",
        threshold=0.5,
        abstention_allowed=True,
        abstention_interval=[0.45, 0.55],
        generalization_target="new_participants",
        scientific_justification="abstain near the boundary",
    )
    decision = dg.apply_gazepoint_decision_rule(
        abstain_rule, [0.2, 0.5, 0.8], positive="target", negative="control"
    )
    assert list(decision.astype(str)) == ["control", ".abstain", "target"]
    audit = dg.audit_gazepoint_abstention(
        ["control", "target", "target"], decision
    )
    assert audit.status == "pass"
    assert audit.coverage == pytest.approx(2 / 3)
    all_abstain = dg.audit_gazepoint_abstention(["a", "b"], [".abstain", ".abstain"])
    assert all_abstain.status == "fail"
    assert np.isnan(all_abstain.covered_error_rate)
    empty = dg.audit_gazepoint_abstention([], [])
    assert np.isnan(empty.coverage)
    with pytest.raises(GP3MLError, match="lengths differ"):
        dg.audit_gazepoint_abstention(["a"], [])
    with pytest.raises(GP3MLError, match="validation failed"):
        dg.apply_gazepoint_decision_rule(
            GP3MLDecisionRule(metric="m"), [0.5], positive="yes", negative="no"
        )

    invalid_evaluations = [
        ({"thresholds": None}, "explicit candidate"),
        ({"thresholds": []}, "explicit candidate"),
        ({"thresholds": [0.0]}, "strictly between"),
        ({"thresholds": [np.nan]}, "strictly between"),
        ({"probability": [0.1]}, "lengths differ"),
        ({"truth": ["control"] * 4}, "exactly two"),
        ({"positive": "missing"}, "exactly two"),
    ]
    base = dict(
        truth=truth,
        probability=probability,
        positive="target",
        thresholds=[0.5],
    )
    for changes, match in invalid_evaluations:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            dg.evaluate_gazepoint_thresholds(**args)

    with pytest.raises(GP3MLError, match="evaluate_gazepoint_thresholds"):
        dg.select_gazepoint_threshold(object(), "accuracy")
    with pytest.raises(GP3MLError, match="direction"):
        dg.select_gazepoint_threshold(evaluation, "accuracy", direction="sideways")
    with pytest.raises(GP3MLError, match="threshold_origin"):
        dg.select_gazepoint_threshold(
            evaluation, "accuracy", threshold_origin="predeclared"
        )
    with pytest.raises(GP3MLError, match="Unknown threshold metric"):
        dg.select_gazepoint_threshold(evaluation, "missing")
    no_finite = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.2, 0.4], "metric": [np.nan, np.nan]}),
        cost_false_positive=1,
        cost_false_negative=1,
    )
    with pytest.raises(GP3MLError, match="no finite"):
        dg.select_gazepoint_threshold(no_finite, "metric")

    tie = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.2, 0.4], "score": [1.0, 1.0]}),
        cost_false_positive=1,
        cost_false_negative=1,
    )
    tie_rule = dg.select_gazepoint_threshold(
        tie,
        "score",
        generalization_target="new_participants",
        scientific_justification="tie uses smallest threshold",
    )
    assert tie_rule.threshold == 0.2


def test_conformal_regression_classification_grouping_and_validation():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 2.7, 4.2])
    reg = cf.fit_gazepoint_conformal(
        truth,
        prediction=pred,
        task_type="regression",
        level=0.75,
        generalization_target="new_trials_known_participants",
    )
    assert cf.validate_gazepoint_conformal(reg).status == "pass"
    interval = cf.predict_gazepoint_interval(reg, [1.5, 2.5])
    assert list(interval.columns) == ["prediction", "lower", "upper"]

    grouped = cf.fit_gazepoint_conformal(
        truth,
        prediction=pred,
        task_type="regression",
        calibration_unit="participant",
        unit=["p1", "p1", "p2", "p2"],
        level=0.75,
        generalization_target="new_participants",
    )
    assert grouped.n_calibration_units == 2

    ctruth = ["control", "target", "control", "target"]
    prob = [0.1, 0.8, 0.4, 0.6]
    cls = cf.fit_gazepoint_conformal(
        ctruth,
        probability=prob,
        task_type="classification",
        positive="target",
        level=0.5,
        generalization_target="new_participants",
    )
    assert cf.validate_gazepoint_conformal(cls).status == "pass"
    sets = cf.predict_gazepoint_set(cls, [0.05, 0.5, 0.95])
    assert len(sets) == 3

    both = GP3MLConformal(
        task_type="classification",
        positive="target",
        negative="control",
        level=0.5,
        calibration_unit="observation",
        generalization_target="new_participants",
        n_calibration_units=3,
        conformity_quantile=0.6,
    )
    labels = cf.predict_gazepoint_set(both, [0.1, 0.5, 0.9]).set.tolist()
    assert labels[0] == "{control}"
    assert labels[1] == "{control, target}"
    assert labels[2] == "{target}"
    none = both.__deepcopy__({})
    none.conformity_quantile = 0.4
    assert cf.predict_gazepoint_set(none, [0.5]).loc[0, "set"] == "{}"

    bad_calls = [
        ({"task_type": "bad"}, "task_type"),
        ({"calibration_unit": "bad"}, "calibration_unit"),
        ({"level": 0}, "level"),
        ({"prediction": None}, "prediction"),
        ({"prediction": [1.0]}, "prediction"),
        ({"prediction": ["x"] * 4}, "numeric"),
    ]
    base = dict(
        truth=truth,
        prediction=pred,
        task_type="regression",
        generalization_target="new_trials_known_participants",
    )
    for changes, match in bad_calls:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            cf.fit_gazepoint_conformal(**args)

    with pytest.raises(GP3MLError, match="probability"):
        cf.fit_gazepoint_conformal(
            ctruth,
            probability=[0.1],
            task_type="classification",
            positive="target",
        )
    with pytest.raises(GP3MLError, match="exactly two"):
        cf.fit_gazepoint_conformal(
            ["a", "b", "c"],
            probability=[0.1, 0.5, 0.9],
            task_type="classification",
            positive="c",
        )
    with pytest.raises(GP3MLError, match="complete `unit`"):
        cf.fit_gazepoint_conformal(
            truth,
            prediction=pred,
            calibration_unit="participant",
            unit=None,
        )
    with pytest.raises(GP3MLError, match="complete `unit`"):
        cf.fit_gazepoint_conformal(
            truth,
            prediction=pred,
            calibration_unit="participant",
            unit=["p1", None, "p2", "p2"],
        )
    with pytest.raises(GP3MLError, match="No finite conformity"):
        cf.fit_gazepoint_conformal(
            [np.nan, np.nan],
            prediction=[np.nan, np.nan],
            task_type="regression",
        )
    with pytest.raises(GP3MLError, match="No finite conformity"):
        cf._type1_quantile(np.array([np.nan]), 0.5)

    assert cf.validate_gazepoint_conformal(object()).status == "fail"
    invalid = GP3MLConformal(
        task_type="bad",
        level=2,
        calibration_unit="bad",
        conformity_quantile=np.nan,
        generalization_target="",
        n_calibration_units=1,
    )
    assert cf.validate_gazepoint_conformal(invalid).status == "fail"
    review = reg.__deepcopy__({})
    review.n_calibration_units = 1
    assert cf.validate_gazepoint_conformal(review).status == "review"
    with pytest.raises(GP3MLError, match="valid regression"):
        cf.predict_gazepoint_interval(cls, [1.0])
    with pytest.raises(GP3MLError, match="valid classification"):
        cf.predict_gazepoint_set(reg, [0.5])


def test_conformal_coverage_row_and_unit_paths():
    reg = GP3MLConformal(
        task_type="regression",
        positive=None,
        negative=None,
        level=0.75,
        calibration_unit="participant",
        generalization_target="new_participants",
        n_calibration_units=3,
        conformity_quantile=0.5,
        caveat="caveat",
    )
    truth = [1.0, 2.0, 3.0, np.nan]
    interval = pd.DataFrame(
        {"lower": [0.5, 1.5, 4.0, 0.0], "upper": [1.5, 2.5, 5.0, 1.0]}
    )
    coverage = cf.assess_gazepoint_conformal_coverage(
        reg, truth, interval=interval, unit=["p1", "p1", "p2", "p2"]
    )
    assert coverage.status == "review"
    assert coverage.row_coverage == 0.5
    assert coverage.unit_coverage == 0.5
    with pytest.raises(GP3MLError, match="interval"):
        cf.assess_gazepoint_conformal_coverage(reg, truth, interval=None)
    with pytest.raises(GP3MLError, match="complete and aligned"):
        cf.assess_gazepoint_conformal_coverage(
            reg, truth, interval=interval, unit=["p1", None, "p2", "p2"]
        )

    cls = GP3MLConformal(
        task_type="classification",
        positive="target",
        negative="control",
        level=0.5,
        calibration_unit="observation",
        generalization_target="new_participants",
        n_calibration_units=4,
        conformity_quantile=0.6,
        caveat="caveat",
    )
    sets = pd.DataFrame(
        {
            "include_negative": [True, False, True, True],
            "include_positive": [False, True, False, True],
        }
    )
    cc = cf.assess_gazepoint_conformal_coverage(
        cls, ["control", "target", "control", None], set=sets
    )
    assert cc.status == "pass"
    assert cc.row_coverage == 0.75
    assert np.isnan(cc.unit_coverage)
    with pytest.raises(GP3MLError, match="classification `set`"):
        cf.assess_gazepoint_conformal_coverage(cls, ["control"], set=None)


def test_dataset_and_missingness_shift_all_severity_paths():
    development = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 3.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "cat": ["a", "a", "b", "b"],
            "missing": [1.0, 2.0, np.nan, 4.0],
        }
    )
    external_pass = development.copy()
    passed = ds.audit_gazepoint_dataset_shift(development, external_pass)
    assert passed.status == "pass"
    assert set(passed.findings.type) == {"numeric", "categorical"}

    external_review = development.copy()
    external_review["num"] = [0.5, 1.5, 2.5, 3.5]
    review = ds.audit_gazepoint_dataset_shift(
        development,
        external_review,
        predictors=["num"],
        thresholds={"smd_review": 0.1, "smd_fail": 10.0, "outside_review": 0.9},
    )
    assert review.status == "review"

    external_fail = development.copy()
    external_fail["num"] = [10.0, 11.0, 12.0, 13.0]
    external_fail["cat"] = ["a", "c", "c", "c"]
    failed = ds.audit_gazepoint_dataset_shift(development, external_fail)
    assert failed.status == "fail"
    assert "c" in failed.findings.loc[failed.findings.predictor == "cat", "novel_levels"].iloc[0]

    empty_dev = pd.DataFrame({"num": pd.Series([], dtype=float), "cat": pd.Series([], dtype=object)})
    empty_ext = empty_dev.copy()
    empty = ds.audit_gazepoint_dataset_shift(empty_dev, empty_ext)
    assert empty.status in {"pass", "review"}
    assert np.isnan(empty.findings.loc[empty.findings.predictor == "num", "outside_training_range"].iloc[0])

    none_predictors = ds.audit_gazepoint_dataset_shift(
        development.iloc[:, :0], external_pass.iloc[:, :0]
    )
    assert none_predictors.status == "review"
    assert none_predictors.findings.empty
    with pytest.raises(GP3MLError, match="absent"):
        ds.audit_gazepoint_dataset_shift(development, external_pass, predictors=["nope"])

    miss_pass = ds.audit_gazepoint_missingness_shift(development, external_pass, ["missing"])
    assert miss_pass.status == "pass"
    ext_miss_review = development.copy()
    ext_miss_review["missing"] = [np.nan, 2.0, np.nan, 4.0]
    miss_review = ds.audit_gazepoint_missingness_shift(
        development, ext_miss_review, ["missing"], review_delta=0.2, fail_delta=0.8
    )
    assert miss_review.status == "review"
    ext_miss_fail = development.copy()
    ext_miss_fail["missing"] = np.nan
    assert ds.audit_gazepoint_missingness_shift(
        development, ext_miss_fail, ["missing"], review_delta=0.1, fail_delta=0.5
    ).status == "fail"
    assert ds.audit_gazepoint_missingness_shift(
        development.iloc[:, :0], external_pass.iloc[:, :0]
    ).status == "review"
    with pytest.raises(GP3MLError, match="absent"):
        ds.audit_gazepoint_missingness_shift(development, external_pass, ["nope"])

    assert ds._status_counts(pd.DataFrame()).empty
    assert ds._status_counts(pd.DataFrame({"x": [1]})).empty
    counts = ds._status_counts(failed.findings)
    assert "n_predictors" in counts
    summary = ds.summarize_gazepoint_shift(failed, miss_pass)
    assert summary.dataset_shift_status == "fail"
    assert summary.missingness_shift_status == "pass"
    summary_without_missing = ds.summarize_gazepoint_shift(passed)
    assert "missingness_shift_status" not in summary_without_missing
    with pytest.raises(GP3MLError, match="dataset-shift"):
        ds.summarize_gazepoint_shift(object())
    with pytest.raises(GP3MLError, match="Invalid missingness"):
        ds.summarize_gazepoint_shift(passed, object())
