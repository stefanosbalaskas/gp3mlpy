from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gp3mlpy import _utils
from gp3mlpy import objects as obj
from gp3mlpy.exceptions import GP3MLError


def test_utility_contracts_cover_success_failure_and_recycling(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GP3MLPY_REPRODUCIBLE_EXAMPLES", "true")
    assert _utils.timestamp() == "<timestamp>"
    monkeypatch.delenv("GP3MLPY_REPRODUCIBLE_EXAMPLES", raising=False)
    assert _utils.timestamp().endswith(" UTC")

    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _utils.assert_data(frame)
    with pytest.raises(GP3MLError, match="data frame"):
        _utils.assert_data([1, 2])
    with pytest.raises(GP3MLError, match="at least 2"):
        _utils.assert_data(frame.iloc[:1])

    assert _utils.clean_columns(["a", None, "", "a", "b"]) == ["a", "b"]
    _utils.assert_columns(frame, ["a", None])
    with pytest.raises(GP3MLError, match="Missing predictors"):
        _utils.assert_columns(frame, ["missing"], argument="predictors")

    assert _utils.worst_status([]) == "fail"
    assert _utils.worst_status(["pass", "review"]) == "review"
    assert _utils.worst_status(["pass", "unexpected"]) == "unexpected"
    clipped = _utils.clip_probability([0.0, 0.5, 1.0], eps=0.1)
    assert np.allclose(clipped, [0.1, 0.5, 0.9])
    assert _utils.seed_from(1, ["a", [2, 3]]) == _utils.seed_from(1, ["a", [2, 3]])
    assert _utils.r_round(2.5) == 2

    digest_sha = _utils.hash_jsonable(
        {
            "frame": frame,
            "series": frame["a"],
            "scalar": np.int64(3),
            "path": tmp_path,
            "object": obj.GP3MLTask(outcome="y"),
        },
        algorithm="sha256",
    )
    digest_md5 = _utils.hash_jsonable(frame, algorithm="md5")
    assert len(digest_sha) == 64
    assert len(digest_md5) == 32
    with pytest.raises(TypeError):
        _utils.hash_jsonable({"bad": object()})

    tables = {"one": frame, "two": pd.DataFrame({"x": [1.0, np.nan]})}
    paths = _utils.write_tables(tables, tmp_path / "tables", "audit")
    assert set(paths) == {"one", "two"}
    assert all(Path(path).exists() for path in paths.values())
    with pytest.raises(GP3MLError, match="overwrite"):
        _utils.write_tables(tables, tmp_path / "tables", "audit")
    _utils.write_tables(tables, tmp_path / "tables", "audit", overwrite=True)

    assert _utils.ensure_choice("a", ["a", "b"], "choice") == "a"
    with pytest.raises(GP3MLError, match="must be one of"):
        _utils.ensure_choice("c", ["a", "b"], "choice")

    assert _utils.as_list("x") == ["x"]
    assert _utils.as_list(None) == [None]
    assert _utils.as_list(pd.Series([1, 2])) == [1, 2]
    assert _utils.as_list(np.array([1, 2])) == [1, 2]
    assert _utils.as_list((1, 2)) == [1, 2]
    assert _utils.as_list(3) == [3]

    assert _utils.recycle("x", 3, "x", kind="character") == ["x", "x", "x"]
    assert _utils.recycle([True, False], 2, "x", kind="logical") == [True, False]
    with pytest.raises(GP3MLError, match="length 1 or length 3"):
        _utils.recycle([1, 2], 3, "x", kind="logical")
    with pytest.raises(GP3MLError, match="character vector"):
        _utils.recycle([1], 1, "x", kind="character")
    with pytest.raises(GP3MLError, match="logical vector"):
        _utils.recycle(["yes"], 1, "x", kind="logical")
    with pytest.raises(GP3MLError, match="must not contain missing"):
        _utils.recycle([None], 1, "x", kind="character", allow_na=False)
    with pytest.raises(GP3MLError, match="must not contain missing"):
        _utils.recycle([float("nan")], 1, "x", kind="logical", allow_na=False)

    assert _utils.is_missing_text(None)
    assert _utils.is_missing_text(float("nan"))
    assert _utils.is_missing_text("   ")
    assert not _utils.is_missing_text("value")
    assert _utils.identifier_like("participant_id")
    assert _utils.identifier_like("filename")
    assert not _utils.identifier_like("pupil_mean")


def test_gp3ml_object_mapping_json_copy_and_locking():
    base = obj.GP3MLObject(a=1, frame=pd.DataFrame({"x": [1, 2]}))
    assert base["a"] == 1
    base["b"] = 2
    base.c = 3
    assert base.b == 2 and base["c"] == 3
    del base["b"]
    assert "b" not in base
    assert list(iter(base)) == ["a", "frame", "c"]
    assert len(base) == 3
    with pytest.raises(AttributeError):
        _ = base.missing

    restored = obj.GP3MLObject.from_dict(base.to_dict())
    restored["a"] = 99
    assert base["a"] == 1
    with pytest.raises(TypeError, match="dictionary"):
        obj.GP3MLObject.from_dict([])

    class BadToList:
        def tolist(self):
            raise ValueError("not listable")

        def __repr__(self):
            return "BadToList()"

    nested = obj.GP3MLObject(
        child=obj.GP3MLTask(outcome="y"),
        frame=pd.DataFrame({"x": [1]}),
        series=pd.Series([1, 2]),
        array=np.array([1, 2]),
        mapping={"x": np.int64(4)},
        tuple_value=(1, 2),
        scalar=True,
        none=None,
        bad=BadToList(),
    )
    payload = nested.to_json(indent=None)
    assert '"outcome": "y"' in payload
    assert "BadToList()" in payload

    clone = deepcopy(nested)
    clone.mapping["x"] = 8
    assert nested.mapping["x"] == 4

    plan = obj.GP3MLAnalysisPlan(locked=False, primary_metric="roc_auc", extra=1)
    plan["primary_metric"] = "brier"
    plan.extra = 2
    del plan["extra"]
    plan.locked = True
    with pytest.raises(GP3MLError, match="immutable"):
        plan["primary_metric"] = "accuracy"
    with pytest.raises(GP3MLError, match="immutable"):
        del plan["primary_metric"]
    with pytest.raises(GP3MLError, match="immutable"):
        plan.primary_metric = "accuracy"


def _df(**columns):
    return pd.DataFrame(columns)


def test_every_r_style_representation_branch_is_exercised():
    checks = _df(check_id=["c1"], status=["review"], n_affected=[1], columns=["x"])
    status_summary = _df(status=["pass", "review", "fail"], n_checks=[1, 2, 3])
    fold_status = _df(status=["pass", "review", "fail"])
    task_dict = {"task_type": "classification", "outcome": "y"}

    instances = [
        obj.GazepointFeatureManifestValidation(
            status="review", n_features=2, issues=checks, summary=_df(check=["x"], status=["review"])
        ),
        obj.GazepointFoldDiagnostics(
            metadata={"generalization_target": "new_participants", "repeats": 1, "outcome_type": "classification"},
            validation={"status": "pass"},
            fold_metrics=_df(fold=[1]),
            repeat_metrics=_df(assessment_size_ratio=[1.25]),
        ),
        obj.GazepointFoldDiagnosticsValidation(status="review", summary=status_summary),
        obj.GazepointGroupFolds(
            metadata={"generalization_target": "new_participants", "repeats": 1, "n_folds_per_repeat": 3, "n_folds_total": 3},
            validation={"status": "pass"},
        ),
        obj.GazepointGroupFoldsAudit(status="pass", summary=_df(fold=[1]), issues=pd.DataFrame()),
        obj.GazepointGroupFoldsValidation(status="pass", issues=pd.DataFrame(), summary=_df(check=["x"], status=["pass"])),
        obj.GazepointMLSplitValidation(status="review", issues=checks, summary=_df(check=["x"], status=["review"])),
        obj.GazepointMLLeakageAudit(
            status="pass",
            generalization_target="new_participants",
            partition_summary=_df(partition=["analysis", "assessment"], n_rows=[10, 5]),
            issues=pd.DataFrame(),
        ),
        obj.GazepointMLLeakageAudit(
            status="review",
            generalization_target="new_participants",
            partition_summary=_df(partition=["analysis", "assessment"], n_rows=[10, 5]),
            issues=checks,
        ),
        obj.GazepointMLSplit(
            metadata={"generalization_target": "new_participants", "seed": 7},
            validation={"status": "pass"},
            analysis=_df(x=[1, 2]), assessment=_df(x=[3]), excluded=pd.DataFrame(),
        ),
        obj.GP3MLAPIContractRegistry(
            exports=_df(stability=["stable", "experimental"]), classes=_df(name=["a", "b"])
        ),
        obj.GP3MLAPIStabilityAudit(status="pass", checks=_df(check=["api"], status=["pass"])),
        obj.GP3MLCalibrationAssessment(summary=_df(brier=[0.1])),
        obj.GP3MLDecisionRule(
            metric="balanced_accuracy", direction="maximize", threshold=None,
            threshold_origin="predeclared", generalization_target="new_trials_known_participants",
            abstention_allowed=True,
        ),
        obj.GP3MLDecisionRule(
            metric="balanced_accuracy", direction="maximize", threshold=0.5,
            threshold_origin="predeclared", generalization_target="new_trials_known_participants",
            abstention_allowed=False,
        ),
        obj.GP3MLEngineCapabilities(table=_df(engine=["glm"], available=[True])),
        obj.GP3MLEngineCapabilities(engine=["glm"], available=[True]),
        obj.GP3MLExternalDatasetDeclaration(label="ext", independent=True, origin="site", n_rows=4),
        obj.GP3MLExternalValidation(label="ext", metrics=_df(roc_auc=[0.8])),
        obj.GP3MLHandoff(source_package="gp3tools", data=_df(x=[1]), predictors=["x"]),
        obj.GP3MLHandoffBundle(handoffs=[1, 2], data=_df(x=[1, 2])),
        obj.GP3MLHandoffValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLMetricUncertainty(bootstrap=10, conf_level=0.95, intervals=_df(metric=["brier"], estimate=[0.1])),
        obj.GP3MLModel(engine="glm", task=task_dict, training_n=10, predictors=["x"]),
        obj.GP3MLModelCard(title="card"),
        obj.GP3MLModelSelection(
            candidate_id="c1", primary_metric="roc_auc", direction="maximize",
            primary_value=0.8, rationale="declared",
        ),
        obj.GP3MLModelTuning(
            grid={"candidates": _df(candidate_id=["c1"])}, results=[{"status": "fail"}]
        ),
        obj.GP3MLModelTuningValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLNestedEvaluationValidation(status="review", checks=checks),
        obj.GP3MLNestedFoldsValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLNestedResamplingAudit(status="review", checks=checks),
        obj.GP3MLResampleEvaluationValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLTransportabilityValidation(status="review", checks=checks),
        obj.GP3MLUncertaintyValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLNestedEvaluation(
            generalization_target="new_participants", fold_status=fold_status, predictions=_df(y=[1, 2])
        ),
        obj.GP3MLNestedFolds(
            folds=[1, 2], inner_v=2, inner_repeats=1,
            outer_metadata={"generalization_target": "new_participants"}, audit={"status": "pass"},
        ),
        obj.GP3MLObjectContractValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLPreprocessor(predictors=["x", "z"], columns=["x", "z"]),
        obj.GP3MLReleaseEvidence(version="0.1", object_hashes={"x": "a"}, file_md5={"f": "b"}),
        obj.GP3MLReleaseModelCard(
            task={"outcome": "y"}, generalization_target="new_participants",
            selection_procedure_recorded=True, uncertainty_unit="participant",
            external_validation_status="externally_validated",
        ),
        obj.GP3MLReproducibilityAudit(status="review", files_scanned=2, findings=checks),
        obj.GP3MLReproducibilityReport(created_at="<timestamp>"),
        obj.GP3MLResampleEvaluation(
            generalization_target="new_participants", engine="glm", fold_status=fold_status,
            predictions=_df(y=[1, 2]),
        ),
        obj.GP3MLResamplePerformanceSummary(
            aggregation="fold", generalization_target="new_participants",
            summary=_df(metric=["roc_auc"], mean=[0.8]),
        ),
        obj.GP3MLResampleUncertainty(unit="fold", summary=_df(metric=["roc_auc"], mean=[0.8])),
        obj.GP3MLResearchBundle(handoffs=[1], outcome="y", generalization_target="new_participants"),
        obj.GP3MLResearchBundleValidation(status="pass", checks=_df(check=["x"], status=["pass"])),
        obj.GP3MLRoleValidation(status="review", checks=checks),
        obj.GP3MLTargetUncertainty(
            unit="participant", generalization_target="new_participants",
            successful_replicates=9, failed_replicates=1,
            intervals=_df(metric=["roc_auc"], estimate=[0.8]),
        ),
        obj.GP3MLTask(
            task_type="classification", outcome="y", generalization_target="new_participants", purpose="research"
        ),
        obj.GP3MLTransportabilityReport(
            status="externally_validated", reason="ok",
            performance_comparison=_df(metric=["roc_auc"], difference=[0.01]),
        ),
        obj.GP3MLTuningGrid(
            candidates=_df(
                candidate_id=["c1"], label=["glm"], engine=["glm"], threshold=[0.5],
                complexity=["low"], interpretability=["high"],
            )
        ),
        obj.GP3MLObject(other="fallback"),
    ]

    for instance in instances:
        rendered = repr(instance)
        assert isinstance(rendered, str)
        assert rendered
        assert str(instance) == rendered

    # Helper branches not guaranteed by the object collection above.
    from gp3mlpy import _reprs

    assert _reprs._nrow(None) == 0
    assert _reprs._nrow([1, 2]) == 2
    assert _reprs._table_text(None) == ""
    assert _reprs._table_text(pd.DataFrame()) == ""
    assert _reprs._status_counts(None) == (0, 0, 0)
    assert _reprs._component(None, "x", 4) == 4
    assert _reprs._component({"x": 3}, "x") == 3
    assert _reprs._component({}, "x", 5) == 5
