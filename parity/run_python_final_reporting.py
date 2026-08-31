from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

import gp3mlpy as gp

PREDICTORS = ["fixation_duration", "pupil_change"]
QUALITY = [
    "pass", "review", "pass", "review", "review", "pass", "review", "pass",
    "pass", "review", "review", "pass", "review", "pass", "review", "pass",
    "pass", "review", "pass", "review", "review", "pass", "pass", "review",
]


def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [_jsonable(v) for v in list(x)]
    return x if isinstance(x, str) else str(x)


def _capture(fn):
    try:
        return {"status": "success", "value": _jsonable(fn())}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}


def _training() -> pd.DataFrame:
    n = np.arange(1, 25, dtype=float)
    return pd.DataFrame(
        {
            "participant_id": [f"P{i:02d}" for i in range(1, 13) for _ in range(2)],
            "trial_id": [f"T{i:02d}" for i in range(1, 25)],
            "stimulus_id": ["S01", "S02"] * 12,
            "fixation_duration": 180.0 + n,
            "pupil_change": np.sin(n / 3.0),
            "quality_status": pd.Categorical(QUALITY, categories=["pass", "review"]),
        }
    )


def _model(data: pd.DataFrame, seed: int):
    task = gp.declare_gazepoint_task(
        data=data,
        outcome="quality_status",
        purpose="Predict predefined recording-quality review status",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        positive="review",
    )
    model = gp.train_gazepoint_classifier(
        data,
        task,
        PREDICTORS,
        engine="glm",
        seed=seed,
    )
    return task, model


def _card_summary(card) -> dict[str, Any]:
    evaluation = card.evaluation
    return {
        "class": card.r_class,
        "title": str(card.title),
        "intended_use": str(card.intended_use),
        "prohibited_count": len(card.prohibited_uses),
        "outcome": str(card.task.outcome),
        "task_type": str(card.task.task_type),
        "generalization_target": str(card.task.generalization_target),
        "engine": str(card.engine),
        "predictors": [str(x) for x in card.predictors],
        "training_n": int(card.training_n),
        "training_hash_recorded": bool(str(card.training_hash)),
        "evaluation_rows": len(evaluation) if isinstance(evaluation, pd.DataFrame) else 0,
        "evaluation_columns": sorted(map(str, evaluation.columns)) if isinstance(evaluation, pd.DataFrame) else [],
        "limitations": [str(x) for x in card.limitations],
        "external_validation_present": card.external_validation is not None,
    }


def _release_summary(card) -> dict[str, Any]:
    out = _card_summary(card)
    out.update(
        class_=card.r_class,
        selection_recorded=bool(card.selection_procedure_recorded),
        uncertainty_unit=None if card.uncertainty_unit is None else str(card.uncertainty_unit),
        release_generalization_target=str(card.generalization_target),
        external_validation_status=str(card.external_validation_status),
        autonomous_selection=bool(card.autonomous_selection),
        deployment_status=str(card.deployment_status),
    )
    out["class"] = out.pop("class_")
    return out


def _markdown_summary(path: Path, returned: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "exists": path.is_file(),
        "return_basename": Path(returned).name,
        "basename": path.name,
        "headings": [line for line in lines if line.startswith("## ")],
        "line_count_positive": len(lines) > 0,
    }


def _json_summary(path: Path, returned: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["title", "intended_use", "engine", "predictors", "training_n", "limitations"]
    return {
        "exists": path.is_file(),
        "return_basename": Path(returned).name,
        "basename": path.name,
        "valid_json": isinstance(payload, dict),
        "required_fields_present": {name: name in payload for name in required},
        "title": str(payload.get("title")),
        "engine": str(payload.get("engine")),
    }


def _write_model(card, fmt: str) -> dict[str, Any]:
    with TemporaryDirectory() as d:
        suffix = ".json" if fmt == "json" else ".md"
        path = Path(d) / f"model_card{suffix}"
        returned = gp.write_gazepoint_model_card(card, path, format=fmt)
        return _json_summary(path, returned) if fmt == "json" else _markdown_summary(path, returned)


def _write_release(card, fmt: str) -> dict[str, Any]:
    with TemporaryDirectory() as d:
        suffix = ".json" if fmt == "json" else ".md"
        path = Path(d) / f"release_model_card{suffix}"
        returned = gp.write_gazepoint_release_model_card(card, path, format=fmt)
        return _json_summary(path, returned) if fmt == "json" else _markdown_summary(path, returned)


def _evidence_summary(obj) -> dict[str, Any]:
    return {
        "class": obj.r_class,
        "version": str(obj.version),
        "object_hash_count": len(obj.object_hashes),
        "object_hash_names": sorted(map(str, obj.object_hashes)),
        "object_hashes_recorded": all(bool(str(v)) for v in obj.object_hashes.values()),
        "file_checksum_count": len(obj.file_md5),
        "file_checksum_names": sorted(map(str, obj.file_md5)),
        "file_checksums": dict(sorted((str(k), str(v)) for k, v in obj.file_md5.items())),
        "file_basenames": dict(sorted((str(k), Path(v).name) for k, v in obj.file_paths.items())),
        "session_recorded": bool(str(obj.session)),
        "notes": [str(x) for x in obj.notes],
        "prohibited_count": len(obj.prohibited_uses),
    }


def _repro_summary(obj) -> dict[str, Any]:
    return {
        "class": obj.r_class,
        "runtime_recorded": bool(str(obj.python_version)),
        "platform_recorded": bool(str(obj.platform)),
        "session_recorded": bool(str(obj.session)),
        "object_hash_count": len(obj.object_hashes),
        "object_hash_names": sorted(map(str, obj.object_hashes)),
        "object_hashes_recorded": all(bool(str(v)) for v in obj.object_hashes.values()),
        "data_hash_recorded": obj.data_hash is not None and bool(str(obj.data_hash)),
        "seeds": dict(sorted((str(k), int(v)) for k, v in obj.seeds.items())),
        "git": {
            "commit": obj.git.get("commit"),
            "branch": obj.git.get("branch"),
            "clean": obj.git.get("clean"),
        },
        "notes": [str(x) for x in obj.notes],
        "prohibited_count": len(obj.prohibited_uses),
    }


def _write_repro(report) -> dict[str, Any]:
    with TemporaryDirectory() as d:
        path = Path(d) / "reproducibility.md"
        returned = gp.write_gazepoint_reproducibility_report(report, path)
        return _markdown_summary(path, returned)


def _blocked_deep_call(data, task):
    original_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "keras" or name.startswith("keras."):
            raise ImportError("blocked by parity fixture")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked
    try:
        return gp.fit_gazepoint_deep_model(
            data,
            task,
            predictors=PREDICTORS,
            hidden_units=(4,),
            dropout=0,
            epochs=1,
            batch_size=8,
            validation_split=0,
            seed=101,
            verbose=0,
        )
    finally:
        builtins.__import__ = original_import


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_final_reporting.py <fixture.json> <output.json>")
    fixture_path, output_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    seed = int(fixture["seed"])

    data = _training()
    task, model = _model(data, seed)
    evaluation = pd.DataFrame({"metric": ["accuracy"], "estimate": [0.75]})
    intended = "Support manual review of predefined recording-quality status"
    limitations = ["Synthetic parity fixture only."]

    card = gp.create_gazepoint_model_card(
        model, intended, limitations=limitations
    )
    card_eval = gp.create_gazepoint_model_card(
        model, intended, evaluation=evaluation, limitations=limitations
    )
    release = gp.create_gazepoint_release_model_card(
        model, intended, evaluation=evaluation, limitations=limitations
    )

    with TemporaryDirectory() as evidence_dir:
        evidence_path = Path(evidence_dir) / "release-note.txt"
        evidence_path.write_text("gp3ml parity release evidence\n", encoding="utf-8")
        populated_evidence = gp.create_gazepoint_release_evidence(
            objects={"vector": [1, 2, 3], "label": "synthetic"},
            files={"note": evidence_path},
            version="0.3.0-candidate",
            notes=["Frozen R/Python parity evidence."],
        )
        populated_evidence_summary = _evidence_summary(populated_evidence)

        def unnamed_files():
            return gp.create_gazepoint_release_evidence(files=[evidence_path])

        cases: dict[str, Any] = {}
        cases["create_gazepoint_model_card::minimal"] = _capture(lambda: _card_summary(card))
        cases["create_gazepoint_model_card::dataframe_evaluation"] = _capture(lambda: _card_summary(card_eval))
        cases["create_gazepoint_model_card::invalid_model"] = _capture(
            lambda: gp.create_gazepoint_model_card("not model", intended)
        )
        cases["write_gazepoint_model_card::markdown_success"] = _capture(lambda: _write_model(card_eval, "markdown"))
        cases["write_gazepoint_model_card::json_success"] = _capture(lambda: _write_model(card_eval, "json"))

        def overwrite_model():
            with TemporaryDirectory() as d:
                path = Path(d) / "model_card.md"
                gp.write_gazepoint_model_card(card, path)
                return gp.write_gazepoint_model_card(card, path)
        cases["write_gazepoint_model_card::overwrite_rejected"] = _capture(overwrite_model)
        cases["write_gazepoint_model_card::invalid_card"] = _capture(
            lambda: gp.write_gazepoint_model_card("not card", "model_card.md")
        )
        cases["write_gazepoint_model_card::invalid_format"] = _capture(
            lambda: gp.write_gazepoint_model_card(card, "model_card.md", format="yaml")
        )

        cases["create_gazepoint_release_model_card::minimal"] = _capture(lambda: _release_summary(release))
        cases["create_gazepoint_release_model_card::dataframe_evaluation"] = _capture(lambda: _release_summary(release))
        cases["create_gazepoint_release_model_card::missing_limitations"] = _capture(
            lambda: gp.create_gazepoint_release_model_card(model, intended)
        )
        cases["create_gazepoint_release_model_card::blank_limitations"] = _capture(
            lambda: gp.create_gazepoint_release_model_card(model, intended, limitations=[""])
        )
        cases["create_gazepoint_release_model_card::invalid_model"] = _capture(
            lambda: gp.create_gazepoint_release_model_card("not model", intended, limitations=limitations)
        )
        cases["create_gazepoint_release_model_card::invalid_selection"] = _capture(
            lambda: gp.create_gazepoint_release_model_card(model, intended, selection="bad", limitations=limitations)
        )
        cases["create_gazepoint_release_model_card::invalid_uncertainty"] = _capture(
            lambda: gp.create_gazepoint_release_model_card(model, intended, uncertainty="bad", limitations=limitations)
        )
        cases["create_gazepoint_release_model_card::invalid_transportability"] = _capture(
            lambda: gp.create_gazepoint_release_model_card(model, intended, transportability="bad", limitations=limitations)
        )
        cases["write_gazepoint_release_model_card::markdown_success"] = _capture(lambda: _write_release(release, "markdown"))
        cases["write_gazepoint_release_model_card::json_success"] = _capture(lambda: _write_release(release, "json"))

        def overwrite_release():
            with TemporaryDirectory() as d:
                path = Path(d) / "release_model_card.md"
                gp.write_gazepoint_release_model_card(release, path)
                return gp.write_gazepoint_release_model_card(release, path)
        cases["write_gazepoint_release_model_card::overwrite_rejected"] = _capture(overwrite_release)
        cases["write_gazepoint_release_model_card::invalid_card"] = _capture(
            lambda: gp.write_gazepoint_release_model_card("not card", "release.md")
        )
        cases["write_gazepoint_release_model_card::invalid_format"] = _capture(
            lambda: gp.write_gazepoint_release_model_card(release, "release.md", format="yaml")
        )

        cases["create_gazepoint_release_evidence::empty_defaults"] = _capture(
            lambda: _evidence_summary(gp.create_gazepoint_release_evidence())
        )
        cases["create_gazepoint_release_evidence::objects_and_file"] = _capture(
            lambda: populated_evidence_summary
        )
        cases["create_gazepoint_release_evidence::missing_file"] = _capture(
            lambda: gp.create_gazepoint_release_evidence(files={"missing": Path(evidence_dir) / "missing.txt"})
        )
        cases["create_gazepoint_release_evidence::unnamed_files"] = _capture(unnamed_files)
        cases["create_gazepoint_release_evidence::custom_version_notes"] = _capture(
            lambda: _evidence_summary(gp.create_gazepoint_release_evidence(version="9.9.9", notes=["alpha", "beta"]))
        )

        with TemporaryDirectory() as project_dir:
            empty_repro = gp.create_gazepoint_reproducibility_report(project_path=project_dir)
            populated_repro = gp.create_gazepoint_reproducibility_report(
                objects={"vector": [1, 2, 3]},
                data=pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
                seeds={"split": 101, "bootstrap": 202},
                notes=["Synthetic reproducibility parity fixture."],
                project_path=project_dir,
            )
            cases["create_gazepoint_reproducibility_report::empty_defaults"] = _capture(lambda: _repro_summary(empty_repro))
            cases["create_gazepoint_reproducibility_report::populated"] = _capture(lambda: _repro_summary(populated_repro))
            cases["write_gazepoint_reproducibility_report::markdown_success"] = _capture(lambda: _write_repro(populated_repro))

            def overwrite_repro():
                with TemporaryDirectory() as d:
                    path = Path(d) / "repro.md"
                    gp.write_gazepoint_reproducibility_report(populated_repro, path)
                    return gp.write_gazepoint_reproducibility_report(populated_repro, path)
            cases["write_gazepoint_reproducibility_report::overwrite_rejected"] = _capture(overwrite_repro)
            cases["write_gazepoint_reproducibility_report::invalid_report"] = _capture(
                lambda: gp.write_gazepoint_reproducibility_report("not report", "repro.md")
            )

        cases["fit_gazepoint_deep_model::backend_missing"] = _capture(lambda: _blocked_deep_call(data, task))
        cases["fit_gazepoint_deep_model::backend_precedes_validation"] = _capture(lambda: _blocked_deep_call(data, "not task"))

    output = {
        "schema_version": 1,
        "runtime": "python",
        "r_reference_version": fixture["r_reference"]["version"],
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"python final-reporting records: {len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
