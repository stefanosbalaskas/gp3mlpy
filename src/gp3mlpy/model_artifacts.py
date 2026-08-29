from __future__ import annotations

from copy import deepcopy
from importlib import metadata
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from ._utils import hash_jsonable
from .exceptions import GP3MLError
from .objects import GP3MLModelArtifact, GP3MLModelArtifactValidation, GP3MLModelPortability


def _schema(model: Any, reference_data: pd.DataFrame | None = None) -> dict[str, Any]:
    predictors = list(getattr(model, "predictors", []) or [])
    if reference_data is None:
        return {"predictors": predictors, "classes": None}
    missing = [x for x in predictors if x not in reference_data.columns]
    if missing:
        raise GP3MLError("Reference data are missing model predictors: " + ", ".join(missing) + ".")
    classes = {p: str(reference_data[p].dtype) for p in predictors}
    return {"predictors": predictors, "classes": classes}


def _git_sha(path: str = ".") -> str | None:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _safe_model_fingerprint(model: Any) -> dict[str, Any]:
    """Hash model provenance without serializing executable Python objects.

    This intentionally differs from R's serialize()-based object hash. It avoids
    introducing unsafe pickle semantics into the Python artifact format.
    """
    if model is None:
        return {"model": None}
    task = getattr(model, "task", None)
    prep = getattr(model, "preprocessor", None)
    return {
        "r_class": getattr(model, "r_class", model.__class__.__name__),
        "engine": getattr(model, "engine", getattr(model, "engine_name", None)),
        "predictors": list(getattr(model, "predictors", []) or []),
        "training_hash": getattr(model, "training_hash", None),
        "training_n": getattr(model, "training_n", None),
        "seed": getattr(model, "seed", None),
        "threshold": getattr(model, "threshold", None),
        "task": task.to_dict() if hasattr(task, "to_dict") else None,
        "preprocessor": prep.to_dict() if hasattr(prep, "to_dict") else None,
        "fit_type": type(getattr(model, "fit", None)).__name__,
    }


def _artifact_payload_for_hash(artifact_or_payload: dict[str, Any]) -> dict[str, Any]:
    x = dict(artifact_or_payload)
    model = x.get("model")
    x["model"] = _safe_model_fingerprint(model)
    # Do not hash arbitrary executable/callable state hidden in auxiliary objects.
    for key in ("preprocessor", "feature_manifest", "task", "decision_rule", "model_card"):
        value = x.get(key)
        if hasattr(value, "to_dict"):
            x[key] = value.to_dict()
    return x


def _artifact_hash(payload: dict[str, Any]) -> str:
    return hash_jsonable(_artifact_payload_for_hash(payload), algorithm="sha256")


def create_gazepoint_model_artifact(
    model,
    preprocessor=None,
    feature_manifest=None,
    task=None,
    decision_rule=None,
    model_card=None,
    reference_data: pd.DataFrame | None = None,
    bundle_model: bool = True,
) -> GP3MLModelArtifact:
    if model is None:
        raise GP3MLError("`model` is required.")
    if preprocessor is None:
        preprocessor = getattr(model, "preprocessor", None)
    if task is None:
        task = getattr(model, "task", None)
    schema = _schema(model, reference_data)
    engine = getattr(model, "engine", getattr(model, "engine_name", None))
    engine_version = None
    if isinstance(engine, str):
        package = {"glm": "statsmodels", "lm": "statsmodels", "ranger": "scikit-learn", "nnet": "scikit-learn", "xgboost": "xgboost", "keras3": "keras"}.get(engine)
        if package:
            try: engine_version = metadata.version(package)
            except metadata.PackageNotFoundError: engine_version = None
    try: gp3mlpy_version = metadata.version("gp3mlpy")
    except metadata.PackageNotFoundError: gp3mlpy_version = "development"
    metadata_obj = {
        "gp3ml_version": "0.3.0",
        "gp3mlpy_version": gp3mlpy_version,
        "engine": engine,
        "engine_version": engine_version,
        "Python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha("."),
        # Python does not silently pickle/bundle executable model state.
        "bundled": False,
        "bundle_error": None if not bundle_model else "Executable model bundling is intentionally disabled; use a safe/native engine format for persisted artifacts.",
        "hash_scope": "governance metadata and model fingerprint; excludes executable fitted-object bytes",
    }
    payload = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_manifest": feature_manifest,
        "task": task,
        "decision_rule": decision_rule,
        "model_card": model_card,
        "predictor_schema": schema,
        "reference_data": reference_data.copy() if isinstance(reference_data, pd.DataFrame) else reference_data,
        "metadata": metadata_obj,
    }
    return GP3MLModelArtifact(**payload, artifact_hash=_artifact_hash(payload))


def validate_gazepoint_model_artifact(artifact, verify_hash: bool = True) -> GP3MLModelArtifactValidation:
    checks = pd.DataFrame({"check": ["class", "model", "task", "schema", "metadata", "hash"], "status": "pass", "detail": ""})
    if not isinstance(artifact, GP3MLModelArtifact):
        checks.loc[:, "status"] = "fail"
    else:
        if artifact.get("model") is None: checks.loc[checks.check == "model", "status"] = "fail"
        if artifact.get("task") is None: checks.loc[checks.check == "task", "status"] = "review"
        if artifact.get("predictor_schema") is None: checks.loc[checks.check == "schema", "status"] = "fail"
        if artifact.get("metadata") is None: checks.loc[checks.check == "metadata", "status"] = "fail"
        if verify_hash:
            expected = artifact.get("artifact_hash")
            if expected is None:
                checks.loc[checks.check == "hash", "status"] = "fail"
            else:
                payload = artifact.to_dict(); payload.pop("artifact_hash", None)
                if _artifact_hash(payload) != expected:
                    checks.loc[checks.check == "hash", "status"] = "fail"
    overall = "fail" if (checks.status == "fail").any() else ("review" if (checks.status == "review").any() else "pass")
    return GP3MLModelArtifactValidation(status=overall, checks=checks)


def restore_gazepoint_model_artifact(artifact):
    validation = validate_gazepoint_model_artifact(artifact, verify_hash=True)
    if validation.status == "fail": raise GP3MLError("Artifact validation failed.")
    return deepcopy(artifact)


def _predict_for_portability(model, newdata):
    if hasattr(model, "predict"):
        try: return np.asarray(model.predict(newdata), dtype=object)
        except Exception: return None
    return None


def test_gazepoint_model_portability(artifact, newdata=None, tolerance: float = 1e-8, fresh_process: bool = False) -> GP3MLModelPortability:
    validation = validate_gazepoint_model_artifact(artifact, verify_hash=True)
    if validation.status == "fail": raise GP3MLError("Artifact is invalid before portability testing.")
    if newdata is None: newdata = artifact.get("reference_data")
    # Trusted in-memory deep-copy round trip is safe. A persisted Python model must use
    # a supported native/skops format; we do not implicitly deserialize pickle/joblib.
    roundtrip = deepcopy(artifact)
    roundtrip_valid = validate_gazepoint_model_artifact(roundtrip, verify_hash=True)
    prediction_equal = None
    if newdata is not None:
        baseline = _predict_for_portability(restore_gazepoint_model_artifact(artifact).model, newdata)
        after = _predict_for_portability(restore_gazepoint_model_artifact(roundtrip).model, newdata)
        if baseline is not None and after is not None:
            try: prediction_equal = bool(np.allclose(np.asarray(baseline, dtype=float), np.asarray(after, dtype=float), rtol=tolerance, atol=tolerance, equal_nan=True))
            except (TypeError, ValueError): prediction_equal = bool(np.array_equal(baseline, after))
    fresh_ok = None
    fresh_error = None
    if fresh_process:
        fresh_ok = False
        fresh_error = "Fresh-process testing requires an explicitly safe/native persisted model format; automatic pickle deserialization is disabled."
    statuses = [roundtrip_valid.status == "pass", prediction_equal is not False, fresh_ok is not False]
    return GP3MLModelPortability(status="pass" if all(statuses) else "fail", roundtrip_valid=roundtrip_valid.status, prediction_equal=prediction_equal, fresh_process_ok=fresh_ok, fresh_process_error=fresh_error, artifact_hash=artifact.artifact_hash)

# The public API name begins with test_, but it is not a pytest test function.
test_gazepoint_model_portability.__test__ = False
