from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ._utils import assert_columns, assert_data, timestamp
from .exceptions import GP3MLError
from .objects import GP3MLRoleValidation, GP3MLTask

GENERALIZATION_TARGETS = (
    "new_trials_known_participants",
    "new_participants",
    "new_stimuli",
    "new_participants_and_new_stimuli",
    "external_validation",
)
TASK_TYPES = ("classification", "regression")


def gp3ml_prohibited_uses() -> list[str]:
    """Return the prohibited-use descriptions frozen in gp3ml 0.3.0."""
    return [
        "person identification or re-identification",
        "biometric authentication or verification",
        "health, disease, disability, or diagnostic inference",
        "protected-attribute prediction or proxy prediction",
        "emotion, stress, personality, deception, cognition, comprehension, intent, or mental-state inference",
        "random row-level evaluation represented as participant- or stimulus-level generalization",
        "outcome-derived or post-outcome feature engineering",
        "preprocessing estimated using assessment or external-validation data",
        "accuracy-only reporting without discrimination, calibration, and uncertainty",
    ]

_PROHIBITED_PATTERN = re.compile(
    "|".join([
        "identif", "re-identif", "authenticat", r"verif.*person", "biometric",
        "diagnos", "disease", "health status", "disability", "protected",
        "race", "ethnic", "religion", "gender identity", "sexual orientation",
        "emotion", "stress", "personality", "deception", "lie detect",
        "cognition", "cognitive", "comprehension", "intent", "mental state",
        "depression", "anxiety", "adhd", "autism", "intelligence",
    ]),
    flags=re.IGNORECASE,
)


def _ordered_class_levels(values: pd.Series) -> list[str]:
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [str(x) for x in values.cat.categories]
    nonmissing = values.dropna().astype(str).unique().tolist()
    return sorted(nonmissing)


def declare_gazepoint_task(
    data: pd.DataFrame,
    outcome: str,
    purpose: str,
    task_type: str = "classification",
    unit_id: str | None = None,
    participant_id: str | None = None,
    stimulus_id: str | None = None,
    generalization_target: str = "new_trials_known_participants",
    positive: str | None = None,
    observed_outcome: bool = True,
    sensitive_outcome: bool = False,
) -> GP3MLTask:
    """Declare a governed Gazepoint prediction task.

    This ports ``declare_gazepoint_task()`` from gp3ml 0.3.0.
    """
    assert_data(data)
    if task_type not in TASK_TYPES:
        raise GP3MLError(f"`task_type` must be one of: {', '.join(TASK_TYPES)}.")
    if generalization_target not in GENERALIZATION_TARGETS:
        raise GP3MLError("`generalization_target` is invalid.")
    if not isinstance(outcome, str) or not outcome:
        raise GP3MLError("Supply one `outcome` column.")
    if not isinstance(unit_id, str) or not unit_id:
        raise GP3MLError("Supply one `unit_id` column.")
    assert_columns(data, [outcome, unit_id, participant_id, stimulus_id], "task columns")
    if not isinstance(purpose, str) or not purpose.strip():
        raise GP3MLError("`purpose` must be one explicit scientific-purpose statement.")
    task = GP3MLTask(
        outcome=outcome,
        purpose=purpose,
        task_type=task_type,
        unit_id=unit_id,
        participant_id=participant_id,
        stimulus_id=stimulus_id,
        generalization_target=generalization_target,
        positive=positive,
        observed_outcome=bool(observed_outcome),
        sensitive_outcome=bool(sensitive_outcome),
        created_at=timestamp(),
    )
    assert_gp3ml_use_case(task, data)
    if task_type == "classification":
        levels = _ordered_class_levels(data[outcome])
        if len(levels) != 2:
            raise GP3MLError("Initial classification support requires exactly two observed outcome levels.")
        task.levels = levels
        task.positive = str(positive) if positive is not None else levels[1]
        if task.positive not in levels:
            raise GP3MLError("`positive` is not an outcome level.")
        task.negative = next(x for x in levels if x != task.positive)
    else:
        if not pd.api.types.is_numeric_dtype(data[outcome]):
            raise GP3MLError("Regression outcomes must be numeric.")
        task.levels = None
        task.positive = None
        task.negative = None
    return task


def assert_gp3ml_use_case(task: GP3MLTask, data: pd.DataFrame | None = None) -> bool:
    """Assert that a task is within the permitted gp3ml scope."""
    if not isinstance(task, GP3MLTask):
        raise GP3MLError("`task` must be a gp3ml task declaration.")
    text = f"{task.purpose} {task.outcome}".lower()
    if bool(task.sensitive_outcome) or _PROHIBITED_PATTERN.search(text):
        raise GP3MLError(
            "This task is prohibited. gp3ml does not support identification, authentication, "
            "health/diagnostic, protected-attribute, or inferred emotion/cognition/intent uses."
        )
    if not bool(task.observed_outcome):
        raise GP3MLError("The outcome must be explicitly observed rather than inferred as a latent mental or sensitive state.")
    if task.generalization_target == "new_participants" and task.participant_id is None:
        raise GP3MLError("Participant-level generalization requires `participant_id`.")
    if task.generalization_target == "new_stimuli" and task.stimulus_id is None:
        raise GP3MLError("Stimulus-level generalization requires `stimulus_id`.")
    if task.generalization_target == "new_participants_and_new_stimuli" and (
        task.participant_id is None or task.stimulus_id is None
    ):
        raise GP3MLError("Crossed generalization requires participant and stimulus identifiers.")
    if data is not None:
        assert_columns(data, [task.outcome, task.unit_id, task.participant_id, task.stimulus_id], "task columns")
    return True


def validate_gazepoint_ml_roles(
    data: pd.DataFrame,
    task: GP3MLTask,
    predictors: list[str] | tuple[str, ...] | str,
    feature_manifest: pd.DataFrame | None = None,
) -> GP3MLRoleValidation:
    """Validate outcome, predictor, identifier and grouping roles."""
    assert_data(data)
    assert_gp3ml_use_case(task, data)
    if isinstance(predictors, str):
        predictor_list = [predictors]
    else:
        predictor_list = list(dict.fromkeys(str(x) for x in predictors))
    missing_predictors = [p for p in predictor_list if p not in data.columns]
    identifiers = [x for x in [task.unit_id, task.participant_id, task.stimulus_id] if x]
    class_counts = data[task.outcome].value_counts(dropna=True) if task.task_type == "classification" else None
    grouping_column = {
        "new_trials_known_participants": task.unit_id,
        "new_participants": task.participant_id,
        "new_stimuli": task.stimulus_id,
        "new_participants_and_new_stimuli": task.participant_id,
        "external_validation": task.unit_id,
    }[task.generalization_target]
    manifest_validation: Any = None
    if feature_manifest is not None:
        from .feature_provenance import validate_gazepoint_feature_manifest
        manifest_validation = validate_gazepoint_feature_manifest(feature_manifest)
    if task.task_type != "classification":
        cls_status, cls_detail = "pass", "not applicable"
    else:
        assert class_counts is not None
        if len(class_counts) != 2 or bool((class_counts < 2).any()): cls_status = "fail"
        elif bool((class_counts < 10).any()): cls_status = "review"
        else: cls_status = "pass"
        cls_detail = ", ".join(f"{k}={v}" for k, v in class_counts.items())
    n_groups = 0 if grouping_column is None else int(data[grouping_column].nunique(dropna=False))
    checks = pd.DataFrame([
        ("predictors_exist", "fail" if missing_predictors else "pass", ", ".join(missing_predictors)),
        ("outcome_not_predictor", "fail" if task.outcome in predictor_list else "pass", task.outcome if task.outcome in predictor_list else ""),
        ("identifiers_not_predictors", "fail" if set(predictor_list) & set(identifiers) else "pass", ", ".join(sorted(set(predictor_list) & set(identifiers)))),
        ("outcome_complete", "fail" if data[task.outcome].isna().any() else "pass", str(int(data[task.outcome].isna().sum()))),
        ("sufficient_group_levels", "fail" if grouping_column is None or n_groups < 2 else "pass", "missing grouping role" if grouping_column is None else str(n_groups)),
        ("classification_level_support", cls_status, cls_detail),
        ("feature_manifest", "review" if manifest_validation is None else manifest_validation.status, "manifest not supplied" if manifest_validation is None else manifest_validation.status),
    ], columns=["check", "status", "detail"])
    status = "fail" if (checks.status == "fail").any() else "review" if (checks.status == "review").any() else "pass"
    return GP3MLRoleValidation(status=status, checks=checks, issues=checks.loc[checks.status != "pass"].reset_index(drop=True), manifest_validation=manifest_validation)


def _task_repr(self: GP3MLTask) -> str:
    return f"<gp3ml_task>\n  type: {self.task_type}\n  outcome: {self.outcome}\n  target: {self.generalization_target}\n  purpose: {self.purpose}"
GP3MLTask.__repr__ = _task_repr  # type: ignore[method-assign]
