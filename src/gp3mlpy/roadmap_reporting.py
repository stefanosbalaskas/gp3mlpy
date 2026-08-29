from __future__ import annotations

from hashlib import md5
from pathlib import Path
import json
import platform
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

from ._utils import hash_jsonable, timestamp
from .exceptions import GP3MLError
from .governance_reports import (
    _json_ready,
    _markdown_table,
    _safe_write,
    create_gazepoint_model_card,
)
from .objects import (
    GP3MLCalibrationAssessment,
    GP3MLModel,
    GP3MLModelSelection,
    GP3MLReleaseEvidence,
    GP3MLReleaseModelCard,
    GP3MLResampleUncertainty,
    GP3MLTargetUncertainty,
    GP3MLTransportabilityReport,
)
from .resample_evaluation import summarize_gazepoint_resample_performance
from .task_governance import gp3ml_prohibited_uses


def _as_character_vector(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def create_gazepoint_release_model_card(
    model: GP3MLModel,
    intended_use: str,
    evaluation: Any = None,
    selection: GP3MLModelSelection | None = None,
    uncertainty: GP3MLTargetUncertainty | GP3MLResampleUncertainty | None = None,
    calibration: Any = None,
    feature_manifest: Any = None,
    transportability: GP3MLTransportabilityReport | None = None,
    limitations: str | Sequence[str] | None = None,
    ethical_review: Any = None,
    deployment_status: str = "research_review_only",
) -> GP3MLReleaseModelCard:
    """Create the release-ready governed model-card contract from gp3ml 0.3.0."""
    if not isinstance(model, GP3MLModel):
        raise GP3MLError("`model` must be a fitted gp3ml model.")
    limitation_values = _as_character_vector(limitations)
    if not limitation_values or any(not item.strip() for item in limitation_values):
        raise GP3MLError("At least one explicit limitation is required.")
    if selection is not None and not isinstance(selection, GP3MLModelSelection):
        raise GP3MLError("`selection` must be a `gp3ml_model_selection` object.")
    if uncertainty is not None and not isinstance(
        uncertainty, (GP3MLTargetUncertainty, GP3MLResampleUncertainty)
    ):
        raise GP3MLError("`uncertainty` must be a target-aligned gp3ml uncertainty object.")
    if transportability is not None and not isinstance(transportability, GP3MLTransportabilityReport):
        raise GP3MLError("`transportability` must be a `gp3ml_transportability_report`.")

    base = create_gazepoint_model_card(
        model=model,
        intended_use=intended_use,
        evaluation=evaluation,
        calibration=calibration,
        feature_manifest=feature_manifest,
        external_validation=None if transportability is None else transportability.validation,
        limitations=limitation_values,
        ethical_review=ethical_review,
    )
    components = base.to_dict()
    components.update(
        selection=selection,
        uncertainty=uncertainty,
        transportability=transportability,
        deployment_status=deployment_status,
        selection_procedure_recorded=selection is not None,
        uncertainty_unit=None if uncertainty is None else uncertainty.unit,
        generalization_target=model.task.generalization_target,
        external_validation_status=(
            "not_externally_validated" if transportability is None else transportability.status
        ),
        autonomous_selection=False,
    )
    return GP3MLReleaseModelCard(**components)


def _release_card_metrics(card: GP3MLReleaseModelCard) -> pd.DataFrame:
    evaluation = card.evaluation
    if getattr(evaluation, "r_class", None) in {"gp3ml_resample_evaluation", "gp3ml_nested_evaluation"}:
        summary = summarize_gazepoint_resample_performance(evaluation)
        return summary.summary
    if isinstance(evaluation, pd.DataFrame):
        return evaluation
    return pd.DataFrame()


def _release_card_selection(selection: GP3MLModelSelection | None) -> pd.DataFrame:
    if selection is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [{
            "candidate_id": selection.candidate_id,
            "primary_metric": selection.primary_metric,
            "direction": selection.direction,
            "primary_value": selection.primary_value,
            "minimum_success_prop": selection.minimum_success_prop,
            "rationale": selection.rationale,
            "autonomous_selection": selection.autonomous_selection,
            "refit_performed": selection.refit_performed,
        }]
    )


def _release_card_uncertainty(uncertainty: Any) -> pd.DataFrame:
    if uncertainty is None:
        return pd.DataFrame()
    if isinstance(uncertainty, GP3MLTargetUncertainty):
        return uncertainty.intervals
    return uncertainty.summary


def write_gazepoint_release_model_card(
    card: GP3MLReleaseModelCard,
    path: str | Path,
    format: str = "markdown",
    overwrite: bool = False,
) -> str:
    """Write a release-ready governed model card as Markdown or JSON."""
    if not isinstance(card, GP3MLReleaseModelCard):
        raise GP3MLError("`card` must be a `gp3ml_release_model_card`.")
    if format not in {"markdown", "json"}:
        raise GP3MLError("`format` must be one of: markdown, json.")
    target = _safe_write(path, overwrite)
    if format == "json":
        target.write_text(json.dumps(_json_ready(card), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(target)

    selection = _release_card_selection(card.selection)
    uncertainty = _release_card_uncertainty(card.uncertainty)
    transportability = card.transportability
    lines = [
        f"# {card.title}", "",
        f"Generated: {card.created_at}", "",
        "## Intended use", "", str(card.intended_use), "",
        "## Governance contract", "",
        f"- Outcome: `{card.task.outcome}`",
        f"- Task type: `{card.task.task_type}`",
        f"- Generalization target: `{card.generalization_target}`",
        f"- Deployment status: `{card.deployment_status}`",
        "- Autonomous model selection: `FALSE`", "",
        "## Model", "",
        f"- Engine: `{card.engine}`",
        f"- Predictors: {', '.join(f'`{x}`' for x in card.predictors)}",
        f"- Training rows: {card.training_n}",
        f"- Training hash: `{card.training_hash}`", "",
        "## Resampling performance", "",
        *_markdown_table(_release_card_metrics(card)), "",
        "## Model-selection procedure", "",
    ]
    lines += _markdown_table(selection) if not selection.empty else [
        "No governed model-selection procedure was supplied."
    ]
    lines += ["", "## Target-aligned uncertainty", ""]
    lines += _markdown_table(uncertainty) if not uncertainty.empty else [
        "No target-aligned uncertainty object was supplied."
    ]
    if card.uncertainty is not None:
        lines += [
            "",
            f"Resampling unit: `{card.uncertainty.unit}`.",
            "",
            str(card.uncertainty.limitations),
        ]
    lines += ["", "## Calibration", ""]
    if isinstance(card.calibration, GP3MLCalibrationAssessment):
        lines += _markdown_table(card.calibration.summary)
    else:
        lines += ["No calibration assessment was supplied."]
    lines += [
        "", "## External validation and transportability", "",
        f"Status: **{card.external_validation_status}**",
        (
            "No independent external validation has been supplied."
            if transportability is None
            else str(transportability.reason)
        ),
        "", "## Limitations", "",
        *[f"- {x}" for x in card.limitations],
        "", "## Prohibited uses", "",
        *[f"- {x}" for x in card.prohibited_uses],
        "", "## Human oversight", "",
        "Predictions support scientific review only. Selection, interpretation, and any subsequent action remain subject to documented human review.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def _session_info() -> str:
    return f"Python {sys.version}\nPlatform: {platform.platform()}"


def create_gazepoint_release_evidence(
    objects: Mapping[str, Any] | None = None,
    files: Mapping[str, str | Path] | None = None,
    version: str = "0.2.0",
    notes: str | Sequence[str] = (),
) -> GP3MLReleaseEvidence:
    """Fingerprint named analysis objects and release files.

    The seemingly historical default ``version='0.2.0'`` intentionally matches
    the frozen gp3ml 0.3.0 R contract.
    """
    objects = {} if objects is None else dict(objects)
    if files is None:
        file_map: dict[str, str | Path] = {}
    elif isinstance(files, Mapping):
        file_map = dict(files)
    else:
        raise GP3MLError("`files` must be a named vector of existing paths.")
    missing = [name for name, value in file_map.items() if not Path(value).exists()]
    if missing:
        raise GP3MLError("`files` must be a named vector of existing paths.")
    object_hashes = {name: hash_jsonable(value, algorithm="md5") for name, value in objects.items()}
    file_md5 = {
        name: md5(Path(value).read_bytes()).hexdigest()
        for name, value in file_map.items()
    }
    return GP3MLReleaseEvidence(
        version=str(version),
        created_at=timestamp(),
        object_hashes=object_hashes,
        file_md5=file_md5,
        file_paths={name: str(value) for name, value in file_map.items()},
        session=_session_info(),
        notes=_as_character_vector(notes),
        prohibited_uses=gp3ml_prohibited_uses(),
    )


def _release_card_repr(self: GP3MLReleaseModelCard) -> str:
    uncertainty_unit = "NA" if self.uncertainty_unit is None else self.uncertainty_unit
    return (
        "<gp3ml_release_model_card>\n"
        f"  Outcome: {self.task.outcome}\n"
        f"  Target: {self.generalization_target}\n"
        f"  Selection recorded: {self.selection_procedure_recorded}\n"
        f"  Uncertainty unit: {uncertainty_unit}\n"
        f"  External validation: {self.external_validation_status}"
    )


def _release_evidence_repr(self: GP3MLReleaseEvidence) -> str:
    return (
        f"<gp3ml_release_evidence> version={self.version}\n"
        f"  Object hashes: {len(self.object_hashes)}\n"
        f"  File checksums: {len(self.file_md5)}"
    )


GP3MLReleaseModelCard.__repr__ = _release_card_repr  # type: ignore[method-assign]
GP3MLReleaseEvidence.__repr__ = _release_evidence_repr  # type: ignore[method-assign]
