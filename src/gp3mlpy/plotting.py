from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .exceptions import GP3MLError
from .objects import (
    GP3MLAPIStabilityAudit, GP3MLAbstentionAudit, GP3MLConformalCoverage,
    GP3MLDatasetShiftAudit, GP3MLEnvironmentComparison, GP3MLGovernanceProfileAudit,
    GP3MLHandoffValidation, GP3MLModelArtifactValidation, GP3MLModelRobustnessAudit,
    GP3MLPlanDeviationAudit, GP3MLReleaseChecksumValidation, GP3MLReproducibilityAudit,
    GP3MLResearchBundleValidation, GP3MLROCrateValidation, GP3MLThresholdEvaluation,
)


def _new_axes(ax: Axes | None = None) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _status_counts(table: pd.DataFrame, levels=("pass", "review", "fail")) -> list[int]:
    return [int((table["status"] == level).sum()) for level in levels]


def plot_threshold_evaluation(x: GP3MLThresholdEvaluation, metric: str = "balanced_accuracy", ax: Axes | None = None) -> Figure:
    if metric not in x.thresholds.columns: raise GP3MLError(f"Unknown metric `{metric}`.")
    fig, ax = _new_axes(ax); ax.plot(x.thresholds.threshold, x.thresholds[metric], marker="o")
    ax.set(xlabel="Decision threshold", ylabel=metric, title=f"Threshold evaluation: {metric}")
    return fig


def plot_abstention_audit(x: GP3MLAbstentionAudit, ax: Axes | None = None) -> Figure:
    fig, ax = _new_axes(ax); labels=["coverage","abstention","covered_error"]; values=[x.coverage,x.abstention_rate,x.covered_error_rate]
    ax.bar(labels, values); ax.set_ylim(0,1); ax.set(ylabel="Proportion",title="Governed abstention audit")
    return fig


def plot_conformal_coverage(x: GP3MLConformalCoverage, ax: Axes | None = None) -> Figure:
    labels=["nominal","row"]; values=[x.nominal_coverage,x.row_coverage]
    if x.get("unit_coverage") is not None and np.isfinite(x.unit_coverage): labels.append("unit"); values.append(x.unit_coverage)
    fig,ax=_new_axes(ax); ax.bar(labels,values); ax.set_ylim(0,1); ax.set(ylabel="Coverage",title="Conformal coverage audit"); return fig


def plot_dataset_shift_audit(x: GP3MLDatasetShiftAudit, ax: Axes | None = None) -> Figure:
    tab=x.findings
    if tab.empty: raise GP3MLError("No predictor shift findings are available.")
    vals=np.where(tab.type.eq("numeric"),pd.to_numeric(tab.standardized_difference,errors="coerce").abs(),pd.to_numeric(tab.distribution_statistic,errors="coerce"))
    fig,ax=_new_axes(ax); ax.barh(tab.predictor.astype(str),vals); ax.set(xlabel="Shift magnitude (SMD or total variation)",title="Predictor distribution shift"); return fig


def plot_environment_comparison(x: GP3MLEnvironmentComparison, ax: Axes | None = None) -> Figure:
    values=[int((x.packages.status=="pass").sum()+(x.core.status=="pass").sum()),int((x.packages.status=="review").sum()+(x.core.status=="review").sum())]
    fig,ax=_new_axes(ax); ax.bar(["pass","review"],values); ax.set(ylabel="Checks",title="Environment reproducibility comparison"); return fig


def _plot_check_counts(x, title: str, ax: Axes | None = None) -> Figure:
    fig,ax=_new_axes(ax); labels=["pass","review","fail"]; ax.bar(labels,_status_counts(x.checks)); ax.set(ylabel="Checks",title=title); return fig


def plot_handoff_validation(x: GP3MLHandoffValidation, ax: Axes | None = None) -> Figure: return _plot_check_counts(x,f"Handoff validation: {x.status}",ax)
def plot_model_artifact_validation(x: GP3MLModelArtifactValidation, ax: Axes | None = None) -> Figure: return _plot_check_counts(x,"Model artifact validation",ax)
def plot_research_bundle_validation(x: GP3MLResearchBundleValidation, ax: Axes | None = None) -> Figure: return _plot_check_counts(x,f"Integrated research workflow: {x.status}",ax)
def plot_ro_crate_validation(x: GP3MLROCrateValidation, ax: Axes | None = None) -> Figure: return _plot_check_counts(x,"Research-object validation",ax)


def plot_api_stability_audit(x: GP3MLAPIStabilityAudit, ax: Axes | None = None) -> Figure:
    fig,ax=_new_axes(ax); ax.bar(x.checks.check.astype(str),x.checks.n_issues); ax.tick_params(axis="x",rotation=90); ax.set(ylabel="Issues",title=f"gp3ml API stability: {x.status}"); fig.tight_layout(); return fig


def plot_model_robustness_audit(x: GP3MLModelRobustnessAudit, ax: Axes | None = None) -> Figure:
    if x.findings.empty: raise GP3MLError("No robustness findings are available.")
    labels=(x.findings.dimension.astype(str)+": "+x.findings.metric.astype(str)).tolist(); fig,ax=_new_axes(ax); ax.barh(labels,x.findings.indicator); ax.set(xlabel="Sensitivity indicator",title="Model robustness and stability"); return fig


def plot_plan_deviation_audit(x: GP3MLPlanDeviationAudit, ax: Axes | None = None) -> Figure:
    vals=[int((x.deviations.status=="pass").sum()),int((x.deviations.status=="deviation").sum())]; fig,ax=_new_axes(ax); ax.bar(["pass","deviation"],vals); ax.set(ylabel="Fields",title="Analysis-plan deviation audit"); return fig


def plot_release_checksum_validation(x: GP3MLReleaseChecksumValidation, ax: Axes | None = None) -> Figure:
    fig,ax=_new_axes(ax); ax.bar(["pass","fail"],[int((x.files.status=="pass").sum()),int((x.files.status=="fail").sum())]); ax.set(ylabel="Artifacts",title="Release checksum validation"); return fig


def plot_governance_profile_audit(x: GP3MLGovernanceProfileAudit, ax: Axes | None = None) -> Figure:
    fig,ax=_new_axes(ax); labels=["pass","review","fail"]; ax.bar(labels,_status_counts(x.controls)); ax.set(ylabel="Controls",title=f"Governance evidence: {x.framework}"); return fig


def plot_reproducibility_audit(x: GP3MLReproducibilityAudit, ax: Axes | None = None) -> Figure:
    fig,ax=_new_axes(ax)
    if x.summary.empty: ax.set_axis_off(); ax.set_title("No volatile artifact output detected")
    else: ax.bar(x.summary.issue.astype(str),x.summary.n); ax.tick_params(axis="x",rotation=90); ax.set(ylabel="Findings",title=f"Reproducibility audit: {x.status}"); fig.tight_layout()
    return fig


def plot_engine_capabilities(x: pd.DataFrame, ax: Axes | None = None) -> Figure:
    if x.attrs.get("r_class")!="gp3ml_engine_capabilities": raise GP3MLError("`x` must be created by gp3ml_engine_capabilities().")
    fig,ax=_new_axes(ax); ax.bar(x.engine.astype(str),x.package_available.astype(int)); ax.set_ylim(0,1); ax.set(ylabel="Package available",title="gp3ml engine portability"); ax.set_yticks([0,1],labels=["no","yes"]); return fig


_METHODS = {
    GP3MLThresholdEvaluation: plot_threshold_evaluation,
    GP3MLAbstentionAudit: plot_abstention_audit,
    GP3MLConformalCoverage: plot_conformal_coverage,
    GP3MLDatasetShiftAudit: plot_dataset_shift_audit,
    GP3MLEnvironmentComparison: plot_environment_comparison,
    GP3MLHandoffValidation: plot_handoff_validation,
    GP3MLModelArtifactValidation: plot_model_artifact_validation,
    GP3MLResearchBundleValidation: plot_research_bundle_validation,
    GP3MLROCrateValidation: plot_ro_crate_validation,
    GP3MLAPIStabilityAudit: plot_api_stability_audit,
    GP3MLModelRobustnessAudit: plot_model_robustness_audit,
    GP3MLPlanDeviationAudit: plot_plan_deviation_audit,
    GP3MLReleaseChecksumValidation: plot_release_checksum_validation,
    GP3MLGovernanceProfileAudit: plot_governance_profile_audit,
    GP3MLReproducibilityAudit: plot_reproducibility_audit,
}


def _bind_plot(cls, fn):
    def plot(self, *args, **kwargs): return fn(self,*args,**kwargs)
    plot.__name__="plot"; setattr(cls,"plot",plot)

for _cls,_fn in _METHODS.items(): _bind_plot(_cls,_fn)
