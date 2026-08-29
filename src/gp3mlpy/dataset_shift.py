from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from ._utils import assert_data, worst_status
from .exceptions import GP3MLError
from .objects import GP3MLDatasetShiftAudit, GP3MLMissingnessShiftAudit, GP3MLShiftSummary

_DEFAULT_THRESHOLDS={"smd_review":.20,"smd_fail":.50,"outside_review":.05,"outside_fail":.20,"tv_review":.20,"tv_fail":.40}

def _numeric_shift_row(name:str,development:pd.DataFrame,external:pd.DataFrame,thresholds:dict[str,float])->dict[str,Any]:
    d=pd.to_numeric(development[name],errors="coerce").to_numpy(dtype=float);e=pd.to_numeric(external[name],errors="coerce").to_numpy(dtype=float);d_ok=d[np.isfinite(d)];e_ok=e[np.isfinite(e)];var_d=np.var(d_ok,ddof=1) if len(d_ok)>1 else np.nan;var_e=np.var(e_ok,ddof=1) if len(e_ok)>1 else np.nan;sd_pool=np.sqrt((var_d+var_e)/2);smd=(float(np.mean(e_ok))-float(np.mean(d_ok)))/sd_pool if np.isfinite(sd_pool) and sd_pool>0 else 0.0;outside=float(np.mean((e_ok<np.min(d_ok))|(e_ok>np.max(d_ok)))) if len(d_ok) and len(e_ok) else np.nan;ks=float(ks_2samp(d_ok,e_ok,method="auto").statistic) if len(np.unique(d_ok))>1 and len(np.unique(e_ok))>1 else np.nan;severity="fail" if abs(smd)>=thresholds["smd_fail"] or (np.isfinite(outside) and outside>=thresholds["outside_fail"]) else ("review" if abs(smd)>=thresholds["smd_review"] or (np.isfinite(outside) and outside>=thresholds["outside_review"]) else "pass")
    return {"predictor":name,"type":"numeric","development_n":len(d_ok),"external_n":len(e_ok),"development_missing":float(np.mean(pd.isna(development[name]))),"external_missing":float(np.mean(pd.isna(external[name]))),"mean_development":float(np.mean(d_ok)) if len(d_ok) else np.nan,"mean_external":float(np.mean(e_ok)) if len(e_ok) else np.nan,"standardized_difference":float(smd),"distribution_statistic":ks,"support_overlap":np.nan if not np.isfinite(outside) else 1-outside,"outside_training_range":outside,"novel_levels":"","status":severity}

def _categorical_shift_row(name:str,development:pd.DataFrame,external:pd.DataFrame,thresholds:dict[str,float])->dict[str,Any]:
    d=development[name];e=external[name];d_values=[str(x) for x in d if pd.notna(x)];e_values=[str(x) for x in e if pd.notna(x)];levels_all=sorted(set(d_values+e_values));pdist=np.asarray([d_values.count(level)/len(d_values) if d_values else np.nan for level in levels_all],dtype=float);edist=np.asarray([e_values.count(level)/len(e_values) if e_values else np.nan for level in levels_all],dtype=float);tv=float(.5*np.sum(np.abs(pdist-edist))) if len(levels_all) else 0.0;novel=sorted(set(e_values)-set(d_values));severity="fail" if novel or tv>=thresholds["tv_fail"] else ("review" if tv>=thresholds["tv_review"] else "pass")
    return {"predictor":name,"type":"categorical","development_n":len(d_values),"external_n":len(e_values),"development_missing":float(d.isna().mean()),"external_missing":float(e.isna().mean()),"mean_development":np.nan,"mean_external":np.nan,"standardized_difference":np.nan,"distribution_statistic":tv,"support_overlap":1-tv,"outside_training_range":np.nan,"novel_levels":", ".join(novel),"status":severity}

def audit_gazepoint_dataset_shift(development:pd.DataFrame,external:pd.DataFrame,predictors:Sequence[str]|None=None,thresholds:Mapping[str,float]|None=None)->GP3MLDatasetShiftAudit:
    assert_data(development,"development",min_rows=0);assert_data(external,"external",min_rows=0)
    if predictors is None:predictors=[x for x in development.columns if x in external.columns]
    predictors=list(dict.fromkeys(str(x) for x in predictors));missing=[x for x in predictors if x not in development.columns or x not in external.columns]
    if missing:raise GP3MLError("Predictors absent from one dataset: "+", ".join(missing)+".")
    threshold_values=dict(_DEFAULT_THRESHOLDS);threshold_values.update({} if thresholds is None else dict(thresholds));rows=[]
    for name in predictors:rows.append(_numeric_shift_row(name,development,external,threshold_values) if pd.api.types.is_numeric_dtype(development[name]) and pd.api.types.is_numeric_dtype(external[name]) else _categorical_shift_row(name,development,external,threshold_values))
    findings=pd.DataFrame(rows);return GP3MLDatasetShiftAudit(status="review" if findings.empty else worst_status(findings.status),findings=findings,thresholds=threshold_values,terminology=["schema shift","missingness shift","covariate shift","prevalence shift","calibration drift","performance degradation"],note="Predictor-distribution shift is reported separately from outcome prevalence, calibration, and performance.")

def audit_gazepoint_missingness_shift(development:pd.DataFrame,external:pd.DataFrame,predictors:Sequence[str]|None=None,review_delta:float=.10,fail_delta:float=.25)->GP3MLMissingnessShiftAudit:
    assert_data(development,"development",min_rows=0);assert_data(external,"external",min_rows=0)
    if predictors is None:predictors=[x for x in development.columns if x in external.columns]
    rows=[]
    for name in dict.fromkeys(str(x) for x in predictors):
        if name not in development.columns or name not in external.columns:raise GP3MLError(f"Predictor `{name}` is absent from one dataset.")
        d=float(development[name].isna().mean());e=float(external[name].isna().mean());delta=e-d;status="fail" if abs(delta)>=fail_delta else ("review" if abs(delta)>=review_delta else "pass");rows.append({"predictor":name,"development_missing":d,"external_missing":e,"delta":delta,"status":status})
    findings=pd.DataFrame(rows);return GP3MLMissingnessShiftAudit(status="review" if findings.empty else worst_status(findings.status),findings=findings,review_delta=review_delta,fail_delta=fail_delta)

def _status_counts(findings:pd.DataFrame)->pd.DataFrame:
    if findings.empty or "status" not in findings:return pd.DataFrame(columns=["status","n_predictors"])
    counts=findings.status.value_counts(sort=False).rename_axis("status").reset_index(name="n_predictors");return counts.sort_values("status",kind="stable").reset_index(drop=True)

def summarize_gazepoint_shift(shift:GP3MLDatasetShiftAudit,missingness:GP3MLMissingnessShiftAudit|None=None)->GP3MLShiftSummary:
    if not isinstance(shift,GP3MLDatasetShiftAudit):raise GP3MLError("`shift` must be a dataset-shift audit.")
    out={"dataset_shift_status":shift.status,"dataset_shift_counts":_status_counts(shift.findings)}
    if missingness is not None:
        if not isinstance(missingness,GP3MLMissingnessShiftAudit):raise GP3MLError("Invalid missingness audit.")
        out["missingness_shift_status"]=missingness.status;out["missingness_shift_counts"]=_status_counts(missingness.findings)
    return GP3MLShiftSummary(**out)
