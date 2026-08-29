from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Callable
import numpy as np
import pandas as pd
from .exceptions import GP3MLError
from .objects import GP3MLModelRobustnessAudit, GP3MLStabilityEvaluation, GP3MLThresholdEvaluation, GP3MLThresholdStability
from ._utils import worst_status


def _numeric_metrics(result: Any) -> dict[str,float]:
    if isinstance(result, Mapping):
        vals = {str(k): float(v) for k,v in result.items() if isinstance(v,(int,float,np.number)) and not isinstance(v,(bool,np.bool_))}
        if vals: return vals
    if isinstance(result, pd.Series):
        vals={str(k):float(v) for k,v in result.items() if isinstance(v,(int,float,np.number))};
        if vals:return vals
    if isinstance(result,pd.DataFrame) and len(result)==1:
        vals={str(c):float(result.iloc[0][c]) for c in result.columns if pd.api.types.is_numeric_dtype(result[c])};
        if vals:return vals
    raise GP3MLError("Evaluator must return a named numeric vector or one-row data frame of numeric metrics.")


def _stability(kind: str, labels, label_name: str, evaluator: Callable[...,Any], call_name: str, **kwargs):
    if not callable(evaluator): raise GP3MLError("`evaluator` must be a function.")
    rows=[]
    for label in labels:
        call={call_name:label, **kwargs}; vals=_numeric_metrics(evaluator(**call)); rows.append({label_name:label,**vals})
    tab=pd.DataFrame(rows); metrics=[c for c in tab.columns if c!=label_name]
    summary=pd.DataFrame([{"metric":m,"minimum":float(tab[m].min(skipna=True)),"maximum":float(tab[m].max(skipna=True)),"sd":float(tab[m].std(ddof=1,skipna=True))} for m in metrics])
    return GP3MLStabilityEvaluation(kind=kind,results=tab,summary=summary)


def evaluate_gazepoint_seed_stability(seeds, evaluator, **kwargs):
    unique=list(dict.fromkeys(int(x) for x in seeds)); return _stability("seed",unique,"seed",evaluator,"seed",**kwargs)

def evaluate_gazepoint_feature_stability(features, evaluator, **kwargs):
    unique=list(dict.fromkeys(str(x) for x in features)); return _stability("feature",unique,"excluded_feature",evaluator,"excluded_feature",**kwargs)

def evaluate_gazepoint_missingness_sensitivity(scenarios, evaluator, **kwargs):
    if not isinstance(scenarios, Mapping) or not scenarios or any(not str(k) for k in scenarios): raise GP3MLError("`scenarios` must be a named list.")
    rows=[]
    for name,scenario in scenarios.items(): rows.append({"scenario":name, **_numeric_metrics(evaluator(scenario=scenario,name=name,**kwargs))})
    tab=pd.DataFrame(rows); metrics=[c for c in tab.columns if c!="scenario"]
    summary=pd.DataFrame([{"metric":m,"minimum":float(tab[m].min(skipna=True)),"maximum":float(tab[m].max(skipna=True)),"sd":float(tab[m].std(ddof=1,skipna=True))} for m in metrics])
    return GP3MLStabilityEvaluation(kind="missingness",results=tab,summary=summary)

def evaluate_gazepoint_threshold_stability(evaluation, metric, direction="maximize", tolerance=.02):
    if not isinstance(evaluation,GP3MLThresholdEvaluation): raise GP3MLError("Invalid threshold evaluation.")
    if direction not in {"maximize","minimize"}: raise GP3MLError("`direction` must be one of: maximize, minimize.")
    tab=evaluation.thresholds; 
    if metric not in tab.columns: raise GP3MLError("Unknown metric.")
    value=pd.to_numeric(tab[metric],errors="coerce"); optimum=float(value.max() if direction=="maximize" else value.min()); scale=max(abs(optimum),np.finfo(float).eps)
    near=value >= optimum-tolerance*scale if direction=="maximize" else value <= optimum+tolerance*scale
    selected=tab.loc[near].copy(); span=np.array([float(selected.threshold.min()),float(selected.threshold.max())]); width=float(span[1]-span[0]); status="stable" if width>=.10 else ("review" if width>=.04 else "unstable")
    return GP3MLThresholdStability(status=status,metric=metric,optimum=optimum,tolerance=tolerance,near_optimal=selected,threshold_span=span)

def audit_gazepoint_model_robustness(seed_stability=None,feature_stability=None,threshold_stability=None,missingness_stability=None,relative_sd_review=.05,relative_sd_fail=.15):
    rows=[]
    for dim,obj in {"seed":seed_stability,"feature":feature_stability,"threshold":threshold_stability,"missingness":missingness_stability}.items():
        if obj is None: continue
        if isinstance(obj,GP3MLThresholdStability):
            rows.append({"dimension":dim,"metric":obj.metric,"indicator":float(np.diff(obj.threshold_span)[0]),"status":{"stable":"pass","review":"review","unstable":"fail"}[obj.status]})
        elif isinstance(obj,GP3MLStabilityEvaluation):
            for _,s in obj.summary.iterrows():
                center=np.nanmean([abs(s.minimum),abs(s.maximum)]); rel=float(s.sd/center if np.isfinite(center) and center>0 else s.sd)
                status="review" if not np.isfinite(rel) else ("fail" if rel>=relative_sd_fail else ("review" if rel>=relative_sd_review else "pass"))
                rows.append({"dimension":dim,"metric":s.metric,"indicator":rel,"status":status})
    findings=pd.DataFrame(rows,columns=["dimension","metric","indicator","status"])
    return GP3MLModelRobustnessAudit(status="review" if findings.empty else worst_status(findings.status),findings=findings,note="Robustness statuses summarize sensitivity diagnostics; they are not proof of model validity.")
