from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings as pywarnings

import numpy as np
import pandas as pd

from ._utils import hash_jsonable, seed_from, write_tables, worst_status
from .calibration import assess_gazepoint_calibration
from .exceptions import GP3MLError
from .metrics import gazepoint_performance_metrics
from .model_engines import fit_gazepoint_model
from .objects import (
    GP3MLObject, GP3MLResampleEvaluation, GP3MLResampleEvaluationValidation,
    GP3MLResamplePerformanceSummary, GP3MLTask,
)
from .resampling import GazepointGroupFolds, validate_gazepoint_group_folds
from .task_governance import assert_gp3ml_use_case, declare_gazepoint_task, validate_gazepoint_ml_roles


def _redeclare_task(data: pd.DataFrame, task: GP3MLTask) -> GP3MLTask:
    return declare_gazepoint_task(
        data=data, outcome=task.outcome, purpose=task.purpose, task_type=task.task_type,
        unit_id=task.unit_id, participant_id=task.participant_id, stimulus_id=task.stimulus_id,
        generalization_target=task.generalization_target, positive=task.positive,
        observed_outcome=task.observed_outcome, sensitive_outcome=False,
    )


def _metric_long(metrics: pd.DataFrame, identifiers: dict[str, Any] | None = None) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame) or len(metrics) != 1:
        return pd.DataFrame()
    numeric = [c for c in metrics.columns if pd.api.types.is_numeric_dtype(metrics[c]) and c not in {"n", "threshold"}]
    rows=[]
    identifiers={} if identifiers is None else identifiers
    for name in numeric:
        row=dict(identifiers); row.update({"metric":name,"value":float(metrics[name].iloc[0]),"n":int(metrics.n.iloc[0]) if "n" in metrics else np.nan,"threshold":float(metrics.threshold.iloc[0]) if "threshold" in metrics else np.nan}); rows.append(row)
    return pd.DataFrame(rows)


def _predictions_from_model(model, data, task):
    if task.task_type=="classification":
        probability=np.asarray(model.predict(data,type="probability"),dtype=float)
        prediction=model.predict(data,type="class")
        return prediction,probability
    return np.asarray(model.predict(data,type="response"),dtype=float),None


def _prediction_table(data,task,fold_object,prediction,probability,source_row_id,candidate_id=None,stage="assessment"):
    ids=[]
    for c in (source_row_id,task.unit_id,task.participant_id,task.stimulus_id):
        if c and c in data.columns and c not in ids: ids.append(c)
    out=data.loc[:,ids].copy()
    out["repeat"]=fold_object["repeat"]; out["fold"]=fold_object.fold; out["fold_id"]=fold_object.fold_id
    out["candidate_id"]=candidate_id; out["stage"]=stage; out["truth"]=[None if pd.isna(v) else str(v) for v in data[task.outcome]]
    out["prediction"]=[None if pd.isna(v) else str(v) for v in prediction]
    out["probability"]=np.nan if probability is None else probability
    out["prediction_missing"]=out.prediction.isna() & out.probability.isna()
    return out


def evaluate_gazepoint_group_folds(
    folds: GazepointGroupFolds,
    task: GP3MLTask,
    predictors: Sequence[str] | None = None,
    engine: Any = None,
    preprocessor_args: dict[str,Any] | None = None,
    engine_args: dict[str,Any] | None = None,
    threshold: float = .5,
    seed: int = 1,
    assess_calibration: bool = False,
    calibration_bins: int = 10,
    calibration_bootstrap: int = 0,
    keep_models: bool = False,
    continue_on_error: bool = True,
) -> GP3MLResampleEvaluation:
    """Fit fold-local models and evaluate assessment partitions only."""
    if not isinstance(folds,GazepointGroupFolds): raise GP3MLError("`folds` must be a `gazepoint_group_folds` object.")
    assert_gp3ml_use_case(task)
    validation=validate_gazepoint_group_folds(folds)
    if validation.status!="pass": raise GP3MLError("The grouped fold object must pass validation before evaluation.")
    if task.outcome!=folds.metadata["outcome"]: raise GP3MLError("The task outcome does not match the fold outcome.")
    if task.generalization_target!=folds.metadata["generalization_target"]: raise GP3MLError("The task generalization target does not match the folds.")
    predictors=list(folds.metadata["predictors"] if predictors is None else predictors)
    if not predictors: raise GP3MLError("At least one predictor is required.")
    if not set(predictors).issubset(set(folds.metadata["predictors"])): raise GP3MLError("Evaluation predictors must be declared in the fold metadata.")
    if preprocessor_args is not None and not isinstance(preprocessor_args,dict) or engine_args is not None and not isinstance(engine_args,dict): raise GP3MLError("`preprocessor_args` and `engine_args` must be lists.")
    preprocessor_args={} if preprocessor_args is None else dict(preprocessor_args); engine_args={} if engine_args is None else dict(engine_args)
    source_row_id=folds.metadata["source_row_id"]; fold_results=[]
    for fold_object in folds.folds.values():
        fold_seed=seed_from(seed,fold_object["repeat"],fold_object.fold)
        caught=[]; error=None; value=None
        try:
            with pywarnings.catch_warnings(record=True) as ws:
                pywarnings.simplefilter("always")
                analysis_task=_redeclare_task(fold_object.analysis,task)
                role_validation=validate_gazepoint_ml_roles(fold_object.analysis,analysis_task,predictors,feature_manifest=folds.feature_manifest)
                if role_validation.status=="fail": raise GP3MLError(f"Fold `{fold_object.fold_id}` failed role validation.")
                if getattr(fold_object.leakage_audit,"status",None)=="fail": raise GP3MLError(f"Fold `{fold_object.fold_id}` failed its stored leakage audit.")
                model=fit_gazepoint_model(fold_object.analysis,analysis_task,predictors,engine,preprocessor_args=preprocessor_args,engine_args=engine_args,seed=fold_seed,threshold=threshold)
                prediction,probability=_predictions_from_model(model,fold_object.assessment,task)
                prediction_table=_prediction_table(fold_object.assessment,task,fold_object,prediction,probability,source_row_id)
                metrics=gazepoint_performance_metrics(task,fold_object.assessment[task.outcome],prediction,probability,threshold)
                identifiers={"repeat":fold_object["repeat"],"fold":fold_object.fold,"fold_id":fold_object.fold_id}
                metrics_long=_metric_long(metrics,identifiers)
                calibration=None
                if assess_calibration and task.task_type=="classification":
                    calibration=assess_gazepoint_calibration(fold_object.assessment[task.outcome],probability,task.positive,int(calibration_bins),int(calibration_bootstrap),seed=seed_from(fold_seed,"calibration"))
                    cal=pd.DataFrame([{"ece":float(calibration.summary.ece.iloc[0]),"calibration_intercept_abs":abs(float(calibration.summary.intercept.iloc[0])),"calibration_slope_abs_error":abs(float(calibration.summary.slope.iloc[0])-1)}])
                    metrics_long=pd.concat([metrics_long,_metric_long(cal,identifiers)],ignore_index=True)
                caught=list(dict.fromkeys(str(w.message) for w in ws))
                value={"model":model,"predictions":prediction_table,"metrics":metrics,"metrics_long":metrics_long,"calibration":calibration,"role_validation":role_validation,"analysis_hash":hash_jsonable(fold_object.analysis.loc[:,[task.outcome,*predictors]],algorithm="md5"),"assessment_hash":hash_jsonable(fold_object.assessment.loc[:,[task.outcome,*predictors]],algorithm="md5")}
        except Exception as exc:
            error=str(exc)
        failed=error is not None
        status="fail" if failed else "review" if caught or fold_object.leakage_audit.status=="review" or value["role_validation"].status=="review" else "pass"
        fr={"repeat":fold_object["repeat"],"fold":fold_object.fold,"fold_id":fold_object.fold_id,"status":status,"error":error,"warnings":caught,"messages":[],"n_analysis":len(fold_object.analysis),"n_assessment":len(fold_object.assessment),"n_excluded":len(fold_object.excluded),"assessment_class_support":fold_object.assessment[task.outcome].value_counts(dropna=False,sort=False).to_dict(),"analysis_class_support":fold_object.analysis[task.outcome].value_counts(dropna=False,sort=False).to_dict(),"leakage_status":fold_object.leakage_audit.status,"leakage_audit":fold_object.leakage_audit,"model":value["model"] if not failed and keep_models else None,"predictions":value["predictions"] if not failed else pd.DataFrame(),"metrics":value["metrics"] if not failed else pd.DataFrame(),"metrics_long":value["metrics_long"] if not failed else pd.DataFrame(),"calibration":value["calibration"] if not failed else None,"role_validation":value["role_validation"] if not failed else None,"analysis_hash":value["analysis_hash"] if not failed else None,"assessment_hash":value["assessment_hash"] if not failed else None,"excluded":fold_object.excluded.copy()}
        fold_results.append(fr)
        if failed and not continue_on_error: raise GP3MLError(f"Fold `{fold_object.fold_id}` failed: {error}")
    status_rows=[]; preds=[]; mets=[]; excluded=[]
    for r in fold_results:
        status_rows.append({"repeat":r["repeat"],"fold":r["fold"],"fold_id":r["fold_id"],"status":r["status"],"leakage_status":r["leakage_status"],"n_analysis":r["n_analysis"],"n_assessment":r["n_assessment"],"n_excluded":r["n_excluded"],"n_predictions":len(r["predictions"]),"n_missing_predictions":int(r["predictions"].prediction_missing.sum()) if len(r["predictions"]) else r["n_assessment"],"warning_count":len(r["warnings"]),"error":r["error"],"warnings":" | ".join(r["warnings"])})
        if len(r["predictions"]): preds.append(r["predictions"])
        if len(r["metrics_long"]): mets.append(r["metrics_long"])
        if len(r["excluded"]):
            ex=r["excluded"].copy(); ex["repeat"]=r["repeat"]; ex["fold"]=r["fold"]; ex["fold_id"]=r["fold_id"]; excluded.append(ex)
    engine_name=engine.name if isinstance(engine,GP3MLObject) and getattr(engine,"r_class",None)=="gp3ml_engine" else (engine if engine is not None else ("glm" if task.task_type=="classification" else "lm"))
    obj=GP3MLResampleEvaluation(fold_results=fold_results,predictions=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame(),metrics=pd.concat(mets,ignore_index=True) if mets else pd.DataFrame(),fold_status=pd.DataFrame(status_rows),excluded=pd.concat(excluded,ignore_index=True) if excluded else pd.DataFrame(),task=task,predictors=predictors,engine=engine_name,preprocessor_args=preprocessor_args,engine_args=engine_args,threshold=threshold,seed=seed,generalization_target=folds.metadata["generalization_target"],folds_metadata=folds.metadata,folds_audit=folds.audit,folds_validation=validation,keep_models=keep_models,call="evaluate_gazepoint_group_folds")
    obj.validation=validate_gazepoint_resample_evaluation(obj); return obj


def collect_gazepoint_fold_predictions(x,include_failed=True):
    if not isinstance(x,GP3MLResampleEvaluation): raise GP3MLError("`x` must be a `gp3ml_resample_evaluation` object.")
    out=x.predictions.copy()
    if include_failed:
        failed=x.fold_status[x.fold_status.status=="fail"]
        if len(failed):
            placeholders=failed[["repeat","fold","fold_id","status","error"]].copy(); placeholders["stage"]="assessment"; placeholders["prediction_missing"]=True
            out=pd.concat([out,placeholders],ignore_index=True,sort=False)
    return out


def summarize_gazepoint_resample_performance(x,aggregation="fold_distribution",conf_level=.95):
    if not isinstance(x,GP3MLObject) or x.r_class not in {"gp3ml_resample_evaluation","gp3ml_nested_evaluation"}: raise GP3MLError("`x` must be a grouped or nested gp3ml evaluation object.")
    if aggregation not in {"fold_distribution","pooled_rows"}: raise GP3MLError("Unknown aggregation.")
    if x.r_class=="gp3ml_nested_evaluation" and aggregation=="pooled_rows": raise GP3MLError("Nested evaluations use candidate-specific thresholds; summarize their outer-fold distribution instead of pooling rows.")
    if aggregation=="fold_distribution":
        rows=[]; alpha=(1-conf_level)/2
        for metric in pd.unique(x.metrics.metric):
            vals=pd.to_numeric(x.metrics.loc[x.metrics.metric==metric,"value"],errors="coerce"); finite=vals[np.isfinite(vals)]
            direction="minimize" if metric in {"rmse","mae","log_loss","brier","ece","calibration_intercept_abs","calibration_slope_abs_error"} else "maximize"
            rows.append({"metric":metric,"direction":direction,"n_folds":len(finite),"mean":float(finite.mean()) if len(finite) else np.nan,"median":float(finite.median()) if len(finite) else np.nan,"sd":float(finite.std(ddof=1)) if len(finite)>1 else np.nan,"lower":float(finite.quantile(alpha)) if len(finite) else np.nan,"upper":float(finite.quantile(1-alpha)) if len(finite) else np.nan})
        summary=pd.DataFrame(rows)
    else:
        if x.predictions.empty: raise GP3MLError("No predictions are available to pool.")
        if x.task.task_type=="classification": metrics=gazepoint_performance_metrics(x.task,x.predictions.truth,x.predictions.prediction,x.predictions.probability,x.threshold)
        else: metrics=gazepoint_performance_metrics(x.task,pd.to_numeric(x.predictions.truth),pd.to_numeric(x.predictions.prediction),threshold=x.threshold)
        summary=_metric_long(metrics); summary["aggregation_warning"]="Pooled row-level metrics summarize assessment predictions and are not participant- or stimulus-level estimates."
    return GP3MLResamplePerformanceSummary(summary=summary,aggregation=aggregation,conf_level=conf_level,generalization_target=x.generalization_target,n_folds=len(x.fold_status),n_failed_folds=int((x.fold_status.status=="fail").sum()),task=x.task)


def validate_gazepoint_resample_evaluation(x):
    if not isinstance(x,GP3MLResampleEvaluation): raise GP3MLError("`x` must be a `gp3ml_resample_evaluation` object.")
    expected=x.folds_metadata["n_folds_total"]
    checks=[
      ("fold_status_complete","pass" if len(x.fold_status)==expected else "fail",f"Recorded {len(x.fold_status)} of {expected} expected fold statuses."),
      ("assessment_only_predictions","pass" if x.predictions.empty or (x.predictions.stage=="assessment").all() else "fail","Predictions are restricted to assessment partitions."),
      ("prediction_coverage","review" if (x.fold_status.n_missing_predictions>0).any() else "pass",f"{int(x.fold_status.n_missing_predictions.sum())} assessment predictions are missing across folds."),
      ("fold_failures","review" if (x.fold_status.status=="fail").any() else "pass",f"{int((x.fold_status.status=='fail').sum())} folds failed and remain explicitly retained."),
      ("generalization_target_preserved","pass" if x.generalization_target==x.task.generalization_target else "fail",f"Generalization target: {x.generalization_target}."),
    ]
    c=pd.DataFrame(checks,columns=["check_id","status","message"]); return GP3MLResampleEvaluationValidation(status=worst_status(c.status),checks=c,issues=c[c.status!="pass"].reset_index(drop=True))


def write_gazepoint_resample_evaluation(x,directory,prefix="gazepoint_resample_evaluation",overwrite=False):
    if not isinstance(x,GP3MLResampleEvaluation): raise GP3MLError("`x` must be a `gp3ml_resample_evaluation` object.")
    return write_tables({"fold_status":x.fold_status,"predictions":x.predictions,"metrics":x.metrics,"excluded":x.excluded,"validation":x.validation.checks},directory,prefix,overwrite)
