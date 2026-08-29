from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from ._utils import assert_data, hash_jsonable, timestamp
from .calibration import assess_gazepoint_calibration
from .exceptions import GP3MLError
from .metrics import gazepoint_classification_metrics, gazepoint_regression_metrics
from .objects import GP3MLExternalValidation,GP3MLExternalValidationReport,GP3MLMetricUncertainty,GP3MLModel,GP3MLModelCard,GP3MLReproducibilityReport,GP3MLResampleEvaluation
from .task_governance import assert_gp3ml_use_case, gp3ml_prohibited_uses

def _markdown_table(x:pd.DataFrame)->list[str]:
    if not isinstance(x,pd.DataFrame) or x.empty:return ["_No rows._"]
    cols=[str(c) for c in x.columns];lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    for _,row in x.iterrows():
        vals=[]
        for v in row:
            s="" if pd.isna(v) else str(v);vals.append(s.replace("|","\\|"))
        lines.append("| "+" | ".join(vals)+" |")
    return lines

def _safe_write(path:str|Path,overwrite:bool)->Path:
    p=Path(path).expanduser()
    if p.exists() and not overwrite:raise GP3MLError(f"File exists: {p}.")
    p.parent.mkdir(parents=True,exist_ok=True);return p

def create_gazepoint_model_card(model:GP3MLModel,intended_use:str,evaluation=None,calibration=None,feature_manifest=None,external_validation=None,limitations=(),ethical_review=None):
    if not isinstance(model,GP3MLModel):raise GP3MLError("`model` must be a fitted gp3ml model.")
    assert_gp3ml_use_case(model.task);return GP3MLModelCard(title=f"Model card: {model.task.outcome}",created_at=timestamp(),intended_use=intended_use,prohibited_uses=gp3ml_prohibited_uses(),task=model.task,engine=model.engine,predictors=model.predictors,training_n=model.training_n,training_hash=model.training_hash,evaluation=evaluation,calibration=calibration,feature_manifest=feature_manifest,external_validation=external_validation,limitations=list(limitations) if not isinstance(limitations,str) else [limitations],ethical_review=ethical_review)

def _model_card_markdown(card:GP3MLModelCard)->list[str]:
    if isinstance(card.evaluation,GP3MLResampleEvaluation):metrics=card.evaluation.metrics
    elif isinstance(card.evaluation,GP3MLMetricUncertainty):metrics=card.evaluation.intervals
    elif isinstance(card.evaluation,pd.DataFrame):metrics=card.evaluation
    else:metrics=pd.DataFrame()
    calibration=card.calibration.summary if getattr(card.calibration,"r_class",None)=="gp3ml_calibration_assessment" else pd.DataFrame();lines=[f"# {card.title}","",f"Generated: {card.created_at}","","## Intended use","",str(card.intended_use),"","## Task contract","",f"- Outcome: `{card.task.outcome}`",f"- Type: `{card.task.task_type}`",f"- Unit: `{card.task.unit_id}`",f"- Generalization target: `{card.task.generalization_target}`",f"- Scientific purpose: {card.task.purpose}","","## Model","",f"- Engine: `{card.engine}`",f"- Training rows: {card.training_n}",f"- Training hash: `{card.training_hash}`",f"- Predictors: {', '.join(f'`{p}`' for p in card.predictors)}","","## Performance",""];lines+=_markdown_table(metrics)+["","## Calibration",""]+_markdown_table(calibration)+["","## Prohibited uses",""]+[f"- {x}" for x in card.prohibited_uses]+["","## Limitations",""];lines += [f"- {x}" for x in card.limitations] if card.limitations else ["- No limitations were supplied; this must be completed before deployment."];lines += ["","## External validation","","No independent external validation has been supplied." if card.external_validation is None else "An external-validation report is attached to the card.","","## Human oversight","","Predictions must support research review rather than autonomous consequential decisions."];return lines

def _json_ready(x:Any)->Any:
    if x is None:return None
    if isinstance(x,(str,int,float,bool)):return None if isinstance(x,float) and not np.isfinite(x) else x
    if isinstance(x,np.generic):return _json_ready(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,pd.DataFrame):return [{str(k):_json_ready(v) for k,v in row.items()} for row in x.to_dict(orient="records")]
    if isinstance(x,pd.Series):return [_json_ready(v) for v in x.tolist()]
    if isinstance(x,pd.Categorical):return [_json_ready(v) for v in x.astype(object)]
    if isinstance(x,dict):return {str(k):_json_ready(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [_json_ready(v) for v in x]
    if hasattr(x,"to_dict") and hasattr(x,"r_class"):return {"r_class":x.r_class,**{k:_json_ready(v) for k,v in x.to_dict().items()}}
    if callable(x):return "<function>"
    return str(x)

def write_gazepoint_model_card(card,path,format="markdown",overwrite=False):
    if not isinstance(card,GP3MLModelCard):raise GP3MLError("`card` must be created by `create_gazepoint_model_card()`.")
    if format not in {"markdown","json"}:raise GP3MLError("`format` must be one of: markdown, json.")
    p=_safe_write(path,overwrite);p.write_text("\n".join(_model_card_markdown(card))+"\n",encoding="utf-8") if format=="markdown" else p.write_text(json.dumps(_json_ready(card),indent=2,ensure_ascii=False)+"\n",encoding="utf-8");return str(p)

def _shift_diagnostics(model:GP3MLModel,external_data:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for name in model.predictors:
        train=model.predictor_summary[name];x=external_data[name]
        if train["type"]=="numeric":sd=train["sd"];smd=np.nan if not np.isfinite(sd) or sd==0 else (float(pd.to_numeric(x,errors="coerce").mean())-train["mean"])/sd;rows.append({"feature":name,"type":"numeric","standardized_mean_difference":smd,"novel_levels":None,"external_missing":int(x.isna().sum())})
        else:novel=[v for v in pd.unique(x.dropna().astype(str)) if v not in train["levels"]];rows.append({"feature":name,"type":"categorical","standardized_mean_difference":np.nan,"novel_levels":", ".join(novel),"external_missing":int(x.isna().sum())})
    return pd.DataFrame(rows)

def evaluate_external_validation(model:GP3MLModel,external_data:pd.DataFrame,label="external",threshold=None,bootstrap=200,seed=1):
    if not isinstance(model,GP3MLModel):raise GP3MLError("`model` must be a fitted gp3ml model.")
    assert_data(external_data,name="external_data");assert_gp3ml_use_case(model.task,external_data);threshold=model.threshold if threshold is None else threshold
    if model.task.task_type=="classification":probability=np.asarray(model.predict(external_data,type="probability"),dtype=float);prediction=model.predict(external_data,type="class");metrics=gazepoint_classification_metrics(external_data[model.task.outcome],probability,prediction,model.task.positive,threshold);calibration=assess_gazepoint_calibration(external_data[model.task.outcome],probability,model.task.positive,bootstrap=int(bootstrap),seed=seed);predictions=pd.DataFrame({"truth":[str(v) for v in external_data[model.task.outcome]],"prediction":[str(v) for v in prediction],"probability":probability})
    else:prediction=np.asarray(model.predict(external_data),dtype=float);metrics=gazepoint_regression_metrics(external_data[model.task.outcome],prediction);calibration=None;predictions=pd.DataFrame({"truth":external_data[model.task.outcome].to_numpy(),"prediction":prediction})
    subset=external_data.loc[:,[model.task.outcome,*model.predictors]];return GP3MLExternalValidation(label=label,created_at=timestamp(),metrics=metrics,calibration=calibration,shift=_shift_diagnostics(model,external_data),predictions=predictions,external_hash=hash_jsonable(subset,algorithm="md5"),task=model.task,model_engine=model.engine)

def create_external_validation_report(validation,development_metrics=None,limitations=()):
    if not isinstance(validation,GP3MLExternalValidation):raise GP3MLError("Supply `evaluate_external_validation()` output.")
    return GP3MLExternalValidationReport(validation=validation,development_metrics=development_metrics,limitations=[limitations] if isinstance(limitations,str) else list(limitations),prohibited_uses=gp3ml_prohibited_uses())

def write_external_validation_report(report,path,overwrite=False):
    if not isinstance(report,GP3MLExternalValidationReport):raise GP3MLError("`report` must be an external-validation report.")
    p=_safe_write(path,overwrite);v=report.validation;lines=[f"# External validation: {v.label}","",f"Generated: {v.created_at}","","## External performance",""]+_markdown_table(v.metrics)+["","## Development performance",""];lines += ["Not supplied."] if report.development_metrics is None else _markdown_table(report.development_metrics);lines += ["","## Calibration",""]+(["Not applicable."] if v.calibration is None else _markdown_table(v.calibration.summary))+["","## Predictor shift",""]+_markdown_table(v.shift)+["","## Dataset fingerprint","",f"`{v.external_hash}`","","## Limitations",""];lines += [f"- {x}" for x in report.limitations] if report.limitations else ["- External representativeness and transportability require substantive review."];lines += ["","## Prohibited uses",""]+[f"- {x}" for x in report.prohibited_uses];p.write_text("\n".join(lines)+"\n",encoding="utf-8");return str(p)

def _git_info(project_path:str|Path)->dict[str,Any]:
    path=Path(project_path);out={"commit":None,"branch":None,"clean":None}
    if not (path/".git").exists():return out
    def cmd(*args):
        try:return subprocess.run(["git","-C",str(path),*args],capture_output=True,text=True,check=True).stdout.strip()
        except Exception:return None
    out["commit"]=cmd("rev-parse","HEAD");out["branch"]=cmd("branch","--show-current");status=cmd("status","--porcelain");out["clean"]=None if status is None else status=="";return out

def create_gazepoint_reproducibility_report(objects=None,data=None,seeds=None,notes=(),project_path=None):
    objects={} if objects is None else dict(objects);seeds={} if seeds is None else dict(seeds);project_path=os.getcwd() if project_path is None else project_path;object_hashes={name:hash_jsonable(value,algorithm="md5") for name,value in objects.items()};data_hash=None if data is None else hash_jsonable(data,algorithm="md5");session=f"Python {sys.version}\nPlatform: {platform.platform()}";return GP3MLReproducibilityReport(created_at=timestamp(),python_version=sys.version.split()[0],platform=platform.platform(),session=session,object_hashes=object_hashes,data_hash=data_hash,seeds=seeds,git=_git_info(project_path),notes=[notes] if isinstance(notes,str) else list(notes),prohibited_uses=gp3ml_prohibited_uses())

def write_gazepoint_reproducibility_report(report,path,overwrite=False):
    if not isinstance(report,GP3MLReproducibilityReport):raise GP3MLError("`report` must be a gp3ml reproducibility report.")
    p=_safe_write(path,overwrite);lines=["# gp3mlpy reproducibility report","",f"Generated: {report.created_at}","","## Runtime","",f"- Python: {report.python_version}",f"- Platform: {report.platform}","","## Git","",f"- Branch: `{report.git.get('branch')}`",f"- Commit: `{report.git.get('commit')}`",f"- Clean: {report.git.get('clean')}","","## Fingerprints","",f"- Data: `{report.data_hash}`"];lines += [f"- {k}: `{v}`" for k,v in report.object_hashes.items()] if report.object_hashes else ["- No objects supplied."];lines += ["","## Seeds",""]+([f"- {k}: {v}" for k,v in report.seeds.items()] if report.seeds else ["- No seeds supplied."])+["","## Notes",""]+([f"- {x}" for x in report.notes] if report.notes else ["- None."])+["","## Session information","","```",report.session,"```","","## Prohibited uses",""]+[f"- {x}" for x in report.prohibited_uses];p.write_text("\n".join(lines)+"\n",encoding="utf-8");return str(p)
