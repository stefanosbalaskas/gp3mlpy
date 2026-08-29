from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import pandas as pd
from .exceptions import GP3MLError
from .governance_reports import _markdown_table
from .objects import GP3MLGovernanceProfile, GP3MLGovernanceProfileAudit
from ._utils import worst_status

def _domains(framework:str)->pd.DataFrame:
    controls=["scientific purpose","intended and prohibited use","data and feature provenance","generalization target","leakage-resistant validation","performance and calibration","prediction-level uncertainty","external validation and shift","human oversight and decision rule","reproducibility and artifact provenance"];keys=["task","task","feature_manifest","task","folds","performance","conformal","transportability","decision_rule","research_artifact"]
    if framework=="gp3ml-native":domains=["purpose","governance","provenance","generalization","validation","evaluation","uncertainty","transportability","oversight","reproducibility"]
    elif framework=="NIST-AI-RMF-1.0":domains=["GOVERN/MAP","GOVERN/MAP","MAP","MAP","MEASURE","MEASURE","MEASURE","MEASURE/MANAGE","GOVERN/MANAGE","GOVERN/MANAGE"]
    elif framework=="ISO-23894-oriented":domains=["AI risk-management evidence"]*10
    else:domains=["AI management-system evidence"]*10
    return pd.DataFrame({"control":controls,"evidence_key":keys,"framework_domain":domains})

def create_gp3ml_governance_profile(evidence:Mapping[str,object],framework:str="gp3ml-native")->GP3MLGovernanceProfile:
    choices={"gp3ml-native","NIST-AI-RMF-1.0","ISO-23894-oriented","ISO-42001-oriented"}
    if framework not in choices:raise GP3MLError("Unknown governance framework.")
    if not isinstance(evidence,Mapping):raise GP3MLError("`evidence` must be a named list.")
    disclaimer="gp3ml-native governance evidence profile." if framework=="gp3ml-native" else "Documentation crosswalk only; this is not evidence of NIST endorsement, ISO conformity, certification, or legal compliance."
    return GP3MLGovernanceProfile(framework=framework,evidence=dict(evidence),controls=_domains(framework),disclaimer=disclaimer)

def audit_gp3ml_governance_profile(profile:GP3MLGovernanceProfile)->GP3MLGovernanceProfileAudit:
    if not isinstance(profile,GP3MLGovernanceProfile):raise GP3MLError("Invalid governance profile.")
    tab=profile.controls.copy();statuses=[];classes=[]
    for key in tab.evidence_key:
        value=profile.evidence.get(key);statuses.append("review" if value is None else "pass");classes.append("" if value is None else getattr(value,"r_class",type(value).__name__))
    tab["status"]=statuses;tab["evidence_class"]=classes;return GP3MLGovernanceProfileAudit(status=worst_status(statuses),framework=profile.framework,controls=tab,disclaimer=profile.disclaimer)

def write_gp3ml_governance_profile(audit:GP3MLGovernanceProfileAudit,path:str|Path)->str:
    if not isinstance(audit,GP3MLGovernanceProfileAudit):raise GP3MLError("Invalid governance audit.")
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);lines=[f"# gp3ml governance profile - {audit.framework}","",audit.disclaimer,"",*_markdown_table(audit.controls)];target.write_text("\n".join(lines)+"\n",encoding="utf-8");return target.resolve().as_posix()
