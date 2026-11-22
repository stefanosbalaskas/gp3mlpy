"""Reviewer-specific numerical oracles and wholly isolated policy audit.
This is NEW supplementary analysis, not a reconstruction of the 2026 paper's
unarchived seed/prediction ledger. Execute against gp3mlpy 0.1.0 and frozen
R gp3ml 0.3.0, preserving separate model/policy/audit cohort IDs.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import gp3mlpy

EXPECTED = {
    "leakage_clean_new_participants": ("success", "pass", None),
    "leakage_participant_overlap": ("success", "fail", "participant_partition_compatibility"),
    "leakage_known_participant_trials": ("success", "pass", None),
    "leakage_participant_trial_overlap": ("success", "fail", "trial_partition_overlap"),
    "leakage_reused_trial_labels": ("success", "pass", None),
    "leakage_exact_row_overlap": ("success", "fail", "exact_row_overlap"),
    "leakage_duplicate_within": ("success", "review", "duplicate_rows_within_partitions"),
    "leakage_role_failures": ("success", "fail", "target_derived_predictors"),
    "leakage_identifier_like": ("success", "review", "identifier_like_predictor_names"),
    "leakage_missing_participant": ("success", "fail", "participant_id_available"),
    "leakage_mismatched_columns_error": ("error", None, None),
    "leakage_missing_predictor_error": ("error", None, None),
    "leakage_empty_partition_error": ("error", None, None),
}

def dump(path, object):
    Path(path).write_text(json.dumps(object, indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")

def audit_fold_ledger(root: Path):
    source = pd.read_csv(root / "synthetic_source.csv")
    inventory = pd.read_csv(root / "fold_inventory.csv")
    assigns = pd.read_csv(root / "fold_assignments.csv")
    assert len(source) == 480
    assert source.participant_id.nunique() == 30
    assert source.stimulus_id.nunique() == 8
    assert int(source.gaze_dispersion.isna().sum()) == 24
    assert len(inventory) == 150
    count = 0
    for (design, repetition, fold), f in assigns.groupby(["design", "repeat", "fold"], sort=True):
        assert len(f) == 480
        assert f.source_row.nunique() == 480
        assert set(f.partition) <= {"analysis", "assessment", "excluded"}
        lookup = source.iloc[f.source_row.to_numpy(dtype=int) - 1].reset_index(drop=True)
        f = f.reset_index(drop=True)
        rows = {}
        for partition in ["analysis", "assessment", "excluded"]:
            z = lookup.loc[f.partition == partition]
            x = inventory[
                (inventory.design == design)
                & (inventory["repeat"] == repetition)
                & (inventory.fold == fold)
                & (inventory.partition == partition)
            ]
            assert len(x) == 1
            n = x.iloc[0]
            assert int(n.n_rows) == len(z)
            assert int(n.n_participants) == z.participant_id.nunique()
            assert int(n.n_stimuli) == z.stimulus_id.nunique()
            assert int(n.n_review) == int((z.quality_status == "review").sum())
            assert int(n.n_pass) == int((z.quality_status == "pass").sum())
            rows[partition] = z
        p_overlap = set(rows["analysis"].participant_id) & set(rows["assessment"].participant_id)
        assert not p_overlap
        if design == "crossed":
            s_overlap = set(rows["analysis"].stimulus_id) & set(rows["assessment"].stimulus_id)
            assert not s_overlap
            assert len(rows["excluded"]) > 0
        else:
            assert len(rows["excluded"]) == 0
        count += 1
    assert count == 50
    for (design, repetition), grp in assigns.groupby(["design", "repeat"]):
        tested = grp.loc[grp.partition == "assessment"].groupby("source_row").size()
        assert len(tested) == 480 and (tested == 1).all()
    return {"status":"PASS", "data_source":"new R synthetic seed=1, missingness seed=951", "fold_seed":20261122,
            "participants_folds":10, "crossed_folds":40, "fold_partitions_audited":150,
            "source_row_assignments_audited":len(assigns), "participant_overlap":0,
            "crossed_stimulus_overlap":0, "per_repeat_assessment_coverage":"once"}

def audit_negative_controls(root: Path):
    outputs = {language: json.loads((root / f"{language}-leakage.json").read_text())
               for language in ["r", "python"]}
    rows = []
    for case, (expect_exec, expect_status, check_id) in EXPECTED.items():
        for language, content in outputs.items():
            result = content["audits"][case]
            assert result["status"] == expect_exec, (case, language, result)
            found = None
            if expect_exec == "success":
                item = result["value"]
                found = item["status"]
                assert found == expect_status, (case, language, found)
                checks = {c["check_id"]:c["status"] for c in item["checks"]}
                if check_id is not None:
                    assert checks.get(check_id) == expect_status, (case, language,check_id,checks.get(check_id))
                message = str(next((c["message"] for c in item["checks"] if c["check_id"] == check_id), ""))
            else:
                message = str(result.get("message",""))
                assert message
            rows.append({"case":case, "language":language, "execution":expect_exec,
                         "overall_status":found, "check_id":check_id or "",
                         "diagnostic":message, "detection_stage":"pre-fit audit" if expect_exec=="success" else "input validation"})
    path = root / "negative_controls.csv"
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"status":"PASS","matched_cases":len(EXPECTED),"language_case_executions":len(rows),
            "actual_status_and_check_identifier_match":True,
            "conventional_comparator_detections":"not measured",
            "false_alarm_rate_across_scenarios":"not measured"}

def independent_oracles():
    # Each expected value follows directly from the mathematical specification,
    # not from an R-generated golden file.
    truth = np.array([0,0,1,1], dtype=int)
    label = np.array([0,1,1,1], dtype=int)
    probs = np.array([.1,.7,.8,.6], dtype=float)
    tn, fp, tp, fn = 1,1,2,0
    expected_ba = (tp/(tp+fn) + tn/(tn+fp))/2
    expected_brier = (.01 + .49 + .04 + .16)/4
    expected_logloss = -sum(math.log(x) for x in [.9,.3,.8,.6])/4
    assert math.isclose(expected_ba,.75,abs_tol=1e-15)
    assert math.isclose(expected_brier,.175,abs_tol=1e-15)
    assert math.isclose(expected_logloss, -float(np.mean(truth*np.log(probs)+(1-truth)*np.log(1-probs))),abs_tol=1e-14)
    pkg_metrics=gp3mlpy.gazepoint_classification_metrics(
        truth=["pass","pass","review","review"],probability=probs.tolist(),
        positive="review",threshold=.5)
    assert math.isclose(float(pkg_metrics.iloc[0].balanced_accuracy),expected_ba,abs_tol=1e-12)
    assert math.isclose(float(pkg_metrics.iloc[0].brier),expected_brier,abs_tol=1e-12)
    assert math.isclose(float(pkg_metrics.iloc[0].log_loss),expected_logloss,abs_tol=1e-12)
    # Participant maxima = [.4, .5, .6, .2, .25]; m=5 and level=.8.
    scores=np.array([.1,.4,.2,.5,.3,.6,.1,.2,.05,.25])
    unit=[f"P{i}" for i in [1,1,2,2,3,3,4,4,5,5]]
    y=["pass","review"]*5
    probability=[s if yy=="pass" else 1-s for s,yy in zip(scores,y,strict=True)]
    obj=gp3mlpy.fit_gazepoint_conformal(
      truth=y,probability=probability,task_type="classification",positive="review",
      level=.8,calibration_unit="participant",unit=unit,
      generalization_target="new_participants")
    assert math.isclose(obj.conformity_quantile,.6,abs_tol=1e-14)
    assert math.isclose(obj.quantile_probability,1.0,abs_tol=1e-14)
    prediction=gp3mlpy.predict_gazepoint_set(obj,[.5])
    assert bool(prediction.iloc[0].include_positive) and bool(prediction.iloc[0].include_negative)
    # For m=15, desired level .90, index ceil(16*.90)=15, the maximum.
    assert math.ceil(16*.9)==15
    return {"status":"PASS","expected_balanced_accuracy":expected_ba,
            "expected_brier":expected_brier,"expected_logloss":expected_logloss,
            "participant_grouped_conformal_q_expected":.6,
            "participant_grouped_conformal_q_actual":float(obj.conformity_quantile),
            "conformal_m15_level90_order_statistic":15,"numerical_tolerance":1e-12}

def independent_participant_audit(root: Path):
    # A separate, NEW fully participant-disjoint evaluation. No retrospective
    # re-labeling of original out-of-fold predictions as independent holdout.
    data=pd.read_csv(root / "synthetic_source.csv")
    cohort={
        "fit":list(sorted(data.participant_id.unique())[:18]),
        "conformal":list(sorted(data.participant_id.unique())[18:22]),
        "threshold":list(sorted(data.participant_id.unique())[22:26]),
        "audit":list(sorted(data.participant_id.unique())[26:30])
    }
    assert len(set(sum(cohort.values(),[])))==30
    for i,key in enumerate(cohort):
        assert all(set(cohort[key]).isdisjoint(cohort[k]) for k in list(cohort)[i+1:])
    parts={name:data[data.participant_id.isin(ids)].copy() for name,ids in cohort.items()}
    xcols=["tracking_ratio","blink_rate","gaze_dispersion"]
    fit=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                     LogisticRegression(max_iter=1000,random_state=20261122))
    fit.fit(parts["fit"][xcols],(parts["fit"].quality_status=="review").astype(int))
    predict={name:fit.predict_proba(frame[xcols])[:,1]
             for name,frame in parts.items() if name!="fit"}
    # A single frozen fitted model: fit does not consume conformal, threshold or audit cohorts.
    conformal=gp3mlpy.fit_gazepoint_conformal(
      truth=parts["conformal"].quality_status.to_list(),
      probability=predict["conformal"],task_type="classification",positive="review",
      level=.9,calibration_unit="participant",
      unit=parts["conformal"].participant_id.to_list(),
      generalization_target="new_participants")
    candidate=np.arange(.10,.91,.05).round(2).tolist()
    label=(parts["threshold"].quality_status=="review").astype(int).to_numpy()
    score={float(t):float(balanced_accuracy_score(label,(predict["threshold"]>=t).astype(int)))
           for t in candidate}
    winner=max(score.values())
    chosen=min(t for t,value in score.items() if abs(value-winner)<1e-12)
    bound=.05  # fixed before opening audit probabilities
    audit=parts["audit"]
    audit_y=(audit.quality_status=="review").astype(int).to_numpy()
    audit_prob=predict["audit"]
    sets=gp3mlpy.predict_gazepoint_set(conformal,audit_prob)
    set_count=sets.include_negative.astype(int)+sets.include_positive.astype(int)
    covered=np.where(audit_y==1,sets.include_positive.to_numpy(),sets.include_negative.to_numpy())
    participant_coverage=pd.DataFrame({"id":audit.participant_id.to_numpy(),"covered":covered}).groupby("id").covered.all()
    keep=(audit_prob < chosen-bound) | (audit_prob > chosen+bound)
    preds=(audit_prob >= chosen).astype(int)
    n_kept=int(keep.sum())
    n_abstain=int((~keep).sum())
    n_err=int((preds[keep]!=audit_y[keep]).sum())
    out = audit[["participant_id","trial_id","stimulus_id","quality_status"]].copy()
    out["probability_review"]=audit_prob
    out["set_size"]=set_count.to_numpy(dtype=int)
    out["true_class_in_prediction_set"]=covered
    out["decision"]=np.where(keep,np.where(preds==1,"review","pass"),"abstain")
    out.to_csv(root/"isolated_audit_predictions.csv",index=False)
    summary={
        "status":"PASS", "analysis":"new 2026-11-22 supplementary replication; not original paper metrics",
        "fit_participants":cohort["fit"],"conformal_participants":cohort["conformal"],
        "threshold_participants":cohort["threshold"],"audit_participants":cohort["audit"],
        "fit_rows":len(parts["fit"]),"conformal_rows":len(parts["conformal"]),
        "threshold_rows":len(parts["threshold"]),"audit_rows":len(audit),
        "positive":"review","threshold_grid":candidate,
        "threshold_objective":"development balanced accuracy, maximize",
        "tie_break":"smallest candidate threshold","threshold_selected":chosen,
        "threshold_development_balanced_accuracy":winner,
        "abstention_halfwidth":bound,"abstention_choice":"illustrative, set before audit",
        "conformal_level":.9,"conformal_unit":"participant",
        "conformal_q":float(conformal.conformity_quantile),
        "conformal_calibration_units":int(conformal.n_calibration_units),
        "prediction_set_size_counts":{str(i):int((set_count==i).sum()) for i in (0,1,2)},
        "prediction_set_row_coverage":float(covered.mean()),
        "all_rows_participant_coverage":float(participant_coverage.mean()),
        "decisions":n_kept,"abstentions":n_abstain,"errors_in_decisions":n_err,
        "decision_rate":n_kept/len(audit),
        "decision_error_rate":n_err/n_kept if n_kept else None,
        "audit_participant_ancestry_overlap":0,
    }
    assert n_kept+n_abstain==len(audit)
    assert sum(summary["prediction_set_size_counts"].values())==len(audit)
    return summary

def dependence_missingness_sensitivity(root: Path):
    # 3 dependence levels x 3 MCAR proportions x 8 generated sources.
    # One participant-level binary label is randomized independently of
    # participant-specific feature fingerprints. Repeats are separate sources.
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    raw=[]
    for dependence in (0.0, .35, 1.0):
        for missingness in (0.0, .10, .30):
            for replicate in range(8):
                seed=20261122 + 1000*int(dependence*100) + 100*int(missingness*100) + replicate
                rng=np.random.default_rng(seed)
                groups=np.repeat(np.arange(30), 16)
                stimuli=np.tile(np.repeat(np.arange(8),2),30)
                outcome=rng.permutation(np.tile([0,1],15))[groups]
                fingerprints=rng.standard_normal((30,8))
                X=dependence*fingerprints[groups]+.5*rng.standard_normal((480,8))
                X[rng.random(X.shape)<missingness]=np.nan
                features=[f"f{i}" for i in range(8)]
                frame=pd.DataFrame(X,columns=features)
                frame["participant_id"]=[f"P{i+1:02d}" for i in groups]
                frame["stimulus_id"]=[f"S{i+1:02d}" for i in stimuli]
                frame["trial_id"]=[f"T{i+1:05d}" for i in range(len(frame))]
                frame["outcome"]=np.where(outcome==1,"review","pass")
                index=np.arange(len(frame))
                train_row,test_row=train_test_split(index,test_size=.25,random_state=99+replicate,stratify=outcome)
                train_participant,test_participant=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=99+replicate).split(frame,outcome,groups))
                for design,train,test in (("row",train_row,test_row),("participant",train_participant,test_participant)):
                    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),KNeighborsClassifier(n_neighbors=1))
                    model.fit(frame.iloc[train][features],outcome[train])
                    predicted=model.predict(frame.iloc[test][features])
                    audit=gp3mlpy.audit_gazepoint_ml_leakage(
                        analysis=frame.iloc[train], assessment=frame.iloc[test],
                        outcome="outcome",predictors=features,
                        participant_id="participant_id",trial_id="trial_id",
                        stimulus_id="stimulus_id",generalization_target="new_participants")
                    check=audit.checks.set_index("check_id").loc["participant_partition_compatibility","status"]
                    assert check==("fail" if design=="row" else "pass")
                    overlap=len(set(groups[train])&set(groups[test]))
                    assert (overlap>0)==(design=="row")
                    raw.append({"dependence":dependence,"missingness":missingness,"replicate":replicate,
                                "design":design,"balanced_accuracy":float(balanced_accuracy_score(outcome[test],predicted)),
                                "shared_participants":overlap,"gp3mlpy_participant_check":check,
                                "overall_audit":audit.status})
    data=pd.DataFrame(raw)
    data.to_csv(root/"dependence_missingness_sensitivity.csv",index=False)
    invalid=data[data.design=="row"]
    valid=data[data.design=="participant"]
    assert len(invalid)==72 and len(valid)==72
    assert (invalid.gp3mlpy_participant_check=="fail").all()
    assert (valid.gp3mlpy_participant_check=="pass").all()
    means=[]
    for (d,m,s),z in data.groupby(["dependence","missingness","design"],sort=True):
        means.append({"dependence":float(d),"missingness":float(m),"design":s,
                      "n_independent_synthetic_sources":len(z),
                      "mean_balanced_accuracy":float(z.balanced_accuracy.mean()),
                      "mean_overlap":float(z.shared_participants.mean())})
    return {"status":"PASS","generated_sources":72,"pipeline_evaluations":144,
            "intentionally_invalid_participant_splits_flagged":72,
            "participant-disjoint_splits_falsely_flagged":0,
            "prohibited_group_overlap_detection": "structural participant-ID check only",
            "ordinary_sklearn_pipeline":"no automatic participant-ID governance audit",
            "aggregate":means}

def main():
    if len(sys.argv)!=2: raise SystemExit("Usage: python verify.py <output_directory>")
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
    assert gp3mlpy.__version__=="0.1.0",gp3mlpy.__version__
    result={
        "fold_ledger":audit_fold_ledger(root),
        "negative_controls":audit_negative_controls(root),
        "numerical_oracles":independent_oracles(),
        "independent_audit":independent_participant_audit(root),
        "dependence_missingness_sensitivity":dependence_missingness_sensitivity(root),
        "versions":{"gp3mlpy":gp3mlpy.__version__,"gp3ml_R":"0.3.0"}
    }
    dump(root/"reviewer_validation_summary.json",result)
    print("REVIEWER_VALIDATION_JSON="+json.dumps(result,allow_nan=False,separators=(",",":")))
    print("REVIEWER_VALIDATION_PASS")
if __name__=="__main__":
    main()
