from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .exceptions import GP3MLError
from .interoperability import as_gp3ml_data, combine_gazepoint_handoffs, create_gazepoint_handoff, validate_gazepoint_handoff
from .objects import GP3MLResearchBundle, GP3MLResearchBundleValidation


def simulate_gazepoint_research_handoffs(n_participants: int = 24, n_stimuli: int = 6, trials_per_stimulus: int = 1, seed: int = 3001) -> GP3MLResearchBundle:
    n_participants, n_stimuli, trials_per_stimulus = int(n_participants), int(n_stimuli), int(trials_per_stimulus)
    if n_participants < 6 or n_stimuli < 2 or trials_per_stimulus < 1:
        raise GP3MLError("Use at least 6 participants, 2 stimuli, and 1 trial per stimulus.")
    rng = np.random.default_rng(int(seed))
    participants = [f"P{i:03d}" for i in range(1, n_participants + 1)]
    stimuli = [f"S{i:02d}" for i in range(1, n_stimuli + 1)]
    rows = [(p, s, r) for r in range(1, trials_per_stimulus + 1) for s in stimuli for p in participants]
    design = pd.DataFrame(rows, columns=["participant_id", "stimulus_id", "replicate"])
    design["trial_id"] = [f"T{i:05d}" for i in range(1, len(design) + 1)]
    cond = {p: ("A" if i % 2 == 0 else "B") for i, p in enumerate(participants)}
    design["assigned_condition"] = design.participant_id.map(cond)
    b = (design.assigned_condition == "B").astype(int).to_numpy(); n = len(design)
    keys = ["participant_id", "trial_id", "stimulus_id"]
    gaze = design[keys + ["assigned_condition"]].copy()
    gaze["valid_gaze_prop"] = np.clip(rng.normal(0.92 + 0.015*b, .035, n), .70, .999)
    gaze["fixation_count"] = rng.poisson(8 + 1.1*b, n)
    gaze["mean_fixation_ms"] = np.maximum(80, rng.normal(225 + 14*b, 26, n))
    gaze["gaze_dispersion"] = np.maximum(.02, rng.normal(.31 - .025*b, .055, n))
    bio = design[keys].copy(); bio["eda_valid_prop"] = np.clip(rng.normal(.94,.035,n),.65,1); bio["hr_valid_prop"] = np.clip(rng.normal(.96,.025,n),.65,1); bio["ibi_valid_prop"] = np.clip(rng.normal(.90,.05,n),.60,1)
    seq = design[keys].copy(); seq["sequence_length"] = np.maximum(2, rng.poisson(9 + .8*b,n)); seq["unique_state_count"] = np.maximum(1,np.minimum(6,rng.poisson(3.2,n))); seq["transition_rate"] = np.clip(rng.normal(.58+.03*b,.08,n),0,1)
    handoffs = {
        "gp3tools": create_gazepoint_handoff(gaze,"gp3tools",producer="synthetic prepared gaze/fixation summaries",keys=keys,outcome="assigned_condition",predictors=["valid_gaze_prop","fixation_count","mean_fixation_ms","gaze_dispersion"]),
        "gpbiometrics": create_gazepoint_handoff(bio,"gpbiometrics",producer="synthetic signal-quality summaries",keys=keys,predictors=["eda_valid_prop","hr_valid_prop","ibi_valid_prop"]),
        "gp3sequences": create_gazepoint_handoff(seq,"gp3sequences",producer="synthetic sequence summaries",keys=keys,predictors=["sequence_length","unique_state_count","transition_rate"]),
    }
    note = "Synthetic workflow for an experimentally assigned condition. No medical, diagnostic, identity, protected-attribute, emotion, stress, cognition, comprehension, intent, personality, deception, or mental-state inference is represented."
    return GP3MLResearchBundle(handoffs=handoffs, keys=keys, outcome="assigned_condition", generalization_target="new_participants", seed=int(seed), governance_note=note)


def validate_gazepoint_research_bundle(x) -> GP3MLResearchBundleValidation:
    if not isinstance(x, GP3MLResearchBundle): raise GP3MLError("`x` must be created by simulate_gazepoint_research_handoffs().")
    required = {"gp3tools","gpbiometrics","gp3sequences"}; sources_present = required.issubset(x.handoffs)
    validations = {name: validate_gazepoint_handoff(h) for name,h in x.handoffs.items()}; handoffs_pass = all(v.status == "pass" for v in validations.values())
    combined = combine_gazepoint_handoffs(x.handoffs, keys=x.keys, collision="error") if handoffs_pass else None
    data = pd.DataFrame() if combined is None else as_gp3ml_data(combined); outcome_present = len(data)>0 and x.outcome in data.columns
    pattern = re.compile("emotion|stress|deception|personality|diagnos|disease|health_status|protected|identity|intent|cognition|comprehension", re.I)
    prohibited = [c for c in data.columns if pattern.search(c)]
    statuses = ["pass" if sources_present else "fail", "pass" if handoffs_pass else "fail", "pass" if len(data)>0 else "fail", "pass" if outcome_present else "fail", "pass" if not prohibited else "fail", "pass" if x.generalization_target=="new_participants" else "fail"]
    checks = pd.DataFrame({"check":["required_sources_present","all_handoffs_pass","combined_rows_present","assigned_outcome_present","no_prohibited_inference_columns","participant_generalization_declared"],"status":statuses,"detail":[", ".join(x.handoffs),f"{sum(v.status=='pass' for v in validations.values())}/{len(validations)} handoffs passed.",f"{len(data)} combined rows.",x.outcome,"None detected." if not prohibited else ", ".join(prohibited),x.generalization_target]})
    return GP3MLResearchBundleValidation(status="fail" if "fail" in statuses else "pass", checks=checks, handoff_validations=validations, bundle=combined, governance_note=x.governance_note)
