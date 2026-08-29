from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .exceptions import GP3MLError
from .feature_provenance import create_gazepoint_feature_manifest
from .task_governance import declare_gazepoint_task


def simulate_gazepoint_governed_data(
    n_participants: int = 30,
    n_stimuli: int = 8,
    trials_per_cell: int = 2,
    seed: int = 1,
) -> pd.DataFrame:
    """Create deterministic non-sensitive data with gp3ml 0.3.0's synthetic design."""
    n_participants = int(n_participants); n_stimuli = int(n_stimuli); trials_per_cell = int(trials_per_cell)
    if n_participants < 4: raise GP3MLError("`n_participants` must be at least 4.")
    if n_stimuli < 2: raise GP3MLError("`n_stimuli` must be at least 2.")
    if trials_per_cell < 1: raise GP3MLError("`trials_per_cell` must be positive.")
    rng = np.random.RandomState(int(seed))
    participants = [f"P{i:03d}" for i in range(1, n_participants + 1)]
    stimuli = [f"S{i:03d}" for i in range(1, n_stimuli + 1)]
    rows = [(p, s, r) for p in participants for s in stimuli for r in range(1, trials_per_cell + 1)]
    design = pd.DataFrame(rows, columns=["participant_id", "stimulus_id", "replicate"])
    n = len(design)
    design["trial_id"] = [f"T{i:05d}" for i in range(1, n + 1)]
    participant_index = design.participant_id.map({p: i for i, p in enumerate(participants)}).to_numpy()
    stimulus_index = design.stimulus_id.map({s: i for i, s in enumerate(stimuli)}).to_numpy()
    participant_effect = rng.normal(0, .35, n_participants)
    stimulus_effect = rng.normal(0, .25, n_stimuli)
    cond_b = ((participant_index + 1 + stimulus_index + 1 + design.replicate.to_numpy()) % 2 == 0)
    design["assigned_condition"] = pd.Categorical(np.where(cond_b, "B", "A"), categories=["A", "B"])
    design["tracking_ratio"] = np.minimum(1, np.maximum(.45, .92 + participant_effect[participant_index]*.04 - np.abs(stimulus_effect[stimulus_index])*.03 + rng.normal(0,.035,n)))
    design["blink_rate"] = np.maximum(0, 6 + participant_effect[participant_index]*1.2 + stimulus_effect[stimulus_index]*.8 + rng.normal(0,1.2,n))
    design["fixation_duration"] = np.maximum(80, 215 + 18*cond_b + participant_effect[participant_index]*28 + stimulus_effect[stimulus_index]*22 + rng.normal(0,18,n))
    design["gaze_dispersion"] = np.maximum(.05, 1.1 - .35*design.tracking_ratio.to_numpy() + participant_effect[participant_index]*.08 + rng.normal(0,.12,n))
    design["pupil_change"] = .10*cond_b + participant_effect[participant_index]*.18 + stimulus_effect[stimulus_index]*.12 + rng.normal(0,.18,n)
    quality_score = -4.2 + 5.0*(1-design.tracking_ratio.to_numpy()) + .22*design.blink_rate.to_numpy() + .85*design.gaze_dispersion.to_numpy()
    quality_probability = 1/(1+np.exp(-quality_score))
    design["quality_status"] = pd.Categorical(np.where(rng.uniform(size=n)<quality_probability, "review", "pass"), categories=["pass","review"])
    response_probability = 1/(1+np.exp(-(-.25 + .60*cond_b + .003*(design.fixation_duration.to_numpy()-215) - .45*design.gaze_dispersion.to_numpy() + participant_effect[participant_index]*.25)))
    design["observed_response"] = pd.Categorical(np.where(rng.uniform(size=n)<response_probability, "recorded_yes", "recorded_no"), categories=["recorded_no","recorded_yes"])
    design["observed_duration"] = np.maximum(.1, np.exp(1.8 - .10*cond_b + .25*design.gaze_dispersion.to_numpy() + participant_effect[participant_index]*.12 + rng.normal(0,.22,n)))
    design["site_label"] = pd.Categorical(np.where(stimulus_index < int(np.ceil(n_stimuli/2)), "development_site", "external_site"))
    return design[["participant_id","trial_id","stimulus_id","replicate","assigned_condition","tracking_ratio","blink_rate","fixation_duration","gaze_dispersion","pupil_change","quality_status","observed_response","observed_duration","site_label"]].copy()


def create_gazepoint_synthetic_manifest(
    outcome: str,
    predictors: Sequence[str],
    participant_id: str = "participant_id",
    stimulus_id: str = "stimulus_id",
    trial_id: str = "trial_id",
) -> pd.DataFrame:
    predictors = list(predictors)
    n = len(predictors)
    return create_gazepoint_feature_manifest(
        features=predictors,
        scientific_source=["Deterministic synthetic demonstration"]*n,
        source_table=["synthetic_trial_features"]*n,
        transformation=["Predeclared synthetic feature"]*n,
        availability_stage=["during_exposure"]*n,
        prediction_time_available=[True]*n,
        outcome_derived=[False]*n,
        post_outcome=[False]*n,
        identifier=[False]*n,
        preprocessing_scope=["resampling_fold"]*n,
        fold_local_required=[True]*n,
        reviewer_notes=[f"Outcome `{outcome}` is explicitly observed and is not a predictor."]*n,
    )


def create_gazepoint_synthetic_task(
    data: pd.DataFrame,
    workflow: str = "recording_quality",
    generalization_target: str = "new_trials_known_participants",
):
    specs = {
        "recording_quality": ("quality_status", "Predict predefined recording-quality review status", "classification", "review"),
        "assigned_condition": ("assigned_condition", "Discriminate the experimentally assigned condition using predeclared features", "classification", "B"),
        "observed_behavior": ("observed_response", "Predict an explicitly recorded non-sensitive response", "classification", "recorded_yes"),
        "observed_duration": ("observed_duration", "Predict an explicitly recorded non-sensitive duration", "regression", None),
    }
    if workflow not in specs:
        raise GP3MLError("`workflow` must be one of: recording_quality, assigned_condition, observed_behavior, observed_duration.")
    targets = {"new_trials_known_participants","new_participants","new_stimuli","new_participants_and_new_stimuli"}
    if generalization_target not in targets:
        raise GP3MLError("Unknown `generalization_target`.")
    outcome,purpose,task_type,positive = specs[workflow]
    return declare_gazepoint_task(data, outcome, purpose, task_type, "trial_id", "participant_id", "stimulus_id", generalization_target, positive=positive, observed_outcome=True, sensitive_outcome=False)
