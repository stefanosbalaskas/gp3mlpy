# Quickstart

Get from installation to a validated grouped workflow without having to learn the entire API first.

<div class="gp-quick-banner" markdown>
<div markdown>
<span class="gp-kicker">Current release</span>
## gp3mlpy 0.1.0
Python 3.11+ · PyPI · MIT · Zenodo DOI `10.5281/zenodo.22206729`
</div>
<div class="gp-quick-proof" markdown>
**Release validation**

100% statements · 100% branches · 125 tests · 67/4 stable parity matrix
</div>
</div>

## 1. Install

=== "PyPI"

    ```bash
    python -m pip install gp3mlpy
    ```

=== "With an optional backend"

    ```bash
    python -m pip install "gp3mlpy[xgboost]"
    python -m pip install "gp3mlpy[conformal]"
    python -m pip install "gp3mlpy[artifact]"
    ```

=== "Development checkout"

    ```bash
    git clone https://github.com/stefanosbalaskas/gp3mlpy.git
    cd gp3mlpy
    python -m pip install -e ".[dev,docs]"
    ```

Verify the active package:

```python
import gp3mlpy as gp

print(gp.__version__)
print(gp.r_reference_version)
```

Expected release values are `0.1.0` and `0.3.0`.

## 2. Start from the scientific target

The first question is not *which model should I fit?* It is *what claim must the assessment data support?*

<div class="gp-choice-grid" markdown>
<div class="gp-choice" markdown>
<span class="gp-kicker">New people</span>
### New participants
Use participant-disjoint splitting when the claim concerns performance on people not used for fitting.

[Participant generalization →](articles/participant-generalization.md)
</div>
<div class="gp-choice" markdown>
<span class="gp-kicker">New material</span>
### New stimuli
Use stimulus-disjoint splitting when assessment stimuli must be unseen during fitting.

[Stimulus generalization →](articles/stimulus-generalization.md)
</div>
<div class="gp-choice" markdown>
<span class="gp-kicker">Both unseen</span>
### Participants + stimuli
Require both forms of independence when the scientific claim needs both.

[Combined generalization →](articles/participant-stimulus-generalization.md)
</div>
</div>

## 3. Run a minimal governed workflow

This synthetic example uses an explicitly observed experimental assignment and requires new-participant generalization.

```python
import gp3mlpy as gp

predictors = [
    "tracking_ratio",
    "blink_rate",
    "fixation_duration",
    "gaze_dispersion",
    "pupil_change",
]

data = gp.simulate_gazepoint_governed_data(
    n_participants=18,
    n_stimuli=4,
    trials_per_cell=1,
    seed=17,
)

task = gp.create_gazepoint_synthetic_task(
    data,
    workflow="assigned_condition",
    generalization_target="new_participants",
)

manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, predictors)

folds = gp.create_gazepoint_group_folds(
    data=data,
    outcome=task.outcome,
    predictors=predictors,
    feature_manifest=manifest,
    generalization_target=task.generalization_target,
    participant_id=task.participant_id,
    trial_id=task.unit_id,
    stimulus_id=task.stimulus_id,
    v=3,
    repeats=1,
    seed=17,
)

evaluation = gp.evaluate_gazepoint_group_folds(
    folds,
    task,
    predictors,
    engine="glm",
    seed=17,
)

validation = gp.validate_gazepoint_resample_evaluation(evaluation)
assert validation.status == "pass"
```

## 4. Add only the layers your question needs

<div class="gp-route-list" markdown>
<div markdown><strong>Need hyperparameter selection?</strong><br>[Nested grouped resampling](articles/nested-grouped-resampling.md)</div>
<div markdown><strong>Need explicit classification decisions?</strong><br>[Decision governance](articles/decision-governance.md)</div>
<div markdown><strong>Need prediction sets or intervals?</strong><br>[Group-aware conformal prediction](articles/group-aware-conformal-prediction.md)</div>
<div markdown><strong>Need a second dataset?</strong><br>[External validation reporting](articles/external-validation-reporting.md)</div>
<div markdown><strong>Need shift diagnostics?</strong><br>[Dataset shift and robustness](articles/dataset-shift-and-robustness.md)</div>
<div markdown><strong>Need a release-ready evidence bundle?</strong><br>[Reproducibility hardening](articles/reproducibility-hardening.md)</div>
</div>

## 5. Keep the boundary explicit

!!! danger "Permitted-use boundary"
    `gp3mlpy` is for explicitly observed, non-sensitive outcomes. It must not be used for person identification, biometric authentication, protected-attribute inference, health/diagnosis inference, or direct/indirect inference of emotion, stress, personality, deception, cognition, comprehension, intent, or other mental states.

Read [Key concepts](key-concepts.md) before adapting the workflow to a new study, then use the [workflow API map](api-map.md) when you are ready to move from examples to individual functions.
