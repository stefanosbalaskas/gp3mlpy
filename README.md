# gp3mlpy

`gp3mlpy` is the Python port of **gp3ml 0.3.0**, the governance-first R package for leakage-resistant and group-aware predictive modelling with Gazepoint-derived research data.

The package is designed for **explicitly observed, non-sensitive outcomes and declared scientific purposes**. It is not an AutoML system, does not silently choose a winning model, and does not relax grouping, provenance, preprocessing, threshold, uncertainty, or external-validation requirements to make an analysis succeed.

## Frozen compatibility target

The port tracks the gp3ml 0.3.0 CRAN source as its frozen reference:

- 127 exported functions: 71 stable and 56 experimental;
- 38 stable public object classes;
- 16 registered plot contracts;
- 20 source vignettes/articles;
- explicit API, object-schema, reproducibility, and governance contracts.

`gp3mlpy.r_reference_version` is `"0.3.0"`. The repository contains machine-readable inventories under `reference/` and deterministic R-reference generators under `reference/r_fixtures/`.

## Scientific safeguards

The package preserves gp3ml's prohibited-use boundary. It must not be used for person identification, biometric authentication, health or diagnosis inference, protected-attribute inference, or direct or indirect inference of emotion, stress, personality, deception, cognition, comprehension, intent, or other mental states.

Participant overlap is a failure when the declared target requires new-participant generalization. Stimulus overlap is a failure when the target requires unseen stimuli. Fitted preprocessing is analysis/fold-local. Tuning occurs inside the appropriate analysis partition. Threshold selection and uncertainty units remain explicit and auditable.

## Core capabilities

`gp3mlpy` provides task and role governance; feature-provenance manifests; leakage auditing; deterministic group-aware holdouts; repeated grouped and nested resampling; fold diagnostics; fold-local preprocessing; governed GLM/LM and optional ML engines; discrimination, error and calibration metrics; target-aware bootstrap uncertainty; external validation and transportability; decision-threshold governance; conformal prediction; dataset and missingness shift audits; locked analysis plans; model artifacts; robustness diagnostics; environment and release provenance; cross-package handoffs; research bundles; RO-Crate export; model cards; and API stability auditing.

## Installation during development

```bash
python -m pip install -e .
```

or, with `uv`:

```bash
uv sync --extra dev --extra docs
```

Optional extras include `xgboost`, `deep`, `conformal`, `rocrate`, and `artifact`.

## Minimal governed workflow

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

assert gp.validate_gazepoint_resample_evaluation(evaluation).status == "pass"
```

## R/Python parity policy

Parity is separated into **API**, **semantic**, **numerical**, and **algorithmic** parity. Python-native equivalents are not falsely described as bitwise-identical when the underlying engine differs. In particular, the current `ranger` and `nnet` Python adapters preserve gp3ml governance and interface semantics but use scikit-learn backends rather than claiming exact algorithmic equivalence to R `ranger`/`nnet`.

The Python implementation does not depend on `rpy2`, Rscript, or an installed R package at runtime. R is used only for generation of frozen reference fixtures when available.

## Model-artifact security

`gp3mlpy` does not silently deserialize arbitrary pickle/joblib files. In-memory model-artifact validation is supported, and persisted artifacts should use an explicitly safe/native engine format or an audited optional persistence backend such as `skops` where supported.

## Documentation

Documentation source is in `docs/`. All 127 compatibility exports have generated reference pages, and all 20 gp3ml 0.3.0 vignettes have corresponding Python articles with runnable scripts under `examples/`.

## Upstream reference

- R package: `gp3ml` 0.3.0
- CRAN: https://CRAN.R-project.org/package=gp3ml
- R source repository: https://github.com/stefanosbalaskas/gp3ml
- R documentation site: https://stefanosbalaskas.github.io/gp3ml/

## License

MIT © 2026 Stefanos Balaskas.
