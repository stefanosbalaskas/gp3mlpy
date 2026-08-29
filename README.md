<div align="center">
  <img src="docs/assets/brand/gp3mlpy-mark.svg" width="165" alt="gp3mlpy package mark">
  <h1>gp3mlpy</h1>
  <p><strong>Governance-first, leakage-resistant predictive modelling for Gazepoint research workflows in Python.</strong></p>

  [![CI](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/ci.yml)
  [![Documentation](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/pages.yml/badge.svg)](https://stefanosbalaskas.github.io/gp3mlpy/)
  [![Python](https://img.shields.io/badge/Python-%E2%89%A53.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![R reference](https://img.shields.io/badge/R%20reference-gp3ml%200.3.0-276DC3?logo=r&logoColor=white)](https://CRAN.R-project.org/package=gp3ml)
  [![License](https://img.shields.io/badge/license-MIT-2f855a.svg)](LICENSE)
</div>

---

`gp3mlpy` is the Python port of **gp3ml 0.3.0**, the governance-first R package for leakage-resistant and group-aware predictive modelling with Gazepoint-derived research data.

It is designed for **explicitly observed, non-sensitive outcomes and declared scientific purposes**. It is not an AutoML system: it does not silently choose a winning model, weaken participant/stimulus grouping, fit preprocessing on assessment data, invent a threshold, or relax provenance and external-validation requirements to make an analysis succeed.

| Start here | Link |
|---|---|
| Documentation | **https://stefanosbalaskas.github.io/gp3mlpy/** |
| Key concepts | https://stefanosbalaskas.github.io/gp3mlpy/key-concepts/ |
| Plot gallery | https://stefanosbalaskas.github.io/gp3mlpy/plots/ |
| Articles | https://stefanosbalaskas.github.io/gp3mlpy/articles/ |
| API map | https://stefanosbalaskas.github.io/gp3mlpy/api-map/ |
| Complete API index | https://stefanosbalaskas.github.io/gp3mlpy/reference/ |
| R reference package | https://CRAN.R-project.org/package=gp3ml |

## Frozen compatibility target

The port tracks gp3ml 0.3.0 as its frozen reference layer:

- **127 exported functions** — 71 stable and 56 experimental;
- **38 stable public object classes**;
- **16 registered plot contracts**;
- **20 article/vignette companions**;
- explicit API, object-schema, failure, reproducibility, and governance contracts.

`gp3mlpy.r_reference_version` is `"0.3.0"`. Machine-readable inventories are stored under `reference/`, alongside deterministic reference-layer tooling.

## Why gp3mlpy

### Generalization is part of the scientific claim

Participant, stimulus, and participant–stimulus generalization targets imply different independence requirements. Overlap that contradicts the declared target is a validation failure rather than a convenient fallback.

### Leakage-sensitive operations remain local

Fitted preprocessing, tuning, calibration, threshold selection, and related operations stay inside the appropriate analysis/fold structure. External and assessment data are not allowed to influence fitting.

### Important choices become auditable objects

The package provides feature-provenance manifests, leakage audits, grouped holdouts, repeated and nested resampling, fold diagnostics, governed model engines, uncertainty summaries, external-validation/transportability reports, decision rules, conformal prediction, shift audits, analysis plans, model artifacts, reproducibility checks, handoffs, RO-Crate export, model cards, and release evidence.

### Parity is described precisely

Parity is separated into **API**, **semantic**, **numerical**, and **algorithmic** parity. Python-native adapters are not falsely described as bitwise-identical when the underlying implementation differs. In particular, current `ranger` and `nnet` labels preserve gp3ml governance/interface semantics while using scikit-learn backends rather than claiming algorithmic identity with the R engines.

## Scientific safeguards

The package preserves gp3ml's prohibited-use boundary. It must not be used for person identification, biometric authentication, health or diagnosis inference, protected-attribute inference, or direct/indirect inference of emotion, stress, personality, deception, cognition, comprehension, intent, or other mental states.

Participant overlap is a failure when the declared target requires new-participant generalization. Stimulus overlap is a failure when the target requires unseen stimuli. Threshold origin, uncertainty unit, calibration source, and analysis partition remain explicit and inspectable.

## Installation

### Install directly from GitHub

```bash
python -m pip install "git+https://github.com/stefanosbalaskas/gp3mlpy.git@main"
```

### Development checkout

```bash
git clone https://github.com/stefanosbalaskas/gp3mlpy.git
cd gp3mlpy
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

## Visual diagnostics

The documentation site contains a generated [plot gallery](https://stefanosbalaskas.github.io/gp3mlpy/plots/) built from the current Python package. Plot contracts cover decision thresholds, abstention, conformal coverage, dataset shift, environment comparison, handoff/model-artifact/research-bundle validation, API stability, robustness, analysis-plan deviations, checksums, governance evidence, reproducibility, and engine portability.

## Documentation and runnable articles

All 127 compatibility exports have dedicated reference pages. The 20 gp3ml 0.3.0 vignette topics have Python article companions, with runnable scripts under `examples/` and CI coverage for the example suite.

Useful entry points:

- [Key concepts](https://stefanosbalaskas.github.io/gp3mlpy/key-concepts/)
- [Workflow API map](https://stefanosbalaskas.github.io/gp3mlpy/api-map/)
- [Integrated research workflow](https://stefanosbalaskas.github.io/gp3mlpy/articles/integrated-research-workflow/)
- [Participant generalization](https://stefanosbalaskas.github.io/gp3mlpy/articles/participant-generalization/)
- [Nested grouped resampling](https://stefanosbalaskas.github.io/gp3mlpy/articles/nested-grouped-resampling/)
- [Dataset shift and robustness](https://stefanosbalaskas.github.io/gp3mlpy/articles/dataset-shift-and-robustness/)
- [Decision governance](https://stefanosbalaskas.github.io/gp3mlpy/articles/decision-governance/)
- [Reproducibility hardening](https://stefanosbalaskas.github.io/gp3mlpy/articles/reproducibility-hardening/)

## Model-artifact security

`gp3mlpy` does not silently deserialize arbitrary pickle/joblib files. In-memory model-artifact validation is supported, and persisted artifacts should use an explicitly safe/native engine format or an audited optional persistence backend such as `skops` where supported.

## Upstream reference

- **R package:** `gp3ml` 0.3.0
- **CRAN:** https://CRAN.R-project.org/package=gp3ml
- **R source:** https://github.com/stefanosbalaskas/gp3ml
- **R documentation:** https://stefanosbalaskas.github.io/gp3ml/

## License

MIT © 2026 Stefanos Balaskas.
