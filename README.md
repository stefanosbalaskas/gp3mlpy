<div align="center">

<img src="https://raw.githubusercontent.com/stefanosbalaskas/gp3mlpy/main/docs/assets/python-suite-logo.png" width="260" alt="Python Suite research packages logo">

# gp3mlpy

**Governance-first, leakage-resistant predictive modelling for Gazepoint research workflows in Python.**

[![CI](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/ci.yml)
[![Documentation](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/pages.yml/badge.svg)](https://stefanosbalaskas.github.io/gp3mlpy/)
[![Coverage](https://img.shields.io/badge/line%20%2B%20branch%20coverage-100%25-2ea44f.svg)](https://github.com/stefanosbalaskas/gp3mlpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gp3mlpy.svg)](https://pypi.org/project/gp3mlpy/)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22206729.svg)](https://doi.org/10.5281/zenodo.22206729)
[![License](https://img.shields.io/badge/license-MIT-2f855a.svg)](LICENSE)

[**Documentation**](https://stefanosbalaskas.github.io/gp3mlpy/) · [**PyPI**](https://pypi.org/project/gp3mlpy/) · [**API reference**](https://stefanosbalaskas.github.io/gp3mlpy/reference/) · [**Articles**](https://stefanosbalaskas.github.io/gp3mlpy/articles/) · [**Zenodo**](https://doi.org/10.5281/zenodo.22206729)

</div>

---

`gp3mlpy` is the Python port of **gp3ml 0.3.0**. It brings the R package's governance-first modelling contracts to Python while keeping the scientific target, leakage controls, group structure, uncertainty, transportability, and reporting evidence explicit.

It is built for **declared research questions and explicitly observed, non-sensitive outcomes**. It is deliberately not an AutoML system: it will not silently change the generalization target, fit preprocessing on assessment data, invent a threshold, or relax governance constraints simply to produce a better-looking model.

## Why use gp3mlpy?

| Principle | What the package enforces |
|---|---|
| **Scientific target first** | Participant, stimulus, and participant–stimulus generalization are declared before resampling. |
| **Leakage resistance** | Preprocessing, tuning, calibration, threshold selection, and uncertainty operations stay inside the correct partition. |
| **Group-aware validation** | Splits that contradict the declared independence target are rejected rather than silently accepted. |
| **Auditability** | Feature manifests, diagnostics, model cards, external-validation reports, checksums, handoffs, and release evidence are first-class objects. |
| **Visible model selection** | Candidate models and decision rules remain inspectable; no hidden winner selection is performed. |
| **Precise parity claims** | API, semantic, numerical, and algorithmic parity are described separately instead of being conflated. |

## Install

```bash
python -m pip install gp3mlpy
```

Optional extras are available for `xgboost`, deep-learning backends, conformal prediction, RO-Crate export, and safe model-artifact workflows.

```bash
python -m pip install "gp3mlpy[xgboost]"
python -m pip install "gp3mlpy[conformal]"
python -m pip install "gp3mlpy[artifact]"
```

## A governed workflow in a few steps

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

The full workflow extends from task declaration and feature provenance through grouped/nested resampling, calibration, decision governance, robustness, external validation, reproducibility, and release evidence.

## Release quality

| Release property | Validated baseline |
|---|---:|
| **gp3mlpy release** | `0.1.0` |
| **Frozen R reference** | `gp3ml 0.3.0` |
| **Compatibility exports** | 127 |
| **Stable exports** | 71 |
| **Stable public classes** | 38 |
| **Python tests** | 125 passing |
| **Statement coverage** | **4,020 / 4,020 — 100%** |
| **Branch coverage** | **1,700 / 1,700 — 100%** |
| **Partial branches** | **0** |
| **Stable behavioral parity** | **67 PASS / 4 EXPECTED-DIFFERENCE / 0 PENDING / 0 FAIL** |

Coverage is permanently enforced in CI with `--cov-branch --cov-fail-under=100`. The core test matrix runs on Ubuntu, Windows, and macOS across Python 3.11, 3.12, and 3.13, alongside Ruff, mypy, strict documentation builds, package build/Twine checks, and installed-wheel API validation.

## Frozen R/Python behavioral parity

The 71-export stable API has a completed executable behavioral freeze against the SHA-256-verified `gp3ml 0.3.0` release archive.

The four documented expected differences are intentional safety/reference-defect boundaries:

1. Python rejects unequal calibration-vector lengths where frozen R recycles them.
2. Python rejects shortened probability vectors where frozen R recycles them.
3. Python returns the intended repeat-level uncertainty summary instead of reproducing the frozen-R `repeat` formula defect.
4. Python keeps a functioning release-model-card Markdown writer instead of reproducing the frozen-R partial-matching defect.

These are **not unresolved parity failures**. See [PARITY_STATUS.md](PARITY_STATUS.md) and the executable evidence under [`parity/`](parity/).

## Choose your starting point

| Goal | Start here |
|---|---|
| Understand the design philosophy | [Key concepts](https://stefanosbalaskas.github.io/gp3mlpy/key-concepts/) |
| Run an end-to-end analysis | [Integrated research workflow](https://stefanosbalaskas.github.io/gp3mlpy/articles/integrated-research-workflow/) |
| Validate new participants | [Participant generalization](https://stefanosbalaskas.github.io/gp3mlpy/articles/participant-generalization/) |
| Use nested grouped resampling | [Nested grouped resampling](https://stefanosbalaskas.github.io/gp3mlpy/articles/nested-grouped-resampling/) |
| Audit shift / external validity | [Dataset shift and robustness](https://stefanosbalaskas.github.io/gp3mlpy/articles/dataset-shift-and-robustness/) |
| Govern thresholds and decisions | [Decision governance](https://stefanosbalaskas.github.io/gp3mlpy/articles/decision-governance/) |
| Build reproducible release evidence | [Reproducibility hardening](https://stefanosbalaskas.github.io/gp3mlpy/articles/reproducibility-hardening/) |
| Find a function quickly | [Workflow API map](https://stefanosbalaskas.github.io/gp3mlpy/api-map/) · [Full API index](https://stefanosbalaskas.github.io/gp3mlpy/reference/) |

## Scientific safeguards

`gp3mlpy` preserves the prohibited-use boundary of the upstream package. It must not be used for person identification, biometric authentication, protected-attribute inference, health or diagnosis inference, or direct/indirect inference of emotion, stress, personality, deception, cognition, comprehension, intent, or other mental states.

See [PROHIBITED-USE.md](PROHIBITED-USE.md) and [GOVERNANCE.md](GOVERNANCE.md) before adapting the package to a new research context.

## Citation

If you use `gp3mlpy`, cite the software release and the upstream `gp3ml` package.

> Balaskas, S. (2026). **gp3mlpy** (Version 0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22206729

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Development and reproducibility

```bash
git clone https://github.com/stefanosbalaskas/gp3mlpy.git
cd gp3mlpy
python -m pip install -e ".[dev,docs]"
python -m pytest --cov=gp3mlpy --cov-branch --cov-fail-under=100
```

The repository also contains the frozen R/Python parity harness, reference inventories, runnable examples, documentation-generation scripts, governance files, and release evidence.

## Upstream reference

- **R package:** `gp3ml` 0.3.0
- **CRAN:** https://CRAN.R-project.org/package=gp3ml
- **Source:** https://github.com/stefanosbalaskas/gp3ml
- **Documentation:** https://stefanosbalaskas.github.io/gp3ml/

## License

MIT © 2026 Stefanos Balaskas.
