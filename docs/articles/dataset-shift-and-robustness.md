# Dataset Shift and Robustness Auditing

> Source-derived companion to `gp3ml` 0.3.0 vignette `dataset-shift-and-robustness.Rmd`. The runnable Python companion is under `examples/dataset-shift-and-robustness.py`.

Dataset shift is not one scalar drift score. `gp3mlpy` keeps predictor-distribution shift, missingness shift, prevalence shift, calibration drift, and performance degradation conceptually separate so one type of change is not silently used as a proxy for another.

## Predictor-distribution shift

```python
import pandas as pd
import gp3mlpy as gp

development = pd.DataFrame(
    {
        "tracking_ratio": [0.94, 0.91, 0.93, 0.89, 0.96],
        "fixation_duration": [212, 225, 219, 231, 208],
        "condition": ["A", "A", "A", "B", "B"],
    }
)

external = pd.DataFrame(
    {
        "tracking_ratio": [0.86, 0.88, 0.84, 0.90, 0.87],
        "fixation_duration": [238, 244, 251, 235, 247],
        "condition": ["A", "B", "B", "B", "B"],
    }
)

audit = gp.audit_gazepoint_dataset_shift(
    development,
    external,
    predictors=["tracking_ratio", "fixation_duration", "condition"],
)

fig = audit.plot()
```

<div class="gp-plot-card">
  <img src="../assets/plots/dataset-shift.svg" alt="Predictor distribution shift audit">
  <div><strong>Predictor shift.</strong> Numeric predictors use standardized differences while categorical predictors use a distribution statistic. The figure visualizes magnitude; the audit object retains status and supporting fields.</div>
</div>

## Missingness is audited separately

```python
missingness = gp.audit_gazepoint_missingness_shift(
    development,
    external,
    predictors=["tracking_ratio", "fixation_duration", "condition"],
)

summary = gp.summarize_gazepoint_shift(audit, missingness)
```

This separation matters because a dataset can have similar observed predictor distributions but substantially different missingness, or vice versa.

## Robustness is dependence on analytical choices

Robustness diagnostics should examine dependence on seeds, folds, features, thresholds, missingness scenarios, and other declared analytical choices rather than relabelling one successful analysis as robust. The relevant functions include:

- [`evaluate_gazepoint_seed_stability`](../reference/evaluate_gazepoint_seed_stability.md)
- [`evaluate_gazepoint_feature_stability`](../reference/evaluate_gazepoint_feature_stability.md)
- [`evaluate_gazepoint_threshold_stability`](../reference/evaluate_gazepoint_threshold_stability.md)
- [`evaluate_gazepoint_missingness_sensitivity`](../reference/evaluate_gazepoint_missingness_sensitivity.md)
- [`audit_gazepoint_model_robustness`](../reference/audit_gazepoint_model_robustness.md)

## Interpretation boundary

A predictor-shift flag does not by itself establish that model performance degraded, that calibration drifted, or that the external population is scientifically inappropriate. Those are distinct questions and should be evaluated with the corresponding external-validation and transportability objects.

Continue with [External validation reporting](external-validation-reporting.md) or browse the [plot gallery](../plots.md).
