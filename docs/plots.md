# Plot gallery

`gp3mlpy` includes plot contracts for audit, validation, robustness, threshold, shift, reproducibility, and portability objects. The figures below are regenerated during the documentation build from the current Python package, so the gallery stays coupled to the code that users actually install.

<div class="gp-note" markdown>
The gallery data are deliberately small synthetic documentation fixtures. They demonstrate rendering contracts only; they are not scientific results and should not be interpreted substantively.
</div>

## Decision-threshold evaluation

<div class="gp-plot-card">
  <img src="assets/plots/threshold-evaluation.svg" alt="Balanced-accuracy curve across explicit decision thresholds">
  <div><strong>Explicit threshold evaluation.</strong> Candidate thresholds are declared by the analyst; the plot visualizes the resulting metric surface rather than choosing a hidden optimum.</div>
</div>

```python
import numpy as np
import gp3mlpy as gp

thresholds = gp.evaluate_gazepoint_thresholds(
    truth=["control", "control", "target", "target"],
    probability=[0.15, 0.42, 0.63, 0.88],
    positive="target",
    thresholds=np.linspace(0.2, 0.8, 7),
)

fig = thresholds.plot(metric="balanced_accuracy")
```

Related API: [`evaluate_gazepoint_thresholds`](reference/evaluate_gazepoint_thresholds.md), [`select_gazepoint_threshold`](reference/select_gazepoint_threshold.md), and [`create_gazepoint_decision_rule`](reference/create_gazepoint_decision_rule.md).

## Predictor-distribution shift

<div class="gp-plot-card">
  <img src="assets/plots/dataset-shift.svg" alt="Horizontal bars showing predictor shift magnitude">
  <div><strong>Dataset-shift audit.</strong> Numeric predictors are summarized with standardized differences; categorical predictors use a distribution statistic. Shift is reported separately from outcome prevalence, calibration, and performance.</div>
</div>

```python
shift = gp.audit_gazepoint_dataset_shift(
    development,
    external,
    predictors=["tracking_ratio", "fixation_duration", "condition"],
)

fig = shift.plot()
```

Related API: [`audit_gazepoint_dataset_shift`](reference/audit_gazepoint_dataset_shift.md), [`audit_gazepoint_missingness_shift`](reference/audit_gazepoint_missingness_shift.md), and [`summarize_gazepoint_shift`](reference/summarize_gazepoint_shift.md).

## Engine portability

<div class="gp-plot-card">
  <img src="assets/plots/engine-capabilities.svg" alt="Engine availability by gp3ml engine label">
  <div><strong>Engine capability audit.</strong> Core engines and optional backends are made visible so portability assumptions are not buried inside model fitting.</div>
</div>

```python
import gp3mlpy as gp
from gp3mlpy.plotting import plot_engine_capabilities

capabilities = gp.gp3ml_engine_capabilities()
fig = plot_engine_capabilities(capabilities)
```

Related API: [`gp3ml_engine_capabilities`](reference/gp3ml_engine_capabilities.md), [`gp3ml_available_engines`](reference/gp3ml_available_engines.md), and [`assert_gp3ml_engine_available`](reference/assert_gp3ml_engine_available.md).

## Governance evidence

<div class="gp-plot-card">
  <img src="assets/plots/governance-profile.svg" alt="Counts of governance controls with pass review and fail status">
  <div><strong>Governance profile audit.</strong> Controls with available evidence and controls requiring review are surfaced directly rather than collapsed into one opaque score.</div>
</div>

```python
profile = gp.create_gp3ml_governance_profile(evidence)
audit = gp.audit_gp3ml_governance_profile(profile)
fig = audit.plot()
```

Related API: [`create_gp3ml_governance_profile`](reference/create_gp3ml_governance_profile.md) and [`audit_gp3ml_governance_profile`](reference/audit_gp3ml_governance_profile.md).

## Additional plot contracts

The plotting layer also registers visual methods for:

- abstention audits;
- conformal-coverage audits;
- environment comparisons;
- handoff validation;
- model-artifact validation;
- integrated research-bundle validation;
- RO-Crate validation;
- API-stability audits;
- model-robustness audits;
- analysis-plan deviation audits;
- release-checksum validation; and
- reproducibility audits.

The plot methods are intentionally thin. They visualize the state of an existing governed object and do not mutate the analysis, refit models, or change validation decisions.
