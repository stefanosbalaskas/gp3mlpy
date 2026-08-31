# Plot gallery

<div class="gp-diagnostics-hero" markdown>
<div markdown>
<span class="gp-kicker">Visual diagnostics</span>
## Inspect governed objects without changing them

`gp3mlpy` plot methods visualize audit, validation, robustness, threshold, shift, reproducibility, and portability objects. The documentation build regenerates every featured figure from the current Python package.
</div>
<div class="gp-diagnostics-stats" markdown>
<div><strong>16</strong><span>registered plot contracts</span></div>
<div><strong>4</strong><span>featured walkthroughs</span></div>
<div><strong>0</strong><span>hidden model selection</span></div>
</div>
</div>

<div class="gp-note" markdown>
The gallery uses deliberately small synthetic documentation fixtures. These figures demonstrate rendering contracts only; they are not scientific results and should not be interpreted substantively.
</div>

## Featured diagnostics

<div class="gp-gallery-grid">
<a class="gp-gallery-card" href="#decision-threshold-evaluation">
<img src="../assets/plots/threshold-evaluation.svg" alt="Decision-threshold evaluation preview">
<div><span>Decision layer</span><strong>Threshold evaluation</strong><small>Inspect declared candidate thresholds and their consequences.</small></div>
</a>
<a class="gp-gallery-card" href="#predictor-distribution-shift">
<img src="../assets/plots/dataset-shift.svg" alt="Dataset-shift audit preview">
<div><span>Transportability</span><strong>Dataset shift</strong><small>Separate predictor shift from calibration and performance change.</small></div>
</a>
<a class="gp-gallery-card" href="#engine-portability">
<img src="../assets/plots/engine-capabilities.svg" alt="Engine portability preview">
<div><span>Portability</span><strong>Engine capabilities</strong><small>Expose available core and optional modelling backends.</small></div>
</a>
<a class="gp-gallery-card" href="#governance-evidence">
<img src="../assets/plots/governance-profile.svg" alt="Governance evidence preview">
<div><span>Governance</span><strong>Evidence profile</strong><small>Surface controls with evidence and controls still requiring review.</small></div>
</a>
</div>

## Decision-threshold evaluation

<div class="gp-diagnostic-block" markdown>
<div class="gp-diagnostic-visual">
<img src="../assets/plots/threshold-evaluation.svg" alt="Balanced-accuracy curve across explicit decision thresholds">
</div>
<div class="gp-diagnostic-copy" markdown>
<span class="gp-kicker">Decision governance</span>
### Explicit candidate thresholds
Candidate thresholds are declared by the analyst. The plot visualizes the resulting metric surface rather than choosing a hidden optimum.

**Related API:** [`evaluate_gazepoint_thresholds`](reference/evaluate_gazepoint_thresholds.md), [`select_gazepoint_threshold`](reference/select_gazepoint_threshold.md), [`create_gazepoint_decision_rule`](reference/create_gazepoint_decision_rule.md).
</div>
</div>

??? example "Show Python example"

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

## Predictor-distribution shift

<div class="gp-diagnostic-block" markdown>
<div class="gp-diagnostic-visual">
<img src="../assets/plots/dataset-shift.svg" alt="Horizontal bars showing predictor shift magnitude">
</div>
<div class="gp-diagnostic-copy" markdown>
<span class="gp-kicker">External validation</span>
### Keep shift distinct from outcome performance
Numeric predictors are summarized with standardized differences; categorical predictors use a distribution statistic. Shift is reported separately from outcome prevalence, calibration, and predictive performance.

**Related API:** [`audit_gazepoint_dataset_shift`](reference/audit_gazepoint_dataset_shift.md), [`audit_gazepoint_missingness_shift`](reference/audit_gazepoint_missingness_shift.md), [`summarize_gazepoint_shift`](reference/summarize_gazepoint_shift.md).
</div>
</div>

??? example "Show Python example"

    ```python
    shift = gp.audit_gazepoint_dataset_shift(
        development,
        external,
        predictors=["tracking_ratio", "fixation_duration", "condition"],
    )

    fig = shift.plot()
    ```

## Engine portability

<div class="gp-diagnostic-block" markdown>
<div class="gp-diagnostic-visual">
<img src="../assets/plots/engine-capabilities.svg" alt="Engine availability by gp3ml engine label">
</div>
<div class="gp-diagnostic-copy" markdown>
<span class="gp-kicker">Runtime capability</span>
### Make backend assumptions visible
Core engines and optional backends are displayed explicitly so portability assumptions are not buried inside model fitting or silently changed across environments.

**Related API:** [`gp3ml_engine_capabilities`](reference/gp3ml_engine_capabilities.md), [`gp3ml_available_engines`](reference/gp3ml_available_engines.md), [`assert_gp3ml_engine_available`](reference/assert_gp3ml_engine_available.md).
</div>
</div>

??? example "Show Python example"

    ```python
    import gp3mlpy as gp
    from gp3mlpy.plotting import plot_engine_capabilities

    capabilities = gp.gp3ml_engine_capabilities()
    fig = plot_engine_capabilities(capabilities)
    ```

## Governance evidence

<div class="gp-diagnostic-block" markdown>
<div class="gp-diagnostic-visual">
<img src="../assets/plots/governance-profile.svg" alt="Counts of governance controls with pass review and fail status">
</div>
<div class="gp-diagnostic-copy" markdown>
<span class="gp-kicker">Evidence review</span>
### Show what is evidenced and what still needs review
Controls with available evidence and controls requiring review are surfaced directly rather than collapsed into one opaque score.

**Related API:** [`create_gp3ml_governance_profile`](reference/create_gp3ml_governance_profile.md), [`audit_gp3ml_governance_profile`](reference/audit_gp3ml_governance_profile.md).
</div>
</div>

??? example "Show Python example"

    ```python
    profile = gp.create_gp3ml_governance_profile(evidence)
    audit = gp.audit_gp3ml_governance_profile(profile)
    fig = audit.plot()
    ```

## The complete plotting surface

<div class="gp-contract-grid" markdown>
<div markdown><strong>Decision + uncertainty</strong><br>Threshold evaluation · abstention · conformal coverage</div>
<div markdown><strong>Validation + robustness</strong><br>Dataset shift · model robustness · environment comparison</div>
<div markdown><strong>Governance + planning</strong><br>Governance profile · analysis-plan deviations · API stability</div>
<div markdown><strong>Artifacts + reproducibility</strong><br>Handoffs · model artifacts · research bundles · RO-Crate · checksums · reproducibility</div>
</div>

Plot methods are intentionally thin: they visualize the state of an existing governed object. They do **not** mutate the analysis, refit models, select models, or change validation decisions.

<div class="gp-inline-cta" markdown>
**Need the function behind a plot?** Use the [workflow API map](api-map.md) for stage-oriented discovery or the [complete API reference](reference/index.md) for exact-name lookup.
</div>
