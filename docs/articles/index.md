# Workflow articles

Twenty Python article companions cover the scope of the frozen `gp3ml 0.3.0` vignette set. Use this page by **research need**, not by filename order.

<div class="gp-route-grid" markdown>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">End-to-end</span>
### [Integrated research workflow](integrated-research-workflow.md)
Move from task declaration and feature provenance through grouped evaluation, evidence capture, and reporting.
</div>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Generalization</span>
### [Participant generalization](participant-generalization.md)
Build and validate models when assessment participants must be unseen during fitting.
</div>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Governance</span>
### [Decision governance](decision-governance.md)
Keep thresholds, asymmetric costs, calibration sources, and abstention explicit.
</div>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Validation</span>
### [External validation reporting](external-validation-reporting.md)
Declare a second dataset, quantify transportability, and report it without contaminating development.
</div>
</div>

## Pick a learning track

<div class="gp-evidence-layout" markdown>
<div class="gp-evidence-panel" markdown>
<span class="gp-kicker">Track 1</span>
### Build the analysis correctly

1. [Integrated research workflow](integrated-research-workflow.md)
2. [Analysis-plan governance](analysis-plan-governance.md)
3. [Assigned-condition discrimination](assigned-condition-discrimination.md)
4. [Recording-quality review](recording-quality-review.md)
</div>
<div class="gp-evidence-panel" markdown>
<span class="gp-kicker">Track 2</span>
### Make the generalization claim explicit

1. [Participant generalization](participant-generalization.md)
2. [Stimulus generalization](stimulus-generalization.md)
3. [Participant–stimulus generalization](participant-stimulus-generalization.md)
4. [Nested grouped resampling](nested-grouped-resampling.md)
5. [Group-aware conformal prediction](group-aware-conformal-prediction.md)
</div>
<div class="gp-evidence-panel" markdown>
<span class="gp-kicker">Track 3</span>
### Govern decisions and validation

1. [Decision governance](decision-governance.md)
2. [External validation reporting](external-validation-reporting.md)
3. [Dataset shift and robustness](dataset-shift-and-robustness.md)
4. [Contaminated-manifest leakage](contaminated-manifest-leakage.md)
5. [Observed behavioral endpoint](observed-behavioral-endpoint.md)
</div>
<div class="gp-evidence-panel" markdown>
<span class="gp-kicker">Track 4</span>
### Harden reproducibility and portability

1. [Reproducibility hardening](reproducibility-hardening.md)
2. [Portable research artifacts](portable-research-artifacts.md)
3. [Cross-package interoperability](cross-package-interoperability.md)
4. [Optional-engine portability](optional-engine-portability.md)
5. [API stability contracts](api-stability-contracts.md)
</div>
</div>

## Complete article map

### Governance and planning

| Article | What it helps you do |
|---|---|
| [Analysis-plan governance](analysis-plan-governance.md) | Declare, lock, validate, and audit a modelling plan before outcome-driven changes accumulate. |
| [Governance standards profile](governance-standards-profile.md) | Map gp3ml evidence to a governance profile without claiming certification or external endorsement. |
| [Decision governance](decision-governance.md) | Make thresholds, asymmetric costs, calibration sources, and abstention explicit. |
| [API stability contracts](api-stability-contracts.md) | Inspect stable versus experimental interfaces and audit a frozen public API. |

### Generalization and resampling

| Article | What it helps you do |
|---|---|
| [Participant generalization](participant-generalization.md) | Keep assessment participants unseen during fitting. |
| [Stimulus generalization](stimulus-generalization.md) | Evaluate performance on stimuli not used during fitting. |
| [Participant–stimulus generalization](participant-stimulus-generalization.md) | Enforce both participant and stimulus independence. |
| [Nested grouped resampling](nested-grouped-resampling.md) | Separate inner tuning from outer assessment while respecting grouping. |
| [Group-aware conformal prediction](group-aware-conformal-prediction.md) | Add prediction sets or intervals while retaining the declared grouping unit. |

### Validation, shift, and robustness

| Article | What it helps you do |
|---|---|
| [External validation reporting](external-validation-reporting.md) | Declare an external dataset, evaluate transportability, and report it explicitly. |
| [Dataset shift and robustness](dataset-shift-and-robustness.md) | Separate covariate and missingness shift from performance and calibration degradation. |
| [Contaminated-manifest leakage](contaminated-manifest-leakage.md) | See how provenance and role checks detect predictors that should not enter the model. |
| [Observed behavioral endpoint](observed-behavioral-endpoint.md) | Keep the endpoint tied to an explicitly observed, permissible behavioral quantity. |

### Reproducibility and interoperability

| Article | What it helps you do |
|---|---|
| [Reproducibility hardening](reproducibility-hardening.md) | Capture environment differences and audit volatile artifacts. |
| [Portable research artifacts](portable-research-artifacts.md) | Package model and evidence outputs with safer persistence and provenance expectations. |
| [Cross-package interoperability](cross-package-interoperability.md) | Create typed handoffs between gp3mlpy and adjacent research tooling. |
| [Optional-engine portability](optional-engine-portability.md) | Check engine capabilities and make optional-backend assumptions explicit. |

## Runnable and visual companions

Every article has a corresponding Python script under the repository's `examples/` directory. CI executes the example suite so article-facing workflows remain coupled to the package implementation.

Several workflow families also produce auditable plot objects. The [plot gallery](../plots.md) shows generated examples for decision thresholds, dataset shift, engine portability, governance evidence, and other registered plot contracts.
