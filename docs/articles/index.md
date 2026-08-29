# Articles

The 20 article companions mirror the scope of the frozen gp3ml 0.3.0 vignette set while adapting examples and terminology to Python. Use this page as a learning map rather than reading the articles in filename order.

## Start here

<div class="gp-card-grid" markdown>

<div class="gp-card" markdown>
<span class="gp-kicker">End-to-end</span>
### [Integrated research workflow](integrated-research-workflow.md)
Move from declared task and feature provenance through grouped evaluation, evidence capture, and reporting.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">Experimental workflow</span>
### [Assigned-condition discrimination](assigned-condition-discrimination.md)
A governed classification workflow for explicitly observed experimental assignment.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">Generalization</span>
### [Participant generalization](participant-generalization.md)
Build and validate models when assessment participants must be unseen during fitting.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">Quality review</span>
### [Recording-quality review](recording-quality-review.md)
Use explicitly observed recording-quality measures without turning them into prohibited latent-state inference.
</div>

</div>

## Governance and planning

| Article | What it helps you do |
|---|---|
| [Analysis-plan governance](analysis-plan-governance.md) | Declare, lock, validate, and audit a modelling plan before outcome-driven changes accumulate. |
| [Governance standards profile](governance-standards-profile.md) | Map gp3ml evidence to a governance profile without claiming certification or external endorsement. |
| [Decision governance](decision-governance.md) | Make thresholds, asymmetric costs, calibration sources, and abstention explicit. |
| [API stability contracts](api-stability-contracts.md) | Inspect stable vs experimental interfaces and audit a frozen public API. |

## Generalization and resampling

| Article | What it helps you do |
|---|---|
| [Nested grouped resampling](nested-grouped-resampling.md) | Separate inner tuning from outer assessment while respecting grouping. |
| [Stimulus generalization](stimulus-generalization.md) | Evaluate performance on stimuli not used during fitting. |
| [Participant–stimulus generalization](participant-stimulus-generalization.md) | Enforce both participant and stimulus independence. |
| [Group-aware conformal prediction](group-aware-conformal-prediction.md) | Add prediction sets/intervals while retaining the declared grouping unit. |

## Validation, shift, and robustness

| Article | What it helps you do |
|---|---|
| [External validation reporting](external-validation-reporting.md) | Declare an external dataset, evaluate transportability, and report it explicitly. |
| [Dataset shift and robustness](dataset-shift-and-robustness.md) | Separate covariate and missingness shift from performance and calibration degradation. |
| [Contaminated-manifest leakage](contaminated-manifest-leakage.md) | See how provenance and role checks detect predictors that should not enter the model. |
| [Observed behavioral endpoint](observed-behavioral-endpoint.md) | Keep the endpoint tied to an explicitly observed, permissible behavioral quantity. |

## Reproducibility and interoperability

| Article | What it helps you do |
|---|---|
| [Reproducibility hardening](reproducibility-hardening.md) | Capture environment differences and audit volatile artifacts. |
| [Portable research artifacts](portable-research-artifacts.md) | Package model/evidence outputs with safer persistence and provenance expectations. |
| [Cross-package interoperability](cross-package-interoperability.md) | Create typed handoffs between gp3mlpy and adjacent research tooling. |
| [Optional-engine portability](optional-engine-portability.md) | Check engine capabilities and make optional-backend assumptions explicit. |

## Additional workflow articles

- [Participant generalization](participant-generalization.md)
- [Stimulus generalization](stimulus-generalization.md)
- [Participant–stimulus generalization](participant-stimulus-generalization.md)
- [Recording-quality review](recording-quality-review.md)
- [Assigned-condition discrimination](assigned-condition-discrimination.md)
- [Integrated research workflow](integrated-research-workflow.md)

## Visual companion

Several article families produce auditable plot objects. The [plot gallery](../plots.md) shows generated examples for decision thresholds, predictor shift, engine portability, and governance evidence, plus the list of additional registered plot contracts.

## Runnable companions

Every article has a corresponding Python script under the repository's `examples/` directory. CI executes the example suite so article-facing workflows remain coupled to the package implementation.
