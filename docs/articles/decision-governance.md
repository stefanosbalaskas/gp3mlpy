# Governed Decision Thresholds and Abstention

> Source-derived companion to `gp3ml` 0.3.0 vignette `decision-governance.Rmd`. The runnable Python companion is under `examples/decision-governance.py`.

Probability estimation and the scientific decision rule are separate stages. `gp3mlpy` does not treat `0.5` as a universally justified classification threshold, and it does not optimize a threshold on outer assessment or independent external-validation data.

## Why threshold origin matters

A threshold can be:

- **predeclared** from the scientific protocol;
- selected on the **training/analysis partition**; or
- selected inside **inner resampling** when nested evaluation is required.

The origin is stored with the decision rule so a later report can distinguish a prespecified rule from a data-informed one.

## Evaluate explicit candidates

```python
import numpy as np
import gp3mlpy as gp

truth = ["control", "control", "control", "target", "target", "target"]
probability = [0.10, 0.32, 0.56, 0.46, 0.68, 0.91]

evaluation = gp.evaluate_gazepoint_thresholds(
    truth=truth,
    probability=probability,
    positive="target",
    thresholds=np.linspace(0.20, 0.80, 7),
    cost_false_positive=1,
    cost_false_negative=1,
)

fig = evaluation.plot(metric="balanced_accuracy")
```

<div class="gp-plot-card">
  <img src="../assets/plots/threshold-evaluation.svg" alt="Balanced accuracy across explicit threshold candidates">
  <div><strong>Threshold evaluation.</strong> The curve visualizes declared candidates. It does not silently select or validate a threshold.</div>
</div>

A selected threshold can then become an explicit decision-rule object:

```python
rule = gp.select_gazepoint_threshold(
    evaluation,
    metric="balanced_accuracy",
    direction="maximize",
    threshold_origin="inner_resampling",
    training_partition="inner_resampling",
    generalization_target="new_participants",
    scientific_justification="Threshold chosen within inner resampling for the declared classification objective.",
)

assert gp.validate_gazepoint_decision_rule(rule, require_threshold=True).status == "pass"
```

## Abstention is a declared protocol choice

An abstention interval can be used when the scientific protocol permits withholding a forced classification. Abstentions must be reported explicitly, including coverage and error among non-abstained predictions. The package does not reinterpret abstention as a way to discard difficult assessment cases after seeing their labels.

## Key functions

| Function | Role |
|---|---|
| [`evaluate_gazepoint_thresholds`](../reference/evaluate_gazepoint_thresholds.md) | Evaluate explicit candidate thresholds and costs. |
| [`select_gazepoint_threshold`](../reference/select_gazepoint_threshold.md) | Select from an already-evaluated table under an explicit metric/direction. |
| [`create_gazepoint_decision_rule`](../reference/create_gazepoint_decision_rule.md) | Declare a threshold, origin, costs, abstention policy, and justification. |
| [`apply_gazepoint_decision_rule`](../reference/apply_gazepoint_decision_rule.md) | Apply the declared rule to probabilities. |
| [`audit_gazepoint_abstention`](../reference/audit_gazepoint_abstention.md) | Audit coverage, abstention, and covered error. |

See the [plot gallery](../plots.md) for the other audit and validation figures.
