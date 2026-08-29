# Key concepts

`gp3mlpy` is built around a small number of contracts that keep predictive modelling aligned with the scientific question. Understanding these concepts first makes the rest of the API much easier to navigate.

<div class="gp-card-grid" markdown>

<div class="gp-card" markdown>
<span class="gp-kicker">01 · Purpose</span>
### Declare the scientific task
A model starts with a declared observed outcome, unit of analysis, participant/stimulus identifiers, and a permitted scientific use. The package does not infer or broaden the use case for you.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">02 · Generalization</span>
### Define who or what must be unseen
`new_participants`, `new_stimuli`, and combined participant–stimulus targets imply different split and resampling constraints. Overlap that violates the declared target is treated as a failure, not a convenience.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">03 · Provenance</span>
### Make predictors auditable
Feature manifests document where predictors came from and whether they are permitted for the declared endpoint. Provenance is checked before modelling rather than retrofitted after performance is known.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">04 · Leakage resistance</span>
### Keep fitting inside the analysis partition
Preprocessing, tuning, calibration, threshold selection, and other fitted operations stay inside the correct fold or analysis split. Assessment and external data are not allowed to influence fitting.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">05 · Decision governance</span>
### Treat thresholds as scientific choices
Classification thresholds, asymmetric error costs, and abstention regions are explicit objects with origins and justifications. `gp3mlpy` does not silently optimize a threshold on assessment data.
</div>

<div class="gp-card" markdown>
<span class="gp-kicker">06 · Evidence</span>
### Preserve validation and provenance
Model cards, release evidence, environment capture, checksums, handoffs, RO-Crate export, and reproducibility audits make the analysis inspectable after the modelling session ends.
</div>

</div>

## Generalization targets

The generalization target determines the unit that must remain independent between analysis and assessment partitions.

| Target | What assessment data must represent | Typical grouping requirement |
|---|---|---|
| New participants | People not used for fitting | Participant-disjoint |
| New stimuli | Stimuli not used for fitting | Stimulus-disjoint |
| New participants and stimuli | Both unseen people and unseen stimuli | Participant- and stimulus-disjoint |
| Assigned-condition workflow | Prediction of an explicitly observed experimental assignment | Respect the declared participant/stimulus target |

The target should be chosen from the scientific claim, not from whichever split gives the highest score.

## Stable vs experimental API

The frozen gp3ml 0.3.0 compatibility layer contains **127 exports**: **71 stable** and **56 experimental**. Stability is part of the public contract and can be inspected programmatically:

```python
import gp3mlpy as gp

contracts = gp.gp3ml_api_contracts()
print(contracts.exports[["name", "stability"]])
```

Use [`gp3ml_api_contracts`](reference/gp3ml_api_contracts.md) and [`audit_gp3ml_api_stability`](reference/audit_gp3ml_api_stability.md) when a workflow depends on a frozen interface.

## Parity means more than matching names

The Python port distinguishes four kinds of parity:

1. **API parity** — corresponding public functions and object contracts.
2. **Semantic parity** — the same governance rules, validation boundaries, and workflow meaning.
3. **Numerical parity** — equivalent numerical behavior where the same computation is available.
4. **Algorithmic parity** — the same underlying algorithm or engine.

Python-native adapters are not described as algorithmically identical when they use a different backend. For example, the `ranger` and `nnet` labels preserve gp3ml workflow semantics while using scikit-learn implementations in Python.

## Plot contracts

Several audit and evaluation objects expose `.plot()` methods after importing `gp3mlpy`. These figures are intended to visualize an already-declared analysis object; plotting does not perform hidden model selection or alter the object.

See the [plot gallery](plots.md) for generated examples covering threshold evaluation, shift auditing, engine portability, and governance evidence.

## Prohibited-use boundary

!!! danger "Hard scientific boundary"
    `gp3mlpy` is for explicitly observed, non-sensitive outcomes. It must not be used for person identification, biometric authentication, health/diagnosis inference, protected-attribute inference, or direct/indirect inference of emotion, stress, personality, deception, cognition, comprehension, intent, or other mental states.

For the machine-readable and programmatic boundary, see [`gp3ml_prohibited_uses`](reference/gp3ml_prohibited_uses.md) and [`assert_gp3ml_use_case`](reference/assert_gp3ml_use_case.md).
