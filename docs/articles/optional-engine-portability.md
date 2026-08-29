# Optional Engines and Cross-Platform Portability

> Source-derived companion to `gp3ml` 0.3.0 vignette `optional-engine-portability.Rmd`. The runnable Python companion is under `examples/optional-engine-portability.py`.

Optional engines remain optional. Missing packages are reported explicitly rather than changing the scientific task or silently selecting another model.

## Inspect the capability table first

```python
import gp3mlpy as gp

capabilities = gp.gp3ml_engine_capabilities()
print(capabilities)
```

The table records the gp3ml engine label, corresponding Python package, supported task types, package availability, backend information where applicable, and portability notes.

<div class="gp-plot-card">
  <img src="../assets/plots/engine-capabilities.svg" alt="Availability of gp3mlpy modelling engines">
  <div><strong>Engine portability.</strong> Availability is visible before fitting and is not treated as a model-selection criterion.</div>
</div>

For a visual check:

```python
from gp3mlpy.plotting import plot_engine_capabilities

fig = plot_engine_capabilities(capabilities)
```

## Core and optional backends

`glm` and `lm` are part of the core scientific stack. Other engine labels may depend on additional packages or adapters. The current Python port deliberately distinguishes interface/semantic correspondence from algorithmic identity:

- `ranger` uses a governed scikit-learn random-forest adapter and is **not** claimed to be algorithmically identical to R `ranger`;
- `nnet` uses a governed scikit-learn MLP adapter and is **not** claimed to be algorithmically identical to R `nnet`;
- `xgboost` requires the optional `xgboost` package;
- `keras3` requires Keras and, when requested, an explicitly checked usable backend;
- `custom` requires an externally supplied engine plus safety declarations.

## Fail explicitly when a requested engine is unavailable

```python
gp.assert_gp3ml_engine_available("xgboost")
```

This check is preferable to silently substituting a different estimator because model family is part of the declared analysis plan.

## Availability is not model selection

Engine availability does not authorize automatic selection. Candidate comparison remains metric-declared, direction-declared, partition-aware, and reviewable through the model-tuning and selection objects.

## Key functions

| Function | Role |
|---|---|
| [`gp3ml_engine_capabilities`](../reference/gp3ml_engine_capabilities.md) | Inspect supported tasks and backend availability. |
| [`gp3ml_available_engines`](../reference/gp3ml_available_engines.md) | Enumerate available governed engines. |
| [`assert_gp3ml_engine_available`](../reference/assert_gp3ml_engine_available.md) | Fail explicitly when a requested backend is missing. |
| [`fit_gazepoint_model`](../reference/fit_gazepoint_model.md) | Fit through the governed engine interface. |
| [`integrate_black_box_model`](../reference/integrate_black_box_model.md) | Integrate an externally supplied model under explicit declarations. |

See the [plot gallery](../plots.md) for the generated portability figure and other diagnostics.
