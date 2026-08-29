# Optional Engines and Cross-Platform Portability

> Source-derived companion to `gp3ml` 0.3.0 vignette `optional-engine-portability.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/optional-engine-portability.py`.

## Capability table

Optional engines remain optional. Missing packages are reported explicitly
rather than changing the scientific task or silently selecting another model.



`glm` and `lm` are always available. `ranger`, `xgboost`, `nnet`, and `keras3`
are exercised by a dedicated GitHub Actions matrix. Keras backend readiness is
queried only when explicitly requested.



Engine availability is not a model-selection rule. Candidate selection remains
explicit, metric-declared, direction-declared, and reviewable.
