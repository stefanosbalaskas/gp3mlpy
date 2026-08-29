# `create_gazepoint_release_model_card`

**R reference:** gp3ml 0.3.0.

## Create a release-ready governed model card

Extends the existing model-card structure with explicit model-selection, target-aligned uncertainty, nested-resampling, and transportability fields.

## R reference usage

```r
create_gazepoint_release_model_card(
  model,
  intended_use,
  evaluation = NULL,
  selection = NULL,
  uncertainty = NULL,
  calibration = NULL,
  feature_manifest = NULL,
  transportability = NULL,
  limitations,
  ethical_review = NULL,
  deployment_status = "research_review_only"
)

\method{print}{gp3ml_release_model_card}(x, ...)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_release_model_card`. See the runtime docstring for Python-specific typing and semantic adaptations.
