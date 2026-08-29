# `create_gazepoint_model_card`

**R reference:** gp3ml 0.3.0.

## Create a governance-focused model card

Create a governance-focused model card

## R reference usage

```r
create_gazepoint_model_card(
  model,
  intended_use,
  evaluation = NULL,
  calibration = NULL,
  feature_manifest = NULL,
  external_validation = NULL,
  limitations = character(),
  ethical_review = NULL
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_model_card`. See the runtime docstring for Python-specific typing and semantic adaptations.
