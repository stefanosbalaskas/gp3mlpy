# `create_gazepoint_model_artifact`

**R reference:** gp3ml 0.3.0.

## Create a portable governed model artifact

Create a portable governed model artifact

## R reference usage

```r
create_gazepoint_model_artifact(
  model,
  preprocessor = model$preprocessor \%||\% NULL,
  feature_manifest = NULL,
  task = model$task \%||\% NULL,
  decision_rule = NULL,
  model_card = NULL,
  reference_data = NULL,
  bundle_model = TRUE
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_model_artifact`. See the runtime docstring for Python-specific typing and semantic adaptations.
