# `create_gazepoint_handoff`

**R reference:** gp3ml 0.3.0.

## Create a lightweight cross-package Gazepoint handoff

Create a lightweight cross-package Gazepoint handoff

## R reference usage

```r
create_gazepoint_handoff(
  data,
  source_package,
  source_version = NULL,
  producer = NULL,
  keys,
  outcome = NULL,
  predictors = character(),
  feature_manifest = NULL,
  notes = character()
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_handoff`. See the runtime docstring for Python-specific typing and semantic adaptations.
