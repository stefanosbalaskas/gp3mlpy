# `capture_gazepoint_environment`

**R reference:** gp3ml 0.3.0.

## Capture a reproducibility environment record

Capture a reproducibility environment record

## R reference usage

```r
capture_gazepoint_environment(
  packages = unique(c("gp3ml", loadedNamespaces())),
  root = ".",
  include_renv = FALSE
)
```

The Python implementation is exported as `gp3mlpy.capture_gazepoint_environment`. See the runtime docstring for Python-specific typing and semantic adaptations.
