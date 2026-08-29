# `summarize_gazepoint_resample_uncertainty`

**R reference:** gp3ml 0.3.0.

## Summarize uncertainty across folds or repeats

Summarize uncertainty across folds or repeats

## R reference usage

```r
summarize_gazepoint_resample_uncertainty(
  evaluation,
  unit = c("fold", "repeat"),
  conf_level = 0.95
)

\method{print}{gp3ml_resample_uncertainty}(x, ...)
```

The Python implementation is exported as `gp3mlpy.summarize_gazepoint_resample_uncertainty`. See the runtime docstring for Python-specific typing and semantic adaptations.
