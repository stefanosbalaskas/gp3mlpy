# `summarize_gazepoint_resample_performance`

**R reference:** gp3ml 0.3.0.

## Summarize repeated grouped-resampling performance

Summarize repeated grouped-resampling performance

## R reference usage

```r
summarize_gazepoint_resample_performance(
  x,
  aggregation = c("fold_distribution", "pooled_rows"),
  conf_level = 0.95
)

\method{print}{gp3ml_resample_performance_summary}(x, ...)
```

The Python implementation is exported as `gp3mlpy.summarize_gazepoint_resample_performance`. See the runtime docstring for Python-specific typing and semantic adaptations.
