# `bootstrap_gazepoint_metrics_by_unit`

**R reference:** gp3ml 0.3.0.

## Generalization-target-aligned bootstrap uncertainty

Resamples observations or declared clusters while preserving every row that belongs to a sampled cluster. Repeated cluster draws duplicate all associated rows. The returned object records the resampling unit and must not be described as uncertainty for another unit.

## R reference usage

```r
bootstrap_gazepoint_metrics_by_unit(
  task,
  truth,
  prediction = NULL,
  probability = NULL,
  participant_id = NULL,
  stimulus_id = NULL,
  unit = c("observation", "participant", "stimulus", "participant_and_stimulus"),
  bootstrap = 1000L,
  conf_level = 0.95,
  seed = 1L,
  threshold = 0.5,
  stratify_observations = TRUE
)

\method{print}{gp3ml_target_uncertainty}(x, ...)
```

The Python implementation is exported as `gp3mlpy.bootstrap_gazepoint_metrics_by_unit`. See the runtime docstring for Python-specific typing and semantic adaptations.
