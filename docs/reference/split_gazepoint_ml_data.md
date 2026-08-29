# `split_gazepoint_ml_data`

**R reference:** gp3ml 0.3.0.

## Create a deterministic group-aware Gazepoint holdout split

Creates analysis and assessment partitions that preserve the grouping unit implied by an explicit generalization target.

## R reference usage

```r
split_gazepoint_ml_data(
  data,
  outcome,
  predictors,
  feature_manifest,
  generalization_target,
  participant_id = NULL,
  trial_id = NULL,
  stimulus_id = NULL,
  assessment_prop = 0.2,
  seed = 1L,
  source_row_id = ".gp3ml_source_row"
)
```

The Python implementation is exported as `gp3mlpy.split_gazepoint_ml_data`. See the runtime docstring for Python-specific typing and semantic adaptations.
