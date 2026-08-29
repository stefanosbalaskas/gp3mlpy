# `create_gazepoint_group_folds`

**R reference:** gp3ml 0.3.0.

## Create deterministic group-aware Gazepoint resampling folds

Creates repeated grouped assessment folds that preserve the grouping structure implied by an explicit generalization target. A passing feature-provenance manifest is required, and every analysis-assessment pair is evaluated using the leakage audit.

## R reference usage

```r
create_gazepoint_group_folds(
  data,
  outcome,
  predictors,
  feature_manifest,
  generalization_target,
  participant_id = NULL,
  trial_id = NULL,
  stimulus_id = NULL,
  v = 5L,
  repeats = 1L,
  seed = 1L,
  source_row_id = ".gp3ml_source_row"
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_group_folds`. See the runtime docstring for Python-specific typing and semantic adaptations.
