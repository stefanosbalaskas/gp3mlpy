# `write_gazepoint_group_folds_csv`

**R reference:** gp3ml 0.3.0.

## Write group-aware resampling tables to CSV

Write group-aware resampling tables to CSV

## R reference usage

```r
write_gazepoint_group_folds_csv(
  x,
  directory,
  prefix = "gazepoint_group_folds",
  tables = c("assignments", "fold_summary", "group_counts", "group_mapping",
    "validation_checks", "validation_issues", "audit_summary", "audit_checks",
    "audit_issues"),
  include_fold_data = FALSE,
  overwrite = FALSE,
  na = ""
)
```

The Python implementation is exported as `gp3mlpy.write_gazepoint_group_folds_csv`. See the runtime docstring for Python-specific typing and semantic adaptations.
