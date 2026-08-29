# `write_gazepoint_fold_diagnostics_csv`

**R reference:** gp3ml 0.3.0.

## Write Gazepoint fold diagnostics to CSV files

Write Gazepoint fold diagnostics to CSV files

## R reference usage

```r
write_gazepoint_fold_diagnostics_csv(
  x,
  directory,
  prefix = "gazepoint_fold_diagnostics",
  tables = c("fold_metrics", "repeat_metrics", "outcome_balance", "group_balance",
    "assessment_coverage", "exclusion_summary", "validation_checks", "validation_issues"),
  overwrite = FALSE,
  na = ""
)
```

The Python implementation is exported as `gp3mlpy.write_gazepoint_fold_diagnostics_csv`. See the runtime docstring for Python-specific typing and semantic adaptations.
