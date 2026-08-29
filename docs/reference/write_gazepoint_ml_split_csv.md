# `write_gazepoint_ml_split_csv`

**R reference:** gp3ml 0.3.0.

## Write group-aware split tables to CSV

Write group-aware split tables to CSV

## R reference usage

```r
write_gazepoint_ml_split_csv(
  x,
  directory,
  prefix = "gazepoint_ml_split",
  tables = c("analysis", "assessment", "excluded", "assignment", "summary",
    "group_counts", "checks", "issues"),
  overwrite = FALSE,
  na = ""
)
```

The Python implementation is exported as `gp3mlpy.write_gazepoint_ml_split_csv`. See the runtime docstring for Python-specific typing and semantic adaptations.
