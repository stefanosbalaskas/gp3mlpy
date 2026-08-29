# `write_gazepoint_feature_manifest_csv`

**R reference:** gp3ml 0.3.0.

## Write a Gazepoint feature manifest or validation table to CSV

Writes a feature manifest or one table from a validated manifest to a UTF-8 CSV file. Existing files are not replaced unless explicitly permitted.

## R reference usage

```r
write_gazepoint_feature_manifest_csv(
  x,
  file,
  table = c("manifest", "issues", "checks"),
  overwrite = FALSE,
  na = ""
)
```

The Python implementation is exported as `gp3mlpy.write_gazepoint_feature_manifest_csv`. See the runtime docstring for Python-specific typing and semantic adaptations.
