# `write_gazepoint_ml_leakage_audit_csv`

**R reference:** gp3ml 0.3.0.

## Write a Gazepoint ML leakage-audit table to CSV

Writes one machine-readable table from a leakage-audit object to a UTF-8 CSV file. Existing files are not replaced unless explicitly permitted.

## R reference usage

```r
write_gazepoint_ml_leakage_audit_csv(
  x,
  file,
  table = c("issues", "checks", "partition_summary"),
  overwrite = FALSE,
  na = ""
)
```

The Python implementation is exported as `gp3mlpy.write_gazepoint_ml_leakage_audit_csv`. See the runtime docstring for Python-specific typing and semantic adaptations.
