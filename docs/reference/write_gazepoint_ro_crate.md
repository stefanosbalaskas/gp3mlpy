# `write_gazepoint_ro_crate`

**R reference:** gp3ml 0.3.0.

## Write a minimal RO-Crate-oriented research object

This helper writes a conservative RO-Crate-oriented JSON-LD metadata file and SHA-256 file hashes. It does not claim formal RO-Crate conformance; use an independent validator when formal conformance is required.

## R reference usage

```r
write_gazepoint_ro_crate(
  path,
  files,
  name,
  description,
  creator_name,
  creator_orcid = NULL,
  license = "MIT",
  doi = NULL,
  copy_files = TRUE
)
```

The Python implementation is exported as `gp3mlpy.write_gazepoint_ro_crate`. See the runtime docstring for Python-specific typing and semantic adaptations.
