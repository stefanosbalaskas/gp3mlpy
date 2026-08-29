# Portable Models and Research Artifacts

> Source-derived companion to `gp3ml` 0.3.0 vignette `portable-research-artifacts.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/portable-research-artifacts.py`.

A reproducible model is more than an R object. This layer records the model,
preprocessing, task, feature provenance, decision rule, schema, environment,
and cryptographic file hashes. Optional `bundle` support is used when available
for engines that need custom serialization.

Environment records can be compared before reproducing a result:



A minimal RO-Crate-oriented export can package research files with SHA-256
hashes. gp3ml deliberately describes this as RO-Crate-oriented unless an
independent conformance validator has also been run.
