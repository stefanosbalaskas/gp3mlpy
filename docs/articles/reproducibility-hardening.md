# Reproducibility Hardening

> Source-derived companion to `gp3ml` 0.3.0 vignette `reproducibility-hardening.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/reproducibility-hardening.py`.

## Deterministic documentation mode

When `gp3ml.reproducible_examples` is `TRUE`, report timestamps and session
details use deterministic documentation placeholders rather than build-session
values. Research use outside documentation retains the real runtime metadata.



## Audit generated text



Before publication, the same audit can be run over generated `docs/` output to
identify runtime-specific paths, addresses, or generated timestamps.
