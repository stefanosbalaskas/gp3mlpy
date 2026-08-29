# API Stability and Public Object Contracts

> Source-derived companion to `gp3ml` 0.3.0 vignette `api-stability-contracts.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/api-stability-contracts.py`.

## Why an explicit contract layer?

`gp3ml` distinguishes the public API established by version 0.2.0 from new
development APIs. Stable exported names and registered public S3 classes
should not silently disappear or change meaning within the 0.2.x line.



## Audit the currently loaded package



A failure indicates removal of an established export or registered public
class. An undeclared export is reviewable rather than silently accepted.

## Inspect an object schema



The contract policy is additive: established named components retain their
meaning, while compatible additions remain possible.
