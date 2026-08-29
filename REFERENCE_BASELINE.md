# Frozen R reference

The authoritative compatibility baseline is **gp3ml 0.3.0**. The repository does not vendor the complete CRAN source tarball; instead, `reference/` stores machine-readable inventories derived from that frozen source. Exact cross-language execution should be performed in an R-enabled environment using `scripts/generate_r_reference.R`.

Verified source counts: 127 exports (71 stable, 56 experimental), 38 stable public classes, 38 R source files, 154 Rd files, 20 vignettes, 32 `testthat` test files, 49 print methods, 16 plot methods, 49 exported Rd example topics, and 44 explicit `expect_error()` contracts.
