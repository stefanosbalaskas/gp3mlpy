# Frozen R reference

`gp3mlpy` targets **gp3ml 0.3.0** as its frozen compatibility reference.

Authoritative source inventory used by this port:

- 127 exported functions: 71 stable and 56 experimental
- 38 stable public object classes
- 154 Rd help files
- 49 exported help topics with R examples
- 49 registered/implemented print contracts
- 16 plot contracts
- 20 vignettes/articles
- 32 R testthat files
- 44 explicit `expect_error()` contracts

The Python runtime does not require R. R is used only for independent fixture generation and parity verification when an R runtime is available.
