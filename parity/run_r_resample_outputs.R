#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "Usage: Rscript parity/run_r_resample_outputs.R <fixture.json> <output.json>",
    call. = FALSE
  )
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The parity runner requires the R package 'jsonlite'.", call. = FALSE)
}
if (!requireNamespace("gp3ml", quietly = TRUE)) {
  stop("The frozen gp3ml reference package is not installed.", call. = FALSE)
}

fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)
output_path <- args[[2L]]
expected <- fixture$r_reference$version
if (!identical(as.character(utils::packageVersion("gp3ml")), expected)) {
  stop("Installed gp3ml version does not match the frozen reference.", call. = FALSE)
}

source("parity/run_r_resample_outputs_helpers.R", local = TRUE)
source("parity/run_r_resample_outputs_impl.R", local = TRUE)
