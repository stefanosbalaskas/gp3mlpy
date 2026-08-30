args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_target_uncertainty.R <fixture.json> <output.json>", call. = FALSE)
}

error_path <- file.path(dirname(args[[2L]]), "r-target-uncertainty-error.txt")
dir.create(dirname(error_path), recursive = TRUE, showWarnings = FALSE)

options(error = function() {
  message_text <- geterrmessage()
  traceback_text <- capture.output(traceback(20L))
  writeLines(
    c(
      "Uncaught R target-uncertainty runner error",
      message_text,
      "",
      "Traceback:",
      traceback_text
    ),
    error_path,
    useBytes = TRUE
  )
  quit(save = "no", status = 1L, runLast = FALSE)
})

impl_path <- "parity/run_r_target_uncertainty_impl.R"
impl_text <- readLines(impl_path, warn = FALSE)
invalid_repeat_line <- "    repeat = as.integer(row$repeat),"
repeat_matches <- which(impl_text == invalid_repeat_line)
if (length(repeat_matches) != 1L) {
  stop(
    sprintf(
      "Expected exactly one guarded target-uncertainty `repeat` parse repair, found %d.",
      length(repeat_matches)
    ),
    call. = FALSE
  )
}
impl_text[[repeat_matches]] <- "    `repeat` = as.integer(row[[\"repeat\"]]),"
parsed_impl <- parse(text = paste(impl_text, collapse = "\n"), keep.source = TRUE)
eval(parsed_impl, envir = globalenv())
