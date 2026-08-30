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
repeat_line <- repeat_matches[[1L]]
impl_text[[repeat_line]] <- "    `repeat` = as.integer(row[[\"repeat\"]]),"

frame_window <- seq.int(repeat_line, min(length(impl_text), repeat_line + 10L))
strings_matches <- frame_window[impl_text[frame_window] == "    stringsAsFactors = FALSE"]
if (length(strings_matches) != 1L) {
  stop(
    sprintf(
      "Expected exactly one guarded target-uncertainty resample frame repair, found %d.",
      length(strings_matches)
    ),
    call. = FALSE
  )
}
impl_text[[strings_matches[[1L]]]] <- paste(
  "    stringsAsFactors = FALSE,",
  "    check.names = FALSE",
  sep = "\n"
)

repeat_call_line <- "repeat_uncertainty <- gp3ml::summarize_gazepoint_resample_uncertainty("
repeat_call_matches <- which(impl_text == repeat_call_line)
if (length(repeat_call_matches) != 1L) {
  stop(
    sprintf(
      "Expected exactly one guarded frozen-R repeat-summary call, found %d.",
      length(repeat_call_matches)
    ),
    call. = FALSE
  )
}
repeat_call <- repeat_call_matches[[1L]]
expected_repeat_call <- c(
  repeat_call_line,
  "  evaluation, unit = \"repeat\", conf_level = 0.90",
  ")"
)
if (!identical(impl_text[repeat_call:(repeat_call + 2L)], expected_repeat_call)) {
  stop("Frozen-R repeat-summary call no longer matches the guarded evidence shape.", call. = FALSE)
}
impl_text[[repeat_call]] <- paste(
  "repeat_uncertainty <- capture_call(",
  "  function() gp3ml::summarize_gazepoint_resample_uncertainty(",
  "    evaluation, unit = \"repeat\", conf_level = 0.90",
  "  ),",
  "  normalize_resample_uncertainty",
  ")",
  sep = "\n"
)
impl_text[[repeat_call + 1L]] <- ""
impl_text[[repeat_call + 2L]] <- ""

repeat_summary_line <- "  repeat_summary = list(status = \"success\", value = normalize_resample_uncertainty(repeat_uncertainty)),"
repeat_summary_matches <- which(impl_text == repeat_summary_line)
if (length(repeat_summary_matches) != 1L) {
  stop(
    sprintf(
      "Expected exactly one guarded repeat-summary evidence line, found %d.",
      length(repeat_summary_matches)
    ),
    call. = FALSE
  )
}
impl_text[[repeat_summary_matches[[1L]]]] <- "  repeat_summary = repeat_uncertainty,"

parsed_impl <- parse(text = paste(impl_text, collapse = "\n"), keep.source = TRUE)
eval(parsed_impl, envir = globalenv())
