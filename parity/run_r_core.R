args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_core.R <fixture.json> <output.json>", call. = FALSE)
}

fixture_path <- args[[1L]]
output_path <- args[[2L]]

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The parity runner requires the R package 'jsonlite'.", call. = FALSE)
}
if (!requireNamespace("gp3ml", quietly = TRUE)) {
  stop("The frozen gp3ml reference package is not installed.", call. = FALSE)
}

fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)

if (!identical(as.character(utils::packageVersion("gp3ml")), fixture$r_reference$version)) {
  stop(
    sprintf(
      "Installed gp3ml version %s does not match frozen reference %s.",
      as.character(utils::packageVersion("gp3ml")),
      fixture$r_reference$version
    ),
    call. = FALSE
  )
}

normalize_scalar <- function(value) {
  if (length(value) == 0L || is.na(value[[1L]]) || (is.numeric(value[[1L]]) && !is.finite(value[[1L]]))) {
    return(NULL)
  }
  value <- value[[1L]]
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) return(as.numeric(value))
  if (is.logical(value)) return(isTRUE(value))
  as.character(value)
}

normalize_row <- function(data) {
  if (!is.data.frame(data) || nrow(data) != 1L) {
    stop("Expected a one-row data frame from the gp3ml metric function.", call. = FALSE)
  }
  out <- lapply(data, normalize_scalar)
  names(out) <- names(data)
  out
}

run_classification <- function(case) {
  truth <- unlist(case$truth, use.names = FALSE)
  probability <- as.numeric(unlist(case$probability, use.names = FALSE))
  predicted <- if (is.null(case$predicted)) NULL else unlist(case$predicted, use.names = FALSE)
  normalize_row(
    gp3ml::gazepoint_classification_metrics(
      truth = truth,
      probability = probability,
      predicted = predicted,
      positive = case$positive,
      threshold = as.numeric(case$threshold)
    )
  )
}

run_regression <- function(case) {
  truth <- as.numeric(unlist(case$truth, use.names = FALSE))
  prediction <- as.numeric(unlist(case$prediction, use.names = FALSE))
  normalize_row(gp3ml::gazepoint_regression_metrics(truth, prediction))
}

classification <- lapply(fixture$classification_cases, run_classification)
names(classification) <- vapply(fixture$classification_cases, function(x) x$id, character(1))

regression <- lapply(fixture$regression_cases, run_regression)
names(regression) <- vapply(fixture$regression_cases, function(x) x$id, character(1))

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  prohibited_uses = gp3ml::gp3ml_prohibited_uses(),
  classification = classification,
  regression = regression
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  result,
  output_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  digits = 16
)
