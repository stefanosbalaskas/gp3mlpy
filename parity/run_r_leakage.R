args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript parity/run_r_leakage.R <fixture.json> <output.json>", call. = FALSE)
fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)
output_path <- args[[2L]]

normalize_scalar <- function(value) {
  if (length(value) == 0L || is.na(value[[1L]]) || (is.numeric(value[[1L]]) && !is.finite(value[[1L]]))) return(NULL)
  value <- value[[1L]]
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) return(as.numeric(value))
  as.character(value)
}
normalize_frame <- function(data) {
  if (nrow(data) == 0L) return(list())
  lapply(seq_len(nrow(data)), function(index) {
    row <- lapply(data[index, , drop = FALSE], normalize_scalar)
    names(row) <- names(data)
    row
  })
}
normalize_audit <- function(value) {
  list(
    class = class(value)[[1L]],
    status = value$status,
    generalization_target = value$generalization_target,
    outcome = value$outcome,
    predictors = as.character(value$predictors),
    roles = list(
      participant_id = normalize_scalar(value$roles$participant_id),
      trial_id = normalize_scalar(value$roles$trial_id),
      stimulus_id = normalize_scalar(value$roles$stimulus_id),
      target_derived = as.list(as.character(value$roles$target_derived)),
      post_outcome = as.list(as.character(value$roles$post_outcome))
    ),
    partition_summary = normalize_frame(value$partition_summary),
    checks = normalize_frame(value$checks),
    issues = normalize_frame(value$issues)
  )
}
capture_call <- function(call, normalize) {
  tryCatch(
    list(status = "success", value = normalize(call())),
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}
base_partitions <- function() {
  list(
    analysis = data.frame(
      participant_id = c("P01", "P01", "P02", "P02"),
      trial_id = c("T01", "T02", "T03", "T04"),
      stimulus_id = c("S01", "S02", "S03", "S04"),
      outcome = c(0, 1, 0, 1),
      feature_a = c(1.1, 1.2, 1.3, 1.4),
      feature_b = c(2.1, 2.2, 2.3, 2.4),
      stringsAsFactors = FALSE
    ),
    assessment = data.frame(
      participant_id = c("P03", "P03", "P04", "P04"),
      trial_id = c("T05", "T06", "T07", "T08"),
      stimulus_id = c("S05", "S06", "S07", "S08"),
      outcome = c(1, 0, 1, 0),
      feature_a = c(1.5, 1.6, 1.7, 1.8),
      feature_b = c(2.5, 2.6, 2.7, 2.8),
      stringsAsFactors = FALSE
    )
  )
}
case_call <- function(kind, target) {
  parts <- base_partitions()
  analysis <- parts$analysis
  assessment <- parts$assessment
  predictors <- c("feature_a", "feature_b")
  participant_id <- "participant_id"
  target_derived <- character()
  post_outcome <- character()

  if (identical(kind, "participant_overlap")) {
    assessment$participant_id[[1L]] <- "P01"
  } else if (identical(kind, "known_participant_trials")) {
    assessment$participant_id <- c("P01", "P02", "P01", "P02")
  } else if (identical(kind, "participant_trial_overlap")) {
    assessment$participant_id <- c("P01", "P02", "P01", "P02")
    assessment$trial_id[[1L]] <- "T01"
  } else if (identical(kind, "reused_trial_labels")) {
    analysis$trial_id <- c("T01", "T02", "T01", "T02")
    assessment$trial_id <- c("T01", "T02", "T01", "T02")
  } else if (identical(kind, "exact_row_overlap")) {
    assessment[1L, ] <- analysis[1L, ]
  } else if (identical(kind, "duplicate_within")) {
    analysis <- rbind(analysis, analysis[1L, , drop = FALSE])
  } else if (identical(kind, "role_failures")) {
    analysis$target_proxy <- c(1, 0, 1, 0)
    assessment$target_proxy <- c(0, 1, 0, 1)
    analysis$post_metric <- c(4, 3, 2, 1)
    assessment$post_metric <- c(8, 7, 6, 5)
    predictors <- c("outcome", "participant_id", "feature_a", "target_proxy", "post_metric")
    target_derived <- "target_proxy"
    post_outcome <- "post_metric"
  } else if (identical(kind, "identifier_like")) {
    analysis$record_index <- seq_len(nrow(analysis))
    assessment$record_index <- 5:8
    predictors <- c("feature_a", "feature_b", "record_index")
  } else if (identical(kind, "missing_participant")) {
    participant_id <- NULL
  } else if (identical(kind, "mismatched_columns_error")) {
    assessment$extra_column <- 1
  } else if (identical(kind, "missing_predictor_error")) {
    predictors <- "missing_predictor"
  } else if (identical(kind, "empty_partition_error")) {
    analysis <- analysis[0, , drop = FALSE]
  } else if (!identical(kind, "clean")) {
    stop(sprintf("Unknown leakage case kind: %s", kind), call. = FALSE)
  }

  gp3ml::audit_gazepoint_ml_leakage(
    analysis = analysis,
    assessment = assessment,
    outcome = "outcome",
    predictors = predictors,
    participant_id = participant_id,
    trial_id = "trial_id",
    stimulus_id = "stimulus_id",
    generalization_target = target,
    target_derived = target_derived,
    post_outcome = post_outcome
  )
}
writer_case <- function(kind) {
  audit <- case_call("participant_overlap", "new_participants")
  extension <- if (identical(kind, "bad_extension")) ".txt" else ".csv"
  output <- tempfile(fileext = extension)
  on.exit(unlink(output), add = TRUE)
  table <- if (kind %in% c("issues", "bad_extension")) "issues" else "checks"
  result <- gp3ml::write_gazepoint_ml_leakage_audit_csv(audit, output, table = table)
  exported <- utils::read.csv(output, stringsAsFactors = FALSE, check.names = FALSE)
  list(
    returned_extension = paste0(".", tools::file_ext(result)),
    columns = names(exported),
    rows = normalize_frame(exported)
  )
}

audits <- list()
for (case in fixture$audit_cases) {
  audits[[case$id]] <- capture_call(
    function() case_call(case$kind, case$generalization_target),
    normalize_audit
  )
}
writers <- list()
for (case in fixture$writer_cases) {
  writers[[case$id]] <- capture_call(function() writer_case(case$kind), identity)
}
result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  audits = audits,
  writers = writers
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = 16)
