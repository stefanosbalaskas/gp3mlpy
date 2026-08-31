args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_external_validation.R <fixture.json> <output.json>")
}

fixture_path <- args[[1L]]
output_path <- args[[2L]]
fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)
seed <- as.integer(fixture$seed)
bootstrap <- as.integer(fixture$bootstrap)

suppressPackageStartupMessages(library(gp3ml))

predictors <- c("fixation_duration", "pupil_change")
quality <- c(
  "pass", "review", "pass", "review", "review", "pass",
  "review", "pass", "pass", "review", "review", "pass",
  "review", "pass", "review", "pass", "pass", "review",
  "pass", "review", "review", "pass", "pass", "review"
)

capture_case <- function(fun) {
  tryCatch(
    list(status = "success", value = fun()),
    error = function(e) {
      list(
        status = "error",
        error_type = class(e)[[1L]],
        message = conditionMessage(e)
      )
    }
  )
}

training_data <- function() {
  data <- data.frame(
    participant_id = rep(sprintf("P%02d", 1:12), each = 2),
    trial_id = sprintf("T%02d", 1:24),
    stimulus_id = rep(c("S01", "S02"), 12),
    fixation_duration = 180 + seq_len(24),
    pupil_change = sin(seq_len(24) / 3),
    quality_status = factor(quality, levels = c("pass", "review")),
    stringsAsFactors = FALSE
  )
  data$fixation_duration <- as.numeric(data$fixation_duration)
  data
}

external_data <- function(training, overlap_stimulus = FALSE) {
  out <- training
  out$participant_id <- rep(sprintf("E%02d", 1:12), each = 2)
  out$trial_id <- sprintf("ET%02d", 1:24)
  out$stimulus_id <- rep(if (overlap_stimulus) c("S01", "S02") else c("ES01", "ES02"), 12)
  out$fixation_duration <- as.numeric(out$fixation_duration) + 4
  out$pupil_change <- cos(seq_len(24) / 4)
  out$quality_status <- factor(as.character(out$quality_status), levels = c("pass", "review"))
  out
}

make_task <- function(data) {
  declare_gazepoint_task(
    data = data,
    outcome = "quality_status",
    purpose = "Predict predefined recording-quality review status",
    task_type = "classification",
    unit_id = "trial_id",
    participant_id = "participant_id",
    stimulus_id = "stimulus_id",
    generalization_target = "new_participants",
    positive = "review"
  )
}

make_model <- function(data, seed) {
  task <- make_task(data)
  train_gazepoint_classifier(
    data = data,
    task = task,
    predictors = predictors,
    engine = "glm",
    seed = seed
  )
}

finite_metric_values <- function(frame) {
  if (!is.data.frame(frame) || !nrow(frame)) return(list())
  out <- list()
  for (name in names(frame)) {
    value <- frame[[name]][[1L]]
    if (is.numeric(value) && length(value) == 1L && is.finite(value)) {
      out[[name]] <- value
    }
  }
  out[sort(names(out))]
}

shift_smd <- function(frame) {
  if (!is.data.frame(frame) || !nrow(frame)) return(list())
  values <- lapply(seq_len(nrow(frame)), function(index) {
    value <- frame$standardized_mean_difference[[index]]
    if (is.na(value)) NULL else as.numeric(value)
  })
  names(values) <- as.character(frame$feature)
  values[sort(names(values))]
}

declaration_value <- function(value) {
  list(
    class = class(value)[[1L]],
    label = as.character(value$label),
    independent = isTRUE(value$independent),
    origin = as.character(value$origin),
    collection_period = value$collection_period,
    participant_id = value$participant_id,
    stimulus_id = value$stimulus_id,
    n_rows = as.integer(value$n_rows),
    n_participants = if (is.na(value$n_participants)) NULL else as.integer(value$n_participants),
    n_stimuli = if (is.na(value$n_stimuli)) NULL else as.integer(value$n_stimuli),
    notes = as.list(as.character(value$notes)),
    hash_recorded = is.character(value$data_hash) && length(value$data_hash) == 1L && nzchar(value$data_hash)
  )
}

external_validation_value <- function(value) {
  calibration <- value$calibration
  list(
    class = class(value)[[1L]],
    label = as.character(value$label),
    model_engine = as.character(value$model_engine),
    n_predictions = nrow(value$predictions),
    prediction_columns = as.list(sort(names(value$predictions))),
    metric_columns = as.list(sort(names(value$metrics))),
    metric_values = finite_metric_values(value$metrics),
    calibration_present = !is.null(calibration),
    calibration_rows = if (is.null(calibration)) 0L else nrow(calibration$summary),
    calibration_columns = if (is.null(calibration)) list() else as.list(sort(names(calibration$summary))),
    shift_rows = nrow(value$shift),
    shift_features = if (nrow(value$shift)) as.list(sort(as.character(value$shift$feature))) else list(),
    shift_smd = shift_smd(value$shift),
    hash_recorded = is.character(value$external_hash) && length(value$external_hash) == 1L && nzchar(value$external_hash)
  )
}

external_report_value <- function(value) {
  development <- value$development_metrics
  list(
    class = class(value)[[1L]],
    validation_label = as.character(value$validation$label),
    development_rows = if (is.null(development)) 0L else nrow(development),
    development_columns = if (is.null(development)) list() else as.list(sort(names(development))),
    limitations = as.list(as.character(value$limitations)),
    prohibited_count = length(value$prohibited_uses)
  )
}

validation_summary_value <- function(value) {
  checks <- value$checks
  list(
    class = class(value)[[1L]],
    status = as.character(value$status),
    n_checks = nrow(checks),
    check_ids = as.list(as.character(checks$check_id)),
    check_statuses = as.list(as.character(checks$status)),
    issues = nrow(value$issues)
  )
}

transport_value <- function(value) {
  schema <- value$schema
  groups <- value$group_coverage
  group_statuses <- list()
  group_overlaps <- list()
  if (is.data.frame(groups) && nrow(groups)) {
    order_index <- order(as.character(groups$unit))
    for (index in order_index) {
      unit <- as.character(groups$unit[[index]])
      group_statuses[[unit]] <- as.character(groups$status[[index]])
      overlap <- groups$overlapping_groups[[index]]
      group_overlaps[[unit]] <- if (is.na(overlap)) NULL else as.integer(overlap)
    }
  }
  declaration_hash_matches <- if (is.null(value$declaration)) {
    NULL
  } else if (is.na(value$declaration_hash_matches)) {
    NULL
  } else {
    isTRUE(value$declaration_hash_matches)
  }
  list(
    class = class(value)[[1L]],
    status = as.character(value$status),
    reason = as.character(value$reason),
    declaration_attached = !is.null(value$declaration),
    declaration_independent = if (is.null(value$declaration)) NULL else isTRUE(value$declaration$independent),
    declaration_hash_matches = declaration_hash_matches,
    schema_rows = if (is.data.frame(schema)) nrow(schema) else 0L,
    schema_predictor_rows = if (is.data.frame(schema) && nrow(schema)) sum(schema$predictor) else 0L,
    missing_predictors = if (is.data.frame(schema) && nrow(schema)) sum(schema$predictor & !schema$external_present) else 0L,
    type_mismatches = if (is.data.frame(schema) && nrow(schema)) sum(schema$predictor & schema$external_present & !schema$class_match) else 0L,
    group_statuses = group_statuses,
    group_overlaps = group_overlaps,
    metrics_rows = nrow(value$metrics),
    performance_rows = nrow(value$performance_comparison),
    prevalence_rows = nrow(value$prevalence_shift),
    predictor_shift_rows = nrow(value$predictor_shift),
    validation_attached = !is.null(value$validation),
    validation_summary = validation_summary_value(value$validation_summary),
    limitations_count = length(value$limitations)
  )
}

writer_value <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  list(
    basename = basename(path),
    exists = file.exists(path),
    line_count = length(lines),
    headings = as.list(lines[grepl("^## ", lines)]),
    header = if (length(lines)) lines[[1L]] else ""
  )
}

write_external_report_value <- function(report) {
  directory <- tempfile("external-validation-parity-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  path <- file.path(directory, "external_validation.md")
  returned <- write_external_validation_report(report, path, overwrite = FALSE)
  result <- writer_value(path)
  result$return_basename <- basename(returned)
  result
}

write_transport_report_value <- function(report, basename_value) {
  directory <- tempfile("transportability-parity-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  path <- file.path(directory, basename_value)
  returned <- write_gazepoint_transportability_report(report, path, overwrite = FALSE)
  result <- writer_value(path)
  result$return_basename <- basename(returned)
  result
}

training <- training_data()
model <- make_model(training, seed)
external <- external_data(training, overlap_stimulus = FALSE)
overlap_external <- external_data(training, overlap_stimulus = TRUE)
development_metrics <- data.frame(accuracy = 0.75, brier = 0.20)

declaration <- declare_gazepoint_external_dataset(
  external,
  label = "independent_site",
  independent = TRUE,
  origin = "Independent deterministic synthetic site",
  notes = "Synthetic parity fixture."
)
non_independent_declaration <- declare_gazepoint_external_dataset(
  external,
  label = "non_independent_site",
  independent = FALSE,
  origin = "Development-linked synthetic site"
)
overlap_declaration <- declare_gazepoint_external_dataset(
  overlap_external,
  label = "overlap_site",
  independent = TRUE,
  origin = "Synthetic site with overlapping stimulus identifiers"
)
incompatible_external <- external[setdiff(names(external), "pupil_change")]
incompatible_declaration <- declare_gazepoint_external_dataset(
  incompatible_external,
  label = "schema_mismatch_site",
  independent = TRUE,
  origin = "Synthetic schema mismatch site"
)

validation <- evaluate_external_validation(
  model,
  external,
  label = "independent_site",
  bootstrap = bootstrap,
  seed = seed
)
report <- create_external_validation_report(
  validation,
  development_metrics = development_metrics,
  limitations = "Synthetic external-validation parity fixture."
)

transport_no_external <- evaluate_gazepoint_external_transportability(
  model,
  training,
  external_data = NULL,
  declaration = NULL,
  development_evaluation = development_metrics,
  bootstrap = bootstrap,
  seed = seed
)
transport_independent <- evaluate_gazepoint_external_transportability(
  model,
  training,
  external_data = external,
  declaration = declaration,
  development_evaluation = development_metrics,
  bootstrap = bootstrap,
  seed = seed
)
transport_overlap <- evaluate_gazepoint_external_transportability(
  model,
  training,
  external_data = overlap_external,
  declaration = overlap_declaration,
  development_evaluation = development_metrics,
  bootstrap = bootstrap,
  seed = seed
)
transport_non_independent <- evaluate_gazepoint_external_transportability(
  model,
  training,
  external_data = external,
  declaration = non_independent_declaration,
  development_evaluation = development_metrics,
  bootstrap = bootstrap,
  seed = seed
)
mismatched_external <- external
mismatched_external$fixation_duration[[1L]] <- mismatched_external$fixation_duration[[1L]] + 1
transport_mismatch <- evaluate_gazepoint_external_transportability(
  model,
  training,
  external_data = mismatched_external,
  declaration = declaration,
  development_evaluation = development_metrics,
  bootstrap = bootstrap,
  seed = seed
)
transport_incompatible <- evaluate_gazepoint_external_transportability(
  model,
  training,
  external_data = incompatible_external,
  declaration = incompatible_declaration,
  development_evaluation = development_metrics,
  bootstrap = bootstrap,
  seed = seed
)

cases <- list()

cases[["declare_gazepoint_external_dataset::successful_independent"]] <- capture_case(function() declaration_value(declaration))
cases[["declare_gazepoint_external_dataset::non_independent"]] <- capture_case(function() declaration_value(non_independent_declaration))
cases[["declare_gazepoint_external_dataset::invalid_label"]] <- capture_case(function() {
  declare_gazepoint_external_dataset(external, label = "", independent = TRUE, origin = "Synthetic site")
})
cases[["declare_gazepoint_external_dataset::invalid_independent"]] <- capture_case(function() {
  declare_gazepoint_external_dataset(external, label = "bad", independent = "yes", origin = "Synthetic site")
})
cases[["declare_gazepoint_external_dataset::missing_identifier"]] <- capture_case(function() {
  declare_gazepoint_external_dataset(
    external,
    label = "bad",
    independent = TRUE,
    origin = "Synthetic site",
    participant_id = "missing_id"
  )
})

cases[["evaluate_external_validation::classification_success"]] <- capture_case(function() external_validation_value(validation))

cases[["create_external_validation_report::successful_report"]] <- capture_case(function() external_report_value(report))
cases[["create_external_validation_report::invalid_validation"]] <- capture_case(function() {
  create_external_validation_report("not validation")
})

cases[["write_external_validation_report::successful_write"]] <- capture_case(function() write_external_report_value(report))
cases[["write_external_validation_report::overwrite_rejected"]] <- capture_case(function() {
  directory <- tempfile("external-validation-overwrite-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  path <- file.path(directory, "external_validation.md")
  write_external_validation_report(report, path, overwrite = FALSE)
  write_external_validation_report(report, path, overwrite = FALSE)
})

cases[["evaluate_gazepoint_external_transportability::no_external"]] <- capture_case(function() transport_value(transport_no_external))
cases[["evaluate_gazepoint_external_transportability::independent_external"]] <- capture_case(function() transport_value(transport_independent))
cases[["evaluate_gazepoint_external_transportability::overlap_requires_review"]] <- capture_case(function() transport_value(transport_overlap))
cases[["evaluate_gazepoint_external_transportability::non_independent_external"]] <- capture_case(function() transport_value(transport_non_independent))
cases[["evaluate_gazepoint_external_transportability::declaration_mismatch"]] <- capture_case(function() transport_value(transport_mismatch))
cases[["evaluate_gazepoint_external_transportability::incompatible_schema"]] <- capture_case(function() transport_value(transport_incompatible))
cases[["evaluate_gazepoint_external_transportability::missing_declaration"]] <- capture_case(function() {
  evaluate_gazepoint_external_transportability(
    model,
    training,
    external_data = external,
    declaration = NULL,
    development_evaluation = development_metrics,
    bootstrap = bootstrap,
    seed = seed
  )
})
cases[["evaluate_gazepoint_external_transportability::invalid_model"]] <- capture_case(function() {
  evaluate_gazepoint_external_transportability("not model", training, external_data = NULL)
})

cases[["validate_gazepoint_transportability::externally_validated"]] <- capture_case(function() {
  validation_summary_value(validate_gazepoint_transportability(transport_independent))
})
cases[["validate_gazepoint_transportability::not_externally_validated"]] <- capture_case(function() {
  validation_summary_value(validate_gazepoint_transportability(transport_no_external))
})
cases[["validate_gazepoint_transportability::invalid_object"]] <- capture_case(function() {
  validate_gazepoint_transportability("not report")
})

cases[["write_gazepoint_transportability_report::successful_write"]] <- capture_case(function() {
  write_transport_report_value(transport_independent, "transportability.md")
})
cases[["write_gazepoint_transportability_report::no_external_write"]] <- capture_case(function() {
  write_transport_report_value(transport_no_external, "transportability_no_external.md")
})
cases[["write_gazepoint_transportability_report::overwrite_rejected"]] <- capture_case(function() {
  directory <- tempfile("transportability-overwrite-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  path <- file.path(directory, "transportability.md")
  write_gazepoint_transportability_report(transport_independent, path, overwrite = FALSE)
  write_gazepoint_transportability_report(transport_independent, path, overwrite = FALSE)
})
cases[["write_gazepoint_transportability_report::invalid_object"]] <- capture_case(function() {
  write_gazepoint_transportability_report("not report", "transportability.md")
})

output <- list(
  runtime = "R",
  package_version = as.character(utils::packageVersion("gp3ml")),
  cases = cases
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  output,
  output_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  na = "null",
  digits = NA
)
