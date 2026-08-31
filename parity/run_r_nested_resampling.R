args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_nested_resampling.R <fixture.json> <output.json>")
}

fixture_path <- args[[1L]]
output_path <- args[[2L]]
fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)
seed <- as.integer(fixture$seed)

predictors <- c("tracking_ratio", "blink_rate")

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

make_data <- function() {
  rows <- list()
  index <- 1L
  trial <- 0L
  for (participant_index in seq_len(12L)) {
    for (stimulus_index in seq_len(4L)) {
      trial <- trial + 1L
      tracking_ratio <- 0.50 + ((participant_index * 3L + stimulus_index * 2L) %% 13L) / 25
      blink_rate <- 3.0 + ((participant_index * 7L + stimulus_index * 5L) %% 17L) / 2
      quality <- if (
        ((participant_index + 2L * stimulus_index + (participant_index * stimulus_index) %% 3L) %% 4L) %in% c(0L, 1L)
      ) "review" else "pass"
      rows[[index]] <- data.frame(
        participant_id = sprintf("P%02d", participant_index),
        trial_id = sprintf("T%03d", trial),
        stimulus_id = sprintf("S%02d", stimulus_index),
        tracking_ratio = tracking_ratio,
        blink_rate = blink_rate,
        quality_status = quality,
        stringsAsFactors = FALSE
      )
      index <- index + 1L
    }
  }
  out <- do.call(rbind, rows)
  out$quality_status <- factor(out$quality_status, levels = c("pass", "review"))
  rownames(out) <- NULL
  out
}

make_task <- function(data) {
  create_gazepoint_synthetic_task(data, "recording_quality", "new_participants")
}

make_outer_folds <- function(data, seed) {
  manifest <- create_gazepoint_synthetic_manifest("quality_status", predictors)
  create_gazepoint_group_folds(
    data,
    "quality_status",
    predictors,
    manifest,
    "new_participants",
    participant_id = "participant_id",
    trial_id = "trial_id",
    stimulus_id = "stimulus_id",
    v = 3L,
    repeats = 1L,
    seed = seed
  )
}

safe_max <- function(data, columns) {
  values <- numeric()
  for (column in columns) {
    if (!column %in% names(data)) next
    current <- suppressWarnings(as.numeric(data[[column]]))
    current <- current[is.finite(current)]
    if (length(current)) values <- c(values, current)
  }
  if (!length(values)) return(NULL)
  max(values)
}

nested_value <- function(value) {
  audit <- value$audit$checks
  validation <- value$validation$checks
  inner_counts <- sort(vapply(
    value$folds,
    function(item) if (is.null(item$inner)) 0L else length(item$inner$folds),
    integer(1)
  ))
  list(
    class = class(value)[[1L]],
    n_outer = length(value$folds),
    inner_ready = sum(vapply(value$folds, function(item) !is.null(item$inner), logical(1))),
    inner_fold_counts = as.list(as.integer(inner_counts)),
    inner_v = as.integer(value$inner_v),
    inner_repeats = as.integer(value$inner_repeats),
    target = as.character(value$outer_metadata$generalization_target),
    failed_outer = sum(vapply(value$folds, function(item) identical(item$status, "fail"), logical(1))),
    audit_rows = nrow(audit),
    audit_failures = sum(audit$status == "fail"),
    max_outer_assessment_overlap = safe_max(
      audit,
      c(
        "outer_assessment_inner_analysis_overlap",
        "outer_assessment_inner_assessment_overlap",
        "outer_assessment_inner_excluded_overlap",
        "outer_assessment_overlap"
      )
    ),
    max_inner_partition_overlap = safe_max(
      audit,
      c(
        "inner_analysis_assessment_overlap",
        "inner_analysis_excluded_overlap",
        "inner_assessment_excluded_overlap"
      )
    ),
    validation_checks = as.list(sort(as.character(validation$check_id))),
    validation_failures = sum(validation$status == "fail")
  )
}

audit_value <- function(value) {
  checks <- value$checks
  list(
    class = class(value)[[1L]],
    rows = nrow(checks),
    outer_folds = length(unique(stats::na.omit(checks$outer_fold_id))),
    inner_folds = length(unique(stats::na.omit(checks$inner_fold_id))),
    failures = sum(checks$status == "fail"),
    max_outer_assessment_overlap = safe_max(
      checks,
      c(
        "outer_assessment_inner_analysis_overlap",
        "outer_assessment_inner_assessment_overlap",
        "outer_assessment_inner_excluded_overlap",
        "outer_assessment_overlap"
      )
    ),
    max_inner_partition_overlap = safe_max(
      checks,
      c(
        "inner_analysis_assessment_overlap",
        "inner_analysis_excluded_overlap",
        "inner_assessment_excluded_overlap"
      )
    )
  )
}

validation_value <- function(value) {
  checks <- value$checks
  list(
    class = class(value)[[1L]],
    n_checks = nrow(checks),
    check_ids = as.list(as.character(checks$check_id)),
    failures = sum(checks$status == "fail")
  )
}

status_counts <- function(x) {
  counts <- table(as.character(x), useNA = "no")
  values <- as.list(as.integer(counts))
  names(values) <- names(counts)
  values
}

evaluation_value <- function(value) {
  selected <- sort(vapply(
    Filter(function(result) !is.null(result$selection), value$results),
    function(result) as.character(result$selection$candidate_id),
    character(1)
  ))
  prediction_fold_counts <- if (nrow(value$predictions)) {
    as.integer(sort(as.integer(table(as.character(value$predictions$fold_id)))))
  } else {
    integer()
  }
  list(
    class = class(value)[[1L]],
    target = as.character(value$generalization_target),
    predictors = as.list(as.character(value$predictors)),
    n_outer = nrow(value$fold_status),
    status_counts = status_counts(value$fold_status$status),
    failed_outer = sum(value$fold_status$status == "fail"),
    selection_count = length(selected),
    selected_candidates = as.list(selected),
    n_predictions = nrow(value$predictions),
    prediction_fold_counts = as.list(prediction_fold_counts),
    outer_assessment_only = !nrow(value$predictions) || all(value$predictions$stage == "outer_assessment"),
    n_metrics = nrow(value$metrics),
    metric_names = if (nrow(value$metrics)) as.list(sort(unique(as.character(value$metrics$metric)))) else list(),
    keep_models = isTRUE(value$keep_models),
    models_retained = sum(vapply(value$results, function(result) !is.null(result$model), logical(1))),
    validation_failures = sum(value$validation$checks$status == "fail"),
    partition_audit_failures = sum(value$nested_folds_audit$checks$status == "fail")
  )
}

write_value <- function(value) {
  directory <- tempfile("nested-parity-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  paths <- write_gazepoint_nested_evaluation(
    value,
    directory,
    prefix = "nested_parity",
    overwrite = FALSE
  )
  list(
    table_names = as.list(sort(names(paths))),
    basenames = as.list(sort(basename(unname(paths)))),
    n_files = length(paths),
    all_exist = all(file.exists(unname(paths)))
  )
}

data <- make_data()
task <- make_task(data)
outer <- make_outer_folds(data, seed)
nested <- create_gazepoint_nested_folds(
  outer,
  inner_v = 2L,
  inner_repeats = 1L,
  seed = seed
)
failed_nested <- create_gazepoint_nested_folds(
  outer,
  inner_v = 20L,
  inner_repeats = 1L,
  seed = seed,
  continue_on_error = TRUE
)
grid <- create_gazepoint_tuning_grid("glm", thresholds = 0.5)

evaluation <- evaluate_gazepoint_nested_resampling(
  nested,
  task,
  grid,
  selection_metric = "brier",
  direction = "minimize",
  predictors = predictors,
  minimum_success_prop = 0.5,
  seed = seed,
  keep_models = FALSE,
  continue_on_error = TRUE
)
evaluation_keep <- evaluate_gazepoint_nested_resampling(
  nested,
  task,
  grid,
  selection_metric = "brier",
  direction = "minimize",
  predictors = predictors,
  minimum_success_prop = 0.5,
  seed = seed,
  keep_models = TRUE,
  continue_on_error = TRUE
)
failed_evaluation <- evaluate_gazepoint_nested_resampling(
  failed_nested,
  task,
  grid,
  selection_metric = "brier",
  direction = "minimize",
  predictors = predictors,
  minimum_success_prop = 0.5,
  seed = seed,
  keep_models = FALSE,
  continue_on_error = TRUE
)

cases <- list()

cases[["create_gazepoint_nested_folds::successful_nested_folds"]] <- capture_case(function() nested_value(nested))
cases[["create_gazepoint_nested_folds::retain_inner_failures"]] <- capture_case(function() nested_value(failed_nested))
cases[["create_gazepoint_nested_folds::invalid_outer"]] <- capture_case(function() create_gazepoint_nested_folds("not folds"))
cases[["create_gazepoint_nested_folds::invalid_inner_v"]] <- capture_case(function() create_gazepoint_nested_folds(outer, inner_v = 1L))
cases[["create_gazepoint_nested_folds::invalid_inner_repeats"]] <- capture_case(function() create_gazepoint_nested_folds(outer, inner_repeats = 0L))

cases[["audit_gazepoint_nested_resampling::successful_audit"]] <- capture_case(function() audit_value(audit_gazepoint_nested_resampling(nested)))
cases[["audit_gazepoint_nested_resampling::failed_inner_audit"]] <- capture_case(function() audit_value(audit_gazepoint_nested_resampling(failed_nested)))
cases[["audit_gazepoint_nested_resampling::invalid_object"]] <- capture_case(function() audit_gazepoint_nested_resampling("not nested"))

cases[["validate_gazepoint_nested_folds::successful_validation"]] <- capture_case(function() validation_value(validate_gazepoint_nested_folds(nested)))
cases[["validate_gazepoint_nested_folds::failed_inner_validation"]] <- capture_case(function() validation_value(validate_gazepoint_nested_folds(failed_nested)))
cases[["validate_gazepoint_nested_folds::invalid_object"]] <- capture_case(function() validate_gazepoint_nested_folds("not nested"))

cases[["evaluate_gazepoint_nested_resampling::successful_evaluation"]] <- capture_case(function() evaluation_value(evaluation))
cases[["evaluate_gazepoint_nested_resampling::keep_models"]] <- capture_case(function() evaluation_value(evaluation_keep))
cases[["evaluate_gazepoint_nested_resampling::retain_outer_failures"]] <- capture_case(function() evaluation_value(failed_evaluation))
cases[["evaluate_gazepoint_nested_resampling::invalid_nested"]] <- capture_case(function() {
  evaluate_gazepoint_nested_resampling(
    "not nested",
    task,
    grid,
    selection_metric = "brier",
    direction = "minimize"
  )
})
cases[["evaluate_gazepoint_nested_resampling::invalid_grid"]] <- capture_case(function() {
  evaluate_gazepoint_nested_resampling(
    nested,
    task,
    "not grid",
    selection_metric = "brier",
    direction = "minimize"
  )
})
cases[["evaluate_gazepoint_nested_resampling::stop_on_outer_failure"]] <- capture_case(function() {
  evaluate_gazepoint_nested_resampling(
    failed_nested,
    task,
    grid,
    selection_metric = "brier",
    direction = "minimize",
    predictors = predictors,
    seed = seed,
    continue_on_error = FALSE
  )
})

cases[["validate_gazepoint_nested_evaluation::successful_validation"]] <- capture_case(function() validation_value(validate_gazepoint_nested_evaluation(evaluation)))
cases[["validate_gazepoint_nested_evaluation::failed_validation"]] <- capture_case(function() validation_value(validate_gazepoint_nested_evaluation(failed_evaluation)))
cases[["validate_gazepoint_nested_evaluation::invalid_object"]] <- capture_case(function() validate_gazepoint_nested_evaluation("not evaluation"))

cases[["write_gazepoint_nested_evaluation::successful_write"]] <- capture_case(function() write_value(evaluation))
cases[["write_gazepoint_nested_evaluation::overwrite_rejected"]] <- capture_case(function() {
  directory <- tempfile("nested-parity-overwrite-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  write_gazepoint_nested_evaluation(
    evaluation,
    directory,
    prefix = "nested_parity",
    overwrite = FALSE
  )
  write_gazepoint_nested_evaluation(
    evaluation,
    directory,
    prefix = "nested_parity",
    overwrite = FALSE
  )
})
cases[["write_gazepoint_nested_evaluation::invalid_object"]] <- capture_case(function() {
  write_gazepoint_nested_evaluation("not evaluation", ".")
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
