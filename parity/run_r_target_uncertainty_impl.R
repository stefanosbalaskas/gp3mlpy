args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_target_uncertainty.R <fixture.json> <output.json>", call. = FALSE)
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
  stop(
    sprintf(
      "Installed gp3ml version %s does not match frozen reference %s.",
      as.character(utils::packageVersion("gp3ml")),
      expected
    ),
    call. = FALSE
  )
}

scalar <- function(value) {
  if (length(value) == 0L || is.null(value)) return(NULL)
  if (is.factor(value)) value <- as.character(value)
  if (is.na(value[[1L]])) return(NULL)
  value <- value[[1L]]
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) {
    number <- as.numeric(value)
    return(if (is.finite(number)) number else NULL)
  }
  as.character(value)
}

normalize_frame <- function(data, sort_by = character()) {
  data <- as.data.frame(data, stringsAsFactors = FALSE, check.names = FALSE)
  keys <- sort_by[sort_by %in% names(data)]
  if (length(keys) && nrow(data)) {
    ordering <- do.call(order, c(unname(data[keys]), list(na.last = TRUE, method = "radix")))
    data <- data[ordering, , drop = FALSE]
    rownames(data) <- NULL
  }
  rows <- lapply(seq_len(nrow(data)), function(index) {
    row <- lapply(data[index, , drop = FALSE], scalar)
    names(row) <- names(data)
    row
  })
  list(columns = as.list(names(data)), rows = rows)
}

metric_exclude <- c("n", "threshold", "replicate", "resample_n")
normalize_draw_summary <- function(draws) {
  numeric_names <- names(draws)[vapply(draws, is.numeric, logical(1))]
  numeric_names <- setdiff(numeric_names, metric_exclude)
  lapply(numeric_names, function(name) {
    values <- as.numeric(draws[[name]])
    finite <- values[is.finite(values)]
    list(
      metric = name,
      n = as.integer(length(finite)),
      mean = if (length(finite)) mean(finite) else NULL,
      sd = if (length(finite) > 1L) stats::sd(finite) else NULL,
      min = if (length(finite)) min(finite) else NULL,
      max = if (length(finite)) max(finite) else NULL
    )
  })
}

normalize_target_uncertainty <- function(value, global_rng_preserved) {
  intervals <- value$intervals
  if (nrow(intervals)) {
    intervals <- intervals[order(intervals$metric, method = "radix"), , drop = FALSE]
    rownames(intervals) <- NULL
  }
  replicate_sizes <- as.integer(value$replicate_sizes)
  finite_interval <- is.finite(intervals$lower) & is.finite(intervals$upper)
  checks <- list(
    global_rng_preserved = isTRUE(global_rng_preserved),
    successful_plus_failed_equals_bootstrap =
      as.integer(value$successful_replicates) + as.integer(value$failed_replicates) == as.integer(value$bootstrap),
    replicate_sizes_recorded = length(replicate_sizes) == as.integer(value$bootstrap),
    replicate_sizes_positive = all(replicate_sizes > 0L),
    intervals_ordered = all(intervals$lower[finite_interval] <= intervals$upper[finite_interval])
  )
  list(
    class = class(value)[[1L]],
    unit = value$unit,
    generalization_target = value$generalization_target,
    bootstrap = as.integer(value$bootstrap),
    successful_replicates = as.integer(value$successful_replicates),
    failed_replicates = as.integer(value$failed_replicates),
    conf_level = as.numeric(value$conf_level),
    seed = as.integer(value$seed),
    limitations = value$limitations,
    point = normalize_frame(value$point),
    intervals = normalize_frame(intervals),
    draw_columns = as.list(names(value$draws)),
    draw_nrow = as.integer(nrow(value$draws)),
    draw_replicates = as.list(sort(unique(as.integer(value$draws$replicate)))),
    draw_resample_n = list(
      min = as.integer(min(value$draws$resample_n)),
      max = as.integer(max(value$draws$resample_n))
    ),
    draw_summary = normalize_draw_summary(value$draws),
    failure_columns = as.list(names(value$failures)),
    failures = normalize_frame(value$failures, c("replicate")),
    replicate_sizes = as.list(replicate_sizes),
    checks = checks
  )
}

normalize_resample_uncertainty <- function(value) {
  list(
    class = class(value)[[1L]],
    unit = value$unit,
    conf_level = as.numeric(value$conf_level),
    generalization_target = value$generalization_target,
    limitations = value$limitations,
    summary = normalize_frame(value$summary, c("metric")),
    distribution = normalize_frame(
      value$distribution,
      c("metric", "repeat", "fold", "fold_id")
    )
  )
}

normalize_validation <- function(value) {
  checks <- value$checks[c("check_id", "status")]
  issues <- value$issues[c("check_id", "status")]
  list(
    class = class(value)[[1L]],
    status = value$status,
    checks = normalize_frame(checks),
    issues = normalize_frame(issues)
  )
}

capture_call <- function(call, normalize = identity) {
  tryCatch(
    {
      value <- call()
      list(status = "success", value = normalize(value))
    },
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

as_character_vector <- function(x) as.character(unlist(x, use.names = FALSE))
as_numeric_vector <- function(x) as.numeric(unlist(x, use.names = FALSE))

classification_spec <- fixture$classification
truth <- as_character_vector(classification_spec$truth)
prediction <- as_character_vector(classification_spec$prediction)
probability <- as_numeric_vector(classification_spec$probability)
participant_id <- as_character_vector(classification_spec$participant_id)
stimulus_id <- as_character_vector(classification_spec$stimulus_id)
n <- length(truth)
classification_data <- data.frame(
  trial_id = sprintf("T%02d", seq_len(n)),
  participant_id = participant_id,
  stimulus_id = stimulus_id,
  quality_status = factor(truth, levels = c("pass", "review")),
  stringsAsFactors = FALSE
)
classification_task <- gp3ml::declare_gazepoint_task(
  classification_data,
  outcome = "quality_status",
  purpose = "Predict observed recording quality.",
  task_type = "classification",
  unit_id = "trial_id",
  participant_id = "participant_id",
  stimulus_id = "stimulus_id",
  generalization_target = "new_participants_and_new_stimuli",
  positive = "review"
)

regression_spec <- fixture$regression
regression_truth <- as_numeric_vector(regression_spec$truth)
regression_prediction <- as_numeric_vector(regression_spec$prediction)
regression_data <- data.frame(
  trial_id = sprintf("R%02d", seq_along(regression_truth)),
  observed_duration = regression_truth,
  stringsAsFactors = FALSE
)
regression_task <- gp3ml::declare_gazepoint_task(
  regression_data,
  outcome = "observed_duration",
  purpose = "Predict an observed response duration.",
  task_type = "regression",
  unit_id = "trial_id",
  generalization_target = "new_trials_known_participants"
)

run_classification_bootstrap <- function(unit) {
  set.seed(20260831)
  before <- .Random.seed
  value <- gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task,
    truth = classification_data$quality_status,
    prediction = factor(prediction, levels = c("pass", "review")),
    probability = probability,
    participant_id = participant_id,
    stimulus_id = stimulus_id,
    unit = unit,
    bootstrap = as.integer(classification_spec$bootstrap),
    conf_level = as.numeric(classification_spec$conf_level),
    seed = as.integer(classification_spec$seed)
  )
  after <- .Random.seed
  normalize_target_uncertainty(value, identical(before, after))
}

run_regression_bootstrap <- function() {
  set.seed(20260831)
  before <- .Random.seed
  value <- gp3ml::bootstrap_gazepoint_metrics_by_unit(
    regression_task,
    truth = regression_truth,
    prediction = regression_prediction,
    unit = "observation",
    bootstrap = as.integer(regression_spec$bootstrap),
    conf_level = as.numeric(regression_spec$conf_level),
    seed = as.integer(regression_spec$seed)
  )
  after <- .Random.seed
  normalize_target_uncertainty(value, identical(before, after))
}

bootstrap_cases <- list()
for (unit in c("observation", "participant", "stimulus", "participant_and_stimulus")) {
  bootstrap_cases[[paste0("classification_", unit)]] <- capture_call(
    function() run_classification_bootstrap(unit)
  )
}
bootstrap_cases$regression_observation <- capture_call(run_regression_bootstrap)
bootstrap_cases$invalid_unit <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, classification_data$quality_status,
    probability = probability, unit = "rows"
  )
})
bootstrap_cases$bootstrap_zero <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, classification_data$quality_status,
    probability = probability, bootstrap = 0L
  )
})
bootstrap_cases$truth_too_short <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, factor("pass", levels = c("pass", "review")),
    probability = 0.2
  )
})
bootstrap_cases$probability_length_mismatch <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, classification_data$quality_status,
    probability = probability[-length(probability)]
  )
})
bootstrap_cases$prediction_length_mismatch <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    regression_task, regression_truth,
    prediction = regression_prediction[-length(regression_prediction)]
  )
})
bootstrap_cases$participant_length_mismatch <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, classification_data$quality_status,
    probability = probability,
    participant_id = participant_id[-length(participant_id)],
    unit = "participant"
  )
})
bootstrap_cases$stimulus_length_mismatch <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, classification_data$quality_status,
    probability = probability,
    stimulus_id = stimulus_id[-length(stimulus_id)],
    unit = "stimulus"
  )
})
participant_missing <- participant_id
participant_missing[[1L]] <- NA_character_
bootstrap_cases$participant_missing_identifier <- capture_call(function() {
  gp3ml::bootstrap_gazepoint_metrics_by_unit(
    classification_task, classification_data$quality_status,
    probability = probability,
    participant_id = participant_missing,
    unit = "participant",
    bootstrap = 2L
  )
})

metrics <- do.call(rbind, lapply(fixture$resample_metrics, function(row) {
  data.frame(
    repeat = as.integer(row$repeat),
    fold = as.integer(row$fold),
    fold_id = as.character(row$fold_id),
    metric = as.character(row$metric),
    value = as.numeric(row$value),
    stringsAsFactors = FALSE
  )
}))
evaluation <- structure(
  list(metrics = metrics, generalization_target = "new_participants"),
  class = "gp3ml_resample_evaluation"
)
fold_uncertainty <- gp3ml::summarize_gazepoint_resample_uncertainty(
  evaluation, unit = "fold", conf_level = 0.90
)
repeat_uncertainty <- gp3ml::summarize_gazepoint_resample_uncertainty(
  evaluation, unit = "repeat", conf_level = 0.90
)

summarize_cases <- list(
  fold_summary = list(status = "success", value = normalize_resample_uncertainty(fold_uncertainty)),
  repeat_summary = list(status = "success", value = normalize_resample_uncertainty(repeat_uncertainty)),
  invalid_unit = capture_call(function() {
    gp3ml::summarize_gazepoint_resample_uncertainty(evaluation, unit = "participant")
  }),
  invalid_object = capture_call(function() {
    gp3ml::summarize_gazepoint_resample_uncertainty(42)
  }),
  empty_metrics = capture_call(function() {
    gp3ml::summarize_gazepoint_resample_uncertainty(
      structure(
        list(metrics = data.frame(), generalization_target = "new_participants"),
        class = "gp3ml_resample_evaluation"
      )
    )
  }),
  repeat_missing_column = capture_call(function() {
    broken <- structure(
      list(
        metrics = metrics[setdiff(names(metrics), "repeat")],
        generalization_target = "new_participants"
      ),
      class = "gp3ml_resample_evaluation"
    )
    gp3ml::summarize_gazepoint_resample_uncertainty(broken, unit = "repeat")
  })
)

target_clean <- gp3ml::bootstrap_gazepoint_metrics_by_unit(
  classification_task,
  truth = classification_data$quality_status,
  prediction = factor(prediction, levels = c("pass", "review")),
  probability = probability,
  participant_id = participant_id,
  stimulus_id = stimulus_id,
  unit = "observation",
  bootstrap = 10L,
  seed = as.integer(classification_spec$seed)
)
damaged_target <- target_clean
damaged_target$limitations <- ""
damaged_resample <- fold_uncertainty
damaged_resample$generalization_target <- ""

validation_cases <- list(
  target_clean = capture_call(
    function() gp3ml::validate_gazepoint_target_uncertainty(target_clean),
    normalize_validation
  ),
  resample_clean = capture_call(
    function() gp3ml::validate_gazepoint_target_uncertainty(fold_uncertainty),
    normalize_validation
  ),
  target_missing_limitations = capture_call(
    function() gp3ml::validate_gazepoint_target_uncertainty(damaged_target),
    normalize_validation
  ),
  resample_missing_target = capture_call(
    function() gp3ml::validate_gazepoint_target_uncertainty(damaged_resample),
    normalize_validation
  ),
  invalid_object = capture_call(function() {
    gp3ml::validate_gazepoint_target_uncertainty(42)
  })
)

normalize_csv <- function(path) {
  lines <- readLines(path, warn = FALSE)
  if (!length(lines) || all(!nzchar(lines))) return(list(columns = list(), rows = list()))
  data <- tryCatch(
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(error) data.frame()
  )
  normalize_frame(data)
}

writer_evidence <- function(value, prefix) {
  directory <- tempfile("gp3ml-r-parity-")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  paths <- gp3ml::write_gazepoint_target_uncertainty(
    value, directory, prefix = prefix, overwrite = FALSE
  )
  result <- list()
  for (name in sort(names(paths))) {
    path <- paths[[name]]
    result[[name]] <- list(
      basename = basename(path),
      table = normalize_csv(path)
    )
  }
  result
}

writer_cases <- list(
  write_target = capture_call(function() writer_evidence(target_clean, "uncertainty_target")),
  write_resample = capture_call(function() writer_evidence(fold_uncertainty, "uncertainty_resample")),
  invalid_object = capture_call(function() {
    gp3ml::write_gazepoint_target_uncertainty(42, tempdir())
  })
)
overwrite_directory <- tempfile("gp3ml-r-parity-overwrite-")
dir.create(overwrite_directory, recursive = TRUE)
gp3ml::write_gazepoint_target_uncertainty(
  fold_uncertainty,
  overwrite_directory,
  prefix = "overwrite_case",
  overwrite = FALSE
)
writer_cases$overwrite_refusal <- capture_call(function() {
  gp3ml::write_gazepoint_target_uncertainty(
    fold_uncertainty,
    overwrite_directory,
    prefix = "overwrite_case",
    overwrite = FALSE
  )
})
unlink(overwrite_directory, recursive = TRUE, force = TRUE)

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  bootstrap_cases = bootstrap_cases,
  summarize_cases = summarize_cases,
  validation_cases = validation_cases,
  writer_cases = writer_cases
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  result,
  output_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  na = "null",
  digits = 16
)
cat("\n", file = output_path, append = TRUE)
