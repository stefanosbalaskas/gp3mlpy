args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_diagnostics.R <fixture.json> <output.json>")
}

fixture <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
output_path <- args[[2]]

scalar <- function(value) {
  if (length(value) == 0L || is.null(value) || is.na(value[[1L]])) return(NULL)
  value <- value[[1L]]
  if (is.factor(value)) return(as.character(value))
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) {
    number <- as.numeric(value)
    return(if (is.finite(number)) number else NULL)
  }
  as.character(value)
}

frame <- function(data) {
  if (!is.data.frame(data) || nrow(data) == 0L) return(list())
  lapply(seq_len(nrow(data)), function(index) {
    row <- data[index, , drop = FALSE]
    setNames(lapply(names(row), function(name) scalar(row[[name]])), names(row))
  })
}

capture_case <- function(call, normalize) {
  tryCatch(
    {
      value <- call()
      force(value)
      list(status = "success", value = normalize(value))
    },
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

make_data <- function(outcome_type = "categorical") {
  grid <- expand.grid(
    repetition = 1:3,
    stimulus = 1:4,
    participant = 1:6,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid <- grid[order(grid$participant, grid$stimulus, grid$repetition), , drop = FALSE]
  source <- seq_len(nrow(grid))
  outcome <- if (identical(outcome_type, "numeric")) {
    180 + source / 3
  } else {
    factor(
      ifelse((grid$participant + grid$stimulus + grid$repetition) %% 2L == 0L, "yes", "no"),
      levels = c("no", "yes")
    )
  }
  data.frame(
    participant_id = sprintf("P%02d", grid$participant),
    stimulus_id = sprintf("S%02d", grid$stimulus),
    trial_id = sprintf("S%02d_T%d", grid$stimulus, grid$repetition),
    outcome = outcome,
    fixation_duration = as.numeric(180 + source),
    pupil_change = as.numeric(source) / 1000,
    stringsAsFactors = FALSE
  )
}

make_manifest <- function() {
  gp3ml::create_gazepoint_feature_manifest(
    features = c("fixation_duration", "pupil_change"),
    scientific_source = c("Gazepoint fixation export", "Gazepoint all-gaze export"),
    source_table = c("fixations", "all_gaze"),
    transformation = c("Trial-level mean", "Baseline-adjusted change"),
    availability_stage = "during_exposure",
    prediction_time_available = TRUE,
    outcome_derived = FALSE,
    post_outcome = FALSE,
    identifier = FALSE,
    preprocessing_scope = "none",
    fold_local_required = FALSE,
    reviewer_notes = ""
  )
}

make_plan <- function(
    target = "new_participants",
    v = 3L,
    repeats = 1L,
    seed = 101L,
    outcome_type = "categorical") {
  gp3ml::create_gazepoint_group_folds(
    data = make_data(outcome_type),
    outcome = "outcome",
    predictors = c("fixation_duration", "pupil_change"),
    feature_manifest = make_manifest(),
    generalization_target = target,
    participant_id = "participant_id",
    trial_id = "trial_id",
    stimulus_id = "stimulus_id",
    v = v,
    repeats = repeats,
    seed = seed
  )
}

normalize_validation <- function(value) {
  list(
    class = class(value)[[1L]],
    status = value$status,
    summary = frame(value$summary),
    checks = frame(value$checks),
    issues = frame(value$issues)
  )
}

normalize_success <- function(value) {
  outcome <- value$outcome_balance
  numeric_rows <- outcome[outcome$metric_type == "numeric", , drop = FALSE]
  nonempty_numeric <- numeric_rows[numeric_rows$n > 0L, , drop = FALSE]
  categorical <- outcome[outcome$metric_type == "categorical", , drop = FALSE]
  categorical_levels <- sort(unique(as.character(
    categorical$outcome_level[!is.na(categorical$outcome_level)]
  )))
  ratios <- value$repeat_metrics$assessment_size_ratio
  ratios <- sort(round(as.numeric(ratios[!is.na(ratios)]), 12))

  list(
    class = class(value)[[1L]],
    metadata = list(
      outcome = value$metadata$outcome,
      generalization_target = value$metadata$generalization_target,
      repeats = as.integer(value$metadata$repeats),
      n_source_rows = as.integer(value$metadata$n_source_rows),
      n_folds_total = as.integer(value$metadata$n_folds_total),
      imbalance_review = as.numeric(value$metadata$imbalance_review),
      imbalance_fail = as.numeric(value$metadata$imbalance_fail),
      outcome_type = value$metadata$outcome_type,
      source_validation_status = value$metadata$source_validation_status,
      source_audit_status = value$metadata$source_audit_status
    ),
    row_counts = list(
      fold_metrics = as.integer(nrow(value$fold_metrics)),
      repeat_metrics = as.integer(nrow(value$repeat_metrics)),
      outcome_balance = as.integer(nrow(value$outcome_balance)),
      group_balance = as.integer(nrow(value$group_balance)),
      assessment_coverage = as.integer(nrow(value$assessment_coverage)),
      exclusion_summary = as.integer(nrow(value$exclusion_summary))
    ),
    columns = list(
      fold_metrics = as.list(names(value$fold_metrics)),
      repeat_metrics = as.list(names(value$repeat_metrics)),
      outcome_balance = as.list(names(value$outcome_balance)),
      group_balance = as.list(names(value$group_balance)),
      assessment_coverage = as.list(names(value$assessment_coverage)),
      exclusion_summary = as.list(names(value$exclusion_summary))
    ),
    coverage_once_per_repeat = nrow(value$assessment_coverage) > 0L &&
      all(value$assessment_coverage$n_assessment == 1L),
    partition_accounting = all(
      value$fold_metrics$n_total == value$fold_metrics$n_analysis +
        value$fold_metrics$n_assessment + value$fold_metrics$n_excluded
    ),
    nonempty_analysis_assessment = all(
      value$fold_metrics$n_analysis > 0L & value$fold_metrics$n_assessment > 0L
    ),
    assessment_groups_present = nrow(value$group_balance) > 0L &&
      all(value$group_balance$n_assessment_groups > 0L),
    assessment_size_ratios = as.list(ratios),
    total_excluded = as.integer(sum(value$fold_metrics$n_excluded)),
    categorical_levels = as.list(categorical_levels),
    categorical_missing_assessment_level = nrow(categorical) > 0L &&
      any(categorical$n[categorical$partition == "assessment"] == 0L),
    numeric_nonempty_means_finite = nrow(nonempty_numeric) == 0L ||
      all(is.finite(nonempty_numeric$mean)),
    validation = normalize_validation(value$validation)
  )
}

success_case <- function(case) {
  plan <- make_plan(
    target = case$target,
    v = unlist(case$v, use.names = FALSE),
    repeats = as.integer(case$repeats),
    outcome_type = case$outcome_type
  )
  if (isTRUE(case$sparse_first_assessment)) {
    plan$folds[[1L]]$assessment$outcome <- factor(
      rep("no", nrow(plan$folds[[1L]]$assessment)),
      levels = c("no", "yes")
    )
  }
  gp3ml::diagnose_gazepoint_group_folds(plan)
}

error_case <- function(kind) {
  if (identical(kind, "invalid_object")) {
    return(gp3ml::diagnose_gazepoint_group_folds(list()))
  }
  plan <- make_plan()
  if (identical(kind, "bad_review_threshold")) {
    return(gp3ml::diagnose_gazepoint_group_folds(plan, imbalance_review = 0.9))
  }
  if (identical(kind, "threshold_order")) {
    return(gp3ml::diagnose_gazepoint_group_folds(
      plan, imbalance_review = 2, imbalance_fail = 1.5
    ))
  }
  if (identical(kind, "missing_component")) {
    plan$audit <- NULL
    return(gp3ml::diagnose_gazepoint_group_folds(plan))
  }
  if (identical(kind, "missing_summary_column")) {
    plan$fold_summary$n_total <- NULL
    return(gp3ml::diagnose_gazepoint_group_folds(plan))
  }
  if (identical(kind, "missing_outcome_column")) {
    plan$folds[[1L]]$analysis$outcome <- NULL
    return(gp3ml::diagnose_gazepoint_group_folds(plan))
  }
  stop(sprintf("Unknown diagnostics error case: %s", kind), call. = FALSE)
}

validation_case <- function(kind) {
  if (identical(kind, "invalid_object")) {
    return(gp3ml::validate_gazepoint_fold_diagnostics(list()))
  }
  diagnostics <- gp3ml::diagnose_gazepoint_group_folds(make_plan())
  if (identical(kind, "clean")) {
    return(gp3ml::validate_gazepoint_fold_diagnostics(diagnostics))
  }
  if (identical(kind, "damaged_coverage")) {
    diagnostics$assessment_coverage$n_assessment[[1L]] <- 0L
  } else if (identical(kind, "review_imbalance")) {
    diagnostics$repeat_metrics$assessment_size_ratio[[1L]] <- 1.6
  } else if (identical(kind, "fail_imbalance")) {
    diagnostics$repeat_metrics$assessment_size_ratio[[1L]] <- 2.5
  } else if (identical(kind, "missing_level")) {
    selected <- which(
      diagnostics$outcome_balance$metric_type == "categorical" &
        diagnostics$outcome_balance$partition == "assessment" &
        as.character(diagnostics$outcome_balance$outcome_level) == "yes"
    )[[1L]]
    diagnostics$outcome_balance$n[[selected]] <- 0L
  } else if (identical(kind, "missing_component")) {
    diagnostics$group_balance <- NULL
  } else {
    stop(sprintf("Unknown diagnostics validation case: %s", kind), call. = FALSE)
  }
  gp3ml::validate_gazepoint_fold_diagnostics(diagnostics)
}

writer_case <- function(kind) {
  if (identical(kind, "invalid_object")) {
    return(gp3ml::write_gazepoint_fold_diagnostics_csv(list(), tempfile()))
  }
  diagnostics <- gp3ml::diagnose_gazepoint_group_folds(make_plan())
  directory <- tempfile("gp3ml_diagnostics_parity_")
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)

  if (identical(kind, "unknown_table")) {
    return(gp3ml::write_gazepoint_fold_diagnostics_csv(
      diagnostics, directory, tables = "unknown"
    ))
  }
  if (identical(kind, "bad_overwrite")) {
    return(gp3ml::write_gazepoint_fold_diagnostics_csv(
      diagnostics, directory, overwrite = "yes"
    ))
  }
  if (identical(kind, "overwrite")) {
    gp3ml::write_gazepoint_fold_diagnostics_csv(
      diagnostics, directory, prefix = "parity", tables = "fold_metrics"
    )
    return(gp3ml::write_gazepoint_fold_diagnostics_csv(
      diagnostics, directory, prefix = "parity", tables = "fold_metrics"
    ))
  }
  if (identical(kind, "selected")) {
    tables <- c("fold_metrics", "repeat_metrics", "validation_checks")
    paths <- gp3ml::write_gazepoint_fold_diagnostics_csv(
      diagnostics, directory, prefix = "parity", tables = tables
    )
    result <- list()
    for (name in tables) {
      exported <- utils::read.csv(
        paths[[name]], stringsAsFactors = FALSE, check.names = FALSE
      )
      result[[name]] <- list(
        basename = basename(paths[[name]]),
        columns = as.list(names(exported)),
        n_rows = as.integer(nrow(exported))
      )
    }
    return(result)
  }
  stop(sprintf("Unknown diagnostics writer case: %s", kind), call. = FALSE)
}

successes <- setNames(lapply(fixture$success_cases, function(case) {
  capture_case(function() success_case(case), normalize_success)
}), vapply(fixture$success_cases, `[[`, character(1), "id"))

errors <- setNames(lapply(fixture$error_cases, function(case) {
  capture_case(function() error_case(case$kind), function(value) TRUE)
}), vapply(fixture$error_cases, `[[`, character(1), "id"))

validations <- setNames(lapply(fixture$validation_cases, function(case) {
  capture_case(function() validation_case(case$kind), normalize_validation)
}), vapply(fixture$validation_cases, `[[`, character(1), "id"))

writers <- setNames(lapply(fixture$writer_cases, function(case) {
  capture_case(function() writer_case(case$kind), function(value) value)
}), vapply(fixture$writer_cases, `[[`, character(1), "id"))

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  successes = successes,
  errors = errors,
  validations = validations,
  writers = writers
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  result,
  output_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  na = "null",
  null = "null",
  digits = NA
)
cat("\n", file = output_path, append = TRUE)
