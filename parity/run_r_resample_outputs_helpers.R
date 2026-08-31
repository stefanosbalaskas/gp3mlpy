scalar <- function(value) {
  if (length(value) == 0L || is.null(value)) return(NULL)
  if (is.factor(value)) value <- as.character(value)
  value <- value[[1L]]
  if (is.na(value)) return(NULL)
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) {
    value <- as.numeric(value)
    return(if (is.finite(value)) value else NULL)
  }
  as.character(value)
}

normalize_frame <- function(data, sort_by = character()) {
  data <- as.data.frame(data, stringsAsFactors = FALSE, check.names = FALSE)
  keys <- sort_by[sort_by %in% names(data)]
  if (length(keys) && nrow(data)) {
    ordering <- do.call(
      order,
      c(unname(data[keys]), list(na.last = TRUE, method = "radix"))
    )
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

capture_call <- function(call, normalize = identity) {
  tryCatch(
    {
      value <- suppressWarnings(call())
      list(status = "success", value = normalize(value))
    },
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

as_character_vector <- function(x) {
  vapply(x, function(value) {
    if (is.null(value)) NA_character_ else as.character(value[[1L]])
  }, character(1))
}
as_numeric_vector <- function(x) as.numeric(unlist(x, use.names = FALSE))
as_logical_vector <- function(x) as.logical(unlist(x, use.names = FALSE))

task_fixture <- function(spec) {
  truth <- as_character_vector(spec$truth)
  n <- length(truth)
  data <- data.frame(
    trial_id = sprintf("T%02d", seq_len(n)),
    participant_id = sprintf("P%02d", seq_len(n)),
    stimulus_id = sprintf("S%02d", ((seq_len(n) - 1L) %% 2L) + 1L),
    quality_status = factor(truth, levels = c("pass", "review")),
    stringsAsFactors = FALSE
  )
  task <- gp3ml::declare_gazepoint_task(
    data,
    outcome = "quality_status",
    purpose = "Predict observed recording quality.",
    task_type = "classification",
    unit_id = "trial_id",
    participant_id = "participant_id",
    stimulus_id = "stimulus_id",
    generalization_target = "new_participants",
    positive = "review"
  )
  list(data = data, task = task)
}

prediction_table <- function(spec) {
  truth <- as_character_vector(spec$truth)
  prediction <- as_character_vector(spec$prediction)
  probability <- as_numeric_vector(spec$probability)
  n <- length(truth)
  data.frame(
    .gp3ml_source_row = seq_len(n),
    trial_id = sprintf("T%02d", seq_len(n)),
    participant_id = sprintf("P%02d", seq_len(n)),
    stimulus_id = sprintf("S%02d", ((seq_len(n) - 1L) %% 2L) + 1L),
    `repeat` = rep(1L, n),
    fold = c(1L, 1L, 2L, 2L),
    fold_id = c(
      "Repeat01_Fold01", "Repeat01_Fold01",
      "Repeat01_Fold02", "Repeat01_Fold02"
    ),
    candidate_id = rep(NA_character_, n),
    stage = rep("assessment", n),
    truth = truth,
    prediction = prediction,
    probability = probability,
    prediction_missing = rep(FALSE, n),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fold_status <- function(failed = FALSE) {
  out <- data.frame(
    `repeat` = c(1L, 1L),
    fold = c(1L, 2L),
    fold_id = c("Repeat01_Fold01", "Repeat01_Fold02"),
    status = c("pass", "pass"),
    leakage_status = c("pass", "pass"),
    n_analysis = c(2L, 2L),
    n_assessment = c(2L, 2L),
    n_excluded = c(0L, 0L),
    n_predictions = c(2L, 2L),
    n_missing_predictions = c(0L, 0L),
    warning_count = c(0L, 0L),
    error = c(NA_character_, NA_character_),
    warnings = c("", ""),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (failed) {
    out <- rbind(
      out,
      data.frame(
        `repeat` = 1L,
        fold = 3L,
        fold_id = "Repeat01_Fold03",
        status = "fail",
        leakage_status = "pass",
        n_analysis = 2L,
        n_assessment = 2L,
        n_excluded = 0L,
        n_predictions = 0L,
        n_missing_predictions = 2L,
        warning_count = 0L,
        error = "synthetic fold failure",
        warnings = "",
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    )
  }
  rownames(out) <- NULL
  out
}

metrics_table <- function(spec) {
  rows <- lapply(spec, function(row) {
    data.frame(
      `repeat` = as.integer(row[["repeat"]]),
      fold = as.integer(row$fold),
      fold_id = as.character(row$fold_id),
      metric = as.character(row$metric),
      value = as.numeric(row$value),
      n = as.integer(row$n),
      threshold = as.numeric(row$threshold),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

evaluation_fixture <- function(spec, metric_spec, failed = FALSE) {
  task_parts <- task_fixture(spec)
  status <- fold_status(failed)
  object <- structure(
    list(
      fold_results = list(),
      predictions = prediction_table(spec),
      metrics = metrics_table(metric_spec),
      fold_status = status,
      excluded = data.frame(
        trial_id = character(),
        `repeat` = integer(),
        fold = integer(),
        fold_id = character(),
        stringsAsFactors = FALSE,
        check.names = FALSE
      ),
      task = task_parts$task,
      predictors = "tracking_ratio",
      engine = "glm",
      preprocessor_args = list(),
      engine_args = list(),
      threshold = as.numeric(spec$threshold),
      seed = 101L,
      generalization_target = "new_participants",
      folds_metadata = list(n_folds_total = nrow(status)),
      folds_audit = NULL,
      folds_validation = NULL,
      keep_models = FALSE,
      call = quote(parity_fixture())
    ),
    class = "gp3ml_resample_evaluation"
  )
  object$validation <- gp3ml::validate_gazepoint_resample_evaluation(object)
  object
}

normalize_validation <- function(value) {
  list(
    class = class(value)[[1L]],
    status = value$status,
    checks = normalize_frame(value$checks, "check_id"),
    issues = normalize_frame(value$issues, "check_id")
  )
}

normalize_summary <- function(value) {
  list(
    class = class(value)[[1L]],
    aggregation = value$aggregation,
    conf_level = as.numeric(value$conf_level),
    generalization_target = value$generalization_target,
    n_folds = as.integer(value$n_folds),
    n_failed_folds = as.integer(value$n_failed_folds),
    summary = normalize_frame(value$summary, "metric")
  )
}

read_csv_frame <- function(path) {
  if (!file.exists(path) || file.info(path)$size == 0L) {
    return(normalize_frame(data.frame()))
  }
  value <- tryCatch(
    utils::read.csv(
      path,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      na.strings = "",
      strip.white = FALSE
    ),
    error = function(error) data.frame()
  )
  normalize_frame(value)
}

feature_write <- function(call, filename) {
  directory <- tempfile("gp3mlpy_feature_writer_")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  path <- file.path(directory, filename)
  returned <- call(path)
  list(
    basename = basename(returned),
    exists = file.exists(path),
    table = read_csv_frame(path)
  )
}

resample_write <- function(value) {
  directory <- tempfile("gp3mlpy_resample_writer_")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  returned <- gp3ml::write_gazepoint_resample_evaluation(
    value, directory, prefix = "parity_eval", overwrite = FALSE
  )
  tables <- lapply(names(returned), function(name) {
    path <- returned[[name]]
    list(
      basename = basename(path),
      exists = file.exists(path),
      table = read_csv_frame(path)
    )
  })
  names(tables) <- names(returned)
  list(keys = as.list(names(returned)), tables = tables)
}

feature_overwrite <- function(manifest) {
  directory <- tempfile("gp3mlpy_feature_overwrite_")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  path <- file.path(directory, "manifest.csv")
  gp3ml::write_gazepoint_feature_manifest_csv(manifest, path)
  gp3ml::write_gazepoint_feature_manifest_csv(manifest, path, overwrite = FALSE)
}

feature_invalid_extension <- function(manifest) {
  directory <- tempfile("gp3mlpy_feature_extension_")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  gp3ml::write_gazepoint_feature_manifest_csv(
    manifest, file.path(directory, "manifest.txt")
  )
}

feature_plain_checks <- function(manifest) {
  directory <- tempfile("gp3mlpy_feature_plain_")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  gp3ml::write_gazepoint_feature_manifest_csv(
    manifest, file.path(directory, "checks.csv"), table = "checks"
  )
}

resample_overwrite <- function(value) {
  directory <- tempfile("gp3mlpy_resample_overwrite_")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)
  gp3ml::write_gazepoint_resample_evaluation(
    value, directory, prefix = "parity_eval", overwrite = FALSE
  )
  gp3ml::write_gazepoint_resample_evaluation(
    value, directory, prefix = "parity_eval", overwrite = FALSE
  )
}
