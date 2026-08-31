library(jsonlite)

normalize_scalar <- function(value) {
  if (length(value) == 0L) return(NULL)
  if (length(value) > 1L) return(lapply(value, normalize_scalar))
  if (is.na(value)) return(NULL)
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) {
    if (!is.finite(value)) return(NULL)
    return(as.numeric(value))
  }
  as.character(value)
}

normalize_value <- function(value) {
  if (is.null(value)) return(NULL)
  if (is.data.frame(value)) return(normalize_frame(value))
  if (is.list(value)) {
    if (!is.null(names(value)) && all(nzchar(names(value)))) {
      out <- lapply(value, normalize_value)
      names(out) <- names(value)
      return(out)
    }
    return(lapply(value, normalize_value))
  }
  if (length(value) > 1L) return(lapply(as.list(value), normalize_value))
  normalize_scalar(value)
}

normalize_frame <- function(frame) {
  rows <- vector("list", nrow(frame))
  if (nrow(frame)) {
    for (i in seq_len(nrow(frame))) {
      row <- vector("list", ncol(frame))
      names(row) <- names(frame)
      for (j in seq_along(frame)) {
        column <- frame[[j]]
        value <- if (is.list(column)) column[[i]] else column[i]
        row[j] <- list(normalize_value(value))
      }
      rows[[i]] <- row
    }
  }
  list(columns = names(frame), rows = rows)
}

capture_case <- function(expr) {
  tryCatch(
    list(status = "success", value = force(expr)),
    error = function(e) list(
      status = "error",
      error_type = class(e)[[1L]],
      message = conditionMessage(e)
    )
  )
}

grid_value <- function(grid) {
  list(class = class(grid)[[1L]], candidates = normalize_frame(grid$candidates))
}

base_grid <- function(textual_complexity = FALSE) {
  complexity <- if (textual_complexity) c("low", "low", "high") else c(2, 1, 3)
  candidates <- data.frame(
    candidate_id = c("candidate_001", "candidate_002", "candidate_003"),
    label = c("glm A", "glm B", "ranger C"),
    engine = c("glm", "glm", "ranger"),
    threshold = c(0.5, 0.5, 0.5),
    complexity = complexity,
    interpretability = c("high", "high", "medium"),
    stringsAsFactors = FALSE
  )
  candidates$engine_args <- list(
    list(max_iter = 50),
    list(max_iter = 100),
    list(trees = 100)
  )
  candidates$preprocessor_args <- list(
    list(center = TRUE),
    list(center = FALSE),
    list()
  )
  structure(list(candidates = candidates, created = NULL, call = "fixture"), class = "gp3ml_tuning_grid")
}

base_comparison <- function(unique_primary = FALSE, low_success_second = FALSE) {
  primary_1 <- if (unique_primary) 0.82 else 0.80
  success_2 <- if (low_success_second) 0.70 else 1.0
  data.frame(
    candidate_id = c("candidate_001", "candidate_001", "candidate_002", "candidate_002", "candidate_003"),
    label = c("glm A", "glm A", "glm B", "glm B", "ranger C"),
    engine = c("glm", "glm", "glm", "glm", "ranger"),
    threshold = rep(0.5, 5),
    complexity = c("2", "2", "1", "1", "3"),
    interpretability = c("high", "high", "high", "high", "medium"),
    candidate_status = c("pass", "pass", "pass", "pass", "fail"),
    success_prop = c(1, 1, success_2, success_2, 0.5),
    failed_folds = c(0L, 0L, 0L, 0L, 2L),
    error = c(NA_character_, NA_character_, NA_character_, NA_character_, "fixture failure"),
    metric = c("roc_auc", "brier", "roc_auc", "brier", NA_character_),
    mean = c(primary_1, 0.20, 0.80, 0.15, NA_real_),
    sd = c(0.02, 0.01, 0.03, 0.02, NA_real_),
    n_folds = c(3L, 3L, 3L, 3L, 0L),
    direction = c("maximize", "minimize", "maximize", "minimize", NA_character_),
    stringsAsFactors = FALSE
  )
}

base_results <- function() {
  list(
    list(candidate_id = "candidate_001", status = "pass", success_prop = 1, warnings = character(), error = NA_character_),
    list(candidate_id = "candidate_002", status = "pass", success_prop = 1, warnings = "fixture warning", error = NA_character_),
    list(candidate_id = "candidate_003", status = "fail", success_prop = 0.5, warnings = character(), error = "fixture failure")
  )
}

base_tuning <- function(textual_complexity = FALSE, unique_primary = FALSE, low_success_second = FALSE) {
  object <- structure(
    list(
      grid = base_grid(textual_complexity),
      results = base_results(),
      comparison = base_comparison(unique_primary, low_success_second),
      task = structure(list(generalization_target = "new_participants"), class = "gp3ml_task"),
      predictors = "x",
      folds_metadata = list(generalization_target = "new_participants"),
      metrics_requested = NULL,
      seed = 1L,
      keep_evaluations = FALSE,
      selection = NULL,
      call = "fixture"
    ),
    class = "gp3ml_model_tuning"
  )
  object$validation <- gp3ml::validate_gazepoint_model_tuning(object)
  object
}

selection_value <- function(selection) {
  list(
    class = class(selection)[[1L]],
    candidate_id = selection$candidate_id,
    primary_metric = selection$primary_metric,
    direction = selection$direction,
    primary_value = normalize_scalar(selection$primary_value),
    minimum_success_prop = normalize_scalar(selection$minimum_success_prop),
    tie_breakers = normalize_frame(selection$tie_breakers),
    rationale = selection$rationale,
    eligible_candidates = as.list(selection$eligible_candidates),
    refit_performed = isTRUE(selection$refit_performed),
    autonomous_selection = isTRUE(selection$autonomous_selection),
    candidate = normalize_frame(selection$candidate)
  )
}

validation_value <- function(value) {
  list(class = class(value)[[1L]], status = value$status, checks = normalize_frame(value$checks), issues = normalize_frame(value$issues))
}

read_table_value <- function(path) {
  table <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = c(""))
  normalize_frame(table)
}

writer_value <- function(tuning, selection) {
  directory <- tempfile("gp3mlpy-tuning-parity-")
  dir.create(directory)
  paths <- gp3ml::write_gazepoint_model_tuning(tuning, directory, prefix = "parity_tuning", selection = selection, overwrite = FALSE)
  tables <- lapply(paths, read_table_value)
  list(
    path_names = sort(names(paths)),
    basenames = as.list(stats::setNames(basename(paths), names(paths))),
    tables = tables
  )
}
