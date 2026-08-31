args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_governance.R <fixture.json> <output.json>", call. = FALSE)
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
  if (is.factor(value)) return(as.character(value))
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) return(as.numeric(value))
  as.character(value)
}

normalize_frame <- function(data) {
  if (!is.data.frame(data)) stop("Expected a data frame in parity normalization.", call. = FALSE)
  if (nrow(data) == 0L) return(list())
  lapply(seq_len(nrow(data)), function(index) {
    row <- lapply(data[index, , drop = FALSE], normalize_scalar)
    names(row) <- names(data)
    row
  })
}

normalize_task <- function(task) {
  keys <- c(
    "outcome", "purpose", "task_type", "unit_id", "participant_id", "stimulus_id",
    "generalization_target", "positive", "observed_outcome", "sensitive_outcome", "levels", "negative"
  )
  components <- stats::setNames(vector("list", length(keys)), keys)
  for (index in seq_along(keys)) {
    key <- keys[[index]]
    value <- task[[key]]
    if (identical(key, "levels") && !is.null(value)) {
      components[index] <- list(as.character(value))
    } else {
      components[index] <- list(normalize_scalar(value))
    }
  }
  list(class = class(task)[[1L]], components = components)
}

normalize_manifest <- function(manifest) {
  list(
    class = class(manifest)[[1L]],
    columns = names(manifest),
    rows = normalize_frame(manifest)
  )
}

normalize_manifest_validation <- function(validation) {
  list(
    class = class(validation)[[1L]],
    status = validation$status,
    n_features = as.integer(validation$n_features),
    summary = normalize_frame(validation$summary),
    checks = normalize_frame(validation$checks),
    issues = normalize_frame(validation$issues)
  )
}

normalize_role_validation <- function(validation) {
  manifest_status <- if (is.null(validation$manifest_validation)) NULL else validation$manifest_validation$status
  list(
    class = class(validation)[[1L]],
    status = validation$status,
    checks = normalize_frame(validation$checks),
    issues = normalize_frame(validation$issues),
    manifest_validation_status = manifest_status
  )
}

capture_call <- function(call, normalize) {
  tryCatch(
    list(status = "success", value = normalize(call())),
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

make_dataset <- function(spec) {
  kind <- spec$kind
  n <- as.integer(spec$n)
  if (identical(kind, "classification_governance")) {
    pass_n <- as.integer(spec$pass_n)
    levels <- as.character(unlist(spec$outcome_levels, use.names = FALSE))
    outcome <- c(rep(levels[[1L]], pass_n), rep(levels[[2L]], n - pass_n))
    return(data.frame(
      participant_id = sprintf("P%02d", seq_len(n)),
      trial_id = sprintf("T%02d", seq_len(n)),
      stimulus_id = sprintf("S%02d", ((seq_len(n) - 1L) %% 2L) + 1L),
      fixation_duration = as.numeric(180 + seq_len(n)),
      pupil_change = as.numeric((seq_len(n) - 1L) / 10),
      quality_status = factor(outcome, levels = levels),
      stringsAsFactors = FALSE
    ))
  }
  if (identical(kind, "regression_governance")) {
    return(data.frame(
      participant_id = sprintf("R%02d", seq_len(n)),
      trial_id = sprintf("RT%02d", seq_len(n)),
      stimulus_id = sprintf("RS%02d", ((seq_len(n) - 1L) %% 2L) + 1L),
      fixation_duration = as.numeric(200 + 3 * seq_len(n)),
      score = as.numeric(1 + (seq_len(n) - 1L) * 0.5),
      stringsAsFactors = FALSE
    ))
  }
  stop(sprintf("Unknown parity dataset kind: %s", kind), call. = FALSE)
}

vectorize_args <- function(values) {
  lapply(values, function(value) {
    if (is.list(value) && is.null(names(value))) {
      return(unlist(value, use.names = FALSE))
    }
    value
  })
}

datasets <- lapply(fixture$datasets, make_dataset)
task_cases <- stats::setNames(fixture$task_cases, vapply(fixture$task_cases, function(case) case$id, character(1)))
manifest_cases <- stats::setNames(
  fixture$manifest_cases,
  vapply(fixture$manifest_cases, function(case) case$id, character(1))
)

make_task <- function(case_id) {
  case <- task_cases[[case_id]]
  do.call(
    gp3ml::declare_gazepoint_task,
    c(list(data = datasets[[case$dataset]]), vectorize_args(case$args))
  )
}

make_manifest <- function(case_id) {
  case <- manifest_cases[[case_id]]
  do.call(gp3ml::create_gazepoint_feature_manifest, vectorize_args(case$args))
}

task_declarations <- list()
for (case in fixture$task_cases) {
  task_declarations[[case$id]] <- capture_call(
    function() {
      do.call(
        gp3ml::declare_gazepoint_task,
        c(list(data = datasets[[case$dataset]]), vectorize_args(case$args))
      )
    },
    normalize_task
  )
}

use_case_assertions <- list()
for (case in fixture$assert_cases) {
  use_case_assertions[[case$id]] <- capture_call(
    function() {
      if (isTRUE(case$plain_object)) {
        return(gp3ml::assert_gp3ml_use_case(list()))
      }
      task <- make_task(case$task_case)
      if (!is.null(case$mutations)) {
        for (name in names(case$mutations)) {
          task[name] <- list(case$mutations[[name]])
        }
      }
      data <- datasets[[task_cases[[case$task_case]]$dataset]]
      gp3ml::assert_gp3ml_use_case(task, data)
    },
    function(value) isTRUE(value)
  )
}

feature_manifest_create <- list()
feature_manifest_validate <- list()
for (case in fixture$manifest_cases) {
  case_id <- case$id
  operation <- case$operation
  if (operation %in% c("create", "create_validate")) {
    created <- tryCatch(
      list(ok = TRUE, value = do.call(gp3ml::create_gazepoint_feature_manifest, vectorize_args(case$args))),
      error = function(error) list(ok = FALSE, error = error)
    )
    if (isTRUE(created$ok)) {
      feature_manifest_create[[case_id]] <- list(
        status = "success",
        value = normalize_manifest(created$value)
      )
      if (identical(operation, "create_validate")) {
        manifest <- created$value
        feature_manifest_validate[[case_id]] <- capture_call(
          function() gp3ml::validate_gazepoint_feature_manifest(manifest),
          normalize_manifest_validation
        )
      }
    } else {
      feature_manifest_create[[case_id]] <- list(
        status = "error",
        message = conditionMessage(created$error)
      )
    }
  } else if (identical(operation, "validate_raw")) {
    columns <- lapply(case$raw_manifest, function(value) unlist(value, use.names = FALSE))
    raw_manifest <- as.data.frame(columns, stringsAsFactors = FALSE, check.names = FALSE)
    feature_manifest_validate[[case_id]] <- capture_call(
      function() gp3ml::validate_gazepoint_feature_manifest(raw_manifest),
      normalize_manifest_validation
    )
  } else {
    stop(sprintf("Unknown manifest parity operation: %s", operation), call. = FALSE)
  }
}

role_validations <- list()
for (case in fixture$role_cases) {
  role_validations[[case$id]] <- capture_call(
    function() {
      manifest <- NULL
      if (!is.null(case$manifest_case)) manifest <- make_manifest(case$manifest_case)
      gp3ml::validate_gazepoint_ml_roles(
        data = datasets[[case$dataset]],
        task = make_task(case$task_case),
        predictors = unlist(case$predictors, use.names = FALSE),
        feature_manifest = manifest
      )
    },
    normalize_role_validation
  )
}

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  task_declarations = task_declarations,
  use_case_assertions = use_case_assertions,
  role_validations = role_validations,
  feature_manifest_create = feature_manifest_create,
  feature_manifest_validate = feature_manifest_validate
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
