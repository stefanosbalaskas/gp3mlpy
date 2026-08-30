args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_resampling.R <fixture.json> <output.json>")
}

fixture <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
output_path <- args[[2]]

scalar <- function(value) {
  if (length(value) == 0L || is.null(value) || is.na(value[[1L]])) {
    return(NULL)
  }
  if (is.logical(value)) return(as.logical(value[[1L]]))
  if (is.integer(value)) return(as.integer(value[[1L]]))
  if (is.numeric(value)) {
    number <- as.numeric(value[[1L]])
    if (!is.finite(number)) return(NULL)
    return(number)
  }
  as.character(value[[1L]])
}

frame <- function(data) {
  if (!is.data.frame(data) || nrow(data) == 0L) return(list())
  lapply(seq_len(nrow(data)), function(i) {
    row <- data[i, , drop = FALSE]
    setNames(lapply(names(row), function(name) scalar(row[[name]])), names(row))
  })
}

capture_case <- function(call, normalize) {
  tryCatch(
    list(status = "success", value = normalize(call())),
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

make_data <- function() {
  data <- expand.grid(
    participant_id = sprintf("P%02d", 1:6),
    stimulus_id = sprintf("S%02d", 1:4),
    repetition = 1:3,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  data$trial_id <- paste0(data$stimulus_id, "_T", data$repetition)
  index <- seq_len(nrow(data))
  data$outcome <- as.integer(index %% 2L)
  data$fixation_duration <- 180 + index
  data$pupil_change <- index / 1000
  data$repetition <- NULL
  data
}

make_manifest <- function() {
  gp3ml::create_gazepoint_feature_manifest(
    features = c("fixation_duration", "pupil_change"),
    scientific_source = c(
      "Gazepoint fixation export",
      "Gazepoint all-gaze export"
    ),
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
    repeats = 2L,
    seed = 77L,
    data = make_data(),
    feature_manifest = make_manifest(),
    participant_id = "participant_id",
    trial_id = "trial_id",
    stimulus_id = "stimulus_id") {
  gp3ml::create_gazepoint_group_folds(
    data = data,
    outcome = "outcome",
    predictors = c("fixation_duration", "pupil_change"),
    feature_manifest = feature_manifest,
    generalization_target = target,
    participant_id = participant_id,
    trial_id = trial_id,
    stimulus_id = stimulus_id,
    v = v,
    repeats = repeats,
    seed = seed
  )
}

assessment_coverage <- function(plan) {
  result <- stats::aggregate(
    as.integer(plan$assignments$partition == "assessment"),
    by = list(
      `repeat` = plan$assignments[["repeat"]],
      source_row = plan$assignments$source_row
    ),
    FUN = sum
  )
  names(result)[3L] <- "n_assessment"
  result
}

trial_units <- function(data) {
  unique(paste(data$participant_id, data$trial_id, sep = "::"))
}

normalize_success <- function(plan, case) {
  repeated <- make_plan(
    target = case$target,
    v = unlist(case$v, use.names = FALSE),
    repeats = as.integer(case$repeats),
    seed = as.integer(case$seed)
  )
  coverage <- assessment_coverage(plan)
  source_rows <- seq_len(plan$metadata$n_source_rows)
  all_accounted <- TRUE
  all_nonempty <- TRUE
  participant_overlap <- 0L
  stimulus_overlap <- 0L
  trial_overlap <- 0L
  excluded_positive <- TRUE
  assessment_has_all_participants <- TRUE
  all_participants <- sort(unique(make_data()$participant_id))

  for (fold_object in plan$folds) {
    assigned <- plan$assignments[
      plan$assignments[["repeat"]] == fold_object[["repeat"]] &
        plan$assignments$fold == fold_object$fold,
      ,
      drop = FALSE
    ]
    all_accounted <- all_accounted && identical(
      sort(as.integer(assigned$source_row)),
      as.integer(source_rows)
    )
    all_nonempty <- all_nonempty &&
      nrow(fold_object$analysis) > 0L &&
      nrow(fold_object$assessment) > 0L
    participant_overlap <- participant_overlap + length(intersect(
      unique(as.character(fold_object$analysis$participant_id)),
      unique(as.character(fold_object$assessment$participant_id))
    ))
    stimulus_overlap <- stimulus_overlap + length(intersect(
      unique(as.character(fold_object$analysis$stimulus_id)),
      unique(as.character(fold_object$assessment$stimulus_id))
    ))
    trial_overlap <- trial_overlap + length(intersect(
      trial_units(fold_object$analysis),
      trial_units(fold_object$assessment)
    ))
    if (identical(case$target, "new_participants_and_new_stimuli")) {
      excluded_positive <- excluded_positive && nrow(fold_object$excluded) > 0L
    } else {
      excluded_positive <- excluded_positive && nrow(fold_object$excluded) == 0L
    }
    if (identical(case$target, "new_trials_known_participants")) {
      assessment_has_all_participants <- assessment_has_all_participants && identical(
        sort(unique(as.character(fold_object$assessment$participant_id))),
        all_participants
      )
    }
  }

  target <- case$target
  invariants <- list(
    participant_overlap_zero = if (target %in% c(
      "new_participants", "new_participants_and_new_stimuli"
    )) participant_overlap == 0L else TRUE,
    stimulus_overlap_zero = if (target %in% c(
      "new_stimuli", "new_participants_and_new_stimuli"
    )) stimulus_overlap == 0L else TRUE,
    participant_trial_overlap_zero = if (identical(
      target, "new_trials_known_participants"
    )) trial_overlap == 0L else TRUE,
    assessment_has_all_participants = assessment_has_all_participants,
    excluded_behavior_valid = excluded_positive
  )

  fold_ids <- vapply(plan$folds, `[[`, character(1), "fold_id")
  list(
    class = class(plan)[[1L]],
    target = target,
    metadata = list(
      repeats = as.integer(plan$metadata$repeats),
      n_source_rows = as.integer(plan$metadata$n_source_rows),
      n_folds_per_repeat = as.integer(plan$metadata$n_folds_per_repeat),
      n_folds_total = as.integer(plan$metadata$n_folds_total)
    ),
    fold_count = as.integer(length(plan$folds)),
    fold_ids_unique = !anyDuplicated(fold_ids) && all(nzchar(fold_ids)),
    same_seed_reproducible = identical(plan$assignments, repeated$assignments),
    source_rows_accounted_per_fold = all_accounted,
    all_analysis_assessment_nonempty = all_nonempty,
    assessment_once_per_repeat = all(coverage$n_assessment == 1L),
    audit = list(
      class = class(plan$audit)[[1L]],
      status = plan$audit$status,
      n_summary = as.integer(nrow(plan$audit$summary)),
      n_issues = as.integer(nrow(plan$audit$issues))
    ),
    validation = list(
      class = class(plan$validation)[[1L]],
      status = plan$validation$status,
      summary = frame(plan$validation$summary),
      checks = frame(plan$validation$checks),
      issues = frame(plan$validation$issues)
    ),
    invariants = invariants
  )
}

error_case <- function(kind) {
  if (identical(kind, "missing_manifest")) {
    return(make_plan(feature_manifest = NULL))
  }
  if (identical(kind, "missing_participant")) {
    return(make_plan(participant_id = NULL))
  }
  if (identical(kind, "v_vector_single_target")) {
    return(make_plan(v = c(2L, 3L)))
  }
  if (identical(kind, "v_too_many_participants")) {
    return(make_plan(v = 7L))
  }
  if (identical(kind, "v_too_many_stimuli")) {
    return(make_plan(target = "new_stimuli", v = 5L))
  }
  if (identical(kind, "insufficient_trials")) {
    data <- make_data()
    data$trial_id[data$participant_id == "P01"] <- "ONLY"
    return(make_plan(
      target = "new_trials_known_participants",
      v = 3L,
      data = data
    ))
  }
  stop(sprintf("Unknown resampling error case: %s", kind))
}

normalize_audit <- function(value) {
  list(
    class = class(value)[[1L]],
    status = value$status,
    summary = frame(value$summary),
    issues = frame(value$issues)
  )
}

audit_case <- function(kind) {
  if (identical(kind, "invalid_object")) {
    return(gp3ml::audit_gazepoint_group_folds(list()))
  }
  plan <- make_plan(repeats = 1L)
  if (identical(kind, "clean")) {
    return(gp3ml::audit_gazepoint_group_folds(plan))
  }
  if (identical(kind, "empty_folds")) {
    plan$folds <- list()
    return(gp3ml::audit_gazepoint_group_folds(plan))
  }
  stop(sprintf("Unknown audit case: %s", kind))
}

normalize_validation <- function(value) {
  list(
    class = class(value)[[1L]],
    status = value$status,
    summary = frame(value$summary),
    checks = frame(value$checks),
    issues = frame(value$issues),
    assessment_coverage = frame(value$assessment_coverage)
  )
}

validation_case <- function(kind) {
  if (identical(kind, "invalid_object")) {
    return(gp3ml::validate_gazepoint_group_folds(list()))
  }
  plan <- make_plan(repeats = 1L)
  if (identical(kind, "clean")) {
    return(gp3ml::validate_gazepoint_group_folds(plan))
  }
  if (identical(kind, "assignment_damage")) {
    position <- which(plan$assignments$partition == "assessment")[[1L]]
    plan$assignments$partition[position] <- "analysis"
    return(gp3ml::validate_gazepoint_group_folds(plan))
  }
  if (identical(kind, "missing_component")) {
    plan$audit <- NULL
    return(gp3ml::validate_gazepoint_group_folds(plan))
  }
  stop(sprintf("Unknown validation case: %s", kind))
}

writer_case <- function(kind) {
  plan <- make_plan(repeats = 1L)
  directory <- tempfile()
  dir.create(directory, recursive = TRUE)
  on.exit(unlink(directory, recursive = TRUE, force = TRUE), add = TRUE)

  if (identical(kind, "bad_prefix")) {
    return(gp3ml::write_gazepoint_group_folds_csv(
      plan, directory, prefix = "bad/name"
    ))
  }
  if (identical(kind, "bad_table")) {
    return(gp3ml::write_gazepoint_group_folds_csv(
      plan, directory, tables = "unknown"
    ))
  }
  if (identical(kind, "overwrite")) {
    gp3ml::write_gazepoint_group_folds_csv(
      plan, directory, prefix = "parity", tables = "fold_summary"
    )
    tryCatch(
      gp3ml::write_gazepoint_group_folds_csv(
        plan, directory, prefix = "parity", tables = "fold_summary"
      ),
      error = function(error) {
        message <- conditionMessage(error)
        normalized <- gsub(
          normalizePath(directory, winslash = "/", mustWork = TRUE),
          "<TMP>",
          message,
          fixed = TRUE
        )
        stop(normalized, call. = FALSE)
      }
    )
    stop("Expected overwrite protection error.")
  }
  if (identical(kind, "summary_tables")) {
    paths <- gp3ml::write_gazepoint_group_folds_csv(
      plan,
      directory,
      prefix = "parity",
      tables = c("assignments", "fold_summary")
    )
    result <- list()
    for (name in c("assignments", "fold_summary")) {
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
  if (identical(kind, "include_fold_data")) {
    paths <- gp3ml::write_gazepoint_group_folds_csv(
      plan,
      directory,
      prefix = "parity",
      tables = "fold_summary",
      include_fold_data = TRUE
    )
    basenames <- sort(basename(unlist(paths, use.names = FALSE)))
    return(list(
      n_paths = as.integer(length(paths)),
      expected_n_paths = as.integer(1L + length(plan$folds) * 3L),
      all_exist = all(file.exists(unlist(paths, use.names = FALSE))),
      has_fold_summary = "parity_fold_summary.csv" %in% basenames,
      n_analysis_files = as.integer(sum(grepl("_analysis\\.csv$", basenames))),
      n_assessment_files = as.integer(sum(grepl("_assessment\\.csv$", basenames))),
      n_excluded_files = as.integer(sum(grepl("_excluded\\.csv$", basenames)))
    ))
  }
  stop(sprintf("Unknown writer case: %s", kind))
}

successes <- setNames(lapply(fixture$success_cases, function(case) {
  capture_case(
    function() make_plan(
      target = case$target,
      v = unlist(case$v, use.names = FALSE),
      repeats = as.integer(case$repeats),
      seed = as.integer(case$seed)
    ),
    function(value) normalize_success(value, case)
  )
}), vapply(fixture$success_cases, `[[`, character(1), "id"))

errors <- setNames(lapply(fixture$error_cases, function(case) {
  capture_case(function() error_case(case$kind), function(value) TRUE)
}), vapply(fixture$error_cases, `[[`, character(1), "id"))

audits <- setNames(lapply(fixture$audit_cases, function(case) {
  capture_case(function() audit_case(case$kind), normalize_audit)
}), vapply(fixture$audit_cases, `[[`, character(1), "id"))

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
  audits = audits,
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
