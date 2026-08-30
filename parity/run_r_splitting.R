args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript parity/run_r_splitting.R <fixture.json> <output.json>", call. = FALSE)
fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)
output_path <- args[[2L]]

normalize_scalar <- function(value) {
  if (length(value) == 0L || is.na(value[[1L]]) || (is.numeric(value[[1L]]) && !is.finite(value[[1L]]))) return(NULL)
  value <- value[[1L]]
  if (is.factor(value)) return(as.character(value))
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
capture_call <- function(call, normalize) {
  tryCatch(
    list(status = "success", value = normalize(call())),
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

make_data <- function() {
  grid <- expand.grid(
    repetition = 1:2,
    stimulus = 1:4,
    participant = 1:8,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid <- grid[order(grid$participant, grid$stimulus, grid$repetition), , drop = FALSE]
  source <- seq_len(nrow(grid))
  data.frame(
    participant_id = sprintf("P%02d", grid$participant),
    trial_id = sprintf("S%02d_R%d", grid$stimulus, grid$repetition),
    stimulus_id = sprintf("S%02d", grid$stimulus),
    fixation_duration = as.numeric(150 + source),
    pupil_change = as.numeric((source %% 9) - 4) / 10,
    quality_status = factor(
      ifelse((grid$participant + grid$stimulus + grid$repetition) %% 2 == 0, "pass", "review"),
      levels = c("pass", "review")
    ),
    stringsAsFactors = FALSE
  )
}

make_manifest <- function(clean = TRUE, extra_feature = FALSE) {
  features <- c("fixation_duration", "pupil_change")
  if (isTRUE(extra_feature)) features <- c(features, "extra_feature")
  if (!isTRUE(clean)) return(gp3ml::create_gazepoint_feature_manifest(features = features))
  n <- length(features)
  gp3ml::create_gazepoint_feature_manifest(
    features = features,
    scientific_source = rep("Gazepoint export", n),
    source_table = rep("all_gaze", n),
    transformation = rep("Predefined trial-level feature", n),
    availability_stage = rep("during_exposure", n),
    prediction_time_available = rep(TRUE, n),
    outcome_derived = rep(FALSE, n),
    post_outcome = rep(FALSE, n),
    identifier = rep(FALSE, n),
    preprocessing_scope = rep("none", n),
    fold_local_required = rep(FALSE, n),
    reviewer_notes = rep("", n)
  )
}

make_split <- function(target, data = NULL, manifest = NULL, overrides = list()) {
  if (is.null(data)) data <- make_data()
  if (is.null(manifest)) manifest <- make_manifest()
  arguments <- list(
    data = data,
    outcome = "quality_status",
    predictors = c("fixation_duration", "pupil_change"),
    feature_manifest = manifest,
    generalization_target = target,
    participant_id = "participant_id",
    trial_id = "trial_id",
    stimulus_id = "stimulus_id",
    assessment_prop = 0.25,
    seed = 101L
  )
  for (name in names(overrides)) arguments[name] <- list(overrides[[name]])
  do.call(gp3ml::split_gazepoint_ml_data, arguments)
}

trial_units <- function(data) unique(paste(data$participant_id, data$trial_id, sep = "|"))

normalize_validation <- function(value) {
  list(
    class = class(value)[[1L]],
    status = value$status,
    summary = normalize_frame(value$summary),
    checks = normalize_frame(value$checks),
    issues = normalize_frame(value$issues)
  )
}

normalize_success <- function(split, target) {
  repeated <- make_split(target)
  source_id <- split$metadata$source_row_id
  analysis_source <- as.integer(split$analysis[[source_id]])
  assessment_source <- as.integer(split$assessment[[source_id]])
  excluded_source <- as.integer(split$excluded[[source_id]])
  source_all <- c(analysis_source, assessment_source, excluded_source)
  analysis_participant <- unique(as.character(split$analysis$participant_id))
  assessment_participant <- unique(as.character(split$assessment$participant_id))
  analysis_stimulus <- unique(as.character(split$analysis$stimulus_id))
  assessment_stimulus <- unique(as.character(split$assessment$stimulus_id))

  if (identical(target, "new_trials_known_participants")) {
    invariants <- list(
      analysis_participants = length(analysis_participant),
      assessment_participants = length(assessment_participant),
      participant_trial_overlap = length(intersect(trial_units(split$analysis), trial_units(split$assessment)))
    )
  } else if (identical(target, "new_participants")) {
    invariants <- list(
      analysis_participants = length(analysis_participant),
      assessment_participants = length(assessment_participant),
      participant_overlap = length(intersect(analysis_participant, assessment_participant))
    )
  } else if (identical(target, "new_stimuli")) {
    invariants <- list(
      analysis_stimuli = length(analysis_stimulus),
      assessment_stimuli = length(assessment_stimulus),
      stimulus_overlap = length(intersect(analysis_stimulus, assessment_stimulus))
    )
  } else {
    invariants <- list(
      analysis_participants = length(analysis_participant),
      assessment_participants = length(assessment_participant),
      participant_overlap = length(intersect(analysis_participant, assessment_participant)),
      analysis_stimuli = length(analysis_stimulus),
      assessment_stimuli = length(assessment_stimulus),
      stimulus_overlap = length(intersect(analysis_stimulus, assessment_stimulus))
    )
  }

  list(
    class = class(split)[[1L]],
    target = target,
    summary = normalize_frame(split$summary),
    validation = normalize_validation(split$validation),
    feature_manifest_status = split$feature_manifest_validation$status,
    leakage_status = split$leakage_audit$status,
    row_counts = list(
      analysis = nrow(split$analysis),
      assessment = nrow(split$assessment),
      excluded = nrow(split$excluded)
    ),
    source_rows_unique = length(source_all) == length(unique(source_all)),
    source_rows_accounted = identical(sort(source_all), seq_len(as.integer(split$metadata$n_source_rows))),
    same_seed_reproducible = identical(as.character(split$assignment$partition), as.character(repeated$assignment$partition)),
    target_invariants = invariants
  )
}

error_case <- function(kind) {
  data <- make_data()
  if (identical(kind, "missing_manifest")) {
    return(gp3ml::split_gazepoint_ml_data(
      data = data,
      outcome = "quality_status",
      predictors = c("fixation_duration", "pupil_change"),
      feature_manifest = NULL,
      generalization_target = "new_participants",
      participant_id = "participant_id",
      trial_id = "trial_id",
      stimulus_id = "stimulus_id"
    ))
  }
  if (identical(kind, "review_manifest")) return(make_split("new_participants", manifest = make_manifest(FALSE)))
  if (identical(kind, "missing_participant")) return(make_split("new_participants", overrides = list(participant_id = NULL)))
  if (identical(kind, "missing_stimulus")) return(make_split("new_stimuli", overrides = list(stimulus_id = NULL)))
  if (identical(kind, "bad_assessment_prop")) return(make_split("new_participants", overrides = list(assessment_prop = 1)))
  if (identical(kind, "outcome_predictor")) return(make_split("new_participants", overrides = list(predictors = c("quality_status", "fixation_duration"))))
  if (identical(kind, "identifier_predictor")) return(make_split("new_participants", overrides = list(predictors = c("participant_id", "fixation_duration"))))
  if (identical(kind, "reserved_source_row")) {
    data$.gp3ml_source_row <- seq_len(nrow(data))
    return(make_split("new_participants", data = data))
  }
  if (identical(kind, "predictor_missing_manifest")) {
    manifest <- gp3ml::create_gazepoint_feature_manifest(
      features = "fixation_duration",
      scientific_source = "Gazepoint export",
      source_table = "all_gaze",
      transformation = "Predefined trial-level feature",
      availability_stage = "during_exposure",
      prediction_time_available = TRUE,
      outcome_derived = FALSE,
      post_outcome = FALSE,
      identifier = FALSE,
      preprocessing_scope = "none",
      fold_local_required = FALSE,
      reviewer_notes = ""
    )
    return(make_split("new_participants", manifest = manifest))
  }
  if (identical(kind, "too_few_trials")) {
    data$trial_id[data$participant_id == "P01"] <- "ONLY"
    return(make_split("new_trials_known_participants", data = data))
  }
  stop(sprintf("Unknown split error case: %s", kind), call. = FALSE)
}

validation_case <- function(kind) {
  if (identical(kind, "invalid_object")) return(gp3ml::validate_gazepoint_ml_split(list()))
  split <- make_split("new_participants")
  if (identical(kind, "clean")) return(gp3ml::validate_gazepoint_ml_split(split))
  if (identical(kind, "source_overlap")) {
    source <- split$metadata$source_row_id
    split$assessment[[source]][[1L]] <- split$analysis[[source]][[1L]]
    return(gp3ml::validate_gazepoint_ml_split(split))
  }
  if (identical(kind, "missing_component")) {
    split$assignment <- NULL
    return(gp3ml::validate_gazepoint_ml_split(split))
  }
  stop(sprintf("Unknown validation case: %s", kind), call. = FALSE)
}

writer_case <- function(kind) {
  split <- make_split("new_participants")
  root <- tempfile("gp3ml_split_parity_")
  dir.create(root, recursive = TRUE)
  on.exit(unlink(root, recursive = TRUE, force = TRUE), add = TRUE)
  if (identical(kind, "bad_prefix")) return(gp3ml::write_gazepoint_ml_split_csv(split, root, prefix = "bad/name", tables = "summary"))
  if (identical(kind, "bad_table")) return(gp3ml::write_gazepoint_ml_split_csv(split, root, tables = "unknown"))
  if (identical(kind, "overwrite")) {
    gp3ml::write_gazepoint_ml_split_csv(split, root, prefix = "parity", tables = "summary")
    tryCatch(
      gp3ml::write_gazepoint_ml_split_csv(split, root, prefix = "parity", tables = "summary"),
      error = function(error) {
        normalized_root <- normalizePath(root, winslash = "/", mustWork = TRUE)
        message <- gsub(normalized_root, "<TMP>", conditionMessage(error), fixed = TRUE)
        stop(message, call. = FALSE)
      }
    )
    stop("Expected overwrite protection error.", call. = FALSE)
  }
  table <- if (identical(kind, "summary")) "summary" else "checks"
  paths <- gp3ml::write_gazepoint_ml_split_csv(split, root, prefix = "parity", tables = table)
  path <- unname(paths[[table]])
  exported <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  list(table = table, basename = basename(path), columns = names(exported), rows = normalize_frame(exported))
}

successes <- list()
for (case in fixture$success_cases) {
  successes[[case$id]] <- capture_call(
    function() make_split(case$target),
    function(value) normalize_success(value, case$target)
  )
}
errors <- list()
for (case in fixture$error_cases) {
  errors[[case$id]] <- capture_call(function() error_case(case$kind), function(value) TRUE)
}
validations <- list()
for (case in fixture$validation_cases) {
  validations[[case$id]] <- capture_call(function() validation_case(case$kind), normalize_validation)
}
writers <- list()
for (case in fixture$writer_cases) {
  writers[[case$id]] <- capture_call(function() writer_case(case$kind), identity)
}

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
jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = 16)
