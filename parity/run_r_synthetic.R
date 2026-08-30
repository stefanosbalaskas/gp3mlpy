args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_synthetic.R <fixture.json> <output.json>", call. = FALSE)
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
  if (is.factor(value)) {
    if (is.na(value[[1L]])) return(NULL)
    return(as.character(value[[1L]]))
  }
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

normalize_frame <- function(data) {
  if (!is.data.frame(data)) stop("Expected a data frame in parity normalization.", call. = FALSE)
  if (nrow(data) == 0L) return(list())
  lapply(seq_len(nrow(data)), function(index) {
    row <- lapply(data[index, , drop = FALSE], scalar)
    names(row) <- names(data)
    row
  })
}

normalize_task <- function(task) {
  keys <- c(
    "outcome", "purpose", "task_type", "unit_id", "participant_id", "stimulus_id",
    "generalization_target", "positive", "observed_outcome", "sensitive_outcome",
    "levels", "negative"
  )
  components <- stats::setNames(vector("list", length(keys)), keys)
  for (index in seq_along(keys)) {
    key <- keys[[index]]
    value <- task[[key]]
    if (identical(key, "levels") && !is.null(value)) {
      components[index] <- list(as.character(value))
    } else {
      components[index] <- list(scalar(value))
    }
  }
  list(class = class(task)[[1L]], components = components)
}

normalize_manifest <- function(manifest) {
  list(
    class = class(manifest)[[1L]],
    columns = as.list(names(manifest)),
    rows = normalize_frame(manifest)
  )
}

capture_call <- function(call, normalize) {
  tryCatch(
    list(status = "success", value = normalize(call())),
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

named_counts <- function(values) {
  counts <- table(values, useNA = "ifany")
  result <- as.list(as.integer(counts))
  names(result) <- names(counts)
  result
}

simulate_args <- function(spec) {
  gp3ml::simulate_gazepoint_governed_data(
    n_participants = spec$n_participants,
    n_stimuli = spec$n_stimuli,
    trials_per_cell = spec$trials_per_cell,
    seed = spec$seed
  )
}

simulation_evidence <- function(spec, include_distribution = FALSE) {
  set.seed(20260831)
  before <- .Random.seed
  data <- simulate_args(spec)
  after <- .Random.seed

  repeated <- simulate_args(spec)
  alternate <- spec
  alternate$seed <- as.integer(spec$seed) + 1L
  changed <- simulate_args(alternate)

  design_columns <- c(
    "participant_id", "trial_id", "stimulus_id", "replicate",
    "assigned_condition", "site_label"
  )
  random_numeric <- c(
    "tracking_ratio", "blink_rate", "fixation_duration",
    "gaze_dispersion", "pupil_change", "observed_duration"
  )
  participant_values <- as.character(data$participant_id)
  stimulus_values <- as.character(data$stimulus_id)
  trial_values <- as.character(data$trial_id)

  exact <- list(
    columns = as.list(names(data)),
    nrow = as.integer(nrow(data)),
    ncol = as.integer(ncol(data)),
    n_participants = as.integer(length(unique(data$participant_id))),
    n_stimuli = as.integer(length(unique(data$stimulus_id))),
    n_trials = as.integer(length(unique(data$trial_id))),
    participant_first = participant_values[[1L]],
    participant_last = participant_values[[length(participant_values)]],
    stimulus_first = stimulus_values[[1L]],
    stimulus_last = stimulus_values[[length(stimulus_values)]],
    trial_first = trial_values[[1L]],
    trial_last = trial_values[[length(trial_values)]],
    replicate_values = as.list(sort(unique(as.integer(data$replicate)))),
    assigned_condition_levels = as.list(levels(data$assigned_condition)),
    quality_status_levels = as.list(levels(data$quality_status)),
    observed_response_levels = as.list(levels(data$observed_response)),
    site_label_levels = as.list(levels(data$site_label)),
    assigned_condition_counts = named_counts(data$assigned_condition),
    site_label_counts = named_counts(data$site_label),
    design_head = normalize_frame(utils::head(data[design_columns], 8L)),
    design_tail = normalize_frame(utils::tail(data[design_columns], 4L))
  )

  same_seed <- isTRUE(all.equal(data, repeated, check.attributes = TRUE))
  random_changed <- !isTRUE(all.equal(
    data[random_numeric],
    changed[random_numeric],
    check.attributes = TRUE
  ))
  checks <- list(
    same_seed_reproducible = same_seed,
    different_seed_changes_random_columns = random_changed,
    global_rng_preserved = identical(before, after),
    tracking_ratio_bounds = all(data$tracking_ratio >= 0.45 & data$tracking_ratio <= 1),
    blink_rate_nonnegative = all(data$blink_rate >= 0),
    fixation_duration_floor = all(data$fixation_duration >= 80),
    gaze_dispersion_floor = all(data$gaze_dispersion >= 0.05),
    observed_duration_floor = all(data$observed_duration >= 0.1)
  )

  result <- list(exact = exact, checks = checks)
  if (isTRUE(include_distribution)) {
    distribution <- list()
    for (name in random_numeric) {
      distribution[[paste0(name, "_mean")]] <- mean(data[[name]])
      distribution[[paste0(name, "_sd")]] <- stats::sd(data[[name]])
    }
    distribution$quality_review_rate <- mean(as.character(data$quality_status) == "review")
    distribution$response_yes_rate <- mean(as.character(data$observed_response) == "recorded_yes")
    result$distribution <- distribution
  }
  result
}

primary_spec <- fixture$primary_simulation
primary <- simulate_args(primary_spec)

simulations <- list(
  simulator_primary = list(
    status = "success",
    value = simulation_evidence(primary_spec, include_distribution = TRUE)
  ),
  simulator_integer_coercion = list(
    status = "success",
    value = simulation_evidence(fixture$coercion_simulation, include_distribution = FALSE)
  ),
  simulator_n_participants_error = capture_call(
    function() gp3ml::simulate_gazepoint_governed_data(3L, 2L, 1L, 1L),
    identity
  ),
  simulator_n_stimuli_error = capture_call(
    function() gp3ml::simulate_gazepoint_governed_data(4L, 1L, 1L, 1L),
    identity
  ),
  simulator_trials_per_cell_error = capture_call(
    function() gp3ml::simulate_gazepoint_governed_data(4L, 2L, 0L, 1L),
    identity
  )
)

manifest_args <- list(
  outcome = "quality_status",
  predictors = c("tracking_ratio", "blink_rate", "gaze_dispersion")
)
manifests <- list(
  manifest_primary = capture_call(
    function() do.call(gp3ml::create_gazepoint_synthetic_manifest, manifest_args),
    normalize_manifest
  ),
  manifest_custom_identifiers = capture_call(
    function() gp3ml::create_gazepoint_synthetic_manifest(
      outcome = "quality_status",
      predictors = c("tracking_ratio", "blink_rate", "gaze_dispersion"),
      participant_id = "subject_key",
      stimulus_id = "item_key",
      trial_id = "event_key"
    ),
    normalize_manifest
  )
)

tasks <- list(
  task_recording_quality = capture_call(
    function() gp3ml::create_gazepoint_synthetic_task(
      primary, "recording_quality", "new_participants"
    ),
    normalize_task
  ),
  task_assigned_condition = capture_call(
    function() gp3ml::create_gazepoint_synthetic_task(
      primary, "assigned_condition", "new_stimuli"
    ),
    normalize_task
  ),
  task_observed_behavior = capture_call(
    function() gp3ml::create_gazepoint_synthetic_task(
      primary, "observed_behavior", "new_trials_known_participants"
    ),
    normalize_task
  ),
  task_observed_duration = capture_call(
    function() gp3ml::create_gazepoint_synthetic_task(
      primary, "observed_duration", "new_participants_and_new_stimuli"
    ),
    normalize_task
  ),
  task_invalid_workflow = capture_call(
    function() gp3ml::create_gazepoint_synthetic_task(
      primary, "latent_personality", "new_participants"
    ),
    normalize_task
  ),
  task_invalid_generalization_target = capture_call(
    function() gp3ml::create_gazepoint_synthetic_task(
      primary, "recording_quality", "same_rows"
    ),
    normalize_task
  )
)

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  simulations = simulations,
  manifests = manifests,
  tasks = tasks
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
