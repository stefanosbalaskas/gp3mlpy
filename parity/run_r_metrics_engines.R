args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_metrics_engines.R <fixture.json> <output.json>", call. = FALSE)
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

capture_call <- function(call, normalize = identity) {
  tryCatch(
    {
      value <- suppressWarnings(call())
      list(status = "success", value = normalize(value))
    },
    error = function(error) list(status = "error", message = conditionMessage(error))
  )
}

as_character_vector <- function(x) as.character(unlist(x, use.names = FALSE))
as_numeric_vector <- function(x) as.numeric(unlist(x, use.names = FALSE))

classification <- fixture$classification
truth <- as_character_vector(classification$truth)
probability <- as_numeric_vector(classification$probability)
prediction <- as_character_vector(classification$prediction)
n <- length(truth)
class_data <- data.frame(
  trial_id = sprintf("T%02d", seq_len(n)),
  participant_id = sprintf("P%02d", ((seq_len(n) - 1L) %/% 2L) + 1L),
  stimulus_id = sprintf("S%02d", ((seq_len(n) - 1L) %% 3L) + 1L),
  quality_status = factor(truth, levels = c("pass", "review")),
  stringsAsFactors = FALSE
)
class_task <- gp3ml::declare_gazepoint_task(
  class_data,
  outcome = "quality_status",
  purpose = "Predict observed recording quality.",
  task_type = "classification",
  unit_id = "trial_id",
  participant_id = "participant_id",
  stimulus_id = "stimulus_id",
  generalization_target = "new_participants",
  positive = "review"
)

regression <- fixture$regression
regression_truth <- as_numeric_vector(regression$truth)
regression_prediction <- as_numeric_vector(regression$prediction)
regression_data <- data.frame(
  trial_id = sprintf("R%02d", seq_along(regression_truth)),
  observed_duration = regression_truth,
  stringsAsFactors = FALSE
)
regression_task <- gp3ml::declare_gazepoint_task(
  regression_data,
  outcome = "observed_duration",
  purpose = "Predict observed response duration.",
  task_type = "regression",
  unit_id = "trial_id",
  generalization_target = "new_trials_known_participants"
)

normalize_uncertainty <- function(value, global_rng_preserved, reproducible) {
  intervals <- value$intervals
  if (nrow(intervals)) {
    intervals <- intervals[order(intervals$metric, method = "radix"), , drop = FALSE]
    rownames(intervals) <- NULL
  }
  metric_columns <- names(value$draws)[vapply(value$draws, is.numeric, logical(1))]
  metric_columns <- setdiff(metric_columns, c("n", "threshold"))
  point <- value$point
  ordered <- if (nrow(intervals)) {
    all(intervals$lower[is.finite(intervals$lower) & is.finite(intervals$upper)] <=
          intervals$upper[is.finite(intervals$lower) & is.finite(intervals$upper)])
  } else TRUE
  estimate_match <- TRUE
  if (nrow(intervals)) {
    for (i in seq_len(nrow(intervals))) {
      metric <- intervals$metric[[i]]
      estimate <- intervals$estimate[[i]]
      point_value <- point[[metric]][[1L]]
      if (is.na(estimate) && is.na(point_value)) next
      if (!isTRUE(all.equal(as.numeric(estimate), as.numeric(point_value), tolerance = 1e-12))) {
        estimate_match <- FALSE
      }
    }
  }
  list(
    class = class(value)[[1L]],
    bootstrap = as.integer(value$bootstrap),
    conf_level = as.numeric(value$conf_level),
    seed = as.integer(value$seed),
    point = normalize_frame(point),
    intervals = normalize_frame(intervals),
    draw_columns = as.list(names(value$draws)),
    draw_nrow = as.integer(nrow(value$draws)),
    metric_columns = as.list(metric_columns),
    checks = list(
      global_rng_preserved = isTRUE(global_rng_preserved),
      reproducible_with_seed = isTRUE(reproducible),
      intervals_ordered = isTRUE(ordered),
      estimates_match_point = isTRUE(estimate_match),
      draw_count_matches_bootstrap = nrow(value$draws) == as.integer(value$bootstrap)
    )
  )
}

bootstrap_case <- function(task, truth, prediction, probability, bootstrap, conf_level, seed) {
  set.seed(20260831)
  before <- .Random.seed
  first <- gp3ml::bootstrap_gazepoint_metrics(
    task,
    truth = truth,
    prediction = prediction,
    probability = probability,
    bootstrap = as.integer(bootstrap),
    conf_level = as.numeric(conf_level),
    seed = as.integer(seed)
  )
  after <- .Random.seed
  second <- gp3ml::bootstrap_gazepoint_metrics(
    task,
    truth = truth,
    prediction = prediction,
    probability = probability,
    bootstrap = as.integer(bootstrap),
    conf_level = as.numeric(conf_level),
    seed = as.integer(seed)
  )
  reproducible <- identical(first$point, second$point) &&
    identical(first$intervals, second$intervals) &&
    identical(first$draws, second$draws)
  normalize_uncertainty(
    first,
    global_rng_preserved = identical(before, after),
    reproducible = reproducible
  )
}

performance_cases <- list(
  classification_primary = capture_call(
    function() gp3ml::gazepoint_performance_metrics(
      class_task,
      truth = class_data$quality_status,
      prediction = factor(prediction, levels = c("pass", "review")),
      probability = probability,
      threshold = as.numeric(classification$threshold)
    ),
    normalize_frame
  ),
  classification_threshold_dispatch = capture_call(
    function() gp3ml::gazepoint_performance_metrics(
      class_task,
      truth = class_data$quality_status,
      prediction = NULL,
      probability = probability,
      threshold = 0.6
    ),
    normalize_frame
  ),
  regression_primary = capture_call(
    function() gp3ml::gazepoint_performance_metrics(
      regression_task,
      truth = regression_truth,
      prediction = regression_prediction
    ),
    normalize_frame
  ),
  classification_probability_length_mismatch = capture_call(
    function() gp3ml::gazepoint_performance_metrics(
      class_task,
      truth = class_data$quality_status,
      prediction = NULL,
      probability = probability[-length(probability)]
    ),
    normalize_frame
  )
)

bootstrap_cases <- list(
  classification_bootstrap = capture_call(function() bootstrap_case(
    class_task,
    class_data$quality_status,
    factor(prediction, levels = c("pass", "review")),
    probability,
    classification$bootstrap,
    classification$conf_level,
    classification$seed
  )),
  regression_bootstrap = capture_call(function() bootstrap_case(
    regression_task,
    regression_truth,
    regression_prediction,
    NULL,
    regression$bootstrap,
    regression$conf_level,
    regression$seed
  )),
  bootstrap_zero = capture_call(function() {
    gp3ml::bootstrap_gazepoint_metrics(
      class_task, class_data$quality_status,
      probability = probability, bootstrap = 0L
    )
  }),
  truth_too_short = capture_call(function() {
    gp3ml::bootstrap_gazepoint_metrics(
      class_task,
      factor("pass", levels = c("pass", "review")),
      probability = 0.2
    )
  }),
  probability_length_mismatch = capture_call(function() {
    gp3ml::bootstrap_gazepoint_metrics(
      class_task,
      class_data$quality_status,
      probability = probability[-length(probability)]
    )
  }),
  prediction_length_mismatch = capture_call(function() {
    gp3ml::bootstrap_gazepoint_metrics(
      regression_task,
      regression_truth,
      prediction = regression_prediction[-length(regression_prediction)]
    )
  })
)

engine_cases <- list(
  runtime_registry = list(
    status = "success",
    value = normalize_frame(gp3ml::gp3ml_available_engines())
  )
)

result <- list(
  schema_version = 1L,
  runtime = "r",
  package_version = as.character(utils::packageVersion("gp3ml")),
  performance_cases = performance_cases,
  bootstrap_cases = bootstrap_cases,
  engine_cases = engine_cases
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
