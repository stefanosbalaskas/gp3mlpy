args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_calibration.R <fixture.json> <output.json>")
}

fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)
output_path <- args[[2L]]

vec <- function(value, mode = c("character", "numeric")) {
  mode <- match.arg(mode)
  result <- unlist(value, recursive = TRUE, use.names = FALSE)
  if (identical(mode, "numeric")) as.numeric(result) else as.character(result)
}

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

numeric_vector <- function(value) {
  lapply(as.numeric(value), function(item) {
    if (is.finite(item)) item else NULL
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

normalize_calibrator <- function(value, probes) {
  calibrated <- gp3ml::apply_gazepoint_calibrator(value, probes)
  list(
    class = class(value)[[1L]],
    method = value$method,
    positive = value$positive,
    negative = value$negative,
    probe_probability = numeric_vector(probes),
    calibrated_probability = numeric_vector(calibrated)
  )
}

normalize_assessment <- function(value, bootstrap_structure = FALSE) {
  result <- list(
    class = class(value)[[1L]],
    summary = frame(value$summary),
    reliability = frame(value$reliability),
    positive = value$positive,
    bins = as.integer(value$bins),
    bootstrap = as.integer(value$bootstrap),
    seed = as.integer(value$seed)
  )
  if (!bootstrap_structure) {
    result$intervals <- frame(value$intervals)
  } else {
    intervals <- value$intervals
    lower <- if ("lower" %in% names(intervals)) as.numeric(intervals$lower) else numeric()
    upper <- if ("upper" %in% names(intervals)) as.numeric(intervals$upper) else numeric()
    result$interval_structure <- list(
      metrics = if ("metric" %in% names(intervals)) as.character(intervals$metric) else character(),
      n_intervals = as.integer(nrow(intervals)),
      bounds_ordered = isTRUE(all((lower <= upper) | (is.na(lower) & is.na(upper)))),
      finite_bound_pairs = as.integer(sum(is.finite(lower) & is.finite(upper)))
    )
  }
  result
}

primary <- fixture$primary
isotonic <- fixture$isotonic
assessment <- fixture$assessment
bootstrap <- fixture$bootstrap

primary_truth <- vec(primary$truth)
primary_probability <- vec(primary$probability, "numeric")
primary_probes <- vec(primary$probes, "numeric")
isotonic_truth <- vec(isotonic$truth)
isotonic_probability <- vec(isotonic$probability, "numeric")
isotonic_probes <- vec(isotonic$probes, "numeric")
assessment_truth <- vec(assessment$truth)
assessment_probability <- vec(assessment$probability, "numeric")

fits <- list()
fits$fit_platt_primary <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = primary_truth,
      probability = primary_probability,
      positive = primary$positive,
      method = "platt"
    )
  },
  function(value) normalize_calibrator(value, primary_probes)
)
fits$fit_isotonic_ties <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = isotonic_truth,
      probability = isotonic_probability,
      positive = isotonic$positive,
      method = "isotonic"
    )
  },
  function(value) normalize_calibrator(value, isotonic_probes)
)
fits$fit_default_positive <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = primary_truth,
      probability = primary_probability
    )
  },
  function(value) normalize_calibrator(value, primary_probes)
)

fit_errors <- list()
fit_errors$fit_binary_levels_error <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = c("pass", "pass", "pass"),
      probability = c(0.1, 0.2, 0.3),
      method = "platt"
    )
  },
  function(value) TRUE
)
fit_errors$fit_unknown_positive_error <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = primary_truth,
      probability = primary_probability,
      positive = "unknown",
      method = "platt"
    )
  },
  function(value) TRUE
)
fit_errors$fit_invalid_method_error <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = primary_truth,
      probability = primary_probability,
      positive = primary$positive,
      method = "not-a-method"
    )
  },
  function(value) TRUE
)
fit_errors$fit_length_mismatch_error <- capture_case(
  function() {
    gp3ml::fit_gazepoint_calibrator(
      truth = primary_truth,
      probability = primary_probability[-length(primary_probability)],
      positive = primary$positive,
      method = "platt"
    )
  },
  function(value) TRUE
)

platt <- gp3ml::fit_gazepoint_calibrator(
  truth = primary_truth,
  probability = primary_probability,
  positive = primary$positive,
  method = "platt"
)
iso <- gp3ml::fit_gazepoint_calibrator(
  truth = isotonic_truth,
  probability = isotonic_probability,
  positive = isotonic$positive,
  method = "isotonic"
)

applications <- list()
applications$apply_platt_primary <- capture_case(
  function() gp3ml::apply_gazepoint_calibrator(platt, primary_probes),
  numeric_vector
)
applications$apply_isotonic_ties <- capture_case(
  function() gp3ml::apply_gazepoint_calibrator(iso, isotonic_probes),
  numeric_vector
)
applications$apply_boundary_clipping <- capture_case(
  function() gp3ml::apply_gazepoint_calibrator(platt, c(0, 1)),
  numeric_vector
)

apply_errors <- list()
apply_errors$apply_invalid_calibrator_error <- capture_case(
  function() gp3ml::apply_gazepoint_calibrator(42, c(0.2, 0.8)),
  function(value) TRUE
)

assessments <- list()
assessments$assess_no_bootstrap <- capture_case(
  function() {
    gp3ml::assess_gazepoint_calibration(
      truth = assessment_truth,
      probability = assessment_probability,
      positive = assessment$positive,
      bins = 5L,
      bootstrap = 0L,
      conf_level = 0.95,
      seed = 101L
    )
  },
  normalize_assessment
)
assessments$assess_bin_boundaries <- capture_case(
  function() {
    gp3ml::assess_gazepoint_calibration(
      truth = c("pass", "review", "pass", "review", "pass", "review"),
      probability = c(0, 0.2, 0.4, 0.6, 0.8, 1),
      positive = "review",
      bins = 5L,
      bootstrap = 0L,
      seed = 7L
    )
  },
  normalize_assessment
)
assessments$assess_bootstrap_structure <- capture_case(
  function() {
    gp3ml::assess_gazepoint_calibration(
      truth = assessment_truth,
      probability = assessment_probability,
      positive = assessment$positive,
      bins = as.integer(bootstrap$bins),
      bootstrap = as.integer(bootstrap$bootstrap),
      conf_level = as.numeric(bootstrap$conf_level),
      seed = as.integer(bootstrap$seed)
    )
  },
  function(value) normalize_assessment(value, bootstrap_structure = TRUE)
)

assessment_errors <- list()
assessment_errors$assess_binary_levels_error <- capture_case(
  function() {
    gp3ml::assess_gazepoint_calibration(
      truth = c("pass", "pass", "pass"),
      probability = c(0.1, 0.2, 0.3),
      bootstrap = 0L
    )
  },
  function(value) TRUE
)
assessment_errors$assess_unknown_positive_error <- capture_case(
  function() {
    gp3ml::assess_gazepoint_calibration(
      truth = assessment_truth,
      probability = assessment_probability,
      positive = "unknown",
      bootstrap = 0L
    )
  },
  function(value) TRUE
)
assessment_errors$assess_length_mismatch_error <- capture_case(
  function() {
    gp3ml::assess_gazepoint_calibration(
      truth = assessment_truth,
      probability = assessment_probability[-length(assessment_probability)],
      positive = assessment$positive,
      bootstrap = 0L
    )
  },
  function(value) TRUE
)

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  fits = fits,
  fit_errors = fit_errors,
  applications = applications,
  apply_errors = apply_errors,
  assessments = assessments,
  assessment_errors = assessment_errors
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
