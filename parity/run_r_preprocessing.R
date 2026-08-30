args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_preprocessing.R <fixture.json> <output.json>")
}

fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)
output_path <- args[[2L]]

build_column <- function(values, kind) {
  if (identical(kind, "numeric")) {
    return(vapply(values, function(value) {
      if (is.null(value)) NA_real_ else as.numeric(value)
    }, numeric(1)))
  }
  if (identical(kind, "logical")) {
    return(vapply(values, function(value) {
      if (is.null(value)) NA else as.logical(value)
    }, logical(1)))
  }
  vapply(values, function(value) {
    if (is.null(value)) NA_character_ else as.character(value)
  }, character(1))
}

build_frame <- function(spec) {
  values <- lapply(names(spec$columns), function(name) {
    build_column(spec$columns[[name]], spec$types[[name]])
  })
  names(values) <- names(spec$columns)
  as.data.frame(values, stringsAsFactors = FALSE, check.names = FALSE)
}

scalar <- function(value) {
  if (length(value) == 0L || is.null(value) || is.na(value[[1L]])) return(NULL)
  value <- value[[1L]]
  if (is.logical(value)) return(isTRUE(value))
  if (is.integer(value)) return(as.integer(value))
  if (is.numeric(value)) {
    number <- as.numeric(value)
    return(if (is.finite(number)) number else NULL)
  }
  as.character(value)
}

named_numeric <- function(values, order) {
  if (is.null(values) || length(values) == 0L) return(list())
  names_found <- names(values)
  lapply(order[order %in% names_found], function(name) {
    list(name = name, value = scalar(values[[name]]))
  })
}

named_levels <- function(values, order) {
  if (is.null(values) || length(values) == 0L) return(list())
  names_found <- names(values)
  lapply(order[order %in% names_found], function(name) {
    list(name = name, levels = as.character(values[[name]]))
  })
}

normalize_preprocessor <- function(value) {
  predictors <- as.character(value$predictors)
  columns <- as.character(value$columns)
  list(
    class = class(value)[[1L]],
    predictors = predictors,
    numeric_imputation = value$numeric_imputation,
    numeric_imputation_values = named_numeric(
      value$numeric_imputation_values,
      predictors
    ),
    factor_levels = named_levels(value$factor_levels, predictors),
    novel_level = value$novel_level,
    columns = columns,
    center = named_numeric(value$center, columns),
    scale = named_numeric(value$scale, columns),
    remove_zero_variance = isTRUE(value$remove_zero_variance)
  )
}

normalize_matrix <- function(value, columns) {
  matrix <- as.matrix(value)
  rows <- lapply(seq_len(nrow(matrix)), function(index) {
    if (ncol(matrix) == 0L) return(list())
    lapply(as.numeric(matrix[index, , drop = TRUE]), scalar)
  })
  list(
    columns = as.character(columns),
    nrow = as.integer(nrow(matrix)),
    ncol = as.integer(ncol(matrix)),
    values = rows
  )
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

training <- build_frame(fixture$training)
new_data <- build_frame(fixture$new_data)

fits <- list()
fits$fit_median_mixed <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    c("num1", "cat", "constant"),
    numeric_imputation = "median",
    center = TRUE,
    scale = TRUE,
    novel_level = "other",
    remove_zero_variance = TRUE
  ),
  normalize_preprocessor
)
fits$fit_mean_no_center_scale <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    c("num1", "num2", "cat"),
    numeric_imputation = "mean",
    center = FALSE,
    scale = FALSE,
    novel_level = "other",
    remove_zero_variance = FALSE
  ),
  normalize_preprocessor
)
fits$fit_all_missing_numeric <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    "all_missing",
    remove_zero_variance = TRUE
  ),
  normalize_preprocessor
)
fits$fit_boolean_predictor <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    "logical",
    center = FALSE,
    scale = FALSE,
    novel_level = "other",
    remove_zero_variance = FALSE
  ),
  normalize_preprocessor
)
fits$fit_keep_zero_variance <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    "constant",
    center = TRUE,
    scale = TRUE,
    remove_zero_variance = FALSE
  ),
  normalize_preprocessor
)

fit_errors <- list()
fit_errors$fit_missing_predictor_error <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    c("num1", "not_present")
  ),
  function(value) TRUE
)
fit_errors$fit_too_few_rows_error <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training[1L, , drop = FALSE],
    "num1"
  ),
  function(value) TRUE
)
fit_errors$fit_invalid_numeric_imputation <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    "num1",
    numeric_imputation = "mode"
  ),
  function(value) TRUE
)
fit_errors$fit_invalid_novel_level <- capture_case(
  function() gp3ml::fit_gazepoint_preprocessor(
    training,
    "cat",
    novel_level = "silent"
  ),
  function(value) TRUE
)

median_pp <- gp3ml::fit_gazepoint_preprocessor(
  training,
  c("num1", "cat", "constant"),
  numeric_imputation = "median",
  center = TRUE,
  scale = TRUE,
  novel_level = "other",
  remove_zero_variance = TRUE
)
mean_pp <- gp3ml::fit_gazepoint_preprocessor(
  training,
  c("num1", "num2", "cat"),
  numeric_imputation = "mean",
  center = FALSE,
  scale = FALSE,
  novel_level = "other",
  remove_zero_variance = FALSE
)
category_pp <- gp3ml::fit_gazepoint_preprocessor(
  training,
  "cat",
  center = FALSE,
  scale = FALSE,
  novel_level = "other",
  remove_zero_variance = FALSE
)
strict_category_pp <- gp3ml::fit_gazepoint_preprocessor(
  training,
  "cat",
  center = FALSE,
  scale = FALSE,
  novel_level = "error",
  remove_zero_variance = FALSE
)

bakes <- list()
bakes$bake_training_roundtrip <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(median_pp, training),
  function(value) normalize_matrix(value, median_pp$columns)
)
bakes$bake_missing_and_novel <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(mean_pp, new_data),
  function(value) normalize_matrix(value, mean_pp$columns)
)
bakes$bake_novel_to_other <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(category_pp, new_data),
  function(value) normalize_matrix(value, category_pp$columns)
)
only_a <- data.frame(cat = rep("a", 3L), stringsAsFactors = FALSE)
bakes$bake_missing_trained_level <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(category_pp, only_a),
  function(value) normalize_matrix(value, category_pp$columns)
)

bake_errors <- list()
bake_errors$bake_novel_error <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(strict_category_pp, new_data),
  function(value) TRUE
)
bake_errors$bake_missing_predictor_error <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(
    mean_pp,
    new_data[, setdiff(names(new_data), "num2"), drop = FALSE]
  ),
  function(value) TRUE
)
bake_errors$bake_invalid_preprocessor_error <- capture_case(
  function() gp3ml::bake_gazepoint_preprocessor(42, new_data),
  function(value) TRUE
)

result <- list(
  runtime = "R",
  package = "gp3ml",
  package_version = as.character(utils::packageVersion("gp3ml")),
  fits = fits,
  fit_errors = fit_errors,
  bakes = bakes,
  bake_errors = bake_errors
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
