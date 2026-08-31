library(jsonlite)

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

base_data <- function() {
  index <- expand.grid(
    stimulus_index = seq_len(4L),
    participant_index = seq_len(12L),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  i <- index$participant_index
  j <- index$stimulus_index
  n <- nrow(index)
  tracking_ratio <- 0.50 + ((i * 3L + j * 2L) %% 13L) / 25
  blink_rate <- 3 + ((i * 7L + j * 5L) %% 17L) / 2
  fixation_duration <- 150 + ((i * 11L + j * 13L) %% 90L)
  is_review <- ((i + 2L * j + (i * j) %% 3L) %% 4L) %in% c(0L, 1L)
  observed_duration <- 1 + 0.6 * tracking_ratio + 0.04 * blink_rate + ((i * j) %% 5L) / 10
  data.frame(
    participant_id = sprintf("P%02d", i),
    trial_id = sprintf("T%03d", seq_len(n)),
    stimulus_id = sprintf("S%02d", j),
    tracking_ratio = tracking_ratio,
    blink_rate = blink_rate,
    fixation_duration = fixation_duration,
    quality_status = factor(ifelse(is_review, "review", "pass"), levels = c("pass", "review")),
    observed_duration = observed_duration,
    stringsAsFactors = FALSE
  )
}

predictors <- c("tracking_ratio", "blink_rate")

classification_task <- function(data, target = "new_participants") {
  gp3ml::create_gazepoint_synthetic_task(data, "recording_quality", target)
}

regression_task <- function(data, target = "new_participants") {
  gp3ml::create_gazepoint_synthetic_task(data, "observed_duration", target)
}

make_folds <- function(data, seed) {
  manifest <- gp3ml::create_gazepoint_synthetic_manifest("quality_status", predictors)
  gp3ml::create_gazepoint_group_folds(
    data = data,
    outcome = "quality_status",
    predictors = predictors,
    feature_manifest = manifest,
    generalization_target = "new_participants",
    participant_id = "participant_id",
    trial_id = "trial_id",
    stimulus_id = "stimulus_id",
    v = 3L,
    repeats = 1L,
    seed = seed
  )
}

model_value <- function(model, data) {
  prediction_type <- if (model$task$task_type == "classification") "probability" else "response"
  prediction <- as.numeric(stats::predict(model, data, type = prediction_type))
  classes <- if (model$task$task_type == "classification") {
    as.character(stats::predict(model, data, type = "class"))
  } else NULL
  distribution <- model$outcome_distribution
  distribution_names <- sort(names(distribution))
  distribution_value <- lapply(
    distribution_names,
    function(name) list(name, as.integer(distribution[[name]]))
  )
  list(
    class = class(model)[[1L]],
    engine = model$engine,
    task_type = model$task$task_type,
    predictors = as.list(as.character(model$predictors)),
    threshold = as.numeric(model$threshold),
    seed = as.integer(model$seed),
    training_n = as.integer(model$training_n),
    outcome_distribution = distribution_value,
    preprocessor_columns = as.list(as.character(model$preprocessor$columns)),
    prediction = as.list(as.numeric(prediction)),
    classes = if (is.null(classes)) NULL else as.list(classes)
  )
}

engine_value <- function(engine) {
  list(
    class = class(engine)[[1L]],
    name = engine$name,
    supports = as.list(as.character(engine$supports)),
    probability = isTRUE(engine$probability),
    metadata = engine$metadata,
    safety_declaration = engine$safety_declaration
  )
}

safe_engine <- function() {
  fit_fun <- function(x, y, task, args) {
    list(mean = mean(y))
  }
  predict_fun <- function(fit, newdata, type, task, ...) {
    rep(fit$mean, nrow(newdata))
  }
  gp3ml::integrate_black_box_model(
    name = "constant_mean",
    fit_fun = fit_fun,
    predict_fun = predict_fun,
    supports = "classification",
    probability = TRUE,
    metadata = list(backend = "fixture"),
    safety_declaration = list(
      prohibited_uses_acknowledged = TRUE,
      prediction_time_inputs_only = TRUE,
      group_aware_evaluation_required = TRUE
    )
  )
}

evaluation_value <- function(value) {
  statuses <- table(value$fold_status$status)
  status_names <- sort(names(statuses))
  status_counts <- lapply(status_names, function(name) as.integer(statuses[[name]]))
  names(status_counts) <- status_names
  metric_names <- if (nrow(value$metrics)) sort(unique(as.character(value$metrics$metric))) else character()
  fold_sizes <- lapply(seq_len(nrow(value$fold_status)), function(i) {
    list(
      as.integer(value$fold_status$n_analysis[[i]]),
      as.integer(value$fold_status$n_assessment[[i]]),
      as.integer(value$fold_status$n_excluded[[i]])
    )
  })
  models_retained <- sum(vapply(value$fold_results, function(result) !is.null(result$model), logical(1)))
  list(
    class = class(value)[[1L]],
    engine = as.character(value$engine),
    generalization_target = value$generalization_target,
    predictors = as.list(as.character(value$predictors)),
    threshold = as.numeric(value$threshold),
    seed = as.integer(value$seed),
    n_folds = as.integer(nrow(value$fold_status)),
    status_counts = status_counts,
    fold_sizes = fold_sizes,
    n_predictions = as.integer(nrow(value$predictions)),
    n_metrics = as.integer(nrow(value$metrics)),
    metric_names = as.list(metric_names),
    validation_status = value$validation$status,
    keep_models = isTRUE(value$keep_models),
    models_retained = as.integer(models_retained),
    failed_folds = as.integer(sum(value$fold_status$status == "fail")),
    missing_predictions = as.integer(sum(value$fold_status$n_missing_predictions))
  )
}

tuning_value <- function(value) {
  statuses <- lapply(value$results, function(result) {
    list(
      candidate_id = result$candidate_id,
      status = result$status,
      success_prop = as.numeric(result$success_prop),
      seed = as.integer(result$seed),
      has_error = !is.na(result$error)
    )
  })
  metric_names <- if (nrow(value$comparison)) {
    sort(unique(as.character(value$comparison$metric[!is.na(value$comparison$metric)])))
  } else character()
  evaluation_retained <- sum(vapply(value$results, function(result) !is.null(result$evaluation), logical(1)))
  list(
    class = class(value)[[1L]],
    candidate_ids = as.list(as.character(value$grid$candidates$candidate_id)),
    results = statuses,
    comparison_rows = as.integer(nrow(value$comparison)),
    comparison_metrics = as.list(metric_names),
    validation_status = value$validation$status,
    metrics_requested = if (is.null(value$metrics_requested)) NULL else as.list(as.character(value$metrics_requested)),
    keep_evaluations = isTRUE(value$keep_evaluations),
    evaluations_retained = as.integer(evaluation_retained),
    generalization_target = value$folds_metadata$generalization_target
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_model_execution.R <fixture.json> <output.json>")
}
fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = TRUE)
output_path <- args[[2L]]
seed <- as.integer(fixture$seed)

data <- base_data()
cls_task <- classification_task(data)
reg_task <- regression_task(data)
folds <- make_folds(data, seed)
cases <- list()

cases[["integrate_black_box_model::safe_constructor"]] <- capture_case(engine_value(safe_engine()))
cases[["integrate_black_box_model::invalid_functions"]] <- capture_case(
  gp3ml::integrate_black_box_model(
    "bad",
    1,
    function(...) NULL,
    safety_declaration = list(
      prohibited_uses_acknowledged = TRUE,
      prediction_time_inputs_only = TRUE,
      group_aware_evaluation_required = TRUE
    )
  )
)
cases[["integrate_black_box_model::missing_safety"]] <- capture_case(
  gp3ml::integrate_black_box_model(
    "bad",
    function(...) NULL,
    function(...) NULL,
    safety_declaration = list(prohibited_uses_acknowledged = TRUE)
  )
)

cases[["fit_gazepoint_model::classification_glm"]] <- capture_case(
  model_value(
    gp3ml::fit_gazepoint_model(
      data, cls_task, predictors, engine = "glm", seed = seed, threshold = 0.45
    ),
    data
  )
)
cases[["fit_gazepoint_model::regression_glm_maps_lm"]] <- capture_case(
  model_value(
    gp3ml::fit_gazepoint_model(
      data, reg_task, predictors, engine = "glm", seed = seed
    ),
    data
  )
)
cases[["fit_gazepoint_model::custom_engine"]] <- capture_case(
  model_value(
    gp3ml::fit_gazepoint_model(
      data, cls_task, predictors, engine = safe_engine(), seed = seed
    ),
    data
  )
)
cases[["fit_gazepoint_model::classification_lm_rejected"]] <- capture_case(
  gp3ml::fit_gazepoint_model(data, cls_task, predictors, engine = "lm")
)
cases[["fit_gazepoint_model::forbidden_predictor"]] <- capture_case(
  gp3ml::fit_gazepoint_model(data, cls_task, "quality_status", engine = "glm")
)
cases[["fit_gazepoint_model::missing_outcome"]] <- capture_case({
  missing_data <- data
  missing_data$quality_status[[1L]] <- NA
  gp3ml::fit_gazepoint_model(missing_data, cls_task, predictors, engine = "glm")
})
cases[["fit_gazepoint_model::unknown_engine"]] <- capture_case(
  gp3ml::fit_gazepoint_model(data, cls_task, predictors, engine = "unknown")
)

cases[["train_gazepoint_classifier::wrapper_success"]] <- capture_case(
  model_value(
    gp3ml::train_gazepoint_classifier(
      data, cls_task, predictors, engine = "glm", seed = seed
    ),
    data
  )
)
cases[["train_gazepoint_classifier::regression_rejected"]] <- capture_case(
  gp3ml::train_gazepoint_classifier(data, reg_task, predictors)
)

cases[["evaluate_gazepoint_group_folds::successful_glm"]] <- capture_case(
  evaluation_value(
    gp3ml::evaluate_gazepoint_group_folds(
      folds,
      cls_task,
      predictors = predictors,
      engine = "glm",
      seed = seed,
      assess_calibration = FALSE,
      keep_models = FALSE
    )
  )
)
cases[["evaluate_gazepoint_group_folds::keep_models"]] <- capture_case(
  evaluation_value(
    gp3ml::evaluate_gazepoint_group_folds(
      folds,
      cls_task,
      predictors = predictors,
      engine = "glm",
      seed = seed,
      assess_calibration = FALSE,
      keep_models = TRUE
    )
  )
)
cases[["evaluate_gazepoint_group_folds::retain_fold_failures"]] <- capture_case(
  evaluation_value(
    gp3ml::evaluate_gazepoint_group_folds(
      folds,
      cls_task,
      predictors = predictors,
      engine = "unknown",
      seed = seed,
      continue_on_error = TRUE
    )
  )
)
cases[["evaluate_gazepoint_group_folds::invalid_folds"]] <- capture_case(
  gp3ml::evaluate_gazepoint_group_folds("not folds", cls_task, predictors)
)
cases[["evaluate_gazepoint_group_folds::undeclared_predictor"]] <- capture_case(
  gp3ml::evaluate_gazepoint_group_folds(folds, cls_task, "fixation_duration")
)
cases[["evaluate_gazepoint_group_folds::target_mismatch"]] <- capture_case(
  gp3ml::evaluate_gazepoint_group_folds(
    folds,
    classification_task(data, "new_stimuli"),
    predictors
  )
)
cases[["evaluate_gazepoint_group_folds::stop_on_failure"]] <- capture_case(
  gp3ml::evaluate_gazepoint_group_folds(
    folds,
    cls_task,
    predictors = predictors,
    engine = "unknown",
    seed = seed,
    continue_on_error = FALSE
  )
)

grid <- gp3ml::create_gazepoint_tuning_grid(
  "glm",
  thresholds = c(0.4, 0.6),
  complexity = c(1, 2),
  interpretability = "high"
)
cases[["tune_gazepoint_model::successful_candidates"]] <- capture_case(
  tuning_value(
    gp3ml::tune_gazepoint_model(
      folds,
      cls_task,
      grid,
      predictors = predictors,
      metrics = c("roc_auc", "brier"),
      seed = seed,
      keep_evaluations = FALSE
    )
  )
)
cases[["tune_gazepoint_model::keep_evaluations"]] <- capture_case(
  tuning_value(
    gp3ml::tune_gazepoint_model(
      folds,
      cls_task,
      gp3ml::create_gazepoint_tuning_grid("glm", thresholds = 0.5),
      predictors = predictors,
      metrics = "roc_auc",
      seed = seed,
      keep_evaluations = TRUE
    )
  )
)
mixed_grid <- gp3ml::create_gazepoint_tuning_grid(
  c("glm", "unknown"),
  thresholds = 0.5,
  complexity = c(1, 2)
)
cases[["tune_gazepoint_model::retain_candidate_failure"]] <- capture_case(
  tuning_value(
    gp3ml::tune_gazepoint_model(
      folds,
      cls_task,
      mixed_grid,
      predictors = predictors,
      metrics = "roc_auc",
      seed = seed,
      continue_on_error = TRUE,
      keep_evaluations = FALSE
    )
  )
)
cases[["tune_gazepoint_model::invalid_grid"]] <- capture_case(
  gp3ml::tune_gazepoint_model(folds, cls_task, "not grid", predictors = predictors)
)
cases[["tune_gazepoint_model::stop_on_candidate_failure"]] <- capture_case(
  gp3ml::tune_gazepoint_model(
    folds,
    cls_task,
    mixed_grid,
    predictors = predictors,
    metrics = "roc_auc",
    seed = seed,
    continue_on_error = FALSE,
    keep_evaluations = FALSE
  )
)

output <- list(
  runtime = "r",
  package_version = as.character(utils::packageVersion("gp3ml")),
  cases = cases
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  output,
  output_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  na = "null",
  digits = 16
)
