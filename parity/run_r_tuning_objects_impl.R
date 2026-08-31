run_cases <- function(fixture) {
  g <- fixture$grid
  cases <- list()

  cases[["create_gazepoint_tuning_grid::cartesian_grid"]] <- capture_case(
    grid_value(gp3ml::create_gazepoint_tuning_grid(
      engine = unlist(g$engine, use.names = FALSE),
      engine_grid = lapply(g$engine_grid, unlist, use.names = FALSE),
      preprocessor_grid = lapply(g$preprocessor_grid, unlist, use.names = FALSE),
      thresholds = unlist(g$thresholds, use.names = FALSE),
      complexity = g$complexity,
      interpretability = g$interpretability
    ))
  )
  cases[["create_gazepoint_tuning_grid::custom_labels"]] <- capture_case(
    grid_value(gp3ml::create_gazepoint_tuning_grid(
      engine = "glm", thresholds = c(0.4, 0.6), complexity = c(1, 2),
      interpretability = c("high", "medium"), labels = c("A", "B")
    ))
  )
  cases[["create_gazepoint_tuning_grid::invalid_threshold"]] <- capture_case(
    grid_value(gp3ml::create_gazepoint_tuning_grid("glm", thresholds = c(0, 0.5)))
  )
  cases[["create_gazepoint_tuning_grid::metadata_length"]] <- capture_case(
    grid_value(gp3ml::create_gazepoint_tuning_grid("glm", thresholds = c(0.4, 0.6), complexity = c(1, 2, 3)))
  )
  cases[["create_gazepoint_tuning_grid::empty_parameter"]] <- capture_case(
    grid_value(gp3ml::create_gazepoint_tuning_grid("glm", engine_grid = list(alpha = numeric())))
  )

  cases[["compare_gazepoint_models::all_candidates"]] <- capture_case(
    normalize_frame(gp3ml::compare_gazepoint_models(base_tuning()))
  )
  cases[["compare_gazepoint_models::metric_filter"]] <- capture_case(
    normalize_frame(gp3ml::compare_gazepoint_models(base_tuning(), metrics = "brier"))
  )
  cases[["compare_gazepoint_models::invalid_object"]] <- capture_case(
    normalize_frame(gp3ml::compare_gazepoint_models("not tuning"))
  )

  cases[["select_gazepoint_model::tie_breaker_selection"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(), metric = "roc_auc", direction = "maximize",
      tie_breakers = c("accuracy", "brier"), rationale = "predeclared review"
    ))
  )
  cases[["select_gazepoint_model::complexity_selection"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(), metric = "roc_auc", direction = "maximize",
      tie_breakers = character(), rationale = "prefer simpler tied candidate"
    ))
  )
  cases[["select_gazepoint_model::minimum_success_filter"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(low_success_second = TRUE), metric = "roc_auc", direction = "maximize",
      minimum_success_prop = 0.8, rationale = "require stable fold success"
    ))
  )
  cases[["select_gazepoint_model::accuracy_rejected"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(), metric = "accuracy", direction = "maximize", rationale = "not allowed"
    ))
  )
  cases[["select_gazepoint_model::invalid_direction"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(), metric = "roc_auc", direction = "up", rationale = "invalid direction"
    ))
  )
  cases[["select_gazepoint_model::missing_rationale"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(base_tuning(), metric = "roc_auc", direction = "maximize"))
  )
  cases[["select_gazepoint_model::no_eligible_metric"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(), metric = "pr_auc", direction = "maximize", rationale = "requested metric"
    ))
  )
  cases[["select_gazepoint_model::unresolved_tie"]] <- capture_case(
    selection_value(gp3ml::select_gazepoint_model(
      base_tuning(textual_complexity = TRUE), metric = "roc_auc", direction = "maximize",
      tie_breakers = character(), rationale = "tie remains"
    ))
  )

  cases[["validate_gazepoint_model_tuning::clean_validation"]] <- capture_case(
    validation_value(gp3ml::validate_gazepoint_model_tuning(base_tuning()))
  )
  cases[["validate_gazepoint_model_tuning::selection_review"]] <- capture_case({
    x <- base_tuning()
    x$selection <- structure(list(candidate_id = "candidate_001"), class = "gp3ml_model_selection")
    validation_value(gp3ml::validate_gazepoint_model_tuning(x))
  })
  cases[["validate_gazepoint_model_tuning::missing_result"]] <- capture_case({
    x <- base_tuning(); x$results <- x$results[-length(x$results)]
    validation_value(gp3ml::validate_gazepoint_model_tuning(x))
  })
  cases[["validate_gazepoint_model_tuning::incomplete_comparison"]] <- capture_case({
    x <- base_tuning(); x$comparison <- x$comparison[x$comparison$candidate_id != "candidate_003", , drop = FALSE]
    row.names(x$comparison) <- NULL
    validation_value(gp3ml::validate_gazepoint_model_tuning(x))
  })
  cases[["validate_gazepoint_model_tuning::target_mismatch"]] <- capture_case({
    x <- base_tuning(); x$folds_metadata <- list(generalization_target = "new_stimuli")
    validation_value(gp3ml::validate_gazepoint_model_tuning(x))
  })
  cases[["validate_gazepoint_model_tuning::invalid_object"]] <- capture_case(
    validation_value(gp3ml::validate_gazepoint_model_tuning("not tuning"))
  )

  cases[["write_gazepoint_model_tuning::write_with_selection"]] <- capture_case({
    x <- base_tuning()
    selection <- gp3ml::select_gazepoint_model(
      x, metric = "roc_auc", direction = "maximize", tie_breakers = "brier",
      rationale = "record reviewed candidate"
    )
    writer_value(x, selection)
  })
  cases[["write_gazepoint_model_tuning::overwrite_refusal"]] <- capture_case({
    x <- base_tuning(); directory <- tempfile("gp3mlpy-tuning-overwrite-"); dir.create(directory)
    gp3ml::write_gazepoint_model_tuning(x, directory, prefix = "parity_tuning", overwrite = FALSE)
    gp3ml::write_gazepoint_model_tuning(x, directory, prefix = "parity_tuning", overwrite = FALSE)
  })
  cases[["write_gazepoint_model_tuning::invalid_selection"]] <- capture_case({
    directory <- tempfile("gp3mlpy-tuning-invalid-selection-"); dir.create(directory)
    gp3ml::write_gazepoint_model_tuning(base_tuning(), directory, selection = "not selection")
  })
  cases[["write_gazepoint_model_tuning::invalid_object"]] <- capture_case({
    directory <- tempfile("gp3mlpy-tuning-invalid-object-"); dir.create(directory)
    gp3ml::write_gazepoint_model_tuning("not tuning", directory)
  })

  cases
}
