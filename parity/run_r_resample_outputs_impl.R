spec <- fixture$classification
clean <- evaluation_fixture(spec, fixture$metrics, FALSE)
failed <- evaluation_fixture(spec, fixture$metrics, TRUE)

bad_stage <- clean
bad_stage$predictions <- bad_stage$predictions
bad_stage$predictions$stage[[1L]] <- "analysis"

incomplete <- clean
incomplete$folds_metadata <- incomplete$folds_metadata
incomplete$folds_metadata$n_folds_total <- 3L

target_mismatch <- clean
target_mismatch$generalization_target <- "new_stimuli"

no_predictions <- clean
no_predictions$predictions <- no_predictions$predictions[FALSE, , drop = FALSE]

manifest_spec <- fixture$manifest
manifest <- gp3ml::create_gazepoint_feature_manifest(
  features = as_character_vector(manifest_spec$features),
  scientific_source = as_character_vector(manifest_spec$scientific_source),
  source_table = as_character_vector(manifest_spec$source_table),
  transformation = as_character_vector(manifest_spec$transformation),
  availability_stage = as_character_vector(manifest_spec$availability_stage),
  prediction_time_available = as_logical_vector(manifest_spec$prediction_time_available),
  preprocessing_scope = as_character_vector(manifest_spec$preprocessing_scope),
  fold_local_required = as_logical_vector(manifest_spec$fold_local_required)
)
manifest_validation <- gp3ml::validate_gazepoint_feature_manifest(manifest)

feature_cases <- list(
  manifest_success = capture_call(function() {
    feature_write(
      function(path) gp3ml::write_gazepoint_feature_manifest_csv(manifest, path),
      "manifest.csv"
    )
  }),
  validation_checks_success = capture_call(function() {
    feature_write(
      function(path) gp3ml::write_gazepoint_feature_manifest_csv(
        manifest_validation, path, table = "checks"
      ),
      "checks.csv"
    )
  }),
  validation_issues_success = capture_call(function() {
    feature_write(
      function(path) gp3ml::write_gazepoint_feature_manifest_csv(
        manifest_validation, path, table = "issues"
      ),
      "issues.csv"
    )
  }),
  overwrite_refusal = capture_call(function() feature_overwrite(manifest)),
  invalid_extension = capture_call(function() feature_invalid_extension(manifest)),
  plain_checks_rejection = capture_call(function() feature_plain_checks(manifest))
)

collector_cases <- list(
  clean_predictions = capture_call(
    function() gp3ml::collect_gazepoint_fold_predictions(clean),
    function(value) normalize_frame(value, c("repeat", "fold", ".gp3ml_source_row"))
  ),
  failed_include = capture_call(
    function() gp3ml::collect_gazepoint_fold_predictions(failed, include_failed = TRUE),
    function(value) normalize_frame(value, c("repeat", "fold", ".gp3ml_source_row"))
  ),
  failed_exclude = capture_call(
    function() gp3ml::collect_gazepoint_fold_predictions(failed, include_failed = FALSE),
    function(value) normalize_frame(value, c("repeat", "fold", ".gp3ml_source_row"))
  ),
  invalid_object = capture_call(function() gp3ml::collect_gazepoint_fold_predictions(list()))
)

summary_cases <- list(
  fold_distribution = capture_call(
    function() gp3ml::summarize_gazepoint_resample_performance(
      clean, aggregation = "fold_distribution", conf_level = 0.8
    ),
    normalize_summary
  ),
  pooled_rows = capture_call(
    function() gp3ml::summarize_gazepoint_resample_performance(
      clean, aggregation = "pooled_rows", conf_level = 0.8
    ),
    normalize_summary
  ),
  no_predictions = capture_call(function() gp3ml::summarize_gazepoint_resample_performance(
    no_predictions, aggregation = "pooled_rows"
  )),
  invalid_aggregation = capture_call(function() gp3ml::summarize_gazepoint_resample_performance(
    clean, aggregation = "invalid"
  )),
  invalid_object = capture_call(function() gp3ml::summarize_gazepoint_resample_performance(list()))
)

validation_cases <- list(
  clean_validation = capture_call(
    function() gp3ml::validate_gazepoint_resample_evaluation(clean), normalize_validation
  ),
  failed_validation = capture_call(
    function() gp3ml::validate_gazepoint_resample_evaluation(failed), normalize_validation
  ),
  bad_stage_validation = capture_call(
    function() gp3ml::validate_gazepoint_resample_evaluation(bad_stage), normalize_validation
  ),
  incomplete_status_validation = capture_call(
    function() gp3ml::validate_gazepoint_resample_evaluation(incomplete), normalize_validation
  ),
  target_mismatch_validation = capture_call(
    function() gp3ml::validate_gazepoint_resample_evaluation(target_mismatch), normalize_validation
  ),
  invalid_object = capture_call(function() gp3ml::validate_gazepoint_resample_evaluation(list()))
)

writer_cases <- list(
  clean_write = capture_call(function() resample_write(clean)),
  overwrite_refusal = capture_call(function() resample_overwrite(clean)),
  invalid_object = capture_call(
    function() gp3ml::write_gazepoint_resample_evaluation(list(), tempdir())
  )
)

result <- list(
  schema_version = 1L,
  runtime = "r",
  package_version = as.character(utils::packageVersion("gp3ml")),
  feature_writer_cases = feature_cases,
  collector_cases = collector_cases,
  summary_cases = summary_cases,
  validation_cases = validation_cases,
  writer_cases = writer_cases
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
