args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_final_reporting.R <fixture.json> <output.json>")
}

fixture_path <- args[[1L]]
output_path <- args[[2L]]
fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)
seed <- as.integer(fixture$seed)

suppressPackageStartupMessages(library(gp3ml))

predictors <- c("fixation_duration", "pupil_change")
quality <- c(
  "pass", "review", "pass", "review", "review", "pass",
  "review", "pass", "pass", "review", "review", "pass",
  "review", "pass", "review", "pass", "pass", "review",
  "pass", "review", "review", "pass", "pass", "review"
)

capture_case <- function(fun) {
  tryCatch(
    list(status = "success", value = fun()),
    error = function(e) {
      list(
        status = "error",
        error_type = class(e)[[1L]],
        message = conditionMessage(e)
      )
    }
  )
}

training_data <- function() {
  data.frame(
    participant_id = rep(sprintf("P%02d", 1:12), each = 2),
    trial_id = sprintf("T%02d", 1:24),
    stimulus_id = rep(c("S01", "S02"), 12),
    fixation_duration = as.numeric(180 + seq_len(24)),
    pupil_change = sin(seq_len(24) / 3),
    quality_status = factor(quality, levels = c("pass", "review")),
    stringsAsFactors = FALSE
  )
}

make_model <- function(data, seed) {
  task <- declare_gazepoint_task(
    data = data,
    outcome = "quality_status",
    purpose = "Predict predefined recording-quality review status",
    task_type = "classification",
    unit_id = "trial_id",
    participant_id = "participant_id",
    stimulus_id = "stimulus_id",
    generalization_target = "new_participants",
    positive = "review"
  )
  model <- train_gazepoint_classifier(
    data = data,
    task = task,
    predictors = predictors,
    engine = "glm",
    seed = seed
  )
  list(task = task, model = model)
}

card_summary <- function(card) {
  evaluation <- card$evaluation
  list(
    class = class(card)[[1L]],
    title = as.character(card$title),
    intended_use = as.character(card$intended_use),
    prohibited_count = length(card$prohibited_uses),
    outcome = as.character(card$task$outcome),
    task_type = as.character(card$task$task_type),
    generalization_target = as.character(card$task$generalization_target),
    engine = as.character(card$engine),
    predictors = as.list(as.character(card$predictors)),
    training_n = as.integer(card$training_n),
    training_hash_recorded = isTRUE(nzchar(as.character(card$training_hash))),
    evaluation_rows = if (is.data.frame(evaluation)) nrow(evaluation) else 0L,
    evaluation_columns = if (is.data.frame(evaluation)) as.list(sort(names(evaluation))) else list(),
    limitations = as.list(as.character(card$limitations)),
    external_validation_present = !is.null(card$external_validation)
  )
}

release_summary <- function(card) {
  out <- card_summary(card)
  out$class <- class(card)[[1L]]
  out$selection_recorded <- isTRUE(card$selection_procedure_recorded)
  out$uncertainty_unit <- if (is.na(card$uncertainty_unit)) NA_character_ else as.character(card$uncertainty_unit)
  out$release_generalization_target <- as.character(card$generalization_target)
  out$external_validation_status <- as.character(card$external_validation_status)
  out$autonomous_selection <- isTRUE(card$autonomous_selection)
  out$deployment_status <- as.character(card$deployment_status)
  out
}

markdown_summary <- function(path, returned) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  list(
    exists = file.exists(path),
    return_basename = basename(returned),
    basename = basename(path),
    headings = as.list(lines[startsWith(lines, "## ")]),
    line_count_positive = length(lines) > 0L
  )
}

json_summary <- function(path, returned) {
  payload <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  required <- c("title", "intended_use", "engine", "predictors", "training_n", "limitations")
  present <- lapply(required, function(name) name %in% names(payload))
  names(present) <- required
  list(
    exists = file.exists(path),
    return_basename = basename(returned),
    basename = basename(path),
    valid_json = is.list(payload),
    required_fields_present = present,
    title = as.character(payload$title),
    engine = as.character(payload$engine)
  )
}

write_model_summary <- function(card, format) {
  suffix <- if (identical(format, "json")) ".json" else ".md"
  path <- tempfile(pattern = "model_card", fileext = suffix)
  on.exit(unlink(path), add = TRUE)
  returned <- write_gazepoint_model_card(card, path, format = format)
  if (identical(format, "json")) json_summary(path, returned) else markdown_summary(path, returned)
}

write_release_summary <- function(card, format) {
  suffix <- if (identical(format, "json")) ".json" else ".md"
  path <- tempfile(pattern = "release_model_card", fileext = suffix)
  on.exit(unlink(path), add = TRUE)
  returned <- write_gazepoint_release_model_card(card, path, format = format)
  if (identical(format, "json")) json_summary(path, returned) else markdown_summary(path, returned)
}

named_character_list <- function(x) {
  if (!length(x)) return(list())
  out <- as.list(as.character(x))
  names(out) <- names(x)
  out
}

evidence_summary <- function(x) {
  checksums <- named_character_list(x$file_md5)
  paths <- if (!length(x$file_paths)) list() else {
    out <- lapply(as.list(x$file_paths), basename)
    names(out) <- names(x$file_paths)
    out
  }
  list(
    class = class(x)[[1L]],
    version = as.character(x$version),
    object_hash_count = length(x$object_hashes),
    object_hash_names = as.list(sort(names(x$object_hashes))),
    object_hashes_recorded = all(nzchar(as.character(x$object_hashes))),
    file_checksum_count = length(x$file_md5),
    file_checksum_names = as.list(sort(names(x$file_md5))),
    file_checksums = checksums,
    file_basenames = paths,
    session_recorded = length(x$session) > 0L,
    notes = as.list(as.character(x$notes)),
    prohibited_count = length(x$prohibited_uses)
  )
}

repro_summary <- function(x) {
  seed_values <- x$seeds
  if (length(seed_values)) {
    seed_values <- lapply(seed_values, as.integer)
    seed_values <- seed_values[sort(names(seed_values))]
  }
  list(
    class = class(x)[[1L]],
    runtime_recorded = length(x$r_version) > 0L && nzchar(as.character(x$r_version)[[1L]]),
    platform_recorded = length(x$platform) > 0L && nzchar(as.character(x$platform)[[1L]]),
    session_recorded = length(x$session) > 0L,
    object_hash_count = length(x$object_hashes),
    object_hash_names = as.list(sort(names(x$object_hashes))),
    object_hashes_recorded = all(nzchar(as.character(x$object_hashes))),
    data_hash_recorded = !is.na(x$data_hash) && nzchar(as.character(x$data_hash)),
    seeds = seed_values,
    git = list(
      commit = x$git$commit,
      branch = x$git$branch,
      clean = x$git$clean
    ),
    notes = as.list(as.character(x$notes)),
    prohibited_count = length(x$prohibited_uses)
  )
}

write_repro_summary <- function(report) {
  path <- tempfile(pattern = "reproducibility", fileext = ".md")
  on.exit(unlink(path), add = TRUE)
  returned <- write_gazepoint_reproducibility_report(report, path)
  markdown_summary(path, returned)
}

data <- training_data()
objects <- make_model(data, seed)
task <- objects$task
model <- objects$model
evaluation <- data.frame(metric = "accuracy", estimate = 0.75, stringsAsFactors = FALSE)
intended <- "Support manual review of predefined recording-quality status"
limitations <- "Synthetic parity fixture only."

card <- create_gazepoint_model_card(
  model = model,
  intended_use = intended,
  limitations = limitations
)
card_eval <- create_gazepoint_model_card(
  model = model,
  intended_use = intended,
  evaluation = evaluation,
  limitations = limitations
)
release <- create_gazepoint_release_model_card(
  model = model,
  intended_use = intended,
  evaluation = evaluation,
  limitations = limitations
)

cases <- list()
cases[["create_gazepoint_model_card::minimal"]] <- capture_case(function() card_summary(card))
cases[["create_gazepoint_model_card::dataframe_evaluation"]] <- capture_case(function() card_summary(card_eval))
cases[["create_gazepoint_model_card::invalid_model"]] <- capture_case(function() {
  create_gazepoint_model_card("not model", intended)
})
cases[["write_gazepoint_model_card::markdown_success"]] <- capture_case(function() write_model_summary(card_eval, "markdown"))
cases[["write_gazepoint_model_card::json_success"]] <- capture_case(function() write_model_summary(card_eval, "json"))
cases[["write_gazepoint_model_card::overwrite_rejected"]] <- capture_case(function() {
  path <- tempfile(fileext = ".md")
  on.exit(unlink(path), add = TRUE)
  write_gazepoint_model_card(card, path)
  write_gazepoint_model_card(card, path)
})
cases[["write_gazepoint_model_card::invalid_card"]] <- capture_case(function() {
  write_gazepoint_model_card("not card", tempfile(fileext = ".md"))
})
cases[["write_gazepoint_model_card::invalid_format"]] <- capture_case(function() {
  write_gazepoint_model_card(card, tempfile(fileext = ".md"), format = "yaml")
})

cases[["create_gazepoint_release_model_card::minimal"]] <- capture_case(function() release_summary(release))
cases[["create_gazepoint_release_model_card::dataframe_evaluation"]] <- capture_case(function() release_summary(release))
cases[["create_gazepoint_release_model_card::missing_limitations"]] <- capture_case(function() {
  create_gazepoint_release_model_card(model = model, intended_use = intended)
})
cases[["create_gazepoint_release_model_card::blank_limitations"]] <- capture_case(function() {
  create_gazepoint_release_model_card(model = model, intended_use = intended, limitations = "")
})
cases[["create_gazepoint_release_model_card::invalid_model"]] <- capture_case(function() {
  create_gazepoint_release_model_card(model = "not model", intended_use = intended, limitations = limitations)
})
cases[["create_gazepoint_release_model_card::invalid_selection"]] <- capture_case(function() {
  create_gazepoint_release_model_card(model = model, intended_use = intended, selection = "bad", limitations = limitations)
})
cases[["create_gazepoint_release_model_card::invalid_uncertainty"]] <- capture_case(function() {
  create_gazepoint_release_model_card(model = model, intended_use = intended, uncertainty = "bad", limitations = limitations)
})
cases[["create_gazepoint_release_model_card::invalid_transportability"]] <- capture_case(function() {
  create_gazepoint_release_model_card(model = model, intended_use = intended, transportability = "bad", limitations = limitations)
})
cases[["write_gazepoint_release_model_card::markdown_success"]] <- capture_case(function() write_release_summary(release, "markdown"))
cases[["write_gazepoint_release_model_card::json_success"]] <- capture_case(function() write_release_summary(release, "json"))
cases[["write_gazepoint_release_model_card::overwrite_rejected"]] <- capture_case(function() {
  path <- tempfile(fileext = ".md")
  on.exit(unlink(path), add = TRUE)
  write_gazepoint_release_model_card(release, path)
  write_gazepoint_release_model_card(release, path)
})
cases[["write_gazepoint_release_model_card::invalid_card"]] <- capture_case(function() {
  write_gazepoint_release_model_card("not card", tempfile(fileext = ".md"))
})
cases[["write_gazepoint_release_model_card::invalid_format"]] <- capture_case(function() {
  write_gazepoint_release_model_card(release, tempfile(fileext = ".md"), format = "yaml")
})

cases[["create_gazepoint_release_evidence::empty_defaults"]] <- capture_case(function() {
  evidence_summary(create_gazepoint_release_evidence())
})
evidence_path <- tempfile(fileext = ".txt")
writeLines("gp3ml parity release evidence", evidence_path, useBytes = TRUE)
populated_evidence <- create_gazepoint_release_evidence(
  objects = list(vector = 1:3, label = "synthetic"),
  files = c(note = evidence_path),
  version = "0.3.0-candidate",
  notes = "Frozen R/Python parity evidence."
)
cases[["create_gazepoint_release_evidence::objects_and_file"]] <- capture_case(function() evidence_summary(populated_evidence))
cases[["create_gazepoint_release_evidence::missing_file"]] <- capture_case(function() {
  create_gazepoint_release_evidence(files = c(missing = paste0(evidence_path, ".missing")))
})
cases[["create_gazepoint_release_evidence::unnamed_files"]] <- capture_case(function() {
  create_gazepoint_release_evidence(files = unname(c(note = evidence_path)))
})
cases[["create_gazepoint_release_evidence::custom_version_notes"]] <- capture_case(function() {
  evidence_summary(create_gazepoint_release_evidence(version = "9.9.9", notes = c("alpha", "beta")))
})
unlink(evidence_path)

project_dir <- tempfile(pattern = "gp3ml-repro-")
dir.create(project_dir, recursive = TRUE)
on.exit(unlink(project_dir, recursive = TRUE, force = TRUE), add = TRUE)
empty_repro <- create_gazepoint_reproducibility_report(project_path = project_dir)
populated_repro <- create_gazepoint_reproducibility_report(
  objects = list(vector = 1:3),
  data = data.frame(a = 1:3, b = c("x", "y", "z"), stringsAsFactors = FALSE),
  seeds = list(split = 101L, bootstrap = 202L),
  notes = "Synthetic reproducibility parity fixture.",
  project_path = project_dir
)
cases[["create_gazepoint_reproducibility_report::empty_defaults"]] <- capture_case(function() repro_summary(empty_repro))
cases[["create_gazepoint_reproducibility_report::populated"]] <- capture_case(function() repro_summary(populated_repro))
cases[["write_gazepoint_reproducibility_report::markdown_success"]] <- capture_case(function() write_repro_summary(populated_repro))
cases[["write_gazepoint_reproducibility_report::overwrite_rejected"]] <- capture_case(function() {
  path <- tempfile(fileext = ".md")
  on.exit(unlink(path), add = TRUE)
  write_gazepoint_reproducibility_report(populated_repro, path)
  write_gazepoint_reproducibility_report(populated_repro, path)
})
cases[["write_gazepoint_reproducibility_report::invalid_report"]] <- capture_case(function() {
  write_gazepoint_reproducibility_report("not report", tempfile(fileext = ".md"))
})

cases[["fit_gazepoint_deep_model::backend_missing"]] <- capture_case(function() {
  fit_gazepoint_deep_model(
    data = data, task = task, predictors = predictors,
    hidden_units = 4L, dropout = 0, epochs = 1L,
    batch_size = 8L, validation_split = 0, seed = seed, verbose = 0L
  )
})
cases[["fit_gazepoint_deep_model::backend_precedes_validation"]] <- capture_case(function() {
  fit_gazepoint_deep_model(
    data = data, task = "not task", predictors = predictors,
    hidden_units = 4L, dropout = 0, epochs = 1L,
    batch_size = 8L, validation_split = 0, seed = seed, verbose = 0L
  )
})

output <- list(
  schema_version = 1L,
  runtime = "r",
  package_version = as.character(utils::packageVersion("gp3ml")),
  cases = cases
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  output,
  output_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  na = "null"
)
cat("r final-reporting records:", length(cases), "\n")
