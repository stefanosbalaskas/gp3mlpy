args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript parity/run_r_tuning_objects.R <fixture.json> <output.json>", call. = FALSE)
}

fixture_path <- args[[1L]]
output_path <- args[[2L]]
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(file_arg)) stop("Cannot resolve parity runner path.", call. = FALSE)
script_path <- sub("^--file=", "", file_arg[[1L]])
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))

source(file.path(script_dir, "run_r_tuning_objects_helpers.R"), local = TRUE)
source(file.path(script_dir, "run_r_tuning_objects_impl.R"), local = TRUE)

fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)
cases <- run_cases(fixture)

output <- list(
  runtime = "r",
  package_version = as.character(utils::packageVersion("gp3ml")),
  cases = cases
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  output, path = output_path, auto_unbox = TRUE, pretty = TRUE,
  null = "null", na = "null", digits = 16
)
