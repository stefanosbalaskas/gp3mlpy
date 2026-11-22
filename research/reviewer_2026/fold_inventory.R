# Supplementary reproducibility audit for Modelling-4636910.
# Frozen R gp3ml 0.3.0. New explicitly seeded demonstration; it is not
# the unpublished fold ledger underlying the original manuscript metrics.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript fold_inventory.R <output_directory>")
dir.create(args[[1L]], recursive = TRUE, showWarnings = FALSE)
if (as.character(utils::packageVersion("gp3ml")) != "0.3.0") stop("Wrong frozen R version")
dat <- gp3ml::simulate_gazepoint_governed_data(
  n_participants = 30L, n_stimuli = 8L, trials_per_cell = 2L, seed = 1L)
stopifnot(nrow(dat) == 480L, length(unique(dat$participant_id)) == 30L,
          length(unique(dat$stimulus_id)) == 8L)
set.seed(951L)
missing_rows <- sample.int(nrow(dat), size = 24L)
dat$gaze_dispersion[missing_rows] <- NA_real_
stopifnot(sum(is.na(dat$gaze_dispersion)) == 24L)
predictors <- c("tracking_ratio", "blink_rate", "gaze_dispersion")
manifest <- gp3ml::create_gazepoint_synthetic_manifest(
  outcome = "quality_status", predictors = predictors)
record <- list()
assignments <- list()
i <- 0L
j <- 0L
specifications <- list(
  participants = list(target = "new_participants", v = 5L),
  crossed = list(target = "new_participants_and_new_stimuli", v = c(5L, 4L))
)
for (design in names(specifications)) {
  spec <- specifications[[design]]
  plan <- gp3ml::create_gazepoint_group_folds(
    data = dat, outcome = "quality_status", predictors = predictors,
    feature_manifest = manifest, generalization_target = spec$target,
    participant_id = "participant_id", stimulus_id = "stimulus_id",
    trial_id = "trial_id", v = spec$v, repeats = 2L, seed = 20261122L)
  stopifnot(identical(plan$audit$status, "pass"),
            identical(plan$validation$status, "pass"))
  for (fold in plan$folds) {
    an <- fold$analysis
    as <- fold$assessment
    ex <- fold$excluded
    n_part_overlap <- length(intersect(unique(an$participant_id), unique(as$participant_id)))
    n_stim_overlap <- length(intersect(unique(an$stimulus_id), unique(as$stimulus_id)))
    stopifnot(n_part_overlap == 0L,
              if (design == "crossed") n_stim_overlap == 0L else TRUE,
              nrow(an) + nrow(as) + nrow(ex) == 480L)
    i <- i + 1L
    tr <- function(frame, label) {
      data.frame(
        design=design, target=spec$target, `repeat`=fold[["repeat"]],
        fold=fold$fold, fold_id=fold$fold_id, partition=label,
        n_rows=nrow(frame),
        n_participants=length(unique(frame$participant_id)),
        n_stimuli=length(unique(frame$stimulus_id)),
        n_pass=sum(frame$quality_status == "pass", na.rm=TRUE),
        n_review=sum(frame$quality_status == "review", na.rm=TRUE),
        participant_overlap=n_part_overlap,
        stimulus_overlap=n_stim_overlap,
        stringsAsFactors=FALSE)
    }
    record[[i]] <- do.call(rbind, list(tr(an,"analysis"),tr(as,"assessment"),tr(ex,"excluded")))
  }
  a <- plan$assignments
  coverage <- stats::aggregate(as.integer(a$partition == "assessment"),
    by=list(`repeat`=a[["repeat"]], source_row=a$source_row), FUN=sum)
  stopifnot(nrow(coverage) == 2L * nrow(dat), all(coverage$x == 1L))
  a$design <- design
  j <- j + 1L
  assignments[[j]] <- a
}
inventory <- do.call(rbind, record)
assignments <- do.call(rbind, assignments)
stopifnot(nrow(inventory) == (10L + 40L) * 3L,
          all(inventory$n_pass + inventory$n_review == inventory$n_rows),
          all(inventory$participant_overlap == 0L),
          all(inventory$stimulus_overlap[inventory$design == "crossed"] == 0L))
write.csv(inventory, file.path(args[[1L]], "fold_inventory.csv"), row.names=FALSE)
write.csv(assignments, file.path(args[[1L]], "fold_assignments.csv"), row.names=FALSE)
write.csv(dat, file.path(args[[1L]], "synthetic_source.csv"), row.names=FALSE)
cat("R_FROZEN_VERSION=0.3.0\n")
cat("SOURCE_ROWS=480; MISSING_GAZE=24\n")
for (design in names(specifications)) {
  subset <- inventory[inventory$design == design,]
  cat(sprintf("FOLD_DESIGN=%s; FOLDS=%d; ANALYSIS_ROWS=%d; ASSESSMENT_ROWS=%d; EXCLUDED_ROWS=%d; PARTICIPANT_OVERLAPS=%d; STIMULUS_OVERLAPS=%d\n",
     design, sum(subset$partition == "analysis"),
     sum(subset$n_rows[subset$partition == "analysis"]),
     sum(subset$n_rows[subset$partition == "assessment"]),
     sum(subset$n_rows[subset$partition == "excluded"]),
     sum(subset$participant_overlap)/3L, sum(subset$stimulus_overlap[subset$partition == "analysis"])))
}
cat("R_FOLD_INVENTORY=PASS\n")
