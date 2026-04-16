#!/usr/bin/env Rscript
# ChromVAR Vignette Reproducibility - R Reference Script
#
# Follows the official chromVAR vignette workflow exactly:
# https://github.com/GreenleafLab/chromVAR/blob/master/vignettes/Articles/Deviations.Rmd
#
# Uses the example_counts dataset with ALL JASPAR motifs.
# Exports all intermediate and final results for comparison with GATAC.
#
# Run with:
#   pixi run test-chromvar-vignette

library(chromVAR)
library(motifmatchr)
library(Matrix)
library(SummarizedExperiment)
library(BiocParallel)
library(BSgenome.Hsapiens.UCSC.hg19)
library(Biostrings)

# Same seed and serial execution as the vignette
set.seed(2017)
register(SerialParam())

output_dir <- "data/chromvar_vignette_output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== ChromVAR Vignette Reproducibility ===\n\n")

# ============================================================================
# 1. Load example data (same as vignette)
# ============================================================================

cat("1. Loading example_counts...\n")
data(example_counts, package = "chromVAR")

# ============================================================================
# 2. Add GC bias (same as vignette)
# ============================================================================

cat("2. Computing GC bias...\n")
example_counts <- addGCBias(example_counts, genome = BSgenome.Hsapiens.UCSC.hg19)

# ============================================================================
# 3. Filter samples and peaks (same as vignette)
# ============================================================================

cat("3. Filtering samples and peaks...\n")
counts_filtered <- filterSamples(example_counts,
                                 min_depth = 1500,
                                 min_in_peaks = 0.15,
                                 shiny = FALSE)
counts_filtered <- filterPeaks(counts_filtered)

# Set peak names to coordinate format for cross-tool alignment
peaks_gr <- rowRanges(counts_filtered)
peak_names <- paste0(seqnames(peaks_gr), ":", start(peaks_gr), "-", end(peaks_gr))
rownames(counts_filtered) <- peak_names

cat(sprintf("   Filtered: %d peaks x %d samples\n",
            nrow(counts_filtered), ncol(counts_filtered)))

# ============================================================================
# 4. Get ALL JASPAR motifs (same as vignette)
# ============================================================================

cat("4. Loading JASPAR motifs...\n")
motifs <- getJasparMotifs()
cat(sprintf("   Loaded %d motifs\n", length(motifs)))

# ============================================================================
# 5. Match motifs (default p.cutoff = 5e-5)
# ============================================================================

cat("5. Matching motifs to peaks...\n")
motif_ix <- matchMotifs(motifs, counts_filtered,
                        genome = BSgenome.Hsapiens.UCSC.hg19)

motif_matches <- assay(motif_ix, "motifMatches")
motif_names <- names(motifs)

cat(sprintf("   Motif matches: %d (%.2f%% density)\n",
            sum(motif_matches),
            100 * sum(motif_matches) / length(motif_matches)))

# ============================================================================
# 6. Get background peaks (as in vignette Options section)
# ============================================================================

cat("6. Computing background peaks...\n")
bg <- getBackgroundPeaks(object = counts_filtered)
cat(sprintf("   Background peaks matrix: %d x %d\n", nrow(bg), ncol(bg)))

# ============================================================================
# 7. Compute deviations (same as vignette)
# ============================================================================

cat("7. Computing deviations...\n")
t_dev_start <- proc.time()
dev <- computeDeviations(object = counts_filtered,
                         annotations = motif_ix,
                         background_peaks = bg)
r_dev_elapsed <- (proc.time() - t_dev_start)[["elapsed"]]
cat(sprintf("   computeDeviations time: %.2fs\n", r_dev_elapsed))

deviations_mat <- deviations(dev)
scores_mat <- deviationScores(dev)

cat(sprintf("   Deviations: %d motifs x %d samples\n",
            nrow(deviations_mat), ncol(deviations_mat)))

# ============================================================================
# 8. Export all results
# ============================================================================

cat("\n8. Exporting results...\n")

# Peak coordinates (1-based GenomicRanges)
peaks_df <- data.frame(
  chrom = as.character(seqnames(peaks_gr)),
  start = start(peaks_gr),
  end = end(peaks_gr),
  name = peak_names
)
write.csv(peaks_df, file.path(output_dir, "peaks.csv"),
          row.names = FALSE, quote = FALSE)

# Count matrix (peaks x cells)
write.csv(as.matrix(assay(counts_filtered, "counts")),
          file.path(output_dir, "count_matrix.csv"), quote = FALSE)

# GC content
write.csv(data.frame(gc_content = rowData(counts_filtered)$bias,
                     row.names = peak_names),
          file.path(output_dir, "gc_content.csv"), quote = FALSE)

# Peak sequences for GC verification
peak_seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg19, peaks_gr)
write.csv(data.frame(peak = peak_names,
                     sequence = as.character(peak_seqs)),
          file.path(output_dir, "peak_sequences.csv"),
          row.names = FALSE, quote = FALSE)

# Motif names
write.csv(data.frame(motif_name = motif_names),
          file.path(output_dir, "motif_names.csv"),
          row.names = FALSE, quote = FALSE)

# Motif PWMs (for GATAC to load the same motifs)
pwms_list <- lapply(seq_along(motifs), function(i) {
  mat <- as.matrix(motifs[[i]])
  df <- as.data.frame(t(mat))
  colnames(df) <- c("A", "C", "G", "T")
  df$motif <- names(motifs)[i]
  df$pos <- 1:nrow(df)
  return(df)
})
write.csv(do.call(rbind, pwms_list),
          file.path(output_dir, "motifs_pwm.csv"),
          row.names = FALSE, quote = FALSE)

# Motif matches - sparse COO format (compact)
motif_sparse <- summary(as(motif_matches, "dgCMatrix"))
write.csv(motif_sparse, file.path(output_dir, "motif_matches_sparse.csv"),
          row.names = FALSE, quote = FALSE)

# Background peaks (1-based indices)
write.csv(bg, file.path(output_dir, "background_peaks.csv"), quote = FALSE)

# Deviation z-scores (motifs x cells)
write.csv(scores_mat, file.path(output_dir, "deviation_scores.csv"), quote = FALSE)

# Raw deviations (motifs x cells)
write.csv(deviations_mat, file.path(output_dir, "deviations.csv"), quote = FALSE)

# Deviation timing
write.csv(data.frame(deviation_time = r_dev_elapsed),
          file.path(output_dir, "timing.csv"),
          row.names = FALSE, quote = FALSE)

# Summary statistics
summary_stats <- data.frame(
  n_peaks = nrow(counts_filtered),
  n_cells = ncol(counts_filtered),
  n_motifs = length(motifs),
  total_motif_matches = sum(motif_matches),
  deviation_mean = mean(deviations_mat),
  deviation_sd = sd(deviations_mat),
  score_mean = mean(scores_mat),
  score_sd = sd(scores_mat)
)
write.csv(summary_stats, file.path(output_dir, "summary_stats.csv"),
          row.names = FALSE, quote = FALSE)

cat(sprintf("\nAll results exported to %s/\n", output_dir))
cat(sprintf("  Peaks: %d, Samples: %d, Motifs: %d\n",
            nrow(counts_filtered), ncol(counts_filtered), length(motifs)))
cat("=== Done ===\n")
