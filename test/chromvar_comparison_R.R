#!/usr/bin/env Rscript
# chromVAR R Implementation - Comparison Script
# This script runs chromVAR on ATAC-seq data and exports results for comparison with GATAC

library(chromVAR)
library(motifmatchr)
library(Matrix)
library(SummarizedExperiment)
library(BiocParallel)
library(BSgenome.Hsapiens.UCSC.hg19)
library(GenomicRanges)

# Set seed for reproducibility
set.seed(42)

# Configure parallel processing (use SerialParam for reproducibility)
register(SerialParam())

# Output directory for results
output_dir <- "data/chromvar_r_output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== chromVAR R Implementation Comparison ===\n\n")

# ============================================================================
# 1. Load Input Data
# ============================================================================

cat("1. Loading input data...\n")

# Option A: Load from file (user should provide their own data)
# For this script to work, you need to provide:
# - peaks_file: BED file with peak regions
# - bam_files: BAM files or fragment files
# - motif_file: JASPAR motifs or custom motif file

# Example using chromVAR example data
data(example_counts, package = "chromVAR")
counts_obj <- example_counts

# Pre-set peak names to coordinate format for consistency
peaks_gr <- rowRanges(counts_obj)
peak_names <- paste0(seqnames(peaks_gr), ":", start(peaks_gr), "-", end(peaks_gr))
rownames(counts_obj) <- peak_names

cat(sprintf("   Loaded counts: %d peaks x %d samples\n", 
            nrow(counts_obj), ncol(counts_obj)))

# ============================================================================
# 2. Add GC Bias
# ============================================================================

cat("\n2. Computing GC bias...\n")
counts_obj <- addGCBias(counts_obj, genome = BSgenome.Hsapiens.UCSC.hg19)

# ============================================================================
# 3. Filter Samples and Peaks
# ============================================================================

cat("\n3. Filtering samples and peaks...\n")
counts_filtered <- filterSamples(counts_obj, 
                                 min_depth = 1500,
                                 min_in_peaks = 0.15, 
                                 shiny = FALSE)
cat(sprintf("   Samples after filtering: %d\n", ncol(counts_filtered)))

counts_filtered <- filterPeaks(counts_filtered, non_overlapping = TRUE)
cat(sprintf("   Peaks after filtering: %d\n", nrow(counts_filtered)))

# ============================================================================
# 4. Export Filtered Data for Comparison
# ============================================================================

cat("\n4. Exporting filtered data for comparison...\n")

# Export filtered peak regions
peaks_gr_filtered <- rowRanges(counts_filtered)
peaks_df <- data.frame(
  chrom = as.character(seqnames(peaks_gr_filtered)),
  start = start(peaks_gr_filtered),
  end = end(peaks_gr_filtered),
  name = rownames(counts_filtered)
)
write.csv(peaks_df, 
          file.path(output_dir, "peaks.csv"), 
          row.names = FALSE, 
          quote = FALSE)

# Export filtered count matrix
counts_mat <- assay(counts_filtered, "counts")
write.csv(as.matrix(counts_mat), 
          file.path(output_dir, "count_matrix.csv"),
          quote = FALSE)

# Export GC content for filtered peaks
gc_content <- rowData(counts_filtered)$bias
write.csv(data.frame(gc_content = gc_content, row.names = rownames(counts_filtered)), 
          file.path(output_dir, "gc_content.csv"),
          quote = FALSE)

# Export sequences for filtered peaks (for motifs comparison)
cat("\n4b. Exporting sequences for filtered peaks...\n")
library(Biostrings)
peak_seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg19, peaks_gr_filtered)
# Export to a CSV for easier loading in Python
write.csv(data.frame(peak = rownames(counts_filtered), sequence = as.character(peak_seqs)), 
          file.path(output_dir, "peak_sequences.csv"),
          row.names = FALSE,
          quote = FALSE)

cat(sprintf("   Exported filtered peaks, counts, and GC content to %s/\n", output_dir))

# ============================================================================
# 5. Get Motifs
# ============================================================================

cat("\n5. Loading motifs...\n")
# Using JASPAR motifs - subset for faster testing
motifs <- getJasparMotifs()
cat(sprintf("   Loaded %d motifs from JASPAR\n", length(motifs)))

# For testing, use only first 10 motifs
motifs <- motifs[1:10]
cat(sprintf("   Using %d motifs for testing\n", length(motifs)))

# Export motif names and PWMs
motif_names <- names(motifs)
write.csv(data.frame(motif_name = motif_names), 
          file.path(output_dir, "motif_names.csv"),
          row.names = FALSE,
          quote = FALSE)

# Export PWMs as a single CSV (motif, row, A, C, G, T)
pwms_list <- lapply(seq_along(motifs), function(i) {
  m <- motifs[[i]]
  # TFBSTools matrices can be converted to matrix
  mat <- as.matrix(m)
  df <- as.data.frame(t(mat))
  colnames(df) <- c("A", "C", "G", "T")
  df$motif <- names(motifs)[i]
  df$pos <- 1:nrow(df)
  return(df)
})
pwms_df <- do.call(rbind, pwms_list)
write.csv(pwms_df, file.path(output_dir, "motifs_pwm.csv"), row.names = FALSE, quote = FALSE)

cat(sprintf("   Exported motif names and PWMs to %s\n", 
            file.path(output_dir, "motifs_pwm.csv")))

# ============================================================================
# 6. Match Motifs to Peaks
# ============================================================================

cat("\n6. Matching motifs to peaks...\n")
motif_ix <- matchMotifs(motifs, 
                        counts_filtered,
                        genome = BSgenome.Hsapiens.UCSC.hg19,
                        p.cutoff = 5e-5)

# Export motif match matrix
motif_matches <- assay(motif_ix, "motifMatches")
motif_matches_mat <- as.matrix(motif_matches)
rownames(motif_matches_mat) <- rownames(counts_filtered)
colnames(motif_matches_mat) <- motif_names

write.csv(motif_matches_mat, 
          file.path(output_dir, "motif_matches.csv"),
          quote = FALSE)

# Also save as sparse format (indices only)
motif_sparse <- summary(motif_matches)
write.csv(motif_sparse, 
          file.path(output_dir, "motif_matches_sparse.csv"),
          row.names = FALSE,
          quote = FALSE)

cat(sprintf("   Motif matches: %d (%.2f%% density)\n",
            sum(motif_matches),
            100 * sum(motif_matches) / length(motif_matches)))
cat(sprintf("   Exported motif matches to %s\n", 
            file.path(output_dir, "motif_matches.csv")))

# ============================================================================
# 7. Compute Background Peaks
# ============================================================================

cat("\n7. Computing background peaks...\n")
bg_peaks <- getBackgroundPeaks(object = counts_filtered,
                               niterations = 50)

# Export background peaks
write.csv(bg_peaks, 
          file.path(output_dir, "background_peaks.csv"),
          quote = FALSE)
cat(sprintf("   Exported background peaks to %s\n", 
            file.path(output_dir, "background_peaks.csv")))

# ============================================================================
# 8. Compute Deviations
# ============================================================================

cat("\n8. Computing chromVAR deviations...\n")
dev <- computeDeviations(object = counts_filtered, 
                         annotations = motif_ix,
                         background_peaks = bg_peaks)

# Extract deviation matrices
deviations_mat <- deviations(dev)
deviation_scores_mat <- deviationScores(dev)

cat(sprintf("   Deviation matrix shape: %d cells x %d motifs\n",
            nrow(deviations_mat), ncol(deviations_mat)))
cat(sprintf("   Deviation range: [%.3f, %.3f]\n",
            min(deviations_mat), max(deviations_mat)))
cat(sprintf("   Deviation scores range: [%.3f, %.3f]\n",
            min(deviation_scores_mat), max(deviation_scores_mat)))

# ============================================================================
# 9. Export Results
# ============================================================================

cat("\n9. Exporting chromVAR results...\n")

# Export deviations (bias-corrected)
write.csv(deviations_mat, 
          file.path(output_dir, "deviations.csv"),
          quote = FALSE)
cat(sprintf("   Exported deviations to %s\n", 
            file.path(output_dir, "deviations.csv")))

# Export deviation z-scores
write.csv(deviation_scores_mat, 
          file.path(output_dir, "deviation_scores.csv"),
          quote = FALSE)
cat(sprintf("   Exported deviation scores to %s\n", 
            file.path(output_dir, "deviation_scores.csv")))

# Export cell metadata
cell_metadata <- as.data.frame(colData(dev))
write.csv(cell_metadata, 
          file.path(output_dir, "cell_metadata.csv"),
          quote = FALSE)

# Export summary statistics
summary_stats <- data.frame(
  n_peaks = nrow(counts_filtered),
  n_cells = ncol(counts_filtered),
  n_motifs = ncol(deviations_mat),
  total_motif_matches = sum(motif_matches),
  deviation_mean = mean(deviations_mat),
  deviation_sd = sd(deviations_mat),
  deviation_score_mean = mean(deviation_scores_mat),
  deviation_score_sd = sd(deviation_scores_mat)
)
write.csv(summary_stats, 
          file.path(output_dir, "summary_stats.csv"),
          row.names = FALSE,
          quote = FALSE)

# ============================================================================
# 10. Compute Variability (optional)
# ============================================================================

cat("\n10. Computing variability...\n")
variability <- computeVariability(dev)

write.csv(variability, 
          file.path(output_dir, "variability.csv"),
          quote = FALSE)
cat(sprintf("   Exported variability to %s\n", 
            file.path(output_dir, "variability.csv")))

# ============================================================================
# 11. Save R objects for later use
# ============================================================================

cat("\n11. Saving R objects...\n")
saveRDS(counts_filtered, file.path(output_dir, "counts_filtered.rds"))
saveRDS(motif_ix, file.path(output_dir, "motif_ix.rds"))
saveRDS(dev, file.path(output_dir, "chromvar_deviations.rds"))

cat("\n=== chromVAR R workflow completed! ===\n")
cat(sprintf("\nAll results saved to: %s/\n", output_dir))
cat("\nKey output files:\n")
cat("  - deviations.csv: Bias-corrected deviation values\n")
cat("  - deviation_scores.csv: Z-scored deviations\n")
cat("  - motif_matches.csv: Binary motif-peak matching matrix\n")
cat("  - background_peaks.csv: Background peak indices\n")
cat("  - summary_stats.csv: Summary statistics\n")
