#!/usr/bin/env Rscript
# ArchR reference (oracle) for GATAC's LSI port.
#
# Two modes, both driven by a matrix the Python side hands over as raw CSC
# arrays. A cells x features CSR and a features x cells CSC share their index
# arrays byte for byte, so nothing is transposed or reformatted between the two
# tools and every difference is algorithmic.
#
#   MODE=computelsi   ArchR:::.computeLSI for LSIMethod 1, 2 and 3.
#                     outlierQuantiles = NULL disables ArchR's depth-tail
#                     hold-out and its .projectLSI step, so the comparison
#                     isolates TF-IDF + SVD. Zero-sum features are pre-dropped
#                     by the Python side, as .computeLSI would drop them.
#
#   MODE=varfeat      The variable-feature scoring of
#                     ArchR:::.identifyVarFeatures. That function reads its
#                     group matrix from arrow files via .getGroupMatrix and so
#                     cannot be handed an in-memory matrix; the three lines that
#                     do the scoring are reproduced verbatim below, run in R
#                     with the same Matrix and matrixStats the package uses.
#                     Its groupMat is FEATURES x CLUSTERS, so colSums are
#                     per-cluster totals and rowVars are per-feature variances.
#
# Inputs (under $OUTDIR):
#   shape.csv     n_feat, n_cell, nnz, k, n_var_features, scale_to
#   indices.i32   CSC row indices (int32, 0-based)
#   indptr.i32    CSC column pointers (int32)
#   clusters.i32  per-cell cluster label (int32; varfeat mode only)
#
# Outputs (under $OUTDIR):
#   archr_lsi_<m>.csv / archr_sv_<m>.csv / archr_time_<m>.csv   (computelsi)
#   archr_varfeat.csv / archr_var.csv                            (varfeat)
#
# Run:  OUTDIR=... MODE=computelsi Rscript test/lsi_R.R

suppressMessages({
  library(ArchR)
  library(Matrix)
  library(matrixStats)
})

outdir <- Sys.getenv("OUTDIR", "data/lsi_output")
mode   <- Sys.getenv("MODE", "computelsi")
stopifnot(dir.exists(outdir))

meta   <- read.csv(file.path(outdir, "shape.csv"))
n_feat <- meta$n_feat[1]
n_cell <- meta$n_cell[1]
nnz    <- meta$nnz[1]
k      <- meta$k[1]
scaleTo <- meta$scale_to[1]

i <- readBin(file.path(outdir, "indices.i32"), "integer", n = nnz, size = 4L)
p <- readBin(file.path(outdir, "indptr.i32"), "integer", n = n_cell + 1L, size = 4L)

# 0-based i and p go straight into the dgCMatrix slots. x = 1 because the tile
# matrix is binarized, which is also ArchR's default.
mat <- new("dgCMatrix", i = i, p = p, x = rep(1.0, nnz),
           Dim = c(as.integer(n_feat), as.integer(n_cell)),
           Dimnames = list(paste0("f", seq_len(n_feat)),
                           paste0("c", seq_len(n_cell))))
cat(sprintf("matrix: %d features x %d cells, nnz=%d\n",
            nrow(mat), ncol(mat), length(mat@i)))

if (mode == "computelsi") {
  for (m in c(1, 2, 3)) {
    cat(sprintf("--- .computeLSI(LSIMethod = %d) ---\n", m))
    t0 <- Sys.time()
    lsi <- ArchR:::.computeLSI(
      mat              = mat,
      LSIMethod        = m,
      scaleTo          = scaleTo,
      nDimensions      = k,
      binarize         = TRUE,
      outlierQuantiles = NULL,
      seed             = 1,
      verbose          = FALSE
    )
    dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("    %.2f s; matSVD %d x %d\n", dt,
                nrow(lsi$matSVD), ncol(lsi$matSVD)))
    write.csv(lsi$matSVD, file.path(outdir, sprintf("archr_lsi_%d.csv", m)),
              row.names = FALSE)
    write.csv(data.frame(d = lsi$svd$d),
              file.path(outdir, sprintf("archr_sv_%d.csv", m)), row.names = FALSE)
    write.csv(data.frame(seconds = dt),
              file.path(outdir, sprintf("archr_time_%d.csv", m)), row.names = FALSE)
  }
} else if (mode == "varfeat") {
  n_var <- meta$n_var_features[1]
  cl <- readBin(file.path(outdir, "clusters.i32"), "integer",
                n = n_cell, size = 4L)
  cat(sprintf("clusters: %d\n", length(unique(cl))))

  # pseudo-bulk, features x clusters -- the orientation ArchR scores in
  groups <- sort(unique(cl))
  groupMat <- vapply(groups,
                     function(g) Matrix::rowSums(mat[, cl == g, drop = FALSE]),
                     numeric(n_feat))

  # ArchR:::.identifyVarFeatures, selectionMethod = "var", verbatim
  groupMat <- log2(t(t(groupMat) / colSums(groupMat)) * scaleTo + 1)
  v <- matrixStats::rowVars(groupMat)
  idx <- sort(head(order(v, decreasing = TRUE), n_var))

  write.csv(data.frame(idx0 = idx - 1L),      # back to 0-based for Python
            file.path(outdir, "archr_varfeat.csv"), row.names = FALSE)
  write.csv(data.frame(var = v),
            file.path(outdir, "archr_var.csv"), row.names = FALSE)
  cat(sprintf("selected %d of %d features\n", length(idx), n_feat))
} else {
  stop(sprintf("Unknown MODE '%s'; expected 'computelsi' or 'varfeat'.", mode))
}

cat("oracle complete\n")
