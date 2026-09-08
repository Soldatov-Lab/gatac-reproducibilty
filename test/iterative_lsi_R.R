#!/usr/bin/env Rscript
# ArchR reference (oracle) for GATAC's iterative-LSI port, end to end.
#
# Unlike test/lsi_R.R, this one needs a real ArchRProject: addIterativeLSI
# reads its TileMatrix from arrow files, so there is no way to hand it an
# in-memory matrix. Arrow creation is the slow step (tens of minutes on the
# 5k-cell fragment file), which is why the outputs are cached and this test is
# separate from the cheap per-stage gates in test/lsi.py.
#
# Design, following test/gene_score_R.R:
#   * No BSgenome. The gene and genome annotations are built by hand from a
#     cached gencode GFF3, with ArchR's special "nullGenome".
#   * Tn5 offsets disabled (offsetPlus = offsetMinus = 0) so insertions are the
#     raw fragment ends, matching GATAC's fragment convention.
#   * The fragment file is the canonical 10x PBMC 5k one that the AMULET test
#     already downloads via snap.datasets.pbmc5k(); it arrives bgzipped, so it
#     only needs a tabix index.
#
# addIterativeLSI is run at TWO seeds. The seed-1 run is what GATAC is scored
# against; seed 2 vs seed 1 measures ArchR's own run-to-run variability, which
# is the context that makes the GATAC number interpretable -- addIterativeLSI is
# not a stable function of its input.
#
# The exported TileMatrix is restricted to the top `totalFeatures` rows by
# accessibility. That is the pool addIterativeLSI can ever select from, so the
# restriction loses nothing and keeps the handover to a manageable size.
#
# Environment:
#   OUTDIR          where to write (required)
#   FRAGMENTS       path to the bgzipped fragment file (required)
#   GENE_SCORE_GFF  cached gencode GFF3 (default ~/.cache/snapatac2/...)
#   NTHREADS, VAR_FEATURES, TOTAL_FEATURES, N_DIMS, ITERATIONS
#
# Run:  OUTDIR=... FRAGMENTS=... Rscript test/iterative_lsi_R.R

suppressMessages({
  library(ArchR)
  library(GenomicRanges)
  library(rtracklayer)
  library(Rsamtools)
  library(Matrix)
})

set.seed(1)
addArchRThreads(threads = as.integer(Sys.getenv("NTHREADS", "12")), force = TRUE)

outdir  <- Sys.getenv("OUTDIR", "data/iterative_lsi_output")
fragIn  <- Sys.getenv("FRAGMENTS", "")
gffPath <- Sys.getenv("GENE_SCORE_GFF",
                      file.path(Sys.getenv("HOME"),
                                ".cache/snapatac2/gencode_v41_GRCh38.gff3.gz"))
VAR_FEATURES   <- as.integer(Sys.getenv("VAR_FEATURES", "25000"))
TOTAL_FEATURES <- as.integer(Sys.getenv("TOTAL_FEATURES", "500000"))
N_DIMS         <- as.integer(Sys.getenv("N_DIMS", "30"))
ITERATIONS     <- as.integer(Sys.getenv("ITERATIONS", "2"))

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
stopifnot(nzchar(fragIn), file.exists(fragIn), file.exists(gffPath))

# Resolve to absolute paths *before* any setwd(). Joining an absolute path onto
# the old working directory silently produces a doubled path that only fails at
# the first write -- after all the compute is done.
outAbs  <- normalizePath(outdir, mustWork = TRUE)
fragAbs <- normalizePath(fragIn, mustWork = TRUE)

# ---- tabix index ----------------------------------------------------------
if (!file.exists(paste0(fragAbs, ".tbi"))) {
  cat("indexing fragments with tabix ...\n")
  # The 10x fragment file is already bgzipped, so it only needs an index. If it
  # is not indexable in place (read-only cache), copy it into OUTDIR first.
  ok <- tryCatch({
    Rsamtools::indexTabix(fragAbs, format = "bed"); TRUE
  }, error = function(e) FALSE)
  if (!ok) {
    local_frag <- file.path(outAbs, basename(fragAbs))
    if (!file.exists(local_frag)) file.copy(fragAbs, local_frag)
    Rsamtools::indexTabix(local_frag, format = "bed")
    fragAbs <- local_frag
  }
}
cat(sprintf("fragments: %s\n", fragAbs))

# ---- annotations (nullGenome) ---------------------------------------------
cat("building annotations ...\n")
gff <- rtracklayer::import(gffPath)
seqlevelsStyle(gff) <- "UCSC"
standardChroms <- paste0("chr", c(1:22, "X"))
genes <- gff[gff$type == "gene"]
genes <- genes[as.character(seqnames(genes)) %in% standardChroms]
genes$symbol <- genes$gene_name
mcols(genes) <- DataFrame(gene_id = genes$gene_id, symbol = genes$symbol)
exons <- gff[gff$type == "exon"]
exons <- exons[as.character(seqnames(exons)) %in% standardChroms]
exons$symbol <- exons$gene_name
exons <- exons[exons$symbol %in% genes$symbol]
mcols(exons) <- DataFrame(symbol = exons$symbol)
tss <- resize(genes, width = 1, fix = "start")
geneAnnotation <- createGeneAnnotation(genes = genes, exons = exons, TSS = tss)

chromMax <- tapply(end(genes), as.character(seqnames(genes)), max)
chromSizes <- GRanges(seqnames = names(chromMax),
                      ranges = IRanges(start = 1, end = as.integer(chromMax) + 1e6L))
seqlengths(chromSizes) <- end(chromSizes)
genomeAnnotation <- SimpleList(genome = "nullGenome",
                               chromSizes = chromSizes, blacklist = GRanges())

# ---- arrow with a TileMatrix ---------------------------------------------
work <- file.path(outAbs, "archr_work")
dir.create(work, showWarnings = FALSE, recursive = TRUE)
oldwd <- getwd(); setwd(work); on.exit(setwd(oldwd), add = TRUE)

arrowPath <- "pbmc5k.arrow"
if (!file.exists(arrowPath)) {
  cat("creating arrow file (the slow step; tens of minutes) ...\n")
  createArrowFiles(
    inputFiles       = fragAbs,
    sampleNames      = "pbmc5k",
    geneAnnotation   = geneAnnotation,
    genomeAnnotation = genomeAnnotation,
    minTSS = 0, minFrags = 500, filterTSS = 0, filterFrags = 500,
    offsetPlus = 0, offsetMinus = 0,
    addTileMat = TRUE, addGeneScoreMat = FALSE,
    TileMatParams = list(tileSize = 500, binarize = TRUE),
    force = TRUE, subThreading = FALSE
  )
}
stopifnot(file.exists(arrowPath))

proj <- ArchRProject(ArrowFiles = arrowPath, outputDirectory = "proj",
                     geneAnnotation = geneAnnotation,
                     genomeAnnotation = genomeAnnotation,
                     copyArrows = FALSE, showLogo = FALSE)
cat(sprintf("project: %d cells; matrices: %s\n", nCells(proj),
            paste(getAvailableMatrices(proj), collapse = ", ")))
stopifnot("TileMatrix" %in% getAvailableMatrices(proj))

# ---- addIterativeLSI at two seeds ----------------------------------------
for (sd in c(1, 2)) {
  cat(sprintf("addIterativeLSI seed=%d ...\n", sd))
  nm <- paste0("IterLSI", sd)
  t0 <- Sys.time()
  proj <- addIterativeLSI(
    ArchRProj = proj, useMatrix = "TileMatrix", name = nm,
    iterations = ITERATIONS, varFeatures = VAR_FEATURES,
    totalFeatures = TOTAL_FEATURES, dimsToUse = seq_len(N_DIMS),
    clusterParams = list(resolution = 2, sampleCells = 10000,
                         maxClusters = 6, n.start = 10),
    filterQuantile = 0.995, binarize = TRUE, LSIMethod = 2, scaleTo = 10^4,
    seed = sd, force = TRUE, verbose = FALSE
  )
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  rd <- proj@reducedDims[[nm]]
  write.csv(rd$matSVD,
            file.path(outAbs, sprintf("archr_iterlsi_seed%d_matSVD.csv", sd)))
  write.csv(as.data.frame(rd$LSIFeatures),
            file.path(outAbs, sprintf("archr_iterlsi_seed%d_features.csv", sd)),
            row.names = FALSE)
  write.csv(data.frame(seconds = dt),
            file.path(outAbs, sprintf("archr_iterlsi_seed%d_time.csv", sd)),
            row.names = FALSE)
  cat(sprintf("  seed %d: %.1f s, matSVD %d x %d, %d features\n", sd, dt,
              nrow(rd$matSVD), ncol(rd$matSVD),
              nrow(as.data.frame(rd$LSIFeatures))))
}

# ---- export the TileMatrix (accessibility pool only) ---------------------
cat("exporting TileMatrix ...\n")
se <- getMatrixFromProject(proj, useMatrix = "TileMatrix", binarize = TRUE,
                           verbose = FALSE)
mat <- as(assay(se), "dgCMatrix")
rd  <- as.data.frame(rowData(se))
rs  <- Matrix::rowSums(mat)
keep <- sort(head(order(rs, decreasing = TRUE), TOTAL_FEATURES))
mat <- mat[keep, ]
rd  <- rd[keep, ]
cat(sprintf("TileMatrix (top %d by accessibility): %d x %d, nnz=%d\n",
            TOTAL_FEATURES, nrow(mat), ncol(mat), length(mat@i)))

writeBin(as.integer(mat@i), file.path(outAbs, "tile_i.i32"), size = 4L)
writeBin(as.integer(mat@p), file.path(outAbs, "tile_p.i32"), size = 4L)
write.csv(data.frame(n_feat = nrow(mat), n_cell = ncol(mat), nnz = length(mat@i)),
          file.path(outAbs, "tile_shape.csv"), row.names = FALSE)
write.csv(rd, file.path(outAbs, "tile_features.csv"), row.names = FALSE)
write.csv(data.frame(cell = colnames(mat),
                     nFrags = proj$nFrags[match(colnames(mat), proj$cellNames)]),
          file.path(outAbs, "cells.csv"), row.names = FALSE)
write.csv(data.frame(var_features = VAR_FEATURES, total_features = TOTAL_FEATURES,
                     n_dims = N_DIMS, iterations = ITERATIONS, lsi_method = 2,
                     scale_to = 1e4, filter_quantile = 0.995,
                     resolution = 2, max_clusters = 6),
          file.path(outAbs, "archr_iterlsi_params.csv"), row.names = FALSE)
cat("oracle complete\n")
