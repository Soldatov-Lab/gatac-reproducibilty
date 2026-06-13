#!/usr/bin/env Rscript
# Gene Score Reproducibility - ArchR Reference (oracle) Script
#
# Produces the ground-truth ArchR `addGeneScoreMatrix` output that GATAC's
# future GPU-accelerated gene-score port is validated against.
#
# Design goals (so the comparison isolates the *algorithm*, not the inputs):
#   * No heavy genome dependencies. We build ArchR's geneAnnotation and
#     genomeAnnotation by hand from a cached gencode GFF3, and use the
#     special genome string "nullGenome" so ArchR never needs a BSgenome
#     (see ArchR ValidationUtils.R::.validGenomeAnnotation).
#   * Tn5 insertion offsets are DISABLED (offsetPlus = offsetMinus = 0) so
#     insertions are the raw fragment ends, matching GATAC's fragment
#     convention. This removes a +4/-5 bp confound.
#   * We export the exact gene coordinates AND the per-gene regulatory
#     regions/weights ArchR used, so GATAC can be driven by identical genes
#     instead of re-parsing a GTF (which could differ).
#
# Outputs (all under $GENE_SCORE_OUTDIR, default data/gene_score_output/):
#   gene_score.mtx        sparse genes x cells normalized gene-score matrix
#   genes.csv             gene-score features: idx, name, seqnames, start, end, strand
#   cells.csv             cell barcodes (matrix column order)
#   gene_annotation.csv   gene-body coords used as input (drives GATAC)
#   gene_regions.csv      per-gene regulatory window + geneWeight (ArchR internals)
#   gene_regions.rds      same, as a GRanges (saveGeneRegions)
#   params.csv            the gene-score parameters used
#   fragments_archr.tsv.gz  the coordinate-sorted, bgzipped, tabix-indexed
#                           fragment file ArchR ingested (GATAC reads the same)
#
# Run with:
#   pixi run Rscript test/gene_score_R.R
# (invoked automatically by test/gene_score.py when outputs are missing)

suppressMessages({
  library(ArchR)
  library(GenomicRanges)
  library(rtracklayer)
  library(data.table)
  library(Matrix)
  library(Rsamtools)
})

set.seed(1)
addArchRThreads(threads = max(1L, parallel::detectCores() - 1L))

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
home      <- Sys.getenv("HOME")
fragRaw   <- Sys.getenv("GENE_SCORE_FRAGMENTS",
                        file.path(home, ".cache/snapatac2/atac_pbmc_500_downsample.tsv.gz"))
gffPath   <- Sys.getenv("GENE_SCORE_GFF",
                        file.path(home, ".cache/snapatac2/gencode_v41_GRCh38.gff3.gz"))
outDir    <- Sys.getenv("GENE_SCORE_OUTDIR", "data/gene_score_output")
workDir   <- file.path(outDir, "archr_work")     # ArrowFiles / project scratch

# Gene-score parameters (ArchR addGeneScoreMatrix defaults, made explicit)
tileSize        <- 500L
geneModel       <- "exp(-abs(x)/5000) + exp(-1)"
extendUpstream  <- c(1000L, 100000L)
extendDownstream<- c(1000L, 100000L)
geneUpstream    <- 5000L
geneDownstream  <- 0L
useGeneBoundaries <- TRUE
useTSS          <- FALSE
ceiling         <- 4L
geneScaleFactor <- 5L
scaleTo         <- 10000
excludeChr      <- c("chrY", "chrM")

dir.create(outDir, showWarnings = FALSE, recursive = TRUE)
dir.create(workDir, showWarnings = FALSE, recursive = TRUE)

cat("=== Gene Score ArchR Oracle ===\n")
cat(sprintf("ArchR %s\n", as.character(packageVersion("ArchR"))))
cat(sprintf("Fragments: %s\n", fragRaw))
cat(sprintf("GFF3:      %s\n", gffPath))
cat(sprintf("Output:    %s\n\n", outDir))

stopifnot(file.exists(fragRaw), file.exists(gffPath))

standardChroms <- paste0("chr", c(1:22, "X"))

# ---------------------------------------------------------------------------
# 1. Build gene / exon / TSS annotation from the gencode GFF3
# ---------------------------------------------------------------------------
cat("1. Loading gene annotation from GFF3...\n")
gff <- rtracklayer::import(gffPath)
seqlevelsStyle(gff) <- "UCSC"

genes <- gff[gff$type == "gene"]
if (!is.null(genes$gene_type)) {
  genes <- genes[genes$gene_type == "protein_coding"]
}
genes <- genes[as.character(seqnames(genes)) %in% standardChroms]
genes$symbol <- genes$gene_name
genes <- genes[!is.na(genes$symbol)]
# One range per gene symbol (ArchR expects unique symbols)
genes <- genes[!duplicated(genes$symbol)]
mcols(genes) <- DataFrame(gene_id = genes$gene_id, symbol = genes$symbol)
genes <- sort(sortSeqlevels(genes), ignore.strand = TRUE)
cat(sprintf("   %d protein-coding genes\n", length(genes)))

exons <- gff[gff$type == "exon"]
exons <- exons[as.character(seqnames(exons)) %in% standardChroms]
exons$symbol <- exons$gene_name
exons <- exons[exons$symbol %in% genes$symbol]
mcols(exons) <- DataFrame(symbol = exons$symbol)

tss <- resize(genes, width = 1, fix = "start")

geneAnnotation <- createGeneAnnotation(genes = genes, exons = exons, TSS = tss)

# ---------------------------------------------------------------------------
# 2. Build a genome annotation WITHOUT a BSgenome ("nullGenome")
# ---------------------------------------------------------------------------
cat("2. Building chromSizes / genomeAnnotation (nullGenome)...\n")
chromMax <- tapply(end(genes), as.character(seqnames(genes)), max)
# pad past the last gene so tile/extension windows are not clipped
chromSizes <- GRanges(
  seqnames = names(chromMax),
  ranges   = IRanges(start = 1, end = as.integer(chromMax) + 1e6L)
)
seqlengths(chromSizes) <- end(chromSizes)
genomeAnnotation <- SimpleList(
  genome     = "nullGenome",
  chromSizes = chromSizes,
  blacklist  = GRanges()
)

# ---------------------------------------------------------------------------
# 3. Prepare a coordinate-sorted, bgzipped, tabix-indexed fragment file
# ---------------------------------------------------------------------------
cat("3. Preparing fragment file (sort + bgzip + tabix)...\n")
fragOut <- file.path(outDir, "fragments_archr.tsv.gz")
if (!file.exists(fragOut) || !file.exists(paste0(fragOut, ".tbi"))) {
  frag <- data.table::fread(cmd = paste("zcat", shQuote(fragRaw)), header = FALSE,
                            col.names = c("chr", "start", "end", "barcode", "count"))
  frag <- frag[chr %in% standardChroms]
  data.table::setorder(frag, chr, start)
  tmpTsv <- file.path(outDir, "fragments_archr.tsv")
  data.table::fwrite(frag, tmpTsv, sep = "\t", col.names = FALSE)
  if (file.exists(fragOut)) file.remove(fragOut)
  bgz <- Rsamtools::bgzip(tmpTsv, dest = fragOut, overwrite = TRUE)
  file.remove(tmpTsv)
  Rsamtools::indexTabix(bgz, format = "bed")
  rm(frag); gc()
}
cat(sprintf("   %s\n", fragOut))

# ---------------------------------------------------------------------------
# 4. Create Arrow file + ArchRProject
# ---------------------------------------------------------------------------
cat("4. Creating Arrow file...\n")
oldwd <- getwd()
setwd(workDir)
on.exit(setwd(oldwd), add = TRUE)

ArrowFiles <- createArrowFiles(
  inputFiles      = file.path(oldwd, fragOut),
  sampleNames     = "pbmc",
  geneAnnotation  = geneAnnotation,
  genomeAnnotation= genomeAnnotation,
  minTSS          = 0,
  minFrags        = 100,
  filterTSS       = 0,
  filterFrags     = 100,
  offsetPlus      = 0,   # disable Tn5 shift to match GATAC fragment convention
  offsetMinus     = 0,
  addTileMat      = FALSE,
  addGeneScoreMat = FALSE,
  force           = TRUE,
  subThreading    = FALSE
)
stopifnot(length(ArrowFiles) == 1, file.exists(ArrowFiles))

proj <- ArchRProject(
  ArrowFiles       = ArrowFiles,
  geneAnnotation   = geneAnnotation,
  genomeAnnotation = genomeAnnotation,
  outputDirectory  = "ArchRProject",
  copyArrows       = FALSE
)
cat(sprintf("   %d cells passed ArchR QC\n", nCells(proj)))

# ---------------------------------------------------------------------------
# 5. Run addGeneScoreMatrix (the oracle) and save the regulatory regions
# ---------------------------------------------------------------------------
cat("5. Computing ArchR gene scores...\n")
geneRegionsRds <- file.path(oldwd, outDir, "gene_regions.rds")
proj <- addGeneScoreMatrix(
  input             = proj,
  genes             = geneAnnotation$genes,
  geneModel         = geneModel,
  matrixName        = "GeneScoreMatrix",
  extendUpstream    = extendUpstream,
  extendDownstream  = extendDownstream,
  geneUpstream      = geneUpstream,
  geneDownstream    = geneDownstream,
  useGeneBoundaries = useGeneBoundaries,
  useTSS            = useTSS,
  tileSize          = tileSize,
  ceiling           = ceiling,
  geneScaleFactor   = geneScaleFactor,
  scaleTo           = scaleTo,
  excludeChr        = excludeChr,
  blacklist         = NULL,
  saveGeneRegions   = geneRegionsRds,
  force             = TRUE
)

se <- getMatrixFromProject(proj, useMatrix = "GeneScoreMatrix")
mat <- assay(se)                       # genes x cells, dgCMatrix
rd  <- as.data.frame(rowData(se))
cells <- colnames(se)
cat(sprintf("   matrix: %d genes x %d cells, nnz=%d\n", nrow(mat), ncol(mat), length(mat@x)))

setwd(oldwd)  # write outputs relative to the workspace root

# ---------------------------------------------------------------------------
# 6. Export everything for the Python harness
# ---------------------------------------------------------------------------
cat("6. Exporting oracle outputs...\n")
Matrix::writeMM(as(mat, "CsparseMatrix"), file.path(outDir, "gene_score.mtx"))

# Feature table (matrix row order)
genesDF <- data.frame(
  idx      = rd$idx,
  name     = rd$name,
  seqnames = rd$seqnames,
  start    = rd$start,
  end      = rd$end,
  strand   = rd$strand,
  stringsAsFactors = FALSE
)
data.table::fwrite(genesDF, file.path(outDir, "genes.csv"))
data.table::fwrite(data.table::data.table(barcode = cells), file.path(outDir, "cells.csv"))

# Gene-body coords used as input -> drives GATAC's gene set exactly
annoDF <- data.frame(
  symbol   = genes$symbol,
  seqnames = as.character(seqnames(genes)),
  start    = start(genes),
  end      = end(genes),
  strand   = as.character(strand(genes)),
  stringsAsFactors = FALSE
)
data.table::fwrite(annoDF, file.path(outDir, "gene_annotation.csv"))

# Per-gene regulatory regions + weights (ArchR internals, for exact reproduction)
gr <- readRDS(geneRegionsRds)
regDF <- data.frame(
  symbol     = mcols(gr)$symbol,
  seqnames   = as.character(seqnames(gr)),
  start      = start(gr),
  end        = end(gr),
  strand     = as.character(strand(gr)),
  geneStart  = mcols(gr)$geneStart,
  geneEnd    = mcols(gr)$geneEnd,
  geneWeight = mcols(gr)$geneWeight,
  stringsAsFactors = FALSE
)
data.table::fwrite(regDF, file.path(outDir, "gene_regions.csv"))

paramsDF <- data.frame(
  param = c("tileSize", "geneModel", "extendUpstreamMin", "extendUpstreamMax",
            "extendDownstreamMin", "extendDownstreamMax", "geneUpstream",
            "geneDownstream", "useGeneBoundaries", "useTSS", "ceiling",
            "geneScaleFactor", "scaleTo", "excludeChr", "offsetPlus", "offsetMinus"),
  value = c(tileSize, geneModel, extendUpstream[1], extendUpstream[2],
            extendDownstream[1], extendDownstream[2], geneUpstream,
            geneDownstream, useGeneBoundaries, useTSS, ceiling,
            geneScaleFactor, scaleTo, paste(excludeChr, collapse = ","), 0, 0),
  stringsAsFactors = FALSE
)
data.table::fwrite(paramsDF, file.path(outDir, "params.csv"))

cat("\nDone. Oracle written to ", outDir, "\n", sep = "")
