# Reproducibility

Full test suite for comparing gatac with existing tools, mainly snapatac2, but also macs3 and chromvar.

## Setup as a Submodule (within gatac repo)

Working within the main `gatac` repository:

```bash
cd reproducibility
pixi install --all
# or, to also fetch the AMULET v1.1 scripts (only needed for the
# amulet_doublet test):
pixi run install-all
```

The `pixi.toml` automatically references the GATAC installation at the parent directory level.

The workspace has two pixi environments:

| Env | Command | Purpose | Python / key deps |
|---|---|---|---|
| `default` | `pixi run python ...` | GATAC, SnapATAC2, chromVAR, ArchR, full pipeline | Python 3.13, numpy 2.x |
| `amulet` | `pixi run --environment amulet python ...` | Original AMULET v1.1 tool only (pinned for compatibility) | Python 3.11, numpy<1.24, pandas<2.0 |

The `amulet` env is auto-installed on first use, so `pixi install` is enough if you only plan to run the default-env tests.

## Running Tests

Once setup is complete, run tests in the following order:

```bash
pixi run python test/fragment_loading.py
pixi run python test/tss_enrichment.py
pixi run python test/tile_matrix.py
pixi run python test/feature_selection.py
pixi run python test/peak_calling.py
pixi run python test/make_peak_matrix.py
pixi run python test/motif_enrichment.py
pixi run python test/gsea_motif_enrichment.py
pixi run python test/chromvar_vignette.py
pixi run python test/amulet_doublet.py          # uses both envs (GATAC + original AMULET)
pixi run python test/gene_score.py              # GATAC vs ArchR addGeneScoreMatrix (runs R oracle on first use)
pixi run python test/lsi.py                     # GATAC vs ArchR .computeLSI + variable features (runs R oracle on first use)
pixi run python test/iterative_lsi.py           # GATAC vs ArchR addIterativeLSI end to end (builds an arrow file: slow, cached)
```

The `gene_score` test generates an ArchR ground-truth gene-score matrix via
`test/gene_score_R.R` (no BSgenome needed — it builds the annotation from a
cached gencode GFF3 and uses ArchR's `nullGenome`), caches it under
`data/gene_score_output/`, then compares GATAC's `make_gene_score_matrix`
against it. Regenerate the oracle with `--regenerate`; build the oracle only
(skip GATAC) with `--skip-gatac`.

The `lsi` test builds a binarized tile matrix from the fragment parquet the
gene-score test vendors (so run that one first), then hands the *same* matrix
to ArchR via `test/lsi_R.R` in two modes: `ArchR:::.computeLSI` for all three
`LSIMethod` variants, and the variable-feature scoring of
`ArchR:::.identifyVarFeatures`. The handover is raw CSC index arrays — a
cells x features CSR and a features x cells CSC are byte-identical — so no
transpose or reformatting sits between the two tools and every difference is
algorithmic. Neither gate needs an ArchR arrow file or an ArchRProject, which
keeps this test cheap enough to run on every change. `outlierQuantiles = NULL`
on the ArchR side disables its depth-tail hold-out so the comparison isolates
TF-IDF + SVD. Same flags as the gene-score test: `--regenerate`, `--skip-gatac`.

The `iterative_lsi` test is the end-to-end counterpart: `addIterativeLSI`
reads its TileMatrix from arrow files, so unlike `lsi` it needs a real
`ArchRProject`. Its oracle builds an arrow file from the same PBMC 5k
fragments the AMULET test downloads and runs `addIterativeLSI` at **two**
seeds — arrow creation takes tens of minutes, so everything is cached and
`--regenerate` forces a rebuild. The second seed is the point: `addIterativeLSI`
is not a stable function of its input (its own seed-to-seed feature Jaccard
measured 0.93 on this ~4.4k-cell fixture but 0.12 on a 643-cell one), so ArchR's
self-agreement, not a fixed threshold, is what makes GATAC's number
interpretable.

The `amulet_doublet` test downloads the canonical 10x Genomics PBMC 5k
fragment file via `snap.datasets.pbmc5k()` (cached at
`~/.cache/snapatac2/atac_pbmc_5k.tsv.gz`), then compares GATAC against
the original AMULET v1.1 release. To run GATAC only (skip the AMULET
comparison):

```bash
pixi run python test/amulet_doublet.py --run-gatac-only
```

## Results Summary

| Test | Speedup | Result | Notes |
|------|---------|--------|-------|
| Fragment Loading | x3.5 | ✅ Full Match | Identical cell barcodes and fragment counts |
| TSS Enrichment | x1.2 | ✅ Correlation: 1.000 | Cell count match, perfect TSSe correlation |
| Tile Matrix | x4.5 | ✅ Full Match | Sum=123,061,807 for both tools |
| Feature Selection | x1.1 | ⚠️ Overlap: 99.8% | 80 features differ due to tie-breaking at boundary count (272 features at count=204) |
| Peak Calling | x4.1 | ⚠️ Jaccard: 0.963 | Different algorithms (gmacs vs MACS2); Recall snap=97.3%, Recall gatac=98.9% |
| Peak Matrix | x4.8 | ✅ Full Match | Shape match, Peak/Cell correlation=1.0 |
| Motif Enrichment | x7.5 | ⚠️ Avg Corr: 0.984 | Minor numerical differences in p-value calculation |
| GSEA | x11.2 | ✅ ES Match | GPU implementation vs GSEApy; 100% sign agreement |
| ChromVAR Deviations | x10.0 | ✅ Correlation: 0.975 | GATAC vs R chromVAR `computeDeviations`; 36 cells × 28,596 peaks × 386 motifs |
| AMULET Doublet Detection | x3.5 | ✅ Full Match | GATAC vs original AMULET v1.1: Jaccard 1.000, q-value Pearson r 1.000 on 13,735 cells × 22 autosomes |
| Gene Score (ArchR) | x28.4 | ✅ Entry-wise corr: 1.000 | GATAC `make_gene_score_matrix` (3.2s) vs ArchR `addGeneScoreMatrix` (90.3s); per-cell 0.99989, per-gene 0.99972, entry-wise 0.99992 on 643 cells × 19,933 genes |
| LSI (ArchR) | x20.5–x31.3 | ✅ Per-component \|r\|: 1.000 | GATAC `tl.lsi` (0.13–0.16s) vs ArchR `.computeLSI` (2.6–4.4s) for all three `LSIMethod` variants; every one of 30 components matches, max relative singular-value error 1.8e-07, on 643 cells × 290,170 tiles |
| LSI Variable Features (ArchR) | — | ✅ Full Match | GATAC `tl.cluster_var_features` (0.03s) vs ArchR `.identifyVarFeatures` scoring given the same clusters: identical feature sets (Jaccard 1.000000), variance agreeing to 1.3e-15 |
| Iterative LSI (ArchR) | — | ✅ Jaccard 0.893, subspace corr 0.972 | GATAC `tl.iterative_lsi` (74s) vs ArchR `addIterativeLSI` on 4,437 cells × 500,000 tiles. ArchR's *own* seed-to-seed agreement on the same data is 0.927 / 0.969, so GATAC lands within 4% on features and matches on the embedding. Remaining difference is the clustering step (cuGraph Leiden vs Seurat Louvain, ARI ~0.95 on a shared embedding) and Seurat's `n.start=10` restarts, which cuGraph has no equivalent for; `filterBias` was ruled out (byte-identical output with it disabled) |
