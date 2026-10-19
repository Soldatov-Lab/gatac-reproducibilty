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

The workspace has three pixi environments:

| Env | Command | Purpose | Python / key deps |
|---|---|---|---|
| `default` | `pixi run python ...` | GATAC, SnapATAC2, chromVAR, ArchR, full pipeline | Python 3.13, numpy 2.x |
| `amulet` | `pixi run --environment amulet python ...` | Original AMULET v1.1 tool only (pinned for compatibility) | Python 3.11, numpy<1.24, pandas<2.0 |
| `tobias` | `pixi run --environment tobias python ...` | Original TOBIAS 0.17.5, for the footprinting test | Python 3.11, tobias==0.17.5 |

The `amulet` and `tobias` envs are auto-installed on first use, so `pixi install` is enough if you only plan to run the default-env tests.

## Running Tests

Once setup is complete, run tests in the following order:

`fragment_loading.py` writes the fragment Parquet files the other tests
read (`data/atac_pbmc_5k.parquet`, `data/atac_pbmc_5k_filtered.parquet`).
GATAC rejects Parquet files written before its current fragment format, so
after updating GATAC from an older version, run it again first; the
gene-score and AMULET tests convert their own cached Parquet again by
themselves.


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
pixi run python test/amulet_doublet.py          # uses both envs (GATAC + original AMULET; AMULET output cached)
pixi run python test/gene_score.py              # GATAC vs ArchR addGeneScoreMatrix (runs R oracle on first use)
pixi run python test/lsi.py                     # GATAC vs ArchR .computeLSI + variable features (runs R oracle on first use)
pixi run python test/iterative_lsi.py           # GATAC vs ArchR addIterativeLSI end to end (builds an arrow file: slow, cached)
pixi run python test/footprinting.py            # GATAC vs TOBIAS ATACorrect/ScoreBigwig/BINDetect (runs TOBIAS on first use: ~1 h, cached)
```

The `footprinting` test compares `tobias.footprint` + `tobias.bindetect` with TOBIAS
0.17.5 on three PBMC 5k pseudobulks (CD14 Mono, CD4 Memory, NK) on chr19 with
the cisBP motifs. `test/footprinting_tobias.py` writes one BAM per pseudobulk
from the same fragments parquet and runs TOBIAS in the `tobias` env. The
outputs go to `data/footprinting_tobias/`; `--rerun-tobias` regenerates them.
Each stage is first checked in isolation: given TOBIAS's own input (its fp64
bias scores, its corrected track, its footprints), GATAC must match TOBIAS
exactly. End to end, the DWM bias scores differ in their last bits (summation
order), so a second TOBIAS run with those scores perturbed by ±1e-13 sets the
tolerance. GATAC's deviation from TOBIAS must stay within twice that control's.

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
is 0.45 on this 6,566-cell fixture), so ArchR's self-agreement, not a fixed
threshold, is what makes GATAC's number interpretable. The remaining
difference is the clustering step (cuGraph Leiden vs Seurat Louvain, which
also restarts `n.start=10` times).

The `amulet_doublet` test downloads the canonical 10x Genomics PBMC 5k
fragment file via `snap.datasets.pbmc5k()` (cached at
`~/.cache/snapatac2/atac_pbmc_5k.tsv.gz`), then compares GATAC against
the original AMULET v1.1 release. AMULET v1.1 takes about 5 minutes, so its
results and timings are cached in `data/amulet_v11_output/`; `--rerun-amulet`
runs it again. To run GATAC only (skip the AMULET comparison):

```bash
pixi run python test/amulet_doublet.py --run-gatac-only
```

## Results Summary

One run of every test on an NVIDIA A100 80GB with 6 CPU cores (2026-10-19,
GATAC `5d313ee`).

| Test | Speedup | Result | Notes |
|------|---------|--------|-------|
| Fragment Loading | x3.2 | ✅ Full Match | 14,225 cells in both; GATAC `make_parquet` + `filter_fragments` 66.2s vs SnapATAC2 `import_fragments` 209.1s |
| TSS Enrichment | x1.9 | ✅ Correlation: 1.000 | Cell count match |
| Tile Matrix | x5.2 | ✅ Full Match | Sum=123,061,807 for both tools |
| Feature Selection | x1.7 | ⚠️ Overlap: 99.8% | 80 features differ by tie-breaking at the boundary count |
| Peak Calling | x2.9 | ⚠️ Jaccard: 0.963 | Different algorithms (gmacs vs MACS2); recall SnapATAC2 97.3%, GATAC 98.9% |
| Peak Matrix | x4.4 | ✅ Full Match | 40,876,001 nnz in both; peak/cell correlation 1.0 |
| Motif Enrichment | x8.5 | ⚠️ Avg Corr: 0.985 | Minor numerical differences in p-value calculation |
| GSEA | x26.1 | ✅ ES Match | GPU vs GSEApy, 300 sets; 100% sign agreement |
| ChromVAR Deviations | x9.5 | ✅ Correlation: 0.975 | vs R chromVAR `computeDeviations`; 36 cells × 28,596 peaks × 386 motifs |
| AMULET Doublet Detection | x62.7 | ✅ Full Match | vs AMULET v1.1: Jaccard 1.000, q-value r 1.000 on 13,735 cells |
| Gene Score (ArchR) | x21.0 | ✅ Entry-wise corr: 1.000 | vs `addGeneScoreMatrix`; 643 cells × 19,933 genes |
| LSI (ArchR) | x25.7–x40.9 | ✅ Per-component \|r\|: 1.000 | vs `.computeLSI`, all three `LSIMethod` variants; 643 cells × 290,170 tiles |
| LSI Variable Features (ArchR) | x21.1 | ✅ Full Match | vs `.identifyVarFeatures` scoring: identical feature sets |
| Iterative LSI (ArchR) | x140.9 | ✅ Within ArchR's seed spread | vs `addIterativeLSI`: Jaccard 0.47, subspace corr 0.74; ArchR seed to seed 0.45 / 0.73 |
| Footprinting (TOBIAS) | x71.8 | ✅ Stage-wise Full Match | vs TOBIAS 0.17.5 (6 cores); 3 pseudobulks × chr19 × 3,935 motifs; end to end within TOBIAS's own noise |
