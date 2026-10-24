# Reproducibility

Full test suite for comparing gatac with existing tools, mainly snapatac2, but also macs3 and chromvar.

## Results Summary

One run of every test on an NVIDIA A100 80GB with 6 CPU cores (2026-10-19,
GATAC `5d313ee`).

| Test | Speedup | Result | Notes |
|------|---------|--------|-------|
| Fragment Loading | x3.2 | ✅ Full Match | 14,225 cells in both; GATAC `make_parquet` + `filter_fragments` 66.2s vs SnapATAC2 `import_fragments` 209.1s |
| TSS Enrichment | x1.9 | ✅ Correlation: 1.000 | Cell count match |
| Tile Matrix | x5.2 | ✅ Full Match | Sum=123,061,807 for both tools |
| Feature Selection | x1.7 | ⚠️ Overlap: 99.8% | 80 features differ by tie-breaking at the boundary count |
| Peak Calling | x24.4 | ✅ Full Match | vs `snap.tl.macs3` (MACS3 3.0.4): all 234,758 summits of 3 groups identical (positions, scores, fold enrichment, p/q); merged peaks identical (119,141); L40S |
| Peak Matrix | x4.3 | ✅ Full Match | 42,932,832 nnz in both on the 119,141 merged peaks; L40S |
| Motif Enrichment | x11.2 | ✅ Full Match | vs `snap.tl.motif_enrichment`: same matched regions, so fold changes and p-values of 3 groups × 1,165 cisBP motifs agree to rounding; L40S |
| GSEA | x26.1 | ✅ ES Match | GPU vs GSEApy, 300 sets; 100% sign agreement |
| ChromVAR Deviations | x9.5 | ✅ Correlation: 0.975 | vs R chromVAR `computeDeviations`; 36 cells × 28,596 peaks × 386 motifs |
| AMULET Doublet Detection | x62.7 | ✅ Full Match | vs AMULET v1.1: Jaccard 1.000, q-value r 1.000 on 13,735 cells |
| Gene Score (ArchR) | x21.0 | ✅ Entry-wise corr: 1.000 | vs `addGeneScoreMatrix`; 643 cells × 19,933 genes |
| LSI (ArchR) | x25.7–x40.9 | ✅ Per-component \|r\|: 1.000 | vs `.computeLSI`, all three `LSIMethod` variants; 643 cells × 290,170 tiles |
| LSI Variable Features (ArchR) | x21.1 | ✅ Full Match | vs `.identifyVarFeatures` scoring: identical feature sets |
| Iterative LSI (ArchR) | x140.9 | ✅ Within ArchR's seed spread | vs `addIterativeLSI`: Jaccard 0.47, subspace corr 0.74; ArchR seed to seed 0.45 / 0.73 |
| Footprinting (TOBIAS) | x71.8 | ✅ Stage-wise Full Match | vs TOBIAS 0.17.5 (6 cores); 3 pseudobulks × chr19 × 3,935 motifs; end to end within TOBIAS's own noise |

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

Run the tests in this order. `fragment_loading.py` writes the fragment Parquet
files the other tests read, so run it again after updating GATAC's fragment
format; `lsi.py` reads the fragments `gene_score.py` writes.

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
pixi run python test/amulet_doublet.py
pixi run python test/gene_score.py
pixi run python test/lsi.py
pixi run python test/iterative_lsi.py
pixi run python test/footprinting.py
```

Most tests take `--run-gatac-only` to skip the other tool. The slow reference
tools run once and their outputs are cached under `data/`:

| Test | Reference | Cache | Rerun with |
|------|-----------|-------|------------|
| `amulet_doublet` | AMULET v1.1 | `data/amulet_v11_output/` | `--rerun-amulet` |
| `gene_score` | ArchR `addGeneScoreMatrix` | `data/gene_score_output/` | `--regenerate` |
| `lsi` | ArchR `.computeLSI`, `.identifyVarFeatures` | `data/lsi_output/` | `--regenerate` |
| `iterative_lsi` | ArchR `addIterativeLSI`, two seeds | `data/iterative_lsi_output/` | `--regenerate` |
| `footprinting` | TOBIAS 0.17.5 | `data/footprinting_tobias/` | `--rerun-tobias` |

`addIterativeLSI` changes with its seed, so GATAC is judged against ArchR's own
seed-to-seed agreement. For footprinting, each TOBIAS stage given TOBIAS's own
input must match exactly; end to end, GATAC must stay within twice the
deviation of a TOBIAS rerun with ±1e-13 bias-score noise.
