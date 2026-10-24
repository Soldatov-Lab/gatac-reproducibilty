# Reproducibility

Full test suite for comparing gatac with existing tools, mainly snapatac2, but also macs3 and chromvar.

## Results Summary

One run of every test on an NVIDIA L40S with 20 CPU cores (Xeon Platinum 8358), data on
local disk (2026-10-24, GATAC `e0a53b0`).
AMULET, ArchR and TOBIAS were rerun on the same node with all 20 cores. Speedups use a warm
GATAC call; **Cold start** is GATAC's first call in a fresh process.

Data: the 10x Genomics PBMC 5k scATAC fragments (`snap.datasets.pbmc5k()`, 14,225 cells after filtering) for every test except chromVAR (the chromVAR vignette's `example_counts`) and GSEA (synthetic ranked lists).

| Test | Speedup | Cold start | Result | Notes |
|------|---------|------------|--------|-------|
| Fragment Loading | x3.5 | x3.2 | ✅ Full Match | 14,225 cells in both; GATAC `make_parquet` + `filter_fragments` 41.5s vs SnapATAC2 `import_fragments` 145.2s |
| TSS Enrichment | x2.5 | x2.1 | ✅ Correlation: 1.000 | Cell count match |
| Tile Matrix | x2.7 | x2.4 | ✅ Full Match | Sum=123,061,807 for both tools |
| Gene Matrix | x3.4 | x3.2 | ✅ Full Match | vs `snap.pp.make_gene_matrix`: same 14,225 cells × 60,606 genes in the same order, all 52,805,125 nonzero entries identical |
| Feature Selection | x2.1 | x1.5 | ⚠️ Overlap: 99.8% | 80 features differ by tie-breaking at the boundary count |
| Spectral Embedding | x3.1 | x2.8 | ✅ Eigenvalue corr: 1.000 | vs `snap.tl.spectral`, 30 components; 14,107 cells × 6,062,095 tiles |
| Peak Calling | x43.1 | x35.2 | ✅ Full Match | vs `snap.tl.macs3` (MACS3 3.0.4): all 234,758 summits of 3 groups identical (positions, scores, fold enrichment, p/q); merged peaks identical (119,141) |
| Peak Matrix | x2.5 | x2.5 | ✅ Full Match | 42,932,832 nnz in both on the 119,141 merged peaks |
| Motif Enrichment | x11.2 | x10.6 | ✅ Full Match | vs `snap.tl.motif_enrichment`: same matched regions, so fold changes and p-values of 3 groups × 1,165 cisBP motifs agree to rounding |
| GSEA | x25.6 | x15.6 | ✅ ES Match | GPU vs GSEApy, 300 sets; 100% sign agreement |
| ChromVAR Deviations | x18.6 | x5.0 | ✅ Correlation: 1.000 | vs R chromVAR `computeDeviations` given R's motif matches and background peaks: z-scores within 0.002 (float32 rounding); 36 cells × 28,596 peaks × 386 motifs. With GATAC's own motif scan and background peaks: 0.965 |
| AMULET Doublet Detection | x142.7 | x114.8 | ✅ Full Match | vs AMULET v1.1: Jaccard 1.000, q-value r 1.000 on 13,735 cells |
| Gene Score (ArchR) | x17.2 | x17.1 | ✅ Entry-wise corr: 1.000 | vs `addGeneScoreMatrix`; 14,153 cells × 19,933 genes |
| LSI (ArchR) | x37.4–x101.1 | x37.2–x91.6 | ✅ Per-component \|r\|: 1.000 | vs `.computeLSI`, all three `LSIMethod` variants; 14,225 cells × 566,222 tiles |
| LSI Variable Features (ArchR) | x8.3 | x8.1 | ✅ Full Match | vs `.identifyVarFeatures` scoring: identical feature sets |
| Iterative LSI (ArchR) | x120.7 | x42.4 | ⚠️ Within ArchR's own variability | vs `addIterativeLSI`: feature Jaccard 0.45, subspace corr 0.78; ArchR against itself 0.37–0.93 / 0.70–0.97 across seeds and thread counts (see below) |
| Footprinting (TOBIAS) | x42.6 | x41.4 | ✅ Stage-wise Full Match | vs TOBIAS 0.17.5; 3 pseudobulks × chr19 × 3,935 motifs; end to end within TOBIAS's own noise |

SnapATAC2, GSEApy, R chromVAR and TOBIAS use every core they are given, so the speedups
depend on the CPU.

A cold start also pays a one-off GPU start-up (CUDA context, CUDA libraries
loaded on first use, cuML and cuGraph imports). It does not depend on the data,
so it only matters for the shortest runs (0.35 s for chromVAR, 1.9 s for
iterative LSI), and a script that has already run other GATAC functions has
paid most of it. The reference tools are timed with their libraries loaded; see
`test/_timing.py`.

Iterative LSI is marked ⚠️ because `addIterativeLSI` is not a stable reference. Given the
same clusters, GATAC selects exactly ArchR's features, but ArchR's clusters change with its
seed and thread count, and the feature selection is very sensitive to them. Across 12 ArchR
runs, GATAC vs ArchR (Jaccard 0.29–0.85) falls within ArchR vs ArchR (0.37–0.93).

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

`run_suite.py` runs the tests in order, each in its own process:

```bash
pixi run python run_suite.py                         # all tests
pixi run python run_suite.py tss_enrichment lsi      # some, still in order
pixi run python run_suite.py --rerun-references      # rerun the cached tools
```

The order matters: `fragment_loading` writes the fragment files the other tests
read, so rerun it after a change to GATAC's fragment format, and `lsi` reads the
fragments `gene_score` writes. A single test runs as
`pixi run python test/<name>.py`; most take `--run-gatac-only` to skip the
other tool.

All data lives in `data/`: what the tests write, the cached reference outputs
below, and three footprinting inputs no test downloads: `GRCh38.p13.genome.fa`,
`cisBP_human.meme` and `pbmc_peak_matrix.h5ad`. The slow reference tools run
once and are cached there:

| Test | Reference | Cache | Rerun with |
|------|-----------|-------|------------|
| `amulet_doublet` | AMULET v1.1 | `amulet_v11_output/` | `--rerun-amulet` |
| `gene_score` | ArchR `addGeneScoreMatrix` | `gene_score_output/` | `--regenerate` |
| `lsi` | ArchR `.computeLSI`, `.identifyVarFeatures` | `lsi_output/` | `--regenerate` |
| `iterative_lsi` | ArchR `addIterativeLSI`, two seeds | `iterative_lsi_output/` | `--regenerate` |
| `footprinting` | TOBIAS 0.17.5 | `footprinting_tobias/` | `--rerun-tobias` |

`addIterativeLSI` changes with its seed, so GATAC is judged against ArchR's own
seed-to-seed agreement. For footprinting, each TOBIAS stage given TOBIAS's own
input must match exactly; end to end, GATAC must stay within twice the
deviation of a TOBIAS rerun with ±1e-13 bias-score noise.
