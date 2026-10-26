# Reproducibility

Full test suite for comparing gatac with existing tools, mainly snapatac2, but also macs3 and chromvar.

## Results Summary

One run of every test on an NVIDIA L40S with 20 CPU cores (Xeon Platinum 8358), data on
local disk (2026-10-26, GATAC `6d16b26`; the tile matrix test was rerun at `6285b33`, which
fixes a crash of its new GPU tile statistics).
AMULET, ArchR and TOBIAS were rerun on the same node with all 20 cores. Speedups use a warm
GATAC call; **Cold start** is GATAC's first call in a fresh process.

Data: the 10x Genomics PBMC 5k scATAC fragments (`snap.datasets.pbmc5k()`, 14,225 cells after filtering) for every test except chromVAR (the chromVAR vignette's `example_counts`) and GSEA (synthetic ranked lists).

| Test | Speedup | Cold start | Result | Notes |
|------|---------|------------|--------|-------|
| Fragment Loading | x5.7 | x5.5 | ✅ Full Match | 14,225 cells in both; GATAC `make_parquet` + `filter_fragments` 27.3s vs SnapATAC2 `import_fragments` 155.4s |
| TSS Enrichment | x4.3 | x1.7 | ✅ Correlation: 1.000 | Cell count match |
| Tile Matrix | x6.2 | x3.1 | ✅ Full Match | Sum=123,061,807 for both tools |
| Gene Matrix | x18.1 | x7.6 | ✅ Full Match | vs `snap.pp.make_gene_matrix`: same 14,225 cells × 60,606 genes in the same order, all 52,805,125 nonzero entries identical |
| Feature Selection | x2.1 | x1.0 | ⚠️ Overlap: 99.8% | 80 features differ by tie-breaking at the boundary count |
| Spectral Embedding | x16.7 | x8.5 | ✅ Eigenvalue corr: 1.000 | vs `snap.tl.spectral`, 30 components; 14,107 cells × 6,062,095 tiles |
| Peak Calling | x45.7 | x34.0 | ✅ Full Match | vs `snap.tl.macs3` (MACS3 3.0.4): all 234,758 summits of 3 groups identical (positions, scores, fold enrichment, p/q); merged peaks identical (119,141) |
| Peak Matrix | x13.4 | x11.0 | ✅ Full Match | 42,932,832 nnz in both on the 119,141 merged peaks |
| Motif Enrichment | x11.5 | x11.1 | ✅ Full Match | vs `snap.tl.motif_enrichment`: same matched regions, so fold changes and p-values of 3 groups × 1,165 cisBP motifs agree to rounding |
| GSEA | x24.7 | x2.5 | ✅ ES Match | GPU vs GSEApy, 300 sets; 100% sign agreement |
| ChromVAR Deviations | x18.1 | x1.7 | ✅ Correlation: 1.000 | vs R chromVAR `computeDeviations` given R's motif matches and background peaks: z-scores within 0.002 (float32 rounding); 36 cells × 28,596 peaks × 386 motifs. With GATAC's own motif scan and background peaks: 0.965 |
| AMULET Doublet Detection | x145.1 | x65.5 | ✅ Full Match | vs AMULET v1.1: Jaccard 1.000, q-value r 1.000 on 13,735 cells |
| Gene Score (ArchR) | x16.4 | x14.5 | ✅ Entry-wise corr: 1.000 | vs `addGeneScoreMatrix`; 14,153 cells × 19,933 genes |
| LSI (ArchR) | x34.3–x96.0 | x24.9–x89.4 | ✅ Per-component \|r\|: 1.000 | vs `.computeLSI`, all three `LSIMethod` variants; 14,225 cells × 566,222 tiles |
| LSI Variable Features (ArchR) | x8.2 | x1.5 | ✅ Full Match | vs `.identifyVarFeatures` scoring: identical feature sets |
| Iterative LSI (ArchR) | x131.4 | x8.4 | ⚠️ Within ArchR's own variability | vs `addIterativeLSI`: feature Jaccard 0.45, subspace corr 0.78; ArchR against itself 0.37–0.93 / 0.70–0.97 across seeds and thread counts (see below) |
| Footprinting (TOBIAS) | x49.0 | x35.3 | ✅ Stage-wise Full Match | vs TOBIAS 0.17.5; 3 pseudobulks × chr19 × 3,935 motifs; end to end, bound-site differences of 0.003–0.004, about TOBIAS's own noise (see below) |

SnapATAC2, GSEApy, R chromVAR and TOBIAS use every core they are given, so the speedups
depend on the CPU.

A cold start also pays a one-off GPU start-up (CUDA context, CUDA libraries
loaded on first use, cuML and cuGraph imports). It does not depend on the data,
so it only matters for the shortest runs, and a script that has already run
other GATAC functions has paid most of it. It also varies between runs: 1.3 s
for chromVAR and 15 s for iterative LSI in this one, against 0.35 s and 1.9 s
in the previous one, with the same code for both tests. The reference tools are
timed with their libraries loaded; see `test/_timing.py`.

Footprinting compares GATAC end to end against a noise control, TOBIAS rerun
with its bias perturbed by ±1e-13, and asserts that each difference is at most
twice the control's. GATAC's differences are the same in every run (bound
sites 0.0033, 0.0027 and 0.0036 for the three pseudobulks), but the control
varies between runs (0.0029, 0.0040 and 0.0024 in the previous one; 0.0015,
0.0027 and 0.0015 in this one), so the assertion failed in this run for two
pseudobulks.

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
