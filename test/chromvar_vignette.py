"""
GATAC vs R chromVAR Reproducibility Test

Compares GATAC's chromVAR implementation against R chromVAR using the official
example_counts dataset from the chromVAR package vignette:
https://github.com/GreenleafLab/chromVAR/blob/master/vignettes/Articles/Deviations.Rmd

Prerequisites:
    Run the R reference script first to generate reference outputs:
        pixi run test-chromvar-vignette

Tests:
    1. test_chromvar_with_r_inputs:
       Uses R's exact motif matches and background peaks in GATAC to isolate
       the deviation kernel comparison.

    2. test_chromvar_full_pipeline:
       Runs GATAC's full pipeline (motif scanning, background peaks, deviations)
       and compares against R's results.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
import argparse
import logging
import os
import subprocess
import sys
import time
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GATAC"))
import gatac as ga

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

R_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "chromvar_vignette_output"
GENOME_PATH = Path(__file__).parent.parent / "data" / "hg19.fa.gz"

# ---------------------------------------------------------------------------
# Correlation thresholds (based on observed reproducibility)
# ---------------------------------------------------------------------------

ISOLATION_CORR_THRESHOLD = 0.95
PIPELINE_CORR_THRESHOLD = 0.93
PER_MOTIF_CORR_THRESHOLD = 0.93
MATCH_CORR_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_r_outputs():
    """Skip tests if R reference data is missing."""
    if not R_OUTPUT_DIR.exists():
        pytest.skip(
            f"R output directory not found: {R_OUTPUT_DIR}. "
            "Run 'pixi run test-chromvar-vignette' first."
        )


def load_r_data():
    """Load R's filtered count matrix and GC content into AnnData."""
    count_mat = pd.read_csv(R_OUTPUT_DIR / "count_matrix.csv", index_col=0)
    r_gc = pd.read_csv(R_OUTPUT_DIR / "gc_content.csv", index_col=0)

    adata = ad.AnnData(
        X=count_mat.T.values.astype(np.float32),
        obs=pd.DataFrame(index=count_mat.columns),
        var=pd.DataFrame(index=count_mat.index),
    )
    adata.var["gc_content"] = r_gc.loc[adata.var_names].values.flatten()
    return adata


def load_r_motif_matches():
    """Load R motif matches from sparse COO format and return as CSR + names."""
    sparse_df = pd.read_csv(R_OUTPUT_DIR / "motif_matches_sparse.csv")
    motif_names_df = pd.read_csv(R_OUTPUT_DIR / "motif_names.csv")
    peaks_df = pd.read_csv(R_OUTPUT_DIR / "peaks.csv")

    n_peaks = len(peaks_df)
    n_motifs = len(motif_names_df)

    # R's summary() gives 1-based indices
    rows = sparse_df["i"].values - 1
    cols = sparse_df["j"].values - 1
    data = np.ones(len(rows), dtype=bool)

    matches = sp.csr_matrix((data, (rows, cols)), shape=(n_peaks, n_motifs))
    motif_names = motif_names_df["motif_name"].values.tolist()

    return matches, motif_names


def load_r_bg_peaks():
    """Load R background peaks and convert 1-based to 0-based."""
    r_bg = pd.read_csv(R_OUTPUT_DIR / "background_peaks.csv", index_col=0)
    return (r_bg.values - 1).astype(np.int32)


def load_r_deviation_scores():
    """Load R deviation z-scores (transposed to cells × motifs)."""
    return pd.read_csv(R_OUTPUT_DIR / "deviation_scores.csv", index_col=0).T


def load_motifs_from_csv(path):
    """Load motifs from R-exported PWM CSV (raw counts)."""
    df = pd.read_csv(path)
    motifs = []
    for motif_id in df["motif"].unique():
        motif_df = df[df["motif"] == motif_id].sort_values("pos")
        raw_counts = motif_df[["A", "C", "G", "T"]].values.astype(np.float64)
        pwm = raw_counts / raw_counts.sum(axis=1, keepdims=True)
        motifs.append(
            ga.tl.DNAMotif(id=motif_id, pwm=pwm, name=motif_id, pfm=raw_counts)
        )
    return motifs


def compare_deviation_scores(r_scores, dev_adata):
    """Compare GATAC deviation scores against R reference.

    Returns (overall_corr, mean_per_motif_corr, n_common_cells, n_common_motifs).
    """
    common_cells = sorted(set(r_scores.index) & set(dev_adata.obs_names))
    common_motifs = sorted(set(r_scores.columns) & set(dev_adata.var_names))

    r_aligned = r_scores.loc[common_cells, common_motifs].values
    gatac_aligned = pd.DataFrame(
        dev_adata.X, index=dev_adata.obs_names, columns=dev_adata.var_names
    ).loc[common_cells, common_motifs].values

    overall_corr = np.corrcoef(
        r_aligned.flatten(), gatac_aligned.flatten()
    )[0, 1]

    per_motif_corrs = []
    for i in range(len(common_motifs)):
        r_col = r_aligned[:, i]
        g_col = gatac_aligned[:, i]
        if np.std(r_col) > 0 and np.std(g_col) > 0:
            corr = np.corrcoef(r_col, g_col)[0, 1]
            per_motif_corrs.append(corr)

    mean_per_motif = np.mean(per_motif_corrs) if per_motif_corrs else 0.0

    return overall_corr, mean_per_motif, len(common_cells), len(common_motifs)


# ---------------------------------------------------------------------------
# Test 1: Deviation kernel isolation
# ---------------------------------------------------------------------------


def test_chromvar_with_r_inputs():
    """Use R's exact motif matches and background peaks in GATAC.

    Isolates the deviation kernel: identical inputs, only implementation differs.
    """
    _check_r_outputs()

    print("\n" + "=" * 70)
    print("TEST: ChromVAR Deviation Kernel (R inputs -> GATAC)")
    print("=" * 70)

    adata = load_r_data()
    r_matches, motif_names = load_r_motif_matches()
    r_bg = load_r_bg_peaks()
    r_scores = load_r_deviation_scores()

    print(f"  Data: {adata.n_obs} cells x {adata.n_vars} peaks")
    print(f"  Motifs: {len(motif_names)}, R matches: {r_matches.sum():,}")

    # Inject R's inputs into AnnData
    adata.varm["motif_match"] = r_matches
    adata.uns["motif_name"] = motif_names
    adata.varm["bg_peaks"] = r_bg

    # Run GATAC chromVAR
    t0 = time.time()
    dev_adata = ga.tl.chromvar(adata)
    elapsed = time.time() - t0
    print(f"  GATAC chromVAR time: {elapsed:.2f}s")

    # Compare
    overall_corr, per_motif_corr, n_cells, n_motifs = compare_deviation_scores(
        r_scores, dev_adata
    )

    print(f"\n  Common cells: {n_cells}, Common motifs: {n_motifs}")
    print(f"  GATAC  mean={np.nanmean(dev_adata.X):.4f}  std={np.nanstd(dev_adata.X):.4f}")
    print(f"  R      mean={np.nanmean(r_scores.values):.4f}  std={np.nanstd(r_scores.values):.4f}")
    print(f"  Overall Deviation Correlation:  {overall_corr:.6f}")
    print(f"  Mean Per-Motif Correlation:     {per_motif_corr:.6f}")

    assert overall_corr > ISOLATION_CORR_THRESHOLD, (
        f"Deviation kernel correlation {overall_corr:.4f} < {ISOLATION_CORR_THRESHOLD}"
    )
    assert per_motif_corr > PER_MOTIF_CORR_THRESHOLD, (
        f"Per-motif correlation {per_motif_corr:.4f} < {PER_MOTIF_CORR_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Test 2: Full pipeline
# ---------------------------------------------------------------------------


def test_chromvar_full_pipeline():
    """Run GATAC's full pipeline and compare to R chromVAR.

    GATAC performs motif scanning, background peak sampling, and deviation
    computation from scratch. Uses R's GC content and motif PWMs as shared
    inputs. Differences from motif scanning and random background sampling
    are expected, so thresholds are relaxed.
    """
    _check_r_outputs()

    if not GENOME_PATH.exists():
        pytest.skip(f"Genome file not found: {GENOME_PATH}")

    print("\n" + "=" * 70)
    print("TEST: Full ChromVAR Pipeline (GATAC vs R)")
    print("=" * 70)

    adata = load_r_data()
    r_scores = load_r_deviation_scores()
    r_matches, r_motif_names = load_r_motif_matches()
    r_matches_dense = r_matches.toarray()

    motifs = load_motifs_from_csv(R_OUTPUT_DIR / "motifs_pwm.csv")
    print(f"  Data: {adata.n_obs} cells x {adata.n_vars} peaks, {len(motifs)} motifs")

    # --- GC content verification ---
    peak_seqs_df = pd.read_csv(R_OUTPUT_DIR / "peak_sequences.csv")
    r_gc = pd.read_csv(R_OUTPUT_DIR / "gc_content.csv", index_col=0)

    peak_seqs_df["python_gc"] = peak_seqs_df["sequence"].apply(
        lambda s: (s.upper().count("G") + s.upper().count("C")) / len(s)
        if len(s) > 0
        else 0.0
    )
    peak_seqs_df.set_index("peak", inplace=True)
    gc_comp = r_gc.join(peak_seqs_df[["python_gc"]])
    gc_corr = gc_comp.iloc[:, 0].corr(gc_comp["python_gc"])
    print(f"\n  GC Content Correlation (R vs Python): {gc_corr:.6f}")

    # --- GATAC motif scanning ---
    adata_pipeline = adata.copy()

    t0 = time.time()
    ga.tl.scan_motifs(
        adata_pipeline,
        motifs,
        str(GENOME_PATH),
        pvalue=5e-5,
        mode="motifmatchr",
        coordinate_system="1-based",
    )
    scan_time = time.time() - t0

    gatac_matches = adata_pipeline.varm["motif_match"]
    if sp.issparse(gatac_matches):
        gatac_matches = gatac_matches.toarray()
    gatac_motif_names = adata_pipeline.uns["motif_name"]

    # Align motif matrices by name
    common_motifs = sorted(set(r_motif_names) & set(gatac_motif_names))
    r_idx = {m: i for i, m in enumerate(r_motif_names)}
    g_idx = {m: i for i, m in enumerate(gatac_motif_names)}

    r_aligned = r_matches_dense[:, [r_idx[m] for m in common_motifs]]
    g_aligned = gatac_matches[:, [g_idx[m] for m in common_motifs]]

    match_corr = np.corrcoef(
        r_aligned.flatten().astype(float),
        g_aligned.flatten().astype(float),
    )[0, 1]

    both = np.sum(r_aligned & g_aligned)
    either = np.sum(r_aligned | g_aligned)
    jaccard = both / either if either > 0 else 0

    print(f"  Motif Scan Time: {scan_time:.2f}s")
    print(f"  GATAC matches: {np.sum(gatac_matches):,},  R matches: {np.sum(r_matches_dense):,}")
    print(f"  Motif Match Correlation: {match_corr:.6f}")
    print(f"  Motif Match Jaccard:     {jaccard:.4f}")

    # --- Background peaks (chromvar binning method) ---
    ga.tl.sample_bg_peaks(
        adata_pipeline,
        method="chromvar",
        n_iterations=50,
    )

    # --- Compute deviations ---
    t0 = time.time()
    dev_adata = ga.tl.chromvar(adata_pipeline)
    dev_time = time.time() - t0

    overall_corr, per_motif_corr, n_cells, n_motifs = compare_deviation_scores(
        r_scores, dev_adata
    )

    print(f"\n  ChromVAR Time: {dev_time:.2f}s")
    print(f"  Common cells: {n_cells}, Common motifs: {n_motifs}")
    print(f"  Deviation Score Correlation: {overall_corr:.6f}")
    print(f"  Per-Motif Correlation:       {per_motif_corr:.6f}")

    assert gc_corr > 0.999, f"GC correlation {gc_corr:.4f} too low"
    assert match_corr > MATCH_CORR_THRESHOLD, (
        f"Motif match correlation {match_corr:.4f} < {MATCH_CORR_THRESHOLD}"
    )
    assert overall_corr > PIPELINE_CORR_THRESHOLD, (
        f"Full pipeline deviation correlation {overall_corr:.4f} < {PIPELINE_CORR_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Test 3: Deviation speedup (R vs GATAC)
# ---------------------------------------------------------------------------


def test_chromvar_deviation_speedup(run_gatac_only=False):
    """Run R chromVAR from Python, time computeDeviations, compare to GATAC.

    Uses R's exact motif matches and background peaks in both tools so the
    only difference is the deviation kernel implementation.
    """
    r_script = Path(__file__).parent / "chromvar_vignette_R.R"
    r_dev_time = None

    if not run_gatac_only:
        print("\n" + "=" * 70)
        print("Running R chromVAR vignette script...")
        print("=" * 70)
        result = subprocess.run(
            ["Rscript", str(r_script)],
            capture_output=True,
            text=True,
            cwd=str(r_script.parent.parent),
        )
        if result.returncode != 0:
            raise RuntimeError(f"R script failed:\n{result.stderr}")
        print(result.stdout)

        timing_path = R_OUTPUT_DIR / "timing.csv"
        if not timing_path.exists():
            raise FileNotFoundError(
                f"R timing file not found: {timing_path}. "
                "Check that the R script completed successfully."
            )
        r_dev_time = pd.read_csv(timing_path)["deviation_time"].iloc[0]
    else:
        _check_r_outputs()

    print("\n" + "=" * 70)
    print("TEST: ChromVAR Deviation Speedup (R vs GATAC)")
    print("=" * 70)

    adata = load_r_data()
    r_matches, motif_names = load_r_motif_matches()
    r_bg = load_r_bg_peaks()
    r_scores = load_r_deviation_scores()

    n_cells = adata.n_obs
    n_peaks = adata.n_vars
    n_motifs = len(motif_names)

    print(f"  Data: {n_cells:,} cells x {n_peaks:,} peaks x {n_motifs:,} motifs")

    adata.varm["motif_match"] = r_matches
    adata.uns["motif_name"] = motif_names
    adata.varm["bg_peaks"] = r_bg

    t0 = time.perf_counter()
    dev_adata = ga.tl.chromvar(adata)
    gatac_time = time.perf_counter() - t0

    overall_corr, per_motif_corr, n_common_cells, n_common_motifs = (
        compare_deviation_scores(r_scores, dev_adata)
    )

    results = [
        "=== ChromVAR Deviation Benchmark ===",
        f"Matrix: {n_cells:,} cells x {n_peaks:,} peaks x {n_motifs:,} motifs",
        "",
    ]

    if not run_gatac_only:
        results += [
            f"R chromVAR (computeDeviations):\t{r_dev_time:.2f}s",
        ]

    results += [
        f"GATAC:\t{gatac_time:.2f}s",
        "",
        f"Correlation (deviations):    {overall_corr:.6f}",
    ]

    if not run_gatac_only:
        speedup = r_dev_time / gatac_time
        results.append(f"Speedup: {speedup:.1f}x")

    log_path = os.path.join(os.path.dirname(__file__), "chromvar_vignette.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in results:
            print(line)
            f.write(line + "\n")

    assert overall_corr > ISOLATION_CORR_THRESHOLD, (
        f"Deviation correlation {overall_corr:.4f} < {ISOLATION_CORR_THRESHOLD}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC chromVAR deviation speedup vs R")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip R script run and speedup comparison",
    )
    args = parser.parse_args()
    test_chromvar_deviation_speedup(run_gatac_only=args.run_gatac_only)
