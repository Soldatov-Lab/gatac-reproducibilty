"""
Peak matrix test demonstrating GATAC matches SnapATAC2.

Uses GATAC peaks (from peak_calling.py) and compares construction
speed and matrix properties between both tools.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import snapatac2 as snap
from anndata import AnnData

import gatac as ga

PEAK_FILE = Path("data/gatac_peaks.bed")
H5AD_FILE = Path("data/pbmc.h5ad")
PARQUET_FILE = Path("data/atac_pbmc_5k.parquet")


def _check_inputs():
    if not PEAK_FILE.exists():
        pytest.skip(f"{PEAK_FILE} not found; run peak_calling.py first")
    if not H5AD_FILE.exists() or not PARQUET_FILE.exists():
        pytest.skip(f"{H5AD_FILE} or {PARQUET_FILE} not found; run fragment_loading.py first")
    try:
        ga.io.FragmentFile(PARQUET_FILE)
    except ValueError:
        pytest.skip(f"{PARQUET_FILE} predates the current GATAC fragment format; "
                    "run fragment_loading.py again")


def _gatac_peak_matrix(peaks_df):
    filtered = ga.pp.filter_fragments(PARQUET_FILE, min_fragments_per_cell=200, chrom_sizes="hg38")
    cells = AnnData(obs=pd.DataFrame(index=ga.io.FragmentFile(filtered).barcodes.tolist()))
    cells.uns["peaks"] = peaks_df
    start = time.time()
    mat = ga.peak.make_matrix(cells, parquet_path=str(filtered), use_rep="peaks",
                              inplace=False, verbose=True)
    return mat, time.time() - start


def test_make_peak_matrix(run_gatac_only=False):
    """Test GATAC peak matrix creation against SnapATAC2 on the same peaks.

    Args:
        run_gatac_only: If True, run GATAC only (skip SnapATAC2 and the comparison)
    """
    _check_inputs()
    peaks_df = pd.read_csv(PEAK_FILE, sep="\t", header=None, names=["chrom", "start", "end"])
    print(f"Loaded {len(peaks_df):,} peaks from {PEAK_FILE}")

    gatac_mat, gatac_time = _gatac_peak_matrix(peaks_df)
    results = [
        f"Peaks:\t{len(peaks_df):,}",
        f"GATAC:\t{gatac_time:.2f}s\t{gatac_mat.shape}\t{gatac_mat.X.nnz:,} nnz",
    ]

    if not run_gatac_only:
        snap_data = snap.read(H5AD_FILE, backed="r")
        snap_peaks = [f"{c}:{s}-{e}" for c, s, e in peaks_df.itertuples(index=False)]
        start = time.time()
        snap_mat = snap.pp.make_peak_matrix(snap_data, use_rep=snap_peaks,
                                            file="data/snap_peak_matrix.h5ad")
        snap_time = time.time() - start
        snap_X = snap_mat.X[:]
        snap_obs, snap_var = list(snap_mat.obs_names), list(snap_mat.var_names)
        snap_mat.close()
        snap_data.close()

        # Align both axes by name; the tools need not order cells or peaks alike
        cells = gatac_mat.obs_names.get_indexer(snap_obs)
        peaks = gatac_mat.var_names.get_indexer(snap_var)
        assert (cells >= 0).all() and (peaks >= 0).all(), "cells or peaks differ between the tools"
        gatac_X = gatac_mat.X[cells][:, peaks]
        n_diff = (gatac_X != snap_X).nnz

        results += [
            f"SnapATAC2:\t{snap_time:.2f}s\t{snap_X.shape}\t{snap_X.nnz:,} nnz",
            f"Speedup:\t{snap_time / gatac_time:.2f}x",
            f"Shape Match:\t{gatac_mat.shape == snap_X.shape}",
            f"Differing entries:\t{n_diff:,}",
        ]

    log_path = os.path.join(os.path.dirname(__file__), "make_peak_matrix.log")
    with open(log_path, "w") as f:
        for result in results:
            print(result)
            f.write(result + "\n")

    if not run_gatac_only:
        # Same peaks, same counting strategy (paired-insertion): every count must match
        assert gatac_mat.shape == snap_X.shape, f"Shape mismatch: GATAC {gatac_mat.shape} vs SnapATAC2 {snap_X.shape}"
        assert n_diff == 0, f"{n_diff:,} entries differ from SnapATAC2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC peak matrix creation")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 peak matrix creation and comparison"
    )
    args = parser.parse_args()

    test_make_peak_matrix(run_gatac_only=args.run_gatac_only)
