"""
Tile matrix test demonstrating GATAC matches SnapATAC2.
"""
import os
import sys
import time
import argparse
import numpy as np
import snapatac2 as snap
import gatac as ga
import pandas as pd


def test_tile_matrix(skip_snapatac2=False):
    """Test GATAC tile matrix creation against SnapATAC2.

    Args:
        skip_snapatac2: If True, skip SnapATAC2 tile matrix creation and comparison
    """
    # SnapATAC2 tile matrix (optional)
    snap_time = None
    snap_sum = None
    snap_tile = None

    if not skip_snapatac2:
        snap_tile = snap.read("data/pbmc.h5ad")

        start_snap = time.time()
        snap.pp.add_tile_matrix(snap_tile)
        end_snap = time.time()
        snap_time = end_snap - start_snap

    # Use the fixed parquet file that includes all fragments from the source TSV
    parquet_filtered = "data/atac_pbmc_5k_filtered.parquet"

    start_gatac = time.time()
    gatac_tile = ga.pp.make_tile_matrix(parquet_filtered, chrom_sizes="hg38", tile_size=500)
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac

    # Compare results (only if not skipping SnapATAC2)
    if not skip_snapatac2:
        snap_tile = snap_tile.to_memory()
        gatac_tile = gatac_tile[snap_tile.obs_names]

        gatac_sum = gatac_tile.X.sum(axis=1).A1
        snap_sum = snap_tile.X.sum(axis=1).A1

        # Calculate statistics
        gatac_total = gatac_sum.sum()
        snap_total = snap_sum.sum()
        match = (gatac_sum == snap_sum).all()
        diff = abs(gatac_total - snap_total)
        rel_diff = diff / snap_total if snap_total > 0 else 0

        # Log results with comparison
        results = [
            f"SnapATAC2:\t{snap_time:.2f}s",
            f"GATAC:\t{gatac_time:.2f}s",
            f"GATAC sum:\t{gatac_total}",
            f"SnapATAC2 sum:\t{snap_total}",
            f"Match:\t{match}",
            f"Difference:\t{diff} ({rel_diff:.3%})"
        ]

        log_path = os.path.join(os.path.dirname(__file__), "tile_matrix.log")
        with open(log_path, 'w') as f:
            for result in results:
                print(result)
                f.write(result + '\n')

        # Assertions to verify tile matrix quality
        # GATAC should exactly match SnapATAC2
        assert gatac_tile.shape == snap_tile.shape, f"Shape mismatch: GATAC {gatac_tile.shape} vs SnapATAC2 {snap_tile.shape}"
        assert match, f"Sum mismatch: GATAC {gatac_total} vs SnapATAC2 {snap_total} (difference: {diff})"
    else:
        # Log GATAC-only results
        gatac_total = gatac_tile.X.sum()

        results = [
            f"GATAC:\t{gatac_time:.2f}s",
            f"GATAC sum:\t{gatac_total}",
            f"Shape:\t{gatac_tile.shape[0]:,} cells × {gatac_tile.shape[1]:,} tiles",
            f"(SnapATAC2 comparison skipped)"
        ]

        log_path = os.path.join(os.path.dirname(__file__), "tile_matrix.log")
        with open(log_path, 'w') as f:
            for result in results:
                print(result)
                f.write(result + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC tile matrix creation")
    parser.add_argument(
        "--skip-snapatac2",
        action="store_true",
        help="Skip SnapATAC2 tile matrix creation and comparison"
    )
    args = parser.parse_args()

    test_tile_matrix(skip_snapatac2=args.skip_snapatac2)




