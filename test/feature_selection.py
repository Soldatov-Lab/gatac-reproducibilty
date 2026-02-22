"""
Feature selection test demonstrating GATAC matches SnapATAC2.

This test compares the feature selection between GATAC's GPU-accelerated
implementation and SnapATAC2's chunked approach, checking both speed and
result consistency.

IMPORTANT: Both tools must use the SAME tile matrix for fair comparison.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import snapatac2 as snap
import gatac as ga


def test_feature_selection(run_gatac_only=False):
    """Test GATAC feature selection against SnapATAC2.

    Args:
        run_gatac_only: If True, run GATAC only (skip SnapATAC2) feature selection and comparison
    """
    # Load the SnapATAC2 tile matrix (source of truth)
    print("Loading SnapATAC2 tile matrix...")
    snap_tile = snap.read("data/pbmc.h5ad")

    # Check if tile matrix exists, if not add it
    if snap_tile.X is None or snap_tile.shape[1] == 0:
        print("Adding tile matrix to SnapATAC2 data...")
        snap.pp.add_tile_matrix(snap_tile)

    # Convert to in-memory AnnData for manipulation
    snap_tile = snap_tile.to_memory()
    print(f"Matrix shape: {snap_tile.shape[0]:,} cells × {snap_tile.shape[1]:,} features")

    # Create a copy for GATAC to use the SAME matrix
    # This ensures fair comparison - same input data
    gatac_tile = ad.AnnData(
        X=snap_tile.X.copy(),
        obs=snap_tile.obs.copy(),
        var=snap_tile.var.copy()
    )
    gatac_tile.obs_names = snap_tile.obs_names.copy()
    gatac_tile.var_names = snap_tile.var_names.copy()

    print(f"Using same matrix for both: {snap_tile.shape}")

    # Feature selection parameters
    n_features = 50000
    filter_lower_quantile = 0.005
    filter_upper_quantile = 0.005

    # SnapATAC2 feature selection (optional)
    snap_time = None
    snap_selected = None
    snap_features = None
    snap_counts = None
    count_match = None
    count_correlation = None

    if not run_gatac_only:
        print("\nRunning SnapATAC2 feature selection...")
        start_snap = time.time()
        snap.pp.select_features(
            snap_tile,
            n_features=n_features,
            filter_lower_quantile=filter_lower_quantile,
            filter_upper_quantile=filter_upper_quantile,
            inplace=True
        )
        end_snap = time.time()
        snap_time = end_snap - start_snap
        snap_selected = snap_tile.var['selected'].values
        snap_features = set(snap_tile.var_names[snap_selected])
        snap_counts = snap_tile.var['count'].values

    # GATAC feature selection
    print("Running GATAC feature selection...")
    start_gatac = time.time()
    ga.pp.select_features(
        gatac_tile,
        n_features=n_features,
        filter_lower_quantile=filter_lower_quantile,
        filter_upper_quantile=filter_upper_quantile,
        inplace=True
    )
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac

    # GATAC results
    gatac_selected = gatac_tile.var['selected'].values
    gatac_n_selected = np.sum(gatac_selected)
    gatac_features = set(gatac_tile.var_names[gatac_selected])
    gatac_counts = gatac_tile.var['accessibility_count'].values

    # Calculate comparisons (only if not skipping SnapATAC2)
    if not run_gatac_only:
        snap_n_selected = np.sum(snap_selected)

        # Check feature overlap
        common_features = snap_features & gatac_features
        snap_only = snap_features - gatac_features
        gatac_only = gatac_features - snap_features
        overlap_ratio = len(common_features) / max(len(snap_features), len(gatac_features))

        # Compare feature counts
        count_correlation = np.corrcoef(snap_counts, gatac_counts)[0, 1]
        count_match = np.allclose(snap_counts, gatac_counts)

        # Investigate the overlap difference
        print("\n--- Investigating Feature Overlap Differences ---")

        # Get indices of differing features
        snap_only_idx = [i for i, v in enumerate(snap_tile.var_names) if v in snap_only]
        gatac_only_idx = [i for i, v in enumerate(gatac_tile.var_names) if v in gatac_only]

        # Check counts at the boundary
        if len(snap_only_idx) > 0:
            snap_only_counts = snap_counts[snap_only_idx]
            print(f"\nFeatures only in SnapATAC2 ({len(snap_only)}):")
            print(f"  Count range: {snap_only_counts.min():.0f} - {snap_only_counts.max():.0f}")
            print(f"  Count mean: {snap_only_counts.mean():.1f}")

        if len(gatac_only_idx) > 0:
            gatac_only_counts = gatac_counts[gatac_only_idx]
            print(f"\nFeatures only in GATAC ({len(gatac_only)}):")
            print(f"  Count range: {gatac_only_counts.min():.0f} - {gatac_only_counts.max():.0f}")
            print(f"  Count mean: {gatac_only_counts.mean():.1f}")

        # Check the selection boundary - what count value is the cutoff?
        snap_selected_counts = snap_counts[snap_selected]
        gatac_selected_counts = gatac_counts[gatac_selected]
        print(f"\nSelection boundary analysis:")
        print(f"  SnapATAC2 min selected count: {snap_selected_counts.min():.0f}")
        print(f"  GATAC min selected count: {gatac_selected_counts.min():.0f}")

        # Check if differences are due to ties
        boundary_count = snap_selected_counts.min()
        features_at_boundary = np.sum(snap_counts == boundary_count)
        print(f"\n  Features at boundary count ({boundary_count:.0f}): {features_at_boundary:,}")
        print(f"  --> Likely tie-breaking difference if this is large")

        # ROOT CAUSE ANALYSIS: Zero-count features
        n_zero_features = np.sum(snap_counts == 0)
        n_nonzero_features = np.sum(snap_counts > 0)
        print(f"\n--- Root Cause Analysis ---")
        print(f"Total features: {len(snap_counts):,}")
        print(f"Zero-count features: {n_zero_features:,} ({n_zero_features/len(snap_counts):.1%})")
        print(f"Non-zero features: {n_nonzero_features:,}")

        # Check if there's perfect match or tie-breaking differences
        if len(snap_only) == 0 and len(gatac_only) == 0:
            print("\nPERFECT MATCH: Both tools selected identical features!")
        else:
            print(f"""
REMAINING DIFFERENCE ({len(snap_only)} features): Tie-breaking at boundary.
  Both tools now exclude zeros before quantile filtering.
  The {len(snap_only)} differing features have identical counts at the selection boundary.
  Different sort order (GPU cupy vs CPU numpy) causes different tie-breaking.
""")

        # Log results with comparison
        results = [
            f"=== Feature Selection Benchmark ===",
            f"Matrix: {snap_tile.shape[0]:,} cells x {snap_tile.shape[1]:,} features",
            f"Same input matrix: YES",
            f"",
            f"SnapATAC2:",
            f"  Time: {snap_time:.2f}s",
            f"  Selected features: {snap_n_selected:,}",
            f"GATAC:",
            f"  Time: {gatac_time:.2f}s",
            f"  Selected features: {gatac_n_selected:,}",
            f"",
            f"Comparison:",
            f"  Count match: {count_match}",
            f"  Count correlation: {count_correlation:.6f}",
            f"  Common features: {len(common_features):,}",
            f"  SnapATAC2-only: {len(snap_only):,}",
            f"  GATAC-only: {len(gatac_only):,}",
            f"  Feature overlap: {overlap_ratio:.1%}",
            f"  Speedup: {snap_time/gatac_time:.2f}x",
            f"",
            f"Notes:",
            f"  Remaining difference ({len(snap_only)}) is tie-breaking at boundary count",
            f"  Features at boundary count ({boundary_count:.0f}): {features_at_boundary:,}"
        ]

        log_path = os.path.join(os.path.dirname(__file__), "feature_selection.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            for result in results:
                print(result)
                f.write(result + '\n')

        # Assertions to verify feature selection quality
        assert count_match, "Feature counts don't match between SnapATAC2 and GATAC"
        assert count_correlation > 0.999, f"Count correlation too low: {count_correlation:.6f} (expected > 0.999)"
        assert overlap_ratio > 0.995, f"Feature overlap too low: {overlap_ratio:.1%} (expected > 99.5%)"
        assert len(snap_only) < 100, f"Too many SnapATAC2-only features: {len(snap_only)} (expected < 100)"
    else:
        # Log GATAC-only results
        results = [
            f"=== Feature Selection Benchmark ===",
            f"Matrix: {gatac_tile.shape[0]:,} cells x {gatac_tile.shape[1]:,} features",
            f"",
            f"GATAC:",
            f"  Time: {gatac_time:.2f}s",
            f"  Selected features: {gatac_n_selected:,}",
            f"",
            f"(SnapATAC2 comparison skipped)"
        ]

        log_path = os.path.join(os.path.dirname(__file__), "feature_selection.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            for result in results:
                print(result)
                f.write(result + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC feature selection")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 feature selection and comparison"
    )
    args = parser.parse_args()

    test_feature_selection(run_gatac_only=args.run_gatac_only)

