"""
Spectral embedding test comparing GATAC (GPU) vs SnapATAC2.

Verifies that GATAC's GPU-accelerated spectral embedding produces
results consistent with SnapATAC2's cosine-based spectral embedding.

Both implementations use the same algorithm:
  1. IDF weighting + L2 row normalization
  2. Matrix-free Laplacian eigenmaps via eigsh
  3. Optional SD weighting

The test checks:
  - Eigenvalue correlation and relative ordering
  - Eigenvector subspace alignment (via canonical angles)
  - Performance (wall-clock time)
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
import anndata as ad
import snapatac2 as snap
import gatac as ga


def test_spectral_embedding(run_gatac_only=False):
    """Compare GATAC spectral embedding against SnapATAC2.

    Args:
        run_gatac_only: If True, run GATAC only (skip SnapATAC2) spectral embedding and comparison
    """

    # ---- Load data ----
    print("Loading SnapATAC2 tile matrix...")
    snap_tile = snap.read("data/pbmc.h5ad")
    snap_tile = snap_tile.to_memory()
    print(f"Matrix shape: {snap_tile.shape[0]:,} cells × {snap_tile.shape[1]:,} features")

    # ---- Feature selection ----
    n_features = 50_000
    n_comps = 30

    print("\nRunning SnapATAC2 feature selection...")
    snap.pp.select_features(snap_tile, n_features=n_features)
    snap_selected = snap_tile.var["selected"].values.astype(bool)
    print(f"Selected {snap_selected.sum():,} features")

    # ---- Filter out cells with zero features in the selected set ----
    # GATAC raises an exception for such cells (degree = -1 → ill-defined
    # Laplacian). SnapATAC2 silently produces degenerate eigenvalues.
    # Pre-filter so both tools receive the same valid input.
    X_sel = snap_tile.X[:, snap_selected]
    row_nnz = np.diff(X_sel.indptr) if sp.issparse(X_sel) else (X_sel != 0).sum(axis=1)
    cell_mask = np.asarray(row_nnz > 0).ravel()
    n_dropped = int((~cell_mask).sum())
    if n_dropped > 0:
        print(f"Filtering {n_dropped} cells with zero features in the selected set")

    # GATAC copy (filtered, same features)
    gatac_filt = ad.AnnData(
        X=snap_tile[cell_mask].X.copy(),
        obs=snap_tile[cell_mask].obs.copy(),
        var=snap_tile.var.copy(),
    )
    gatac_filt.obs_names = snap_tile[cell_mask].obs_names.copy()
    gatac_filt.var_names = snap_tile.var_names.copy()
    gatac_filt.var["selected"] = snap_selected

    # ---- GATAC spectral (GPU) ----
    print(f"\nRunning GATAC spectral (n_comps={n_comps})...")
    t0 = time.perf_counter()
    ga.tl.spectral(gatac_filt, n_comps=n_comps, weighted_by_sd=False, chunk_size=10_000)
    gatac_time = time.perf_counter() - t0
    gatac_evals = gatac_filt.uns["spectral_eigenvalue"]
    gatac_evecs = gatac_filt.obsm["X_spectral"]
    print(f"  Time: {gatac_time:.2f}s")
    print(f"  Eigenvalues (top 5): {gatac_evals[:5]}")
    print(f"  Embedding shape: {gatac_evecs.shape}")

    snap_time = None
    eval_corr = None
    eval_rmse = None
    eval_max_diff = None
    mean_angle_deg = None
    max_angle_deg = None
    per_vec_cos = None

    if not run_gatac_only:
        # SnapATAC2 copy (filtered)
        snap_filt = snap_tile[cell_mask].copy()
        snap_filt.var["selected"] = snap_selected

        print(f"Using {snap_filt.shape[0]:,} cells for comparison")

        # ---- SnapATAC2 spectral ----
        print(f"\nRunning SnapATAC2 spectral (n_comps={n_comps})...")
        t0 = time.perf_counter()
        snap.tl.spectral(snap_filt, n_comps=n_comps, weighted_by_sd=False)
        snap_time = time.perf_counter() - t0
        snap_evals = snap_filt.uns["spectral_eigenvalue"]
        snap_evecs = snap_filt.obsm["X_spectral"]
        print(f"  Time: {snap_time:.2f}s")
        print(f"  Eigenvalues (top 5): {snap_evals[:5]}")
        print(f"  Embedding shape: {snap_evecs.shape}")

        # ---- Compare eigenvalues ----
        print("\n=== Eigenvalue Comparison ===")
        min_k = min(len(snap_evals), len(gatac_evals))
        se = snap_evals[:min_k]
        ge = gatac_evals[:min_k]

        eval_corr = np.corrcoef(se, ge)[0, 1]
        eval_rmse = np.sqrt(np.mean((se - ge) ** 2))
        eval_max_diff = np.max(np.abs(se - ge))

        print(f"  Correlation:  {eval_corr:.6f}")
        print(f"  RMSE:         {eval_rmse:.6f}")
        print(f"  Max abs diff: {eval_max_diff:.6f}")
        print(f"  SnapATAC2 evals: {se[:10]}")
        print(f"  GATAC evals:     {ge[:10]}")

        # ---- Compare eigenvectors (subspace alignment) ----
        print("\n=== Eigenvector Subspace Comparison ===")
        k_compare = min(min_k, 20)
        U = snap_evecs[:, :k_compare]
        V = gatac_evecs[:, :k_compare]

        # Canonical angles between the two subspaces
        Qu, _ = np.linalg.qr(U)
        Qv, _ = np.linalg.qr(V)
        _, svals, _ = np.linalg.svd(Qu.T @ Qv)
        svals = np.clip(svals, 0, 1)
        angles = np.arccos(svals)
        mean_angle_deg = np.degrees(angles.mean())
        max_angle_deg = np.degrees(angles.max())

        print(f"  Canonical angles (degrees, top {k_compare} subspace):")
        print(f"    Mean: {mean_angle_deg:.2f}°")
        print(f"    Max:  {max_angle_deg:.2f}°")
        print(f"    Min:  {np.degrees(angles.min()):.2f}°")

        # Per-eigenvector cosine similarity (up to sign)
        per_vec_cos = np.array([
            abs(np.dot(U[:, i], V[:, i])) / (
                np.linalg.norm(U[:, i]) * np.linalg.norm(V[:, i])
            )
            for i in range(k_compare)
        ])

        print(f"\n  Per-eigenvector |cos similarity| (top {k_compare}):")
        for i in range(min(k_compare, 10)):
            print(f"    Component {i}: {per_vec_cos[i]:.4f}")
        if k_compare > 10:
            print(f"    ... (mean of remaining: {per_vec_cos[10:].mean():.4f})")

        # ---- Log results ----
        results = [
            f"=== Spectral Embedding Benchmark ===",
            f"Matrix: {snap_filt.shape[0]:,} cells × {snap_filt.shape[1]:,} features",
            f"n_components: {n_comps}",
            f"",
            f"SnapATAC2:",
            f"  Time: {snap_time:.2f}s",
            f"GATAC:",
            f"  Time: {gatac_time:.2f}s",
            f"",
            f"Comparison:",
            f"  Eigenvalue correlation: {eval_corr:.6f}",
            f"  Eigenvalue RMSE:        {eval_rmse:.6f}",
            f"  Eigenvalue max diff:    {eval_max_diff:.6f}",
            f"  Subspace mean angle:    {mean_angle_deg:.2f}°",
            f"  Subspace max angle:     {max_angle_deg:.2f}°",
            f"  Per-eigenvector |cos| (top {min(k_compare, 10)}):",
        ] + [
            f"    Component {i}: {per_vec_cos[i]:.4f}"
            for i in range(min(k_compare, 10))
        ] + [
            f"  Speedup: {snap_time / gatac_time:.1f}x",
        ]

        log_path = os.path.join(os.path.dirname(__file__), "spectral_embedding.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            for result in results:
                print(result)
                f.write(result + '\n')

        # ---- Assertions ----
        assert eval_corr > 0.95, (
            f"Eigenvalue correlation too low: {eval_corr:.4f}"
        )
        assert mean_angle_deg < 30, (
            f"Subspace misalignment too large: {mean_angle_deg:.1f}°"
        )
        assert per_vec_cos[1] > 0.5, (
            f"First non-trivial eigenvector mismatch: cos={per_vec_cos[1]:.4f}"
        )

    else:
        # Log GATAC-only results
        results = [
            f"=== Spectral Embedding Benchmark ===",
            f"Matrix: {gatac_filt.shape[0]:,} cells × {gatac_filt.shape[1]:,} features",
            f"n_components: {n_comps}",
            f"",
            f"GATAC:",
            f"  Time: {gatac_time:.2f}s",
            f"  Eigenvalues (top 5): {list(gatac_evals[:5])}",
            f"",
            f"(SnapATAC2 comparison skipped)",
        ]

        log_path = os.path.join(os.path.dirname(__file__), "spectral_embedding.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            for result in results:
                print(result)
                f.write(result + '\n')

    print("\n✓ All assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC spectral embedding")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 spectral embedding and comparison"
    )
    args = parser.parse_args()

    test_spectral_embedding(run_gatac_only=args.run_gatac_only)
