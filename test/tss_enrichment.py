"""
TSS enrichment test demonstrating GATAC matches SnapATAC2.
"""
import os
import sys
import time
import argparse
import snapatac2 as snap
import gatac as ga
import pandas as pd


def test_tss_enrichment(run_gatac_only=False):
    """Test GATAC TSS enrichment against SnapATAC2.
    
    Args:
        run_gatac_only: If True, run GATAC only (skip SnapATAC2) comparison
    """
    # Load data
    print("Loading data...")
    data = snap.read("data/pbmc.h5ad")

    # SnapATAC2 TSSe (optional)
    snap_time = None
    df_snap = None
    if not run_gatac_only:
        start_snap = time.time()
        snap.metrics.tsse(data, snap.genome.hg38)
        end_snap = time.time()
        snap_time = end_snap - start_snap

        df_snap = data.obs[:].to_pandas()
        df_snap.index = data.obs_names

    # GATAC TSSe
    GTF_FILE = snap.genome.hg38.annotation
    parquet_filtered = "data/atac_pbmc_5k_filtered.parquet"

    start_gatac = time.time()
    metrics = ga.pp.compute_metrics(parquet_filtered, GTF_FILE)
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac

    df_gatac = metrics.set_index("barcode").to_pandas()

    if not run_gatac_only:
        correlation = pd.concat([df_gatac.tsse_score, df_snap.tsse], axis=1).corr().iloc[0, 1]

        cell_idx = snap.pp.filter_cells(data, min_counts=5000, min_tsse=10, max_counts=100000, inplace=False)
        snap_cells = cell_idx.shape[0]

        metrics_filtered = metrics.query("n_unique >= 5000 and n_unique <= 100000 and tsse_score >= 10")
        gatac_cells = metrics_filtered.shape[0]

        results = [
            f"SnapATAC2:\t{snap_time:.2f}s",
            f"GATAC:\t{gatac_time:.2f}s",
            f"TSSe correlation:\t{correlation:.3f}",
            f"Cell Count Match:\t{snap_cells == gatac_cells}"
        ]

        log_path = os.path.join(os.path.dirname(__file__), "tss_enrichment.log")
        with open(log_path, 'w') as f:
            for result in results:
                print(result)
                f.write(result + '\n')

        # Assertions to verify TSSe quality
        assert correlation > 0.99, f"TSSe correlation too low: {correlation:.3f} (expected > 0.99)"
        assert snap_cells == gatac_cells, f"Cell count mismatch: Snap={snap_cells}, GATAC={gatac_cells}"
    else:
        results = [
            f"GATAC:\t{gatac_time:.2f}s",
            "(SnapATAC2 comparison skipped)"
        ]

        log_path = os.path.join(os.path.dirname(__file__), "tss_enrichment.log")
        with open(log_path, 'w') as f:
            for result in results:
                print(result)
                f.write(result + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC TSS enrichment")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 comparison"
    )
    args = parser.parse_args()
    
    test_tss_enrichment(run_gatac_only=args.run_gatac_only)

