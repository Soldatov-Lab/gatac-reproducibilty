"""
Peak calling test demonstrating GATAC matches SnapATAC2.

GATAC's ``peak.call`` runs MACS3 on the GPU with SnapATAC2's settings, so each
group's summits (start, end, summit, score, fold enrichment, p- and q-values)
must be those of ``snap.tl.macs3``, and the merged peaks those of
``snap.tl.merge_peaks``.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import snapatac2 as snap

import gatac as ga

from _timing import N_CPUS, cold_warm
from _paths import data_path

SELECTED_CELL_TYPES = ['CD4 Memory', 'CD14 Mono', 'NK']
SUMMIT_KEY = ["chrom", "start", "end", "peak"]
SUMMIT_VALUES = ["score", "signal_value", "p_value", "q_value"]


def _summit_differences(gatac_df: pd.DataFrame, snap_df: pd.DataFrame) -> tuple[int, int, int]:
    """(summits only GATAC calls, summits only SnapATAC2 calls, shared summits with any value differing)."""
    snap_df = snap_df.astype({"start": np.int64, "end": np.int64, "peak": np.int64, "score": np.int64})
    m = gatac_df[SUMMIT_KEY + SUMMIT_VALUES].merge(
        snap_df[SUMMIT_KEY + SUMMIT_VALUES], on=SUMMIT_KEY, how="outer",
        suffixes=("", "_snap"), indicator=True)
    both = m[m["_merge"] == "both"]
    differ = np.zeros(len(both), dtype=bool)
    for c in SUMMIT_VALUES:
        differ |= both[c].to_numpy() != both[c + "_snap"].to_numpy()
    return int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum()), int(differ.sum())


def test_peak_calling(run_gatac_only=False):
    """Test GATAC peak calling against SnapATAC2.

    Args:
        run_gatac_only: If True, run GATAC only (skip SnapATAC2 peak calling and comparison)
    """
    # Load data
    print("Loading annotated PBMC 5k dataset...")
    data = snap.read(snap.datasets.pbmc5k(type="annotated_h5ad"), backed=None)

    # Focus on subset of cell types for faster testing
    data = data[data.obs['cell_type'].isin(SELECTED_CELL_TYPES)].copy()

    # GATAC peak calling, first: run after SnapATAC2's MACS3 in the same
    # process, its first group took ~5x longer (71 s instead of 14 s)
    parquet_path = data_path("atac_pbmc_5k_filtered.parquet")

    def run_gatac():
        ga.peak.call(
            data,
            groupby='cell_type',
            parquet_path=parquet_path,
            genome='hg38',
            key_added='gmacs',
            verbose=False,
        )
        return ga.peak.merge(data.uns['gmacs'], chrom_sizes='hg38')

    gatac_merged, gatac_cold, gatac_time = cold_warm(run_gatac)
    gatac_df = gatac_merged[['chrom', 'start', 'end']].copy()

    # Save GATAC peaks to BED file for later use
    gatac_df.to_csv(data_path("gatac_peaks.bed"), sep='\t', header=False, index=False)
    print(f"Saved {len(gatac_df):,} GATAC peaks to {data_path('gatac_peaks.bed')}")

    results = [f"GATAC:\t{gatac_time:.2f}s", f"GATAC cold start:\t{gatac_cold:.2f}s"]
    if run_gatac_only:
        results += [f"Peaks called:\t{len(gatac_df):,}", "(SnapATAC2 comparison skipped)"]
    else:
        start_snap = time.time()
        snap.tl.macs3(data, groupby='cell_type', n_jobs=N_CPUS)
        snap_merged = snap.tl.merge_peaks(data.uns['macs3'], snap.genome.hg38).to_pandas()
        snap_time = time.time() - start_snap
        coords = snap_merged['Peaks'].str.extract(r'^(.+):(\d+)-(\d+)$')
        snap_df = pd.DataFrame({'chrom': coords[0], 'start': coords[1].astype(np.int64),
                                'end': coords[2].astype(np.int64)})

        results.insert(0, f"SnapATAC2:\t{snap_time:.2f}s")
        results.append(f"Speedup:\t{snap_time / gatac_time:.1f}x")
        results.append(f"Speedup (cold start):\t{snap_time / gatac_cold:.1f}x")
        per_group = {}
        for group in SELECTED_CELL_TYPES:
            only_gatac, only_snap, differ = _summit_differences(data.uns['gmacs'][group], data.uns['macs3'][group])
            per_group[group] = (only_gatac, only_snap, differ)
            results.append(f"{group}:\t{len(data.uns['gmacs'][group]):,} summits (SnapATAC2 "
                           f"{len(data.uns['macs3'][group]):,}); only GATAC {only_gatac}, "
                           f"only SnapATAC2 {only_snap}, values differ {differ}")
        merged = gatac_df.merge(snap_df, how='outer', indicator=True)
        merged_only_gatac = int((merged['_merge'] == 'left_only').sum())
        merged_only_snap = int((merged['_merge'] == 'right_only').sum())
        results.append(f"Merged:\t{len(gatac_df):,} peaks (SnapATAC2 {len(snap_df):,}); "
                       f"only GATAC {merged_only_gatac}, only SnapATAC2 {merged_only_snap}")

    log_path = os.path.join(os.path.dirname(__file__), "peak_calling.log")
    with open(log_path, 'w') as f:
        for result in results:
            print(result)
            f.write(result + '\n')

    if not run_gatac_only:
        # Same MACS3 computation: every summit and every value must match
        for group, (only_gatac, only_snap, differ) in per_group.items():
            assert only_gatac == only_snap == differ == 0, (
                f"{group}: {only_gatac} summits only in GATAC, {only_snap} only in SnapATAC2, "
                f"{differ} with differing values")
        assert merged_only_gatac == merged_only_snap == 0, (
            f"merged peaks differ: {merged_only_gatac} only in GATAC, {merged_only_snap} only in SnapATAC2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC peak calling")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 peak calling and comparison"
    )
    args = parser.parse_args()

    test_peak_calling(run_gatac_only=args.run_gatac_only)
