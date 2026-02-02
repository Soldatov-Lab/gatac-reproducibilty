"""
Peak calling test demonstrating GATAC matches SnapATAC2.
Includes peak calling and peak matrix construction benchmarking.
"""
import sys
import time
import numpy as np
import snapatac2 as snap
import gatac as ga
import pandas as pd
from intervaltree import IntervalTree

if __name__ == '__main__':
    # Load data
    print("Loading annotated PBMC 5k dataset...")
    data = snap.read(snap.datasets.pbmc5k(type="annotated_h5ad"), backed=None)

    # Focus on subset of cell types for faster testing
    SELECTED_CELL_TYPES = ['CD4 Memory', 'CD14 Mono', 'NK']
    data = data[data.obs['cell_type'].isin(SELECTED_CELL_TYPES)].copy()

    # SnapATAC2 peak calling
    start_snap = time.time()
    snap.tl.macs3(data, groupby='cell_type')
    snapatac_merged = snap.tl.merge_peaks(data.uns['macs3'], snap.genome.hg38)
    snapatac_merged = snapatac_merged.to_pandas()
    # Parse Peaks column to get chrom, start, end
    snap_peaks = []
    for peak in snapatac_merged['Peaks']:
        chrom, rest = peak.split(':')
        start, end = rest.split('-')
        snap_peaks.append({'chrom': chrom, 'start': int(start), 'end': int(end)})
    snapatac_df = pd.DataFrame(snap_peaks)
    end_snap = time.time()
    snap_time = end_snap - start_snap

    # GATAC peak calling
    parquet_path = "data/atac_pbmc_5k_filtered.parquet"
    start_gatac = time.time()
    ga.tl.call_peaks(
        data,
        groupby='cell_type',
        parquet_path=parquet_path,
        genome='hg38',
        key_added='gmacs',
        verbose=False,
    )
    gatac_merged = ga.tl.merge_peaks(data.uns['gmacs'], chrom_sizes='hg38')
    gatac_df = gatac_merged[['chrom', 'start', 'end']].copy()
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac

    # Calculate overlap metrics
    def calculate_overlap_intervals(peaks1, peaks2):
        trees1 = {}
        for _, row in peaks1.iterrows():
            chrom = str(row['chrom'])
            if chrom not in trees1:
                trees1[chrom] = IntervalTree()
            trees1[chrom][row['start']:row['end']] = True
        
        overlap_count = 0
        for _, row in peaks2.iterrows():
            chrom = str(row['chrom'])
            if chrom in trees1 and trees1[chrom].overlaps(row['start'], row['end']):
                overlap_count += 1
        
        n1 = len(peaks1)
        n2 = len(peaks2)
        jaccard = overlap_count / (n1 + n2 - overlap_count) if (n1 + n2 - overlap_count) > 0 else 0
        recall_1 = overlap_count / n1 if n1 > 0 else 0
        recall_2 = overlap_count / n2 if n2 > 0 else 0
        
        return jaccard, recall_1, recall_2

    jaccard, recall_snap, recall_gatac = calculate_overlap_intervals(snapatac_df, gatac_df)

    results = [
        f"SnapATAC2:\t{snap_time:.2f}s",
        f"GATAC:\t{gatac_time:.2f}s",
        f"Jaccard:\t{jaccard:.3f}",
        f"Recall Snap:\t{recall_snap:.3f}",
        f"Recall GATAC:\t{recall_gatac:.3f}"
    ]

    with open(sys.argv[0].split(".")[0] + ".log", 'w') as f:
        for result in results:
            print(result)
            f.write(result + '\n')