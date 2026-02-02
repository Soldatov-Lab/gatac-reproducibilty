"""
Motif enrichment test comparing GATAC vs SnapATAC2.

Following the same pattern as peak_calling.py (timing and saving to log file).
Uses GATAC for peak calling, SnapATAC2 for marker finding, then compares
motif enrichment between both tools.
"""
import sys
import time
import numpy as np
import snapatac2 as snap
import gatac as ga
from gatac.tl.motif import DNAMotif, read_motifs
from scipy.stats import pearsonr


if __name__ == '__main__':
    # Load data
    print("Loading annotated PBMC 5k dataset...")
    data = snap.read(snap.datasets.pbmc5k(type="annotated_h5ad"), backed=None)

    # Focus on subset of cell types for faster testing
    SELECTED_CELL_TYPES = ['CD4 Memory', 'CD14 Mono', 'NK']
    data = data[data.obs['cell_type'].isin(SELECTED_CELL_TYPES)].copy()
    print(f"Filtered to {len(data)} cells from {SELECTED_CELL_TYPES}")

    # Call peaks using GATAC (faster than MACS3)
    print("Calling peaks with GATAC...")
    parquet_path = "data/atac_pbmc_5k_filtered.parquet"
    ga.tl.call_peaks(
        data,
        groupby='cell_type',
        parquet_path=parquet_path,
        genome='hg38',
        key_added='gmacs',
        verbose=False,
    )
    
    # Merge peaks and store in uns['peaks'] for make_peak_matrix
    print("Merging peaks...")
    merged_peaks = ga.tl.merge_peaks(data.uns['gmacs'], chrom_sizes='hg38')
    data.uns['peaks'] = merged_peaks
    print(f"Merged peaks: {len(merged_peaks)} regions")

    # Create peak matrix using GATAC
    print("Creating peak matrix with GATAC...")
    peak_adata = ga.tl.make_peak_matrix(
        data,
        parquet_path=parquet_path,
        use_rep='peaks',
        genome='hg38',
        inplace=False,
    )
    
    # Copy cell type annotations
    peak_adata.obs['cell_type'] = data.obs['cell_type']

    # Find marker regions using SnapATAC2
    print("Finding marker regions...")
    markers = snap.tl.marker_regions(peak_adata, groupby='cell_type', pvalue=0.05)
    
    # Check if markers were found, if not use top peaks per group
    total_markers = sum(len(v) for v in markers.values())
    if total_markers == 0:
        print("  No markers found with pvalue=0.05, using top variable peaks per cell type...")
        from scipy.stats import zscore as scipy_zscore
        
        # Aggregate by cell type and pick top variable peaks  
        count = snap.tl.aggregate_X(peak_adata, 'cell_type', normalize='RPKM')
        z = scipy_zscore(np.log2(1 + count.X), axis=0)
        
        markers = {}
        for i, cell_type in enumerate(count.obs_names):
            # Select top 500 peaks per cell type
            top_idx = np.argsort(z[i, :])[-500:]
            markers[cell_type] = list(count.var_names[top_idx])
    
    for cell_type, regions in markers.items():
        print(f"  {cell_type}: {len(regions)} marker regions")

    # Load motifs - use the same MEME file for both tools
    print("Loading CIS-BP motifs...")
    from snapatac2.datasets import register_datasets
    meme_path = register_datasets().fetch('cisBP_human.meme')
    
    # Load for SnapATAC2 (uses built-in function)
    snap_motifs = snap.datasets.cis_bp(unique=True)
    
    # Load for GATAC (use read_motifs from the same file)
    gatac_motifs = read_motifs(meme_path, unique=True)
    print(f"Loaded {len(snap_motifs)} SnapATAC2 motifs, {len(gatac_motifs)} GATAC motifs")

    # Get genome FASTA path
    genome_fasta = snap.genome.hg38.fasta
    print(f"Using genome: {genome_fasta}")

    # Run SnapATAC2 motif enrichment
    print("\nRunning SnapATAC2 motif enrichment...")
    start_snap = time.time()
    snap_results = snap.tl.motif_enrichment(snap_motifs, markers, genome_fasta)
    end_snap = time.time()
    snap_time = end_snap - start_snap

    # Run GATAC motif enrichment
    print("\nRunning GATAC motif enrichment...")
    start_gatac = time.time()
    gatac_results = ga.tl.motif_enrichment(gatac_motifs, markers, genome_fasta)
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac

    # Compare results
    print("\nComparing results...")
    correlations = {}
    for cell_type in markers.keys():
        snap_df = snap_results[cell_type].sort('id')
        gatac_df = gatac_results[cell_type].sort('id')
        
        # Get fold changes (filter out inf values for correlation)
        snap_fc = np.array(snap_df['log2(fold change)'].to_list())
        gatac_fc = np.array(gatac_df['log2(fold change)'].to_list())
        
        # Filter valid values for correlation
        valid_mask = np.isfinite(snap_fc) & np.isfinite(gatac_fc)
        if valid_mask.sum() > 2:
            corr, _ = pearsonr(snap_fc[valid_mask], gatac_fc[valid_mask])
            correlations[cell_type] = corr
        else:
            correlations[cell_type] = np.nan

    # Calculate speedup
    speedup = snap_time / gatac_time if gatac_time > 0 else float('inf')
    avg_corr = np.nanmean(list(correlations.values()))

    # Print and save results
    results = [
        f"SnapATAC2:\t{snap_time:.2f}s",
        f"GATAC:\t{gatac_time:.2f}s",
        f"Speedup:\t{speedup:.1f}x",
        f"Avg Correlation:\t{avg_corr:.3f}",
    ]
    for cell_type, corr in correlations.items():
        results.append(f"Corr {cell_type}:\t{corr:.3f}")

    with open(sys.argv[0].replace('.py', '.log'), 'w') as f:
        for result in results:
            print(result)
            f.write(result + '\n')
