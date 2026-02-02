"""
Peak matrix benchmark comparing GATAC and SnapATAC2.

Uses GATAC peaks (from peak_calling.py) and compares construction
speed and matrix properties between both tools.
"""
import sys
import time
import numpy as np
import snapatac2 as snap
import gatac as ga
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    # File paths
    PEAK_FILE = Path("data/gatac_peaks.bed")
    H5AD_FILE = Path("data/pbmc.h5ad")
    PARQUET_FILE = Path("data/atac_pbmc_5k.parquet")
    
    # Validate inputs exist
    if not PEAK_FILE.exists():
        print(f"ERROR: Peak file not found: {PEAK_FILE}")
        print("Run peak_calling.py first to generate GATAC peaks.")
        sys.exit(1)
    if not H5AD_FILE.exists():
        print(f"ERROR: H5AD file not found: {H5AD_FILE}")
        print("Run fragment_loading.py first to import fragments.")
        sys.exit(1)
    if not PARQUET_FILE.exists():
        print(f"ERROR: Parquet file not found: {PARQUET_FILE}")
        print("Run fragment_loading.py first to create parquet file.")
        sys.exit(1)
    
    # Load peaks
    peaks_df = pd.read_csv(
        PEAK_FILE,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end']
    )
    n_peaks = len(peaks_df)
    print(f"Loaded {n_peaks:,} peaks from {PEAK_FILE}")
    
    # SnapATAC2: Load data and create peak matrix
    print("\n--- SnapATAC2 Peak Matrix ---")
    snap_data = snap.read(H5AD_FILE, backed='r')
    print(f"Loaded h5ad with {snap_data.n_obs:,} cells")
    
    # Format peaks for SnapATAC2 (chr:start-end format)
    snap_peaks = [f"{r['chrom']}:{r['start']}-{r['end']}" for _, r in peaks_df.iterrows()]
    
    start_snap = time.time()
    snap_peak_mat = snap.pp.make_peak_matrix(
        snap_data,
        use_rep=snap_peaks,
        file="data/snap_peak_matrix.h5ad",
    )
    end_snap = time.time()
    snap_time = end_snap - start_snap
    
    snap_shape = snap_peak_mat.shape
    snap_nnz = snap_peak_mat.X[:].nnz if hasattr(snap_peak_mat.X[:], 'nnz') else np.count_nonzero(snap_peak_mat.X[:])
    print(f"Shape: {snap_shape}")
    print(f"Non-zero: {snap_nnz:,}")
    print(f"Time: {snap_time:.2f}s")
    
    # GATAC: Create peak matrix from parquet
    print("\n--- GATAC Peak Matrix ---")
    
    import cudf
    from anndata import AnnData as AD
    
    # Filter fragments
    print("Filtering fragments for GATAC...")
    FILTERED_PARQUET = ga.pp.filter_fragments(
        PARQUET_FILE,
        min_fragments_per_cell=200,
        chrom_sizes="hg38"
    )
    
    # Read parquet to get unique barcodes
    print("Reading filtered parquet file to get barcodes...")
    frag_df = cudf.read_parquet(str(FILTERED_PARQUET), columns=['barcode'])
    unique_barcodes = frag_df['barcode'].unique().to_pandas().tolist()
    del frag_df
    
    # Create minimal AnnData with matching barcodes
    obs_df = pd.DataFrame(index=unique_barcodes)
    tile_data = AD(obs=obs_df)
    print(f"Created AnnData with {tile_data.n_obs:,} cells from parquet")
    
    # Store peaks in uns for make_peak_matrix
    tile_data.uns['peaks'] = peaks_df
    
    start_gatac = time.time()
    gatac_peak_mat = ga.tl.make_peak_matrix(
        tile_data,
        parquet_path=str(FILTERED_PARQUET),
        use_rep='peaks',
        inplace=False,
        verbose=True,
    )
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac
    
    gatac_shape = gatac_peak_mat.shape
    gatac_nnz = gatac_peak_mat.X.nnz if hasattr(gatac_peak_mat.X, 'nnz') else np.count_nonzero(gatac_peak_mat.X)
    print(f"Shape: {gatac_shape}")
    print(f"Non-zero: {gatac_nnz:,}")
    print(f"Time: {gatac_time:.2f}s")
    
    # Comparison summary
    print("\n--- Comparison Summary ---")
    shape_match = snap_shape == gatac_shape
    
    # Calculate overlap of non-zero entries if shapes match
    if shape_match:
        # Align cells by barcode (they may be in different orders)
        snap_obs = list(snap_peak_mat.obs_names) if hasattr(snap_peak_mat.obs_names, '__iter__') else snap_peak_mat.obs_names.tolist()
        gatac_obs = list(gatac_peak_mat.obs_names) if hasattr(gatac_peak_mat.obs_names, '__iter__') else gatac_peak_mat.obs_names.tolist()
        
        # Align peaks by name (they may be in different orders)
        snap_var = list(snap_peak_mat.var_names) if hasattr(snap_peak_mat.var_names, '__iter__') else snap_peak_mat.var_names.tolist()
        gatac_var = list(gatac_peak_mat.var_names) if hasattr(gatac_peak_mat.var_names, '__iter__') else gatac_peak_mat.var_names.tolist()
        
        # Find common cells and peaks
        common_cells = list(set(snap_obs) & set(gatac_obs))
        common_peaks = list(set(snap_var) & set(gatac_var))
        print(f"Common cells: {len(common_cells):,}")
        print(f"Common peaks: {len(common_peaks):,}")
        
        # Build index mappings using dicts for O(1) lookup instead of O(n) list.index()
        snap_obs_map = {c: i for i, c in enumerate(snap_obs)}
        gatac_obs_map = {c: i for i, c in enumerate(gatac_obs)}
        snap_var_map = {p: i for i, p in enumerate(snap_var)}
        gatac_var_map = {p: i for i, p in enumerate(gatac_var)}
        
        snap_cell_idx = [snap_obs_map[c] for c in common_cells]
        gatac_cell_idx = [gatac_obs_map[c] for c in common_cells]
        snap_peak_idx = [snap_var_map[p] for p in common_peaks]
        gatac_peak_idx = [gatac_var_map[p] for p in common_peaks]
        
        # Get aligned matrices (aligned on both axes)
        snap_X = snap_peak_mat.X[snap_cell_idx, :][:, snap_peak_idx]
        gatac_X = gatac_peak_mat.X[gatac_cell_idx, :][:, gatac_peak_idx]
        
        # Compare non-zero counts on aligned matrices
        snap_aligned_nnz = snap_X.nnz if hasattr(snap_X, 'nnz') else np.count_nonzero(snap_X)
        gatac_aligned_nnz = gatac_X.nnz if hasattr(gatac_X, 'nnz') else np.count_nonzero(gatac_X)
        print(f"Aligned SnapATAC2 non-zero: {snap_aligned_nnz:,}")
        print(f"Aligned GATAC non-zero: {gatac_aligned_nnz:,}")
        
        # Both are likely sparse - compute correlation of counts
        snap_sum = np.asarray(snap_X.sum(axis=0)).flatten()
        gatac_sum = np.asarray(gatac_X.sum(axis=0)).flatten()
        
        # Pearson correlation of peak totals
        corr = np.corrcoef(snap_sum, gatac_sum)[0, 1]
        print(f"Peak count correlation: {corr:.4f}")
        
        # Cell-level comparison
        snap_cell_sum = np.asarray(snap_X.sum(axis=1)).flatten()
        gatac_cell_sum = np.asarray(gatac_X.sum(axis=1)).flatten()
        cell_corr = np.corrcoef(snap_cell_sum, gatac_cell_sum)[0, 1]
        print(f"Cell count correlation: {cell_corr:.4f}")
    else:
        print(f"Shape mismatch: SnapATAC2={snap_shape}, GATAC={gatac_shape}")
    
    # Results summary
    results = [
        f"Peaks:\t{n_peaks:,}",
        f"SnapATAC2:\t{snap_time:.2f}s\t{snap_shape}\t{snap_nnz:,} nnz",
        f"GATAC:\t{gatac_time:.2f}s\t{gatac_shape}\t{gatac_nnz:,} nnz",
        f"Speedup:\t{snap_time/gatac_time:.2f}x",
        f"Shape Match:\t{shape_match}",
    ]
    
    if shape_match:
        results.append(f"Peak Corr:\t{corr:.4f}")
        results.append(f"Cell Corr:\t{cell_corr:.4f}")
    
    log_file = sys.argv[0].replace(".py", ".log")
    with open(log_file, 'w') as f:
        for result in results:
            print(result)
            f.write(result + '\n')
    
    print(f"\nResults saved to {log_file}")
