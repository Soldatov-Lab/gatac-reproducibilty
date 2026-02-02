#!/usr/bin/env python3
"""
GATAC vs chromVAR Comparison Script
Compares GATAC's motif scanning and chromVAR scores with R's chromVAR/motifmatchr.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
import sys
import os
import time
import scipy.sparse as sp

# Import GATAC
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GATAC"))
import gatac as ga

# Paths
base_dir = Path(__file__).parent.parent
r_output_dir = base_dir / "chromvar_r_output"
meme_path = r_output_dir / "test_motifs.meme"
genome_path = base_dir / "data/hg19.fa.gz"

if not r_output_dir.exists():
    print(f"Error: R output directory not found: {r_output_dir}")
    sys.exit(1)

if not meme_path.exists():
    print(f"Error: MEME file not found: {meme_path}")
    sys.exit(1)

if not genome_path.exists():
    print(f"Error: Genome file not found: {genome_path}")
    sys.exit(1)

# 1. Load R results for Motif Matches
print("Loading R motif matches...")
r_matches = pd.read_csv(r_output_dir / "motif_matches.csv", index_col=0)
if r_matches.iloc[0,0] in ["FALSE", "TRUE"]:
    r_matches = r_matches == "TRUE"
print(f"R matches shape: {r_matches.shape}")

# 2. Setup GATAC scan
print("Reading motifs...")
motifs = ga.tl.read_motifs(str(meme_path), unique=False)
print(f"Loaded {len(motifs)} motifs")

# Load R Count Matrix and Peaks to build AnnData
print("Loading R count matrix and peaks...")
r_counts_raw = pd.read_csv(r_output_dir / "count_matrix.csv", index_col=0)
# count_matrix.csv is peaks x cells
# GATAC AnnData expects cells x peaks
counts = r_counts_raw.values.T
cell_names = r_counts_raw.columns
peak_names = r_counts_raw.index

# Create AnnData
adata = ad.AnnData(
    X=sp.csr_matrix(counts.astype(np.float32)),
    obs=pd.DataFrame(index=cell_names),
    var=pd.DataFrame(index=peak_names)
)
print(f"Created AnnData: {adata.shape}")

# Run Motif Scan
bg_probs = (0.25, 0.25, 0.25, 0.25)
print(f"Scanning {len(motifs)} motifs in {adata.n_vars} peaks...")
start_scan = time.time()
ga.tl.scan_motifs(
    adata, motifs, str(genome_path), 
    pvalue=5e-5, 
    bg_probs=bg_probs, 
    coordinate_system="1-based"
)
end_scan = time.time()
print(f"Scanning took {end_scan - start_scan:.2f}s")

# 3. Compare Motif Matches
gatac_matches_sparse = adata.varm["motif_match"]
gatac_matches = pd.DataFrame(
    gatac_matches_sparse.toarray(),
    index=adata.var_names,
    columns=adata.uns["motif_name"]
)

r_motif_names = r_matches.columns
gatac_motif_names = gatac_matches.columns

print("\nComparing motif matches...")
motif_comp = []
for motif_name in r_motif_names:
    matching_col = [c for c in gatac_motif_names if c in motif_name or motif_name in c]
    if not matching_col:
        continue
    g_col = matching_col[0]
    r_vec = r_matches[motif_name].values
    g_vec = gatac_matches[g_col].values
    tp = np.sum(r_vec & g_vec)
    fp = np.sum(~r_vec & g_vec)
    fn = np.sum(r_vec & ~g_vec)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0 if np.sum(r_vec) == 0 else 0
    motif_comp.append({"motif": motif_name, "Jaccard": jaccard, "GATAC_count": np.sum(g_vec), "R_count": np.sum(r_vec)})

motif_comp_df = pd.DataFrame(motif_comp)
print(f"Mean Motif Match Jaccard: {motif_comp_df['Jaccard'].mean():.4f}")

# 4. Load R Background Peaks
print("\nLoading R background peaks...")
# background_peaks.csv has 1 column for peaks and 50 columns for bg indices
# R indices are 1-based
r_bg_peaks = pd.read_csv(r_output_dir / "background_peaks.csv", index_col=0)
bg_peaks_idx = r_bg_peaks.values - 1 # Convert to 0-based
adata.varm["bg_peaks"] = bg_peaks_idx.astype(np.int32)
print(f"Loaded background peaks: {adata.varm['bg_peaks'].shape}")

# 5. Run GATAC chromVAR
print("\nRunning GATAC chromVAR...")
start_cv = time.time()
dev_adata = ga.tl.chromvar(adata)
end_cv = time.time()
print(f"GATAC chromVAR took {end_cv - start_cv:.2f}s")

# 5b. Run GATAC chromVAR using R's motif matches for perfect comparison
print("\nRunning GATAC chromVAR using R's motif matches...")
adata_r_matches = adata.copy()

# Reorder R matches to match adata.var_names and align motif names
r_matches_aligned = r_matches.loc[adata.var_names]
# GATAC expects a sparse boolean matrix in varm["motif_match"]
adata_r_matches.varm["motif_match"] = sp.csr_matrix(r_matches_aligned.values.astype(bool))
adata_r_matches.uns["motif_name"] = np.array(r_matches_aligned.columns)

dev_adata_r_matches = ga.tl.chromvar(adata_r_matches)

# 6. Load R Deviation Scores
print("Loading R deviation scores...")
r_devs_raw = pd.read_csv(r_output_dir / "deviation_scores.csv", index_col=0)
# R devs: motifs x cells -> cells x motifs
r_devs = r_devs_raw.T 

# 7. Compare Deviation Scores
print("\nComparing chromVAR Deviation Scores...")
# Align motifs
common_motifs = []
correlations_gatac_scan = []
correlations_r_scan = []

for r_col in r_devs.columns:
    # Compare with GATAC-scanned motifs
    matching_g = [g for g in dev_adata.var_names if g in r_col or r_col in g]
    # Compare with R-scanned motifs (direct GATAC chromvar comparison)
    matching_r = [r for r in dev_adata_r_matches.var_names if r == r_col]
    
    if matching_g and matching_r:
        g_col = matching_g[0]
        r_vals = r_devs[r_col].values
        
        g_scan_vals = dev_adata[:, g_col].X.flatten()
        r_scan_vals = dev_adata_r_matches[:, r_col].X.flatten()
        
        corr_g = np.corrcoef(r_vals, g_scan_vals)[0, 1]
        corr_r = np.corrcoef(r_vals, r_scan_vals)[0, 1]
        
        correlations_gatac_scan.append(corr_g)
        correlations_r_scan.append(corr_r)
        common_motifs.append(r_col)

comparison_results = pd.DataFrame({
    "motif": common_motifs,
    "pearson_GATAC_scan": correlations_gatac_scan,
    "pearson_R_scan": correlations_r_scan
})

print(comparison_results.to_string(index=False))
print(f"\nMean Pearson (GATAC scan): {comparison_results['pearson_GATAC_scan'].mean():.4f}")
print(f"Mean Pearson (R scan):     {comparison_results['pearson_R_scan'].mean():.4f}")

# Export final comparison
comparison_results.to_csv("chromvar_full_comparison.csv", index=False)
print(f"\nSaved full comparison to chromvar_full_comparison.csv")

if comparison_results['pearson_R_scan'].mean() > 0.99:
    print("\nSUCCESS: GATAC chromVAR implementation matches R exactly given same inputs!")
elif comparison_results['pearson_GATAC_scan'].mean() > 0.95:
    print("\nSUCCESS: GATAC chromVAR scores highly correlate with R implementation!")
else:
    print("\nWARNING: Differences found. Motif scanning differences account for some, but check chromVAR implementation too.")

