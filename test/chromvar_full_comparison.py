#!/usr/bin/env python3
"""
GATAC chromVAR Implementation - Full Step-by-Step Comparison Script
This script compares each step of the chromVAR pipeline between GATAC and R.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
import logging
import sys
import os
import time

# Setup logging - Toned down
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import GATAC
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GATAC"))
import gatac as ga

# Output directories
r_output_dir = Path("chromvar_r_output")

# Timing
start_time = time.time()

if not r_output_dir.exists():
    print(f"Error: R output directory not found: {r_output_dir}")
    sys.exit(1)

# 1. GC Content Comparison
r_gc = pd.read_csv(r_output_dir / "gc_content.csv", index_col=0)
peak_seqs_df = pd.read_csv(r_output_dir / "peak_sequences.csv")

def compute_gc(seq):
    seq = seq.upper()
    if len(seq) == 0: return 0.0
    return (seq.count('G') + seq.count('C')) / len(seq)

peak_seqs_df['python_gc'] = peak_seqs_df['sequence'].apply(compute_gc)
peak_seqs_df.set_index('peak', inplace=True)
gc_comp = r_gc.join(peak_seqs_df[['python_gc']])
gc_corr = gc_comp.iloc[:,0].corr(gc_comp['python_gc'])

# 2. Results Loading
r_matches = pd.read_csv(r_output_dir / "motif_matches.csv", index_col=0)
count_mat = pd.read_csv(r_output_dir / "count_matrix.csv", index_col=0)

# 3. Background Peak Selection Comparison
adata = ad.AnnData(
    X=count_mat.T.values.astype(np.float32),
    obs=pd.DataFrame(index=count_mat.columns),
    var=pd.DataFrame(index=count_mat.index)
)
adata.var["gc_content"] = r_gc.loc[adata.var_names].values.flatten()
ga.tl.sample_bg_peaks(adata, method="chromvar", n_iterations=50)

r_bg = pd.read_csv(r_output_dir / "background_peaks.csv", index_col=0)
r_bg_indices = r_bg.values - 1

def get_bg_stats(bg_indices, bias_values):
    return bias_values[bg_indices].mean(axis=1)

r_bg_gc_mean = get_bg_stats(r_bg_indices, adata.var["gc_content"].values)
gatac_bg_gc_mean = get_bg_stats(adata.varm["bg_peaks"], adata.var["gc_content"].values)
gc_mean_corr = np.corrcoef(r_bg_gc_mean, gatac_bg_gc_mean)[0, 1]

r_reads = np.log10(count_mat.sum(axis=1).values)
gatac_reads = adata.var["reads_per_peak"].values
r_bg_reads_mean = get_bg_stats(r_bg_indices, r_reads)
gatac_bg_reads_mean = get_bg_stats(adata.varm["bg_peaks"], gatac_reads)
reads_mean_corr = np.corrcoef(r_bg_reads_mean, gatac_bg_reads_mean)[0, 1]

# 4. Deviation Scores Comparison
adata.varm["motif_match"] = r_matches.values.astype(bool)
adata.uns["motif_name"] = np.array(r_matches.columns)
adata.varm["bg_peaks"] = r_bg_indices.astype(np.int32)
dev_adata = ga.tl.chromvar(adata)

r_scores = pd.read_csv(r_output_dir / "deviation_scores.csv", index_col=0).T
common_cells = list(set(r_scores.index) & set(dev_adata.obs_names))
common_motifs = list(set(r_scores.columns) & set(dev_adata.var_names))
r_scores_aligned = r_scores.loc[common_cells, common_motifs]
gatac_scores_aligned = pd.DataFrame(dev_adata.X, index=dev_adata.obs_names, columns=dev_adata.var_names).loc[common_cells, common_motifs]
overall_corr = np.corrcoef(r_scores_aligned.values.flatten(), gatac_scores_aligned.values.flatten())[0, 1]

# Summary Output
end_time = time.time()
total_time = end_time - start_time
results = [
    f"GC Correlation:\t{gc_corr:.6f}",
    f"BG GC Correlation:\t{gc_mean_corr:.6f}",
    f"BG Reads Correlation:\t{reads_mean_corr:.6f}",
    f"Z-score Correlation:\t{overall_corr:.6f}",
    f"Total Time:\t{total_time:.2f}s"
]

log_file = os.path.splitext(sys.argv[0])[0] + ".log"
with open(log_file, 'w') as f:
    for result in results:
        print(result)
        f.write(result + '\n')
