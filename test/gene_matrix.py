"""
Gene matrix test comparing GATAC vs SnapATAC2.
"""
import sys
import time
import signal
import numpy as np
from scipy import stats
import snapatac2 as snap
import gatac as ga


# ============================================================================
# Configuration
# ============================================================================
PARQUET_FILE = "data/atac_pbmc_5k_filtered.parquet"
SNAP_H5AD = "data/pbmc.h5ad"

# Use SnapATAC2's built-in annotation for fair comparison
GTF_FILE = str(snap.genome.hg38.annotation)

# Timeout for GATAC (20 seconds)
GATAC_TIMEOUT = 30

# ============================================================================
# SnapATAC2
# ============================================================================
snap_data = snap.read(SNAP_H5AD)

start_snap = time.time()
snap_gene = snap.pp.make_gene_matrix(snap_data, gene_anno=snap.genome.hg38)
end_snap = time.time()
snap_time = end_snap - start_snap

# ============================================================================
# GATAC (with timeout)
# ============================================================================
class GATACTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise GATACTimeoutError(f"GATAC gene matrix took longer than {GATAC_TIMEOUT}s!")

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(GATAC_TIMEOUT)

try:
    start_gatac = time.time()
    gatac_gene = ga.pp.make_gene_matrix(
        PARQUET_FILE,
        gene_anno=GTF_FILE,
        upstream=2000,
        downstream=0,
        include_gene_body=True,
    )
    end_gatac = time.time()
    gatac_time = end_gatac - start_gatac
finally:
    # Disable the alarm
    signal.alarm(0)

# ============================================================================
# Comparison
# ============================================================================
# Align cells to common set
snap_gene = snap_gene.to_memory()
common_cells = list(set(snap_gene.obs_names) & set(gatac_gene.obs_names))
snap_gene = snap_gene[common_cells]
gatac_gene = gatac_gene[common_cells]

# Align genes to common set
common_genes = list(set(snap_gene.var_names) & set(gatac_gene.var_names))
snap_gene = snap_gene[:, common_genes]
gatac_gene = gatac_gene[:, common_genes]

# Per-cell sums
gatac_sum = np.asarray(gatac_gene.X.sum(axis=1)).flatten()
snap_sum = np.asarray(snap_gene.X.sum(axis=1)).flatten()

# Correlation
correlation = stats.pearsonr(gatac_sum, snap_sum)[0] if len(snap_sum) > 1 else 0

# Results
results = [
    f"SnapATAC2:\t{snap_time:.2f}s",
    f"GATAC:\t{gatac_time:.2f}s",
    f"Common cells:\t{len(common_cells)}",
    f"Common genes:\t{len(common_genes)}",
    f"GATAC genes total:\t{gatac_gene.shape[1]}",
    f"SnapATAC2 genes total:\t{snap_gene.shape[1]}",
    f"GATAC sum:\t{gatac_sum.sum():.0f}",
    f"SnapATAC2 sum:\t{snap_sum.sum():.0f}",
    f"Per-cell correlation:\t{correlation:.6f}",
]

with open(sys.argv[0].replace(".py", ".log"), 'w') as f:
    for result in results:
        print(result)
        f.write(result + '\n')
