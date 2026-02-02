"""
TSS enrichment test demonstrating GATAC matches SnapATAC2.
"""
import sys
import time
import snapatac2 as snap
import gatac as ga
import pandas as pd


data = snap.read("data/pbmc.h5ad")

start_snap = time.time()
snap.metrics.tsse(data, snap.genome.hg38)
end_snap = time.time()
snap_time = end_snap - start_snap

df_snap = data.obs[:].to_pandas()
df_snap.index = data.obs_names


GTF_FILE = snap.genome.hg38.annotation
parquet_filtered = "data/atac_pbmc_5k_filtered.parquet"

start_gatac = time.time()
tss_df = ga.pp.load_tss_from_gtf(GTF_FILE)
metrics = ga.pp.compute_metrics(parquet_filtered,tss_df)
end_gatac = time.time()
gatac_time = end_gatac - start_gatac

df_gatac = metrics.set_index("barcode").to_pandas()


correlation = pd.concat([df_gatac.tsse_score,df_snap.tsse],axis=1).corr().iloc[0,1]

cell_idx = snap.pp.filter_cells(data, min_counts=5000, min_tsse=10, max_counts=100000,inplace=False)
snap_cells = cell_idx.shape[0]

metrics = metrics.query("n_unique >= 5000 and n_unique <= 100000 and tsse_score >= 10")
gatac_cells = metrics.shape[0]

results = [
    f"SnapATAC2:\t{snap_time:.2f}s",
    f"GATAC:\t{gatac_time:.2f}s",
    f"TSSe correlation:\t{correlation:.3f}",
    f"Cell Count Match:\t{snap_cells == gatac_cells}"
]

with open(sys.argv[0].split(".")[0] + ".log", 'w') as f:
    for result in results:
        print(result)
        f.write(result + '\n')

