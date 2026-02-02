"""
Tile matrix test demonstrating GATAC matches SnapATAC2.
"""
import sys
import time
import snapatac2 as snap
import gatac as ga
import pandas as pd


snap_tile = snap.read("data/pbmc.h5ad")

start_snap = time.time()
snap.pp.add_tile_matrix(snap_tile)
end_snap = time.time()
snap_time = end_snap - start_snap


# Use the fixed parquet file that includes all fragments from the source TSV
parquet_filtered = "data/atac_pbmc_5k_filtered.parquet"

start_gatac = time.time()
gatac_tile = ga.pp.make_tile_matrix(parquet_filtered, chrom_sizes="hg38", tile_size=500) 
end_gatac = time.time()
gatac_time = end_gatac - start_gatac

snap_tile = snap_tile.to_memory()
gatac_tile = gatac_tile[snap_tile.obs_names]

gatac_sum = gatac_tile.X.sum(axis=1).A1
snap_sum = snap_tile.X.sum(axis=1).A1

results = [
    f"SnapATAC2:\t{snap_time:.2f}s",
    f"GATAC:\t{gatac_time:.2f}s",
    f"GATAC sum:\t{gatac_sum.sum()}",
    f"SnapATAC2 sum:\t{snap_sum.sum()}",
    f"Match:\t{(gatac_sum == snap_sum).all()}"
]

with open(sys.argv[0].split(".")[0] + ".log", 'w') as f:
    for result in results:
        print(result)
        f.write(result + '\n')




