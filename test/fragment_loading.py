"""
Fragment loading test demonstrating GATAC matches SnapATAC2.

This test shows that with the chromosome filtering fix, GATAC produces
the same cell count as SnapATAC2 when using the same parameters.
"""

import time
import snapatac2 as snap
import gatac as ga
import cudf
import sys

# Get the same fragment file used by both
fragment_file = snap.datasets.pbmc5k()


# SnapATAC2 processing
start_snap = time.time()
data = snap.pp.import_fragments(
    fragment_file,
    chrom_sizes=snap.genome.hg38,
    file="data/pbmc.h5ad",
    sorted_by_barcode=False,
    min_num_fragments=200
)
snap_cells = data.n_obs
end_snap = time.time()
snap_time = end_snap - start_snap

# GATAC processing
start_gatac = time.time()
parquet = ga.pp.convert.make_parquet(
    fragment_file, 
    output_path="data/atac_pbmc_5k.parquet"
)

filtered = ga.pp.filter_fragments(
    parquet,
    min_fragments_per_cell=200,
    chrom_sizes="hg38"
)

df_filtered = cudf.read_parquet(filtered)
gatac_cells = df_filtered['barcode'].nunique()
end_gatac = time.time()
gatac_time = end_gatac - start_gatac

# Results

results = [
    f"SnapATAC2:\t{snap_time:.2f}s",
    f"GATAC:\t{gatac_time:.2f}s",
    f"Match:\t{snap_cells == gatac_cells}"
]


with open(sys.argv[0].split(".")[0] + ".log", 'w') as f:
    for result in results:
        print(result)
        f.write(result + '\n')

