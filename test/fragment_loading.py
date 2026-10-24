"""
Fragment loading test demonstrating GATAC matches SnapATAC2.

This test shows that with the chromosome filtering fix, GATAC produces
the same cell count as SnapATAC2 when using the same parameters.
"""

import argparse
import os
import time

import snapatac2 as snap
import gatac as ga

from _paths import data_path
from _timing import N_CPUS, cold_warm


def test_fragment_loading(run_gatac_only=False):
    # Get the same fragment file used by both
    fragment_file = snap.datasets.pbmc5k()

    # GATAC processing
    def run_gatac():
        t0 = time.perf_counter()
        parquet = ga.io.make_parquet(fragment_file, output_path=data_path("atac_pbmc_5k.parquet"))
        convert_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        filtered = ga.pp.filter_fragments(
            parquet,
            min_fragments_per_cell=200,
            chrom_sizes="hg38"
        )
        return filtered, convert_time, time.perf_counter() - t0

    (filtered, convert_time, filter_time), gatac_cold, _ = cold_warm(run_gatac)
    gatac_time = convert_time + filter_time

    # Barcodes are ids in the file; its barcode table lists the cells kept
    gatac_cells = ga.io.FragmentFile(filtered).n_barcodes

    results = []
    if not run_gatac_only:
        # SnapATAC2 processing
        t0 = time.perf_counter()
        data = snap.pp.import_fragments(
            fragment_file,
            chrom_sizes=snap.genome.hg38,
            file=data_path("pbmc.h5ad"),
            sorted_by_barcode=False,
            min_num_fragments=200,
            n_jobs=N_CPUS,
        )
        snap_cells = data.n_obs
        snap_time = time.perf_counter() - t0
        results.append(f"SnapATAC2:\t{snap_time:.2f}s")

    results += [
        f"GATAC:\t{gatac_time:.2f}s",
        f"  make_parquet:\t{convert_time:.2f}s",
        f"  filter_fragments:\t{filter_time:.2f}s",
        f"GATAC cold start:\t{gatac_cold:.2f}s",
        f"GATAC cells:\t{gatac_cells:,}",
    ]
    if not run_gatac_only:
        results += [
            f"Match:\t{snap_cells == gatac_cells}",
            f"Speedup: {snap_time / gatac_time:.1f}x",
        ]

    log_path = os.path.join(os.path.dirname(__file__), "fragment_loading.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in results:
            print(line)
            f.write(line + "\n")

    if not run_gatac_only:
        assert snap_cells == gatac_cells, f"Cell count differs: SnapATAC2 {snap_cells:,}, GATAC {gatac_cells:,}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC fragment loading")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 run and comparison",
    )
    args = parser.parse_args()
    test_fragment_loading(run_gatac_only=args.run_gatac_only)
