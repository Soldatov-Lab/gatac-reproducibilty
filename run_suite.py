"""
Run the reproducibility tests in order, each in its own process.

A fresh process per test is what makes GATAC's cold start a real one.

    pixi run python run_suite.py                        # all tests, data in data/
    pixi run python run_suite.py --data-dir /fast/disk  # keep the data elsewhere
    pixi run python run_suite.py tss_enrichment lsi     # some tests, still in order
    pixi run python run_suite.py --rerun-references     # rerun the cached tools too

The data directory holds the inputs the tests cannot download, everything they
write and the cached reference outputs; single tests read it from
$GATAC_REPRO_DATA. Logs go to test/*.log.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# fragment_loading writes the fragment files the others read
TESTS = [
    "fragment_loading",
    "tss_enrichment",
    "tile_matrix",
    "gene_matrix",
    "feature_selection",
    "peak_calling",
    "make_peak_matrix",
    "motif_enrichment",
    "gsea_motif_enrichment",
    "chromvar_vignette",
    "amulet_doublet",
    "gene_score",
    "lsi",
    "iterative_lsi",
    "footprinting",
    "spectral_embedding",
]

# Flags that make a test rerun its cached reference tool
RERUN_FLAGS = {
    "motif_enrichment": ["--rerun-reference"],
    "gsea_motif_enrichment": ["--rerun-reference"],
    "chromvar_vignette": ["--rerun-reference"],
    "amulet_doublet": ["--rerun-amulet"],
    "gene_score": ["--regenerate"],
    "lsi": ["--regenerate"],
    "iterative_lsi": ["--regenerate"],
    "footprinting": ["--rerun-tobias"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("tests", nargs="*", metavar="TEST",
                        help=f"tests to run (default: all): {', '.join(TESTS)}")
    parser.add_argument("--data-dir", help="data directory (default: $GATAC_REPRO_DATA or data/)")
    parser.add_argument("--rerun-references", action="store_true",
                        help="rerun the reference tools (SnapATAC2, GSEApy, R chromVAR, AMULET, ArchR, TOBIAS) "
                             "instead of using their cached outputs")
    args = parser.parse_args()
    unknown = sorted(set(args.tests) - set(TESTS))
    if unknown:
        parser.error(f"unknown test(s): {', '.join(unknown)}")

    env = dict(os.environ)
    if args.data_dir:
        os.makedirs(args.data_dir, exist_ok=True)
        env["GATAC_REPRO_DATA"] = os.path.abspath(args.data_dir)
    print(f"Data directory: {env.get('GATAC_REPRO_DATA', os.path.join(HERE, 'data'))}")

    failed = []
    for test in [t for t in TESTS if t in args.tests or not args.tests]:
        cmd = [sys.executable, os.path.join("test", f"{test}.py")]
        if args.rerun_references:
            cmd += RERUN_FLAGS.get(test, [])
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=HERE, env=env).returncode
        print(f"=== {test}: {'ok' if rc == 0 else f'FAILED ({rc})'} in {time.perf_counter() - t0:.0f}s",
              flush=True)
        if rc:
            failed.append(test)

    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
