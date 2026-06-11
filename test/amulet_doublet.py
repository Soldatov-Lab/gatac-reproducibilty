"""
AMULET doublet detection test (GATAC vs original AMULET v1.1).

Compares GATAC's ``ga.pp.detect_doublets`` against the original AMULET v1.1
release on the canonical 10x Genomics PBMC 5k fragment file
(``snap.datasets.pbmc5k()``), checking both runtime and result consistency
(q-value correlation, Jaccard of doublet calls).

Both tools read the same canonical CellRanger ``fragments.tsv.gz`` — GATAC
via a one-time parquet conversion, AMULET directly. The count column is
ignored by both (each row = one fragment), so semantics are identical.

The original AMULET v1.1 source is downloaded on first use via the pixi
task ``amulet-setup`` (which fetches the v1.1 release from GitHub and
extracts it to ``data/AMULET-v1.1/``, a gitignored project-local path).
The ``AMULET_V11_DIR`` env var can override the location for users who
already have the v1.1 release installed elsewhere. AMULET v1.1 runs in
a dedicated pixi env (``amulet``) with pinned older numpy/pandas
(Python 3.11, numpy<1.24, pandas<2.0) so the unpatched v1.1 code runs
without modification.

Usage::

    pixi run python test/amulet_doublet.py            # full comparison
    pixi run python test/amulet_doublet.py --run-gatac-only  # GATAC only

The ``amulet`` env auto-installs on first use. To pre-install both envs:
``pixi install --all`` or ``pixi run install-all``.
"""
import os
import sys
import time
import shutil
import subprocess
import argparse
import tempfile
import duckdb
import pandas as pd
import gatac as ga


AMULET_DIR = os.environ.get("AMULET_V11_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "AMULET-v1.1",
)
PIXI = shutil.which("pixi") or "pixi"


def _ensure_amulet_v11():
    """Download the original AMULET v1.1 release if not already present.

    The pixi task ``amulet-setup`` downloads and extracts the v1.1 release
    from GitHub into ``data/AMULET-v1.1/`` (a gitignored, project-local path).
    The ``AMULET_V11_DIR`` env var can override the location for users who
    already have the v1.1 release installed elsewhere.
    """
    if os.path.isfile(os.path.join(AMULET_DIR, "AMULET.py")):
        return
    print(f"AMULET v1.1 not found at {AMULET_DIR}, downloading...")
    subprocess.run([PIXI, "run", "amulet-setup"], check=True)
    if not os.path.isfile(os.path.join(AMULET_DIR, "AMULET.py")):
        raise FileNotFoundError(
            f"AMULET v1.1 still not found at {AMULET_DIR} after `pixi run "
            f"amulet-setup`. Set the AMULET_V11_DIR env var to point at a "
            f"v1.1 install, or run `pixi run amulet-setup` manually."
        )


def _pixi_run_amulet(args, **kwargs):
    """Run a command in the `amulet` pixi env (auto-installs on first use)."""
    return subprocess.run(
        [PIXI, "run", "--environment", "amulet", "python", *args],
        check=True,
        **kwargs,
    )


def _prepare_amulet_inputs(fragments_tsv_gz, workdir, min_fragments=200):
    """Build the singlecell.csv and chrs.txt AMULET requires.

    AMULET v1.1's ``FragmentFileOverlapCounter.py`` reads:
      - ``fragments.tsv.gz``: 5 columns (chr, start, end, barcode, count).
        AMULET only uses the first 4 columns, ignoring ``count`` (so it sees
        one fragment per row). This matches the canonical CellRanger output
        and is the documented AMULET usage.
      - ``singlecell.csv``: columns ``barcode, is__cell_barcode`` (header).
        We mark every barcode with >= min_fragments as ``is__cell_barcode=1``.
      - ``chrs.txt``: one chromosome per line. We use the autosomes list
        shipped with AMULET v1.1 (excludes chrX, chrY, and decoy contigs).

    Returns (singlecell_csv, chrs_txt, chroms).
    """
    singlecell_path = os.path.join(workdir, "singlecell.csv")
    chrs_path = os.path.join(workdir, "chrs.txt")

    # Use the autosomes list shipped with AMULET v1.1 (matches the published
    # tool's default behaviour, excludes chrX/chrY and decoys).
    shutil.copy(os.path.join(AMULET_DIR, "human_autosomes.txt"), chrs_path)
    with open(chrs_path) as f:
        chroms = [line.strip() for line in f if line.strip()]
    chrom_list_sql = ",".join(f"'{c}'" for c in chroms)

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT DISTINCT column3 AS barcode, 1 AS is__cell_barcode
            FROM read_csv(
                '{fragments_tsv_gz}',
                delim='\t', header=false, comment='#', ignore_errors=true
            )
            WHERE column0 IN ({chrom_list_sql})
            GROUP BY column3
            HAVING COUNT(*) >= {min_fragments}
        ) TO '{singlecell_path}' (FORMAT CSV, HEADER TRUE);
    """)
    con.close()

    return singlecell_path, chrs_path, chroms


def _run_amulet_original(fragments_tsv_gz, singlecell_csv, chrs_txt, outdir):
    """Run AMULET v1.1 step1 + step2 in the amulet env.

    Returns dict with timing, doublet set, and per-barcode q-values.
    """
    os.makedirs(outdir, exist_ok=True)

    # Step 1: FragmentFileOverlapCounter
    t0 = time.perf_counter()
    _pixi_run_amulet([
        f"{AMULET_DIR}/FragmentFileOverlapCounter.py",
        "--maxinsertsize", "900",
        "--expectedoverlap", "2",
        "--startbases", "0",
        "--endbases", "0",
        fragments_tsv_gz, singlecell_csv, chrs_txt, outdir,
    ])
    step1_time = time.perf_counter() - t0

    # Step 2: AMULET.py (no --rfilter; GATAC infers repeats internally)
    t0 = time.perf_counter()
    _pixi_run_amulet([
        f"{AMULET_DIR}/AMULET.py",
        "--expectedoverlap", "2",
        "--q", "0.01",
        os.path.join(outdir, "Overlaps.txt"),
        os.path.join(outdir, "OverlapSummary.txt"),
        outdir,
    ])
    step2_time = time.perf_counter() - t0

    probs = pd.read_csv(os.path.join(outdir, "MultipletProbabilities.txt"), sep="\t")
    n_cells = len(probs)
    doublet_set = set(probs.loc[probs["q-value"] < 0.01, "barcode"].astype(str))
    n_doublets = len(doublet_set)
    doublet_rate = 100.0 * n_doublets / n_cells if n_cells > 0 else 0.0

    return {
        "step1_time": step1_time,
        "step2_time": step2_time,
        "total_time": step1_time + step2_time,
        "n_cells": n_cells,
        "n_doublets": n_doublets,
        "doublet_rate": doublet_rate,
        "doublet_set": doublet_set,
        "q_by_barcode": dict(zip(probs["barcode"].astype(str), probs["q-value"])),
    }


def test_amulet_doublet(run_gatac_only=False):
    """Test GATAC AMULET doublet detection against the original AMULET v1.1.

    Args:
        run_gatac_only: If True, skip the original AMULET v1.1 run and
            comparison. Useful when the amulet pixi env is unavailable.
    """
    # Canonical 10x Genomics PBMC 5k fragment file (CellRanger tsv.gz, ~1 GB)
    import snapatac2 as snap
    fragments_tsv_gz = snap.datasets.pbmc5k()
    print(f"Using fragment file: {fragments_tsv_gz}")

    min_fragments = 200
    q_threshold = 0.01

    # Verify the original AMULET v1.1 install is reachable (only needed if
    # the user did not pass --run-gatac-only). Auto-download via the
    # `amulet-setup` pixi task if missing.
    if not run_gatac_only:
        _ensure_amulet_v11()

    # Both GATAC and AMULET v1.1 use autosomes only by default (AMULET was
    # never designed for non-autosomes — see AMULET paper Methods).

    # ------------------------------------------------------------------
    # 1. GATAC (default env)
    # ------------------------------------------------------------------
    # GATAC reads parquet; convert the canonical tsv.gz once and cache.
    parquet_path = os.path.join(
        os.path.dirname(__file__), "amulet_doublet_input.parquet"
    )
    if not os.path.exists(parquet_path):
        print("Converting tsv.gz -> parquet (one-time, ~30s)...")
        parquet_path = str(ga.pp.convert.make_parquet(fragments_tsv_gz, parquet_path))

    print("Running GATAC AMULET doublet detection...")
    t0 = time.perf_counter()
    gatac_result = ga.pp.detect_doublets(
        fragment_path=parquet_path,
        chrom_sizes="hg38",
        min_fragments=min_fragments,
        q_threshold=q_threshold,
        n_threads=4,
    )
    gatac_time = time.perf_counter() - t0

    gatac_out = os.path.join(os.path.dirname(__file__), "amulet_doublet_gatac.csv")
    gatac_result.to_csv(gatac_out, index=False)

    gatac_n_cells = len(gatac_result)
    gatac_n_doublets = int(gatac_result["is_doublet"].sum())
    gatac_doublet_rate = 100.0 * gatac_n_doublets / gatac_n_cells
    gatac_doublet_set = set(
        gatac_result.loc[gatac_result["is_doublet"], "cell_id"].astype(str)
    )
    gatac_q_by_barcode = dict(
        zip(
            gatac_result["cell_id"].astype(str),
            gatac_result["q_value"].astype(float),
        )
    )

    # ------------------------------------------------------------------
    # 2. Original AMULET v1.1 (amulet env, if requested)
    # ------------------------------------------------------------------
    amulet_stats = None
    if not run_gatac_only:
        print("Running original AMULET v1.1 (amulet env)...")
        with tempfile.TemporaryDirectory(prefix="amulet_v11_") as workdir:
            singlecell_csv, chrs_txt, _chroms = _prepare_amulet_inputs(
                fragments_tsv_gz, workdir, min_fragments=min_fragments
            )
            amulet_outdir = os.path.join(workdir, "out")
            amulet_stats = _run_amulet_original(
                fragments_tsv_gz, singlecell_csv, chrs_txt, amulet_outdir
            )
            shutil.copy(
                os.path.join(amulet_outdir, "MultipletProbabilities.txt"),
                os.path.join(os.path.dirname(__file__), "amulet_doublet_v1.1.txt"),
            )

    # ------------------------------------------------------------------
    # 3. Build results list + log
    # ------------------------------------------------------------------
    results = [
        "=== AMULET Doublet Detection (GATAC vs original v1.1) ===",
        f"Fragment file: {fragments_tsv_gz}",
        f"Chromosomes:   22 autosomes (chr1-22, AMULET is designed for autosomes only)",
        f"min_fragments: {min_fragments}",
        f"q threshold:   {q_threshold}",
        "",
        "GATAC (default env):",
        f"  Cells tested:        {gatac_n_cells:,}",
        f"  Doublets detected:   {gatac_n_doublets:,}",
        f"  Doublet rate:        {gatac_doublet_rate:.2f}%",
        f"  Time:                {gatac_time:.2f}s",
        f"  Output:              {gatac_out}",
    ]

    speedup = None
    jaccard = None
    q_pearson = None
    if amulet_stats is not None:
        results += [
            "",
            "AMULET v1.1 original (amulet env, Python 3.11, numpy<1.24):",
            f"  Cells tested:        {amulet_stats['n_cells']:,}",
            f"  Doublets detected:   {amulet_stats['n_doublets']:,}",
            f"  Doublet rate:        {amulet_stats['doublet_rate']:.2f}%",
            f"  Time step1:          {amulet_stats['step1_time']:.2f}s",
            f"  Time step2:          {amulet_stats['step2_time']:.2f}s",
            f"  Time total:          {amulet_stats['total_time']:.2f}s",
        ]
        speedup = amulet_stats["total_time"] / gatac_time
        results.append(f"  Speedup (GATAC vs AMULET v1.1): {speedup:.1f}x")

        if amulet_stats["doublet_set"] or gatac_doublet_set:
            inter = gatac_doublet_set & amulet_stats["doublet_set"]
            union = gatac_doublet_set | amulet_stats["doublet_set"]
            jaccard = len(inter) / len(union) if union else 0.0
            results += [
                "",
                "Result comparison:",
                f"  Doublet set intersection:   {len(inter):,}",
                f"  Doublet set union:          {len(union):,}",
                f"  Jaccard index:              {jaccard:.4f}",
            ]

        common = set(gatac_q_by_barcode) & set(amulet_stats["q_by_barcode"])
        if len(common) > 10:
            g_q = pd.Series([gatac_q_by_barcode[b] for b in common])
            a_q = pd.Series([amulet_stats["q_by_barcode"][b] for b in common])
            if g_q.std() > 0 and a_q.std() > 0:
                q_pearson = float(g_q.corr(a_q))
                results.append(
                    f"  q-value Pearson r (n={len(common):,} common barcodes): {q_pearson:.4f}"
                )

    log_path = os.path.join(os.path.dirname(__file__), "amulet_doublet.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in results:
            print(line)
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # 4. Assertions
    # ------------------------------------------------------------------
    assert "cell_id" in gatac_result.columns
    assert "q_value" in gatac_result.columns
    assert "is_doublet" in gatac_result.columns
    assert gatac_result["q_value"].between(0, 1).all()
    assert gatac_n_cells > 1000
    assert 0.5 < gatac_doublet_rate < 20.0, (
        f"GATAC doublet rate {gatac_doublet_rate:.2f}% outside [0.5%, 20%]"
    )

    if amulet_stats is not None:
        assert amulet_stats["n_cells"] > 500
        assert 0.1 < amulet_stats["doublet_rate"] < 25.0, (
            f"AMULET v1.1 doublet rate {amulet_stats['doublet_rate']:.2f}% outside [0.1%, 25%]"
        )
        assert speedup > 1.5, f"GATAC speedup {speedup:.1f}x too low (expected > 1.5x)"
        assert jaccard is not None and jaccard > 0.5, (
            f"Jaccard {jaccard:.4f} too low (expected > 0.5)"
        )
        if q_pearson is not None:
            assert q_pearson > 0.5, (
                f"q-value correlation {q_pearson:.4f} too low (expected > 0.5)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GATAC AMULET doublet detection vs original AMULET v1.1"
    )
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Skip the original AMULET v1.1 run (use when amulet pixi env unavailable)",
    )
    args = parser.parse_args()
    test_amulet_doublet(run_gatac_only=args.run_gatac_only)
