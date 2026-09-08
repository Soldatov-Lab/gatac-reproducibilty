"""
LSI reproducibility test: GATAC (GPU) vs ArchR.

ArchR is the correctness *oracle*. This harness:

  1. Builds a binarized tile matrix from the fragment parquet already vendored
     for the gene-score test, using GATAC's own ``pp.make_tile_matrix``, and
     caches it.

  2. Hands that exact matrix to ArchR by running ``test/lsi_R.R``, which calls
     ``ArchR:::.computeLSI`` for all three ``LSIMethod`` variants and reproduces
     the variable-feature scoring of ``ArchR:::.identifyVarFeatures``. The
     handover is raw CSC index arrays: a cells x features CSR and a
     features x cells CSC are byte-identical, so nothing is transposed or
     reformatted and every difference is algorithmic.

  3. Compares GATAC against it:

       G0  ``tl.lsi`` vs ``.computeLSI``, all three methods -- per-component
           |r| and singular values.
       G2  ``tl.cluster_var_features`` vs ArchR's scoring, given the same
           clusters -- feature Jaccard, which should be exactly 1.

Deliberately **not** covered here: the end-to-end ``addIterativeLSI``
comparison (G3/G4). That needs an ArchR project with arrow files, which these
gates do not -- keeping this test cheap enough to run on every change. See
``iterative_lsi.py`` for the end-to-end gate.

Notes on why the comparison is set up the way it is:

  * ``outlierQuantiles = NULL`` on the ArchR side disables its depth-tail
    hold-out and ``.projectLSI`` step, so G0 isolates TF-IDF + SVD.
  * Zero-sum features are dropped before the handover, as ``.computeLSI``
    would drop them internally.
  * Comparison is **per component**, not just subspace overlap. The singular
    spectrum of ATAC TF-IDF data is nearly flat past the tenth component, which
    suggests per-component comparison is too strict; measured, it is not, as
    long as both sides use a converged solver (ArchR's irlba at ``tol = 1e-5``,
    float32 Lanczos here). Component-wise disagreement is a symptom of an
    under-converged solver, not of LSI.

Run:
    pixi run python test/lsi.py                 # build/validate oracle, compare
    pixi run python test/lsi.py --regenerate    # force re-running the R oracle
    pixi run python test/lsi.py --skip-gatac    # oracle only, never compare
    pixi run pytest test/lsi.py
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUTDIR = os.path.join(REPO, "data", "lsi_output")
FRAGMENTS = os.path.join(REPO, "data", "gene_score_output",
                         "fragments_archr.parquet")
TILE_H5AD = os.path.join(OUTDIR, "tile_matrix.h5ad")

N_COMPS = 30
N_VAR_FEATURES = 2000
N_CLUSTERS = 6
SCALE_TO = 1e4
TILE_SIZE = 5000
TIMING_REPEATS = 3

# G0: measured 1.0000 on every component and 2e-7 on the singular values, so
# these leave a wide margin while still failing on a real regression.
PERCOMP_CORR_MIN = 0.999
REL_SV_ERR_MAX = 1e-6
# G2: ArchR's scoring is deterministic given the clusters, so this is exact.
FEATURE_JACCARD_MIN = 1.0


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def build_tile_matrix(regenerate=False):
    """Binarized tile matrix from the vendored fragments, via GATAC."""
    import anndata as ad

    if os.path.exists(TILE_H5AD) and not regenerate:
        return ad.read_h5ad(TILE_H5AD)

    import gatac as ga

    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(FRAGMENTS):
        raise FileNotFoundError(
            f"{FRAGMENTS} not found. It is produced by the gene-score test; "
            "run test/gene_score.py first."
        )
    adata = ga.pp.make_tile_matrix(
        FRAGMENTS, chrom_sizes="hg38", tile_size=TILE_SIZE,
        min_fragments_per_cell=100, count_strategy="binarize",
    )
    adata.write_h5ad(TILE_H5AD)
    return adata


def export_for_oracle(X, clusters=None):
    """Dump the matrix (and optionally clusters) as raw arrays for R."""
    os.makedirs(OUTDIR, exist_ok=True)
    X = X.tocsr()
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    n_cell, n_feat = X.shape
    # cells x features CSR == features x cells CSC, same index arrays
    X.indices.tofile(os.path.join(OUTDIR, "indices.i32"))
    X.indptr.tofile(os.path.join(OUTDIR, "indptr.i32"))
    pd.DataFrame(
        [{"n_feat": n_feat, "n_cell": n_cell, "nnz": X.nnz, "k": N_COMPS,
          "n_var_features": N_VAR_FEATURES, "scale_to": SCALE_TO}]
    ).to_csv(os.path.join(OUTDIR, "shape.csv"), index=False)
    if clusters is not None:
        np.asarray(clusters, dtype=np.int32).tofile(
            os.path.join(OUTDIR, "clusters.i32")
        )


def run_oracle(mode):
    """
    Invoke test/lsi_R.R in the given mode.

    Goes through ``pixi run``, as the gene-score and AMULET tests do, so the R
    environment is guaranteed regardless of how this script was launched. A
    bare ``Rscript`` picks up whatever R is on PATH, which will not have ArchR
    unless the caller happened to use ``pixi run python``.
    """
    env = dict(os.environ, OUTDIR=OUTDIR, MODE=mode)
    cmd = ["pixi", "run", "Rscript", os.path.join(HERE, "lsi_R.R")]
    print(f"  running ArchR oracle (MODE={mode}) ...", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ArchR oracle (MODE={mode}) failed:\n{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def ensure_oracle(X, clusters, regenerate=False):
    """Build both oracle outputs if missing."""
    needed = [f"archr_lsi_{m}.csv" for m in (1, 2, 3)] + ["archr_varfeat.csv"]
    if regenerate or not all(
        os.path.exists(os.path.join(OUTDIR, f)) for f in needed
    ):
        export_for_oracle(X, clusters)
        run_oracle("computelsi")
        run_oracle("varfeat")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _percomp_corr(A, B):
    return np.array([abs(np.corrcoef(A[:, j], B[:, j])[0, 1])
                     for j in range(A.shape[1])])


def gatac_lsi_available():
    try:
        import gatac as ga
        return hasattr(ga.tl, "lsi") and hasattr(ga.tl, "cluster_var_features")
    except Exception:
        return False


def compare(X, clusters, results):
    """Run GATAC and score it against the oracle. Returns the metrics."""
    import anndata as ad

    import gatac as ga
    from gatac.tl.lsi import cluster_var_features

    metrics = {}

    results.append("")
    results.append("--- G0: tl.lsi vs ArchR:::.computeLSI ---")
    for m in (1, 2, 3):
        e_ref = np.loadtxt(os.path.join(OUTDIR, f"archr_lsi_{m}.csv"),
                           delimiter=",", skiprows=1)
        s_ref = np.loadtxt(os.path.join(OUTDIR, f"archr_sv_{m}.csv"),
                           delimiter=",", skiprows=1)
        t_ref = float(np.loadtxt(os.path.join(OUTDIR, f"archr_time_{m}.csv"),
                                 delimiter=",", skiprows=1))

        # Best of TIMING_REPEATS: the first call in a process pays one-off
        # CUDA JIT compilation, and this GPU is usually shared, so a single
        # sample understates the steady-state cost by several-fold.
        dt = float("inf")
        for _ in range(TIMING_REPEATS):
            a = ad.AnnData(X.copy())
            a.var["selected"] = True
            a.obs["n_unique"] = np.diff(X.tocsr().indptr).astype(float)
            t0 = time.perf_counter()
            ga.tl.lsi(a, N_COMPS, method=m, outlier_quantiles=None,
                      depth_cor_cutoff=None)
            dt = min(dt, time.perf_counter() - t0)

        r = _percomp_corr(a.obsm["X_lsi"], e_ref)
        rel = np.abs(a.uns["lsi"]["singular_values"] - s_ref) / s_ref
        metrics[f"percomp_{m}"] = float(r.min())
        metrics[f"relsv_{m}"] = float(rel.max())
        metrics[f"gatac_s_{m}"] = dt
        metrics[f"archr_s_{m}"] = t_ref
        results.append(
            f"  method {m}: per-component |r| min={r.min():.5f} "
            f"(<{PERCOMP_CORR_MIN}: {(r < PERCOMP_CORR_MIN).sum()}/{len(r)}) | "
            f"max rel sv err={rel.max():.2e} | "
            f"GATAC {dt:.2f}s vs ArchR {t_ref:.2f}s = x{t_ref / dt:.1f}"
        )

    results.append("")
    results.append("--- G2: cluster_var_features vs ArchR's scoring ---")
    ref_idx = np.loadtxt(os.path.join(OUTDIR, "archr_varfeat.csv"),
                         delimiter=",", skiprows=1).astype(int)
    ref_var = np.loadtxt(os.path.join(OUTDIR, "archr_var.csv"),
                         delimiter=",", skiprows=1)
    dt = float("inf")
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        sel, var = cluster_var_features(X, clusters, len(ref_idx),
                                        scale_to=SCALE_TO, binarize=True)
        dt = min(dt, time.perf_counter() - t0)
    metrics["varfeat_gatac_s"] = dt
    jac = len(np.intersect1d(sel, ref_idx)) / len(np.union1d(sel, ref_idx))
    var_err = float(np.abs(var - ref_var).max() / np.abs(ref_var).max())
    metrics["feature_jaccard"] = jac
    metrics["var_rel_err"] = var_err
    results.append(
        f"  feature Jaccard={jac:.6f} (identical sets: "
        f"{np.array_equal(sel, ref_idx)}) | variance max rel diff={var_err:.2e} "
        f"| GATAC {dt:.2f}s"
    )

    speedups = [metrics[f"archr_s_{m}"] / metrics[f"gatac_s_{m}"] for m in (1, 2, 3)]
    results.append("")
    results.append(
        f"Speedup over ArchR .computeLSI: x{min(speedups):.1f}-x{max(speedups):.1f} "
        f"across the three LSIMethod variants "
        f"({X.shape[0]:,} cells x {X.shape[1]:,} tiles, k={N_COMPS})"
    )
    return metrics


def sanity_check_oracle(X, results):
    results.append(
        f"Tile matrix: {X.shape[0]:,} cells x {X.shape[1]:,} tiles, "
        f"nnz={X.nnz:,} (binarized, tile_size={TILE_SIZE})"
    )
    for m in (1, 2, 3):
        e = np.loadtxt(os.path.join(OUTDIR, f"archr_lsi_{m}.csv"),
                       delimiter=",", skiprows=1)
        assert e.shape == (X.shape[0], N_COMPS), (
            f"oracle embedding for method {m} is {e.shape}, expected "
            f"{(X.shape[0], N_COMPS)}"
        )
        assert np.isfinite(e).all(), f"oracle embedding {m} has non-finite values"
    results.append("Oracle sanity checks passed (shape + finiteness).")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_lsi(skip_gatac=False, regenerate=False):
    results = ["=== LSI: ArchR oracle vs GATAC ==="]

    adata = build_tile_matrix(regenerate=regenerate)
    X = adata.X.tocsr()
    # Drop features with no signal, as .computeLSI does internally, so both
    # sides see the identical matrix.
    keep = np.asarray((X != 0).sum(axis=0)).ravel() > 0
    X = X[:, keep].tocsr()
    # Fixed, reproducible cluster labels: G2 is about the scoring given
    # clusters, not about the clustering, so they are injected rather than
    # computed. This is what `cluster_fn` exists for in the iterative test.
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, N_CLUSTERS, X.shape[0]).astype(np.int32)

    ensure_oracle(X, clusters, regenerate=regenerate)
    sanity_check_oracle(X, results)

    metrics = None
    if skip_gatac:
        results.append("(GATAC comparison skipped: --skip-gatac)")
    elif not gatac_lsi_available():
        results.append(
            "(GATAC comparison skipped: ga.tl.lsi not available. The oracle is "
            "built and validated — this test becomes the correctness gate once "
            "the port lands.)"
        )
    else:
        metrics = compare(X, clusters, results)

    log_path = os.path.join(HERE, "lsi.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        for line in results:
            print(line)
            fh.write(line + "\n")

    # Assertions after logging, so the log is always written.
    if metrics is not None:
        for m in (1, 2, 3):
            assert metrics[f"percomp_{m}"] > PERCOMP_CORR_MIN, (
                f"method {m}: per-component |r| {metrics[f'percomp_{m}']:.5f} "
                f"(expected > {PERCOMP_CORR_MIN})"
            )
            assert metrics[f"relsv_{m}"] < REL_SV_ERR_MAX, (
                f"method {m}: max rel sv err {metrics[f'relsv_{m}']:.2e} "
                f"(expected < {REL_SV_ERR_MAX})"
            )
        assert metrics["feature_jaccard"] >= FEATURE_JACCARD_MIN, (
            f"feature Jaccard {metrics['feature_jaccard']:.6f} "
            f"(expected {FEATURE_JACCARD_MIN})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSI: ArchR oracle vs GATAC")
    parser.add_argument("--skip-gatac", action="store_true",
                        help="Build/validate the ArchR oracle only")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force re-running the ArchR R oracle even if cached")
    args = parser.parse_args()
    test_lsi(skip_gatac=args.skip_gatac, regenerate=args.regenerate)
    sys.exit(0)
