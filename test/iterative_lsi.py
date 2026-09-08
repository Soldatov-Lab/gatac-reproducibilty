"""
Iterative LSI reproducibility test: GATAC (GPU) vs ArchR `addIterativeLSI`.

The end-to-end counterpart to ``test/lsi.py``. Where that test gates the
individual stages against ArchR and needs no arrow file, this one runs the
whole loop -- iterative feature selection, clustering, re-decomposition --
against a real ``ArchRProject``.

  1. ``test/iterative_lsi_R.R`` builds an arrow file with a TileMatrix from the
     canonical 10x PBMC 5k fragments (the same file the AMULET test
     downloads), runs ``addIterativeLSI`` at **two** seeds, and exports the
     TileMatrix plus both references. Arrow creation takes tens of minutes, so
     everything is cached; ``--regenerate`` forces a rebuild.

  2. GATAC's ``tl.iterative_lsi`` runs on the *exported* TileMatrix, so both
     sides consume identical input and every difference is algorithmic.

  3. Two things are scored:

       G3  GATAC vs ArchR seed 1 -- feature Jaccard and mean subspace
           correlation, against absolute thresholds.
       G4  the same, relative to **ArchR's own seed-2-vs-seed-1 agreement**.

Why G4 exists, and why it is a ratio rather than a strict inequality:
``addIterativeLSI`` is not a stable function of its input. Its own
seed-to-seed agreement varies enormously with dataset size -- measured at
feature Jaccard 0.93 on this ~4.4k-cell fixture but 0.12 on a 643-cell one.
An absolute threshold is therefore meaningless on small data (nothing,
including ArchR, can pass it), and a strict "GATAC must beat ArchR's own
spread" is unreasonably hard on large data, where ArchR is very
self-consistent. So G3 carries the absolute check on a fixture big enough for
it to mean something, and G4 asks that GATAC land within a documented fraction
of ArchR's self-agreement.

Known unmodelled differences, both documented rather than fixed:
  * GATAC clusters with cuGraph Leiden, ArchR with Seurat's Louvain via
    ``FindClusters``. On a shared embedding these agree at ARI ~0.95, not 1.0.
  * Seurat's ``n.start = 10`` restarts have no cuGraph equivalent.
``filterBias`` was ruled out as a contributor: rerunning the oracle with
``filterBias = FALSE`` produced byte-identical output on this fixture.

Run:
    pixi run python test/iterative_lsi.py               # cached oracle, compare
    pixi run python test/iterative_lsi.py --regenerate  # rebuild the oracle
    pixi run python test/iterative_lsi.py --skip-gatac  # oracle only
    pixi run pytest test/iterative_lsi.py
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUTDIR = os.environ.get(
    "ITERLSI_OUTDIR", os.path.join(REPO, "data", "iterative_lsi_output")
)
R_SCRIPT = os.path.join(HERE, "iterative_lsi_R.R")

# G3: absolute thresholds. Measured 0.895 / 0.945 on the ~4.4k-cell fixture.
FEATURE_JACCARD_MIN = 0.80
SUBSPACE_CORR_MIN = 0.89
# G4: GATAC must land within this fraction of ArchR's own seed-to-seed
# agreement. Measured 0.965 (Jaccard) and 0.975 (correlation), so 0.90 leaves
# margin while still catching a real regression. See the module docstring for
# why this is a ratio rather than a strict inequality.
ARCHR_SELF_FRACTION_MIN = 0.90


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
def fragment_file():
    """The canonical 10x PBMC 5k fragments, as the AMULET test obtains them."""
    import snapatac2 as snap

    return str(snap.datasets.pbmc5k())


def ensure_oracle(regenerate=False):
    needed = ["tile_shape.csv", "tile_i.i32", "tile_p.i32", "tile_features.csv",
              "cells.csv", "archr_iterlsi_params.csv",
              "archr_iterlsi_seed1_matSVD.csv", "archr_iterlsi_seed1_features.csv",
              "archr_iterlsi_seed2_matSVD.csv", "archr_iterlsi_seed2_features.csv"]
    if not regenerate and all(
        os.path.exists(os.path.join(OUTDIR, f)) for f in needed
    ):
        print("Using cached ArchR oracle in", OUTDIR)
        return
    print("Generating the ArchR oracle via iterative_lsi_R.R. This creates an "
          "arrow file and runs addIterativeLSI twice — expect tens of minutes.")
    os.makedirs(OUTDIR, exist_ok=True)
    env = dict(os.environ, OUTDIR=OUTDIR, FRAGMENTS=fragment_file())
    # Through pixi, as the gene-score and AMULET tests do: a bare Rscript picks
    # up whatever R is on PATH, which will not have ArchR.
    subprocess.run(["pixi", "run", "Rscript", R_SCRIPT],
                   cwd=REPO, env=env, check=True)


def load_oracle():
    """The exported TileMatrix as cells x features CSR, plus the references."""
    shape = pd.read_csv(os.path.join(OUTDIR, "tile_shape.csv")).iloc[0]
    n_feat, n_cell, nnz = int(shape.n_feat), int(shape.n_cell), int(shape.nnz)
    i = np.fromfile(os.path.join(OUTDIR, "tile_i.i32"), dtype=np.int32, count=nnz)
    p = np.fromfile(os.path.join(OUTDIR, "tile_p.i32"), dtype=np.int32,
                    count=n_cell + 1)
    # features x cells CSC and cells x features CSR share their index arrays
    X = sp.csr_matrix((np.ones(nnz, np.float32), i, p.astype(np.int64)),
                      shape=(n_cell, n_feat))
    feats = pd.read_csv(os.path.join(OUTDIR, "tile_features.csv"))
    keys = (feats["seqnames"].astype(str) + ":"
            + feats["idx"].astype(int).astype(str)).to_numpy()
    cells = pd.read_csv(os.path.join(OUTDIR, "cells.csv"))
    params = pd.read_csv(os.path.join(OUTDIR, "archr_iterlsi_params.csv")).iloc[0]
    return X, keys, cells, params


def _feature_rows(csv, key_to_row, results):
    df = pd.read_csv(os.path.join(OUTDIR, csv))
    keys = df["seqnames"].astype(str) + ":" + df["idx"].astype(int).astype(str)
    rows = [key_to_row[k] for k in keys if k in key_to_row]
    if len(rows) < len(keys):
        results.append(
            f"  note: {len(keys) - len(rows)} of {len(keys)} features in {csv} "
            "are outside the exported accessibility pool"
        )
    return np.sort(np.asarray(rows))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def jaccard(a, b):
    return len(np.intersect1d(a, b)) / len(np.union1d(a, b))


def subspace_corr(A, B):
    """Canonical correlations between two embeddings: (min, mean)."""
    Qa, _ = np.linalg.qr(A - A.mean(0))
    Qb, _ = np.linalg.qr(B - B.mean(0))
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)
    return float(s.min()), float(s.mean())


def sanity_check_oracle(X, cells, params, results):
    results.append(
        f"TileMatrix: {X.shape[0]:,} cells x {X.shape[1]:,} tiles, nnz={X.nnz:,}"
    )
    results.append(
        f"ArchR params: varFeatures={int(params.var_features)}, "
        f"totalFeatures={int(params.total_features)}, "
        f"dims={int(params.n_dims)}, iterations={int(params.iterations)}"
    )
    assert X.shape[0] == len(cells), "cell table does not match the matrix"
    for sd in (1, 2):
        e = pd.read_csv(os.path.join(OUTDIR, f"archr_iterlsi_seed{sd}_matSVD.csv"),
                        index_col=0)
        assert e.shape[1] == int(params.n_dims), (
            f"seed {sd} embedding has {e.shape[1]} dims, expected "
            f"{int(params.n_dims)}"
        )
        assert np.isfinite(e.to_numpy()).all(), f"seed {sd} embedding not finite"
    results.append("Oracle sanity checks passed (shape, dims, finiteness).")


def gatac_available():
    try:
        import gatac as ga
        return hasattr(ga.tl, "iterative_lsi")
    except Exception:
        return False


def compare(X, keys, cells, params, results):
    import anndata as ad

    import gatac as ga

    key_to_row = {k: r for r, k in enumerate(keys)}
    f1 = _feature_rows("archr_iterlsi_seed1_features.csv", key_to_row, results)
    f2 = _feature_rows("archr_iterlsi_seed2_features.csv", key_to_row, results)
    e1 = pd.read_csv(os.path.join(OUTDIR, "archr_iterlsi_seed1_matSVD.csv"),
                     index_col=0)
    e2 = pd.read_csv(os.path.join(OUTDIR, "archr_iterlsi_seed2_matSVD.csv"),
                     index_col=0)

    # ArchR drops cells with no signal in its own selected features and does
    # not project them back, so its embedding can cover fewer cells than the
    # matrix. Compare on the cells all three runs share.
    order = cells["cell"].to_numpy()
    shared = [c for c in order if c in e1.index and c in e2.index]
    if len(shared) < len(order):
        results.append(
            f"  note: ArchR returned {len(shared)} of {len(order)} cells; "
            "comparing on the shared set"
        )
    pos = {c: i for i, c in enumerate(order)}
    rows = np.asarray([pos[c] for c in shared])
    a1 = e1.loc[shared].to_numpy()
    a2 = e2.loc[shared].to_numpy()

    a = ad.AnnData(X.copy())
    a.obs["n_unique"] = cells["nFrags"].to_numpy(dtype=float)
    t0 = time.perf_counter()
    ga.tl.iterative_lsi(
        a, int(params.n_dims), iterations=int(params.iterations),
        n_features=int(params.var_features),
        total_features=int(params.total_features),
        method=int(params.lsi_method), scale_to=float(params.scale_to),
        filter_quantile=float(params.filter_quantile),
        cluster_params={"resolution": float(params.resolution),
                        "max_clusters": int(params.max_clusters)},
        random_state=1,
    )
    dt = time.perf_counter() - t0
    fg = np.where(a.var["selected_iterative_lsi"].to_numpy())[0]
    eg = a.obsm["X_iterative_lsi"][rows]

    k = min(eg.shape[1], a1.shape[1])
    g3_jac = jaccard(fg, f1)
    _, g3_corr = subspace_corr(eg[:, :k], a1[:, :k])
    g4_jac = jaccard(f2, f1)
    _, g4_corr = subspace_corr(a2[:, :k], a1[:, :k])

    t_ref = None
    tpath = os.path.join(OUTDIR, "archr_iterlsi_seed1_time.csv")
    if os.path.exists(tpath):
        t_ref = float(np.loadtxt(tpath, delimiter=",", skiprows=1))

    results.append("")
    results.append("--- G3/G4: iterative_lsi vs ArchR addIterativeLSI ---")
    results.append(f"  GATAC selected {len(fg)} features in {dt:.1f}s"
                   + (f" (ArchR: {t_ref:.1f}s = x{t_ref / dt:.1f})"
                      if t_ref else ""))
    results.append("                          feature Jaccard   mean subspace corr")
    results.append(f"  GATAC vs ArchR seed1        {g3_jac:.4f}              {g3_corr:.4f}")
    results.append(f"  ArchR seed2 vs seed1        {g4_jac:.4f}              {g4_corr:.4f}")
    results.append(
        f"  GATAC as a fraction of ArchR's own spread: "
        f"{g3_jac / g4_jac:.3f} (Jaccard), {g3_corr / g4_corr:.3f} (corr)"
    )
    return {
        "g3_jac": g3_jac, "g3_corr": g3_corr,
        "g4_jac": g4_jac, "g4_corr": g4_corr,
        "gatac_s": dt, "archr_s": t_ref,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_iterative_lsi(skip_gatac=False, regenerate=False):
    results = ["=== Iterative LSI: ArchR addIterativeLSI oracle vs GATAC ==="]

    ensure_oracle(regenerate=regenerate)
    X, keys, cells, params = load_oracle()
    sanity_check_oracle(X, cells, params, results)

    metrics = None
    if skip_gatac:
        results.append("(GATAC comparison skipped: --skip-gatac)")
    elif not gatac_available():
        results.append(
            "(GATAC comparison skipped: ga.tl.iterative_lsi not available. The "
            "oracle is built and validated — this test becomes the correctness "
            "gate once the port lands.)"
        )
    else:
        metrics = compare(X, keys, cells, params, results)

    log_path = os.path.join(HERE, "iterative_lsi.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        for line in results:
            print(line)
            fh.write(line + "\n")

    # Assertions after logging, so the log is always written.
    if metrics is not None:
        assert metrics["g3_jac"] >= FEATURE_JACCARD_MIN, (
            f"G3 feature Jaccard {metrics['g3_jac']:.4f} "
            f"(expected >= {FEATURE_JACCARD_MIN})"
        )
        assert metrics["g3_corr"] >= SUBSPACE_CORR_MIN, (
            f"G3 mean subspace corr {metrics['g3_corr']:.4f} "
            f"(expected >= {SUBSPACE_CORR_MIN})"
        )
        for name, num, den in (("Jaccard", metrics["g3_jac"], metrics["g4_jac"]),
                               ("corr", metrics["g3_corr"], metrics["g4_corr"])):
            frac = num / den
            assert frac >= ARCHR_SELF_FRACTION_MIN, (
                f"G4 {name}: GATAC reached {frac:.3f} of ArchR's own "
                f"seed-to-seed agreement (expected >= "
                f"{ARCHR_SELF_FRACTION_MIN})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterative LSI: ArchR addIterativeLSI oracle vs GATAC"
    )
    parser.add_argument("--skip-gatac", action="store_true",
                        help="Build/validate the ArchR oracle only")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force rebuilding the arrow file and oracle")
    args = parser.parse_args()
    test_iterative_lsi(skip_gatac=args.skip_gatac, regenerate=args.regenerate)
    sys.exit(0)
