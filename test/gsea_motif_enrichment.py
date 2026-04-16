"""
Test that GPU-accelerated preranked GSEA matches GSEApy's Rust-backed
``gp.prerank`` implementation.

The test creates a synthetic ranked list + synthetic gene sets, runs both
backends with the same seed, and checks that NES/pval/FDR values agree
within tolerance.

Results and timings are written to ``gsea_motif_enrichment.log`` next to this file.
"""

import os
import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Log helpers (shared pattern from gsea_motif_enrichment.py)
# ---------------------------------------------------------------------------

LOG_PATH = os.path.join(os.path.dirname(__file__), "gsea_motif_enrichment.log")


def _write_log(log_lines: list[str]) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_data(
    n_genes: int = 5000,
    n_sets: int = 20,
    set_size_range: tuple[int, int] = (30, 200),
    n_enriched: int = 5,
    enrichment_strength: float = 1.5,
    seed: int = 42,
) -> tuple[pd.Series, dict[str, list[str]]]:
    """Create a synthetic ranked gene list and gene sets.

    ``n_enriched`` sets will have their members biased towards the top of
    the ranking to create a known ground truth.

    Returns
    -------
    ranked_series : pd.Series
        Gene names → ranking values (sorted descending).
    gene_sets : dict[str, list[str]]
        Gene-set name → list of gene names.
    """
    rng = np.random.RandomState(seed)
    gene_names = [f"gene_{i:05d}" for i in range(n_genes)]

    metric = rng.randn(n_genes)

    gene_sets: dict[str, list[str]] = {}
    for s in range(n_sets):
        size = rng.randint(set_size_range[0], set_size_range[1] + 1)
        members = list(rng.choice(gene_names, size=size, replace=False))
        gene_sets[f"set_{s:03d}"] = members

    for s in range(n_enriched):
        for gene in gene_sets[f"set_{s:03d}"]:
            idx = int(gene.split("_")[1])
            metric[idx] += enrichment_strength

    order = np.argsort(-metric)
    sorted_names = [gene_names[i] for i in order]
    sorted_metric = metric[order]

    ranked_series = pd.Series(sorted_metric, index=sorted_names, name="ranking")
    return ranked_series, gene_sets


# ---------------------------------------------------------------------------
# Run GSEApy (Rust backend)
# ---------------------------------------------------------------------------


def run_gseapy(
    ranked_series: pd.Series,
    gene_sets: dict[str, list[str]],
    permutation_num: int = 1000,
    min_size: int = 15,
    max_size: int = 2000,
    seed: int = 42,
    weight: float = 1.0,
) -> pd.DataFrame:
    """Run GSEApy prerank and return results as a DataFrame."""
    import gseapy as gp

    res = gp.prerank(
        rnk=ranked_series,
        gene_sets=gene_sets,
        permutation_num=permutation_num,
        seed=seed,
        min_size=min_size,
        max_size=max_size,
        weight=weight,
        threads=1,
        no_plot=True,
        verbose=False,
    )

    df = res.res2d.copy()
    df = df.rename(columns={
        "Term": "term",
        "ES": "es",
        "NES": "nes",
        "NOM p-val": "pval",
        "FDR q-val": "fdr",
        "Lead_genes": "lead_edge",
    })
    df["es"] = df["es"].astype(float)
    df["nes"] = df["nes"].astype(float)
    df["pval"] = df["pval"].astype(float)
    df["fdr"] = df["fdr"].astype(float)
    df = df.set_index("term")[["es", "nes", "pval", "fdr"]]
    return df


# ---------------------------------------------------------------------------
# Run GPU implementation
# ---------------------------------------------------------------------------


def run_gpu(
    ranked_series: pd.Series,
    gene_sets: dict[str, list[str]],
    permutation_num: int = 1000,
    min_size: int = 15,
    max_size: int = 2000,
    seed: int = 42,
    weight: float = 1.0,
) -> pd.DataFrame:
    """Run GPU prerank and return results as a DataFrame."""
    from gatac.tl.gsea import prerank_gpu

    results = prerank_gpu(
        feature_names=list(ranked_series.index),
        ranking_values=ranked_series.values,
        feature_sets=gene_sets,
        weight=weight,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutation_num,
        seed=seed,
    )

    rows = []
    for r in results:
        rows.append({
            "term": r["term"],
            "es": r["es"],
            "nes": r["nes"],
            "pval": r["pval"],
            "fdr": r["fdr"],
        })

    df = pd.DataFrame(rows).set_index("term")
    return df


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_results(
    gseapy_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    es_atol: float = 0.02,
) -> dict:
    """Compare GSEApy and GPU results.

    Both use random permutations with different RNG implementations
    (Rust's SmallRng vs NumPy's MT19937), so exact match is impossible.
    We check:
      1. ES values match closely (deterministic given same input).
      2. NES values are correlated (depends on permutation null).
      3. Rankings (by NES) agree (Spearman correlation).

    Returns
    -------
    dict with comparison metrics.
    """
    from scipy.stats import spearmanr

    common = sorted(set(gseapy_df.index) & set(gpu_df.index))
    assert len(common) > 0, "No common gene sets between GSEApy and GPU results"

    g = gseapy_df.loc[common]
    u = gpu_df.loc[common]

    es_diff = np.abs(g["es"].values - u["es"].values)
    nes_diff = np.abs(g["nes"].values - u["nes"].values)
    pval_diff = np.abs(g["pval"].values - u["pval"].values)
    fdr_diff = np.abs(g["fdr"].values - u["fdr"].values)

    nes_rho, nes_pval = spearmanr(g["nes"].values, u["nes"].values)

    sign_agree = float(
        (np.sign(g["nes"].values) == np.sign(u["nes"].values)).mean()
    )

    k = min(5, len(common))
    top_gseapy = set(g.sort_values("nes", ascending=False).index[:k])
    top_gpu = set(u.sort_values("nes", ascending=False).index[:k])
    top_k_overlap = len(top_gseapy & top_gpu) / k

    return {
        "n_common": len(common),
        "es_max_diff": float(es_diff.max()),
        "es_mean_diff": float(es_diff.mean()),
        "es_match": float(es_diff.max()) < es_atol,
        "nes_max_diff": float(nes_diff.max()),
        "nes_mean_diff": float(nes_diff.mean()),
        "nes_spearman_rho": float(nes_rho),
        "nes_spearman_pval": float(nes_pval),
        "pval_max_diff": float(pval_diff.max()),
        "pval_mean_diff": float(pval_diff.mean()),
        "fdr_max_diff": float(fdr_diff.max()),
        "fdr_mean_diff": float(fdr_diff.mean()),
        "sign_agreement": sign_agree,
        "top_k_overlap": top_k_overlap,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_gsea_gpu():
    """End-to-end comparison: GPU prerank vs GSEApy (Rust backend).

    Uses a large synthetic dataset (60 000 genes, 300 sets) to exercise
    both correctness and GPU scaling.
    """
    PERMUTATION_NUM = 1000
    SEED = 42

    log_lines: list[str] = []
    log_lines.append("=" * 70)
    log_lines.append("test_gsea_gpu")
    log_lines.append("=" * 70)

    # --- Synthetic data ---
    ranked, gene_sets = make_synthetic_data(
        n_genes=60000,
        n_sets=300,
        set_size_range=(30, 500),
        n_enriched=30,
        enrichment_strength=1.5,
        seed=SEED,
    )
    log_lines.append(f"Genes: {len(ranked):,}   Gene sets: {len(gene_sets)}")

    # --- GSEApy ---
    t0 = time.time()
    gseapy_df = run_gseapy(ranked, gene_sets, permutation_num=PERMUTATION_NUM, seed=SEED)
    gseapy_time = time.time() - t0
    log_lines.append(f"GSEApy time:\t{gseapy_time:.2f}s  ({len(gseapy_df)} sets)")

    # --- GPU ---
    t0 = time.time()
    gpu_df = run_gpu(ranked, gene_sets, permutation_num=PERMUTATION_NUM, seed=SEED)
    gpu_time = time.time() - t0
    log_lines.append(f"GPU time:\t{gpu_time:.2f}s  ({len(gpu_df)} sets)")
    log_lines.append(f"Speedup:\t{gseapy_time / gpu_time:.1f}x")

    # --- Metrics ---
    metrics = compare_results(gseapy_df, gpu_df)

    log_lines.append("")
    log_lines.append(f"Common sets:          {metrics['n_common']}")
    log_lines.append(f"ES  max|diff|:        {metrics['es_max_diff']:.6f}  (match: {metrics['es_match']})")
    log_lines.append(f"ES  mean|diff|:       {metrics['es_mean_diff']:.6f}")
    log_lines.append(f"NES max|diff|:        {metrics['nes_max_diff']:.4f}")
    log_lines.append(f"NES mean|diff|:       {metrics['nes_mean_diff']:.4f}")
    log_lines.append(f"NES Spearman ρ:       {metrics['nes_spearman_rho']:.4f}  (p={metrics['nes_spearman_pval']:.2e})")
    log_lines.append(f"Pval max|diff|:       {metrics['pval_max_diff']:.4f}")
    log_lines.append(f"Pval mean|diff|:      {metrics['pval_mean_diff']:.4f}")
    log_lines.append(f"FDR max|diff|:        {metrics['fdr_max_diff']:.4f}")
    log_lines.append(f"FDR mean|diff|:       {metrics['fdr_mean_diff']:.4f}")
    log_lines.append(f"Sign agreement:       {metrics['sign_agreement']:.1%}")
    log_lines.append(f"Top-5 overlap:        {metrics['top_k_overlap']:.1%}")

    # --- Sample comparison ---
    common = sorted(set(gseapy_df.index) & set(gpu_df.index))
    log_lines.append("")
    log_lines.append(f"{'Term':<12} {'GSEApy ES':>11} {'GPU ES':>11} {'GSEApy NES':>12} {'GPU NES':>12} {'GSEApy pval':>12} {'GPU pval':>12}")
    for term in common[:5]:
        log_lines.append(
            f"{term:<12} "
            f"{gseapy_df.loc[term, 'es']:>11.6f} {gpu_df.loc[term, 'es']:>11.6f} "
            f"{gseapy_df.loc[term, 'nes']:>12.4f} {gpu_df.loc[term, 'nes']:>12.4f} "
            f"{gseapy_df.loc[term, 'pval']:>12.4f} {gpu_df.loc[term, 'pval']:>12.4f}"
        )

    _write_log(log_lines)

    # --- Assertions ---
    assert metrics["es_match"], (
        f"ES values differ by up to {metrics['es_max_diff']:.6f} "
        f"(threshold: 0.02). The enrichment score formula may differ."
    )
    assert metrics["nes_spearman_rho"] > 0.9, (
        f"NES Spearman ρ = {metrics['nes_spearman_rho']:.4f} (expected > 0.9). "
        "The permutation null or normalisation may differ."
    )
    assert metrics["sign_agreement"] >= 0.9, (
        f"Sign agreement = {metrics['sign_agreement']:.1%} (expected ≥ 90%)"
    )
    assert metrics["top_k_overlap"] >= 0.6, (
        f"Top-5 overlap = {metrics['top_k_overlap']:.1%} (expected ≥ 60%)"
    )


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Clear log on fresh run
    open(LOG_PATH, "w").close()
    test_gsea_gpu()
    print(f"\nResults written to {LOG_PATH}")
