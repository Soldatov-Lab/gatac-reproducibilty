"""
Footprinting test: GATAC ``tobias.footprint`` + ``tobias.bindetect`` vs TOBIAS 0.17.5.

Three PBMC 5k pseudobulks (CD14 Mono, CD4 Memory, NK) on chr19, cisBP motifs.
TOBIAS (ATACorrect -> ScoreBigwig -> BINDetect, default parameters) runs in the
``tobias`` pixi env on BAMs written from the same fragments parquet
(``test/footprinting_tobias.py``); its outputs are cached in
``data/footprinting_tobias/``.

Checks
------
Stage by stage, each GATAC stage gets TOBIAS's own input, and must match exactly:

1. bias model: DWM k-mer counts, log2 PWM/DWM, read counts, correction factor;
2. correction with TOBIAS's fp64 bias scores injected -> corrected track;
3. footprint scores with TOBIAS's corrected track injected;
4. BINDetect with TOBIAS's footprints injected -> results table (p-values to
   TOBIAS's 6 printed digits) and motif distance matrix.

End to end, the only difference is the last bits of the DWM bias score
(floating-point summation order). The tolerances come from a noise control:
TOBIAS rerun with that score perturbed by U(-1e-13, 1e-13), once per seed in
CONTROL_SEEDS. GATAC's deviation from TOBIAS must stay within twice the
largest deviation of these reruns. One rerun is not enough: the bound-site
calls that flip are those at the bound threshold, and how many flip varies
from one perturbation to the next.

Usage::

    pixi run python test/footprinting.py                 # full comparison
    pixi run python test/footprinting.py --run-gatac-only
    pixi run python test/footprinting.py --rerun-tobias  # ignore the TOBIAS cache
"""
import argparse
import copy
import importlib
import json
import os
import shutil
import subprocess
import time

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import scipy.stats

import gatac as ga

from _paths import data_path
from _timing import N_CPUS, cold_warm
fpmod = importlib.import_module("gatac.tobias.footprint")     # the module, not ga.tobias.footprint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQUET = data_path("atac_pbmc_5k_filtered.parquet")
H5AD = data_path("pbmc_peak_matrix.h5ad")
FASTA = data_path("GRCh38.p13.genome.fa")
MEME = data_path("cisBP_human.meme")
CACHE = data_path("footprinting_tobias")
CHROMS = ["chr19"]
GROUPS = {"CD14 Mono": "mono", "CD4 Memory": "cd4", "NK": "nk"}
# Noise-control reruns of TOBIAS, one per seed, cached in CACHE/<dir>
CONTROL_SEEDS = (0, 1, 2)


def _control_dir(seed):
    return os.path.join(CACHE, "control" if seed == 0 else f"control_{seed}")
PIXI = shutil.which("pixi") or "pixi"


def _tobias(*args):
    subprocess.run([PIXI, "run", "--environment", "tobias", "python",
                    os.path.join(HERE, "footprinting_tobias.py"), *args], check=True, cwd=ROOT)


def _run_tobias(adata, peaks_bed, layouts):
    """TOBIAS outputs (cached); returns the TOBIAS wall times."""
    timings_f = os.path.join(CACHE, "timings.json")
    groups = list(GROUPS.values())
    timings = json.load(open(timings_f)) if os.path.exists(timings_f) else {}
    if "run" not in timings:
        for g in groups:
            bc = os.path.join(CACHE, f"{g}.txt")
            with open(bc, "w") as f:
                f.write("\n".join(adata.obs_names[adata.obs["group"] == g]) + "\n")
            _tobias("bam", PARQUET, bc, os.path.join(CACHE, f"{g}.bam"), "--chroms", ",".join(CHROMS),
                    "--fasta", FASTA, "--cores", str(N_CPUS))
        t0 = time.perf_counter()
        _tobias("run", CACHE, "--groups", ",".join(groups), "--peaks", peaks_bed, "--fasta", FASTA,
                "--motifs", MEME, "--cores", str(N_CPUS))
        timings["run"] = time.perf_counter() - t0
        timings["cores"] = N_CPUS
        json.dump(timings, open(timings_f, "w"))
    for seed in CONTROL_SEEDS:
        # Cached once its BINDetect results are written, which may be by a
        # separate job running footprinting_tobias.py run --perturb SEED
        if os.path.exists(os.path.join(_control_dir(seed), "bindetect", "bindetect_results.txt")):
            continue
        _tobias("run", _control_dir(seed), "--bamdir", CACHE, "--perturb", str(seed), "--groups", ",".join(groups),
                "--peaks", peaks_bed, "--fasta", FASTA, "--motifs", MEME, "--cores", str(N_CPUS))
    # inputs for the stage-isolated checks (cheap; rebuilt when the layouts change)
    np.savez(os.path.join(CACHE, "out_regions.npz"), **layouts["corrected"].intervals)
    np.savez(os.path.join(CACHE, "fp_regions.npz"), **layouts["footprint"].intervals)
    for g in groups:
        _tobias("bias64", CACHE, g, os.path.join(CACHE, "out_regions.npz"),
                os.path.join(CACHE, f"{g}_bias64.npz"), "--fasta", FASTA, "--cores", str(N_CPUS))
    specs = []
    for g in groups:
        for tag, d in (("", CACHE), *((f"control{seed}_", _control_dir(seed)) for seed in CONTROL_SEEDS)):
            specs += [f"{tag}{g}_corrected={os.path.join(d, g, g + '_corrected.bw')}={os.path.join(CACHE, 'out_regions.npz')}",
                      f"{tag}{g}_footprints={os.path.join(d, g, g + '_footprints.bw')}={os.path.join(CACHE, 'fp_regions.npz')}"]
    _tobias("dump", CACHE, os.path.join(CACHE, "dump.npz"), "--groups", ",".join(groups), "--bigwig", *specs)
    return timings


def _exact_correction(tracks, g, gi, bias64, cuts):
    """Corrected track from GATAC's correction given TOBIAS's fp64 bias scores (float32, as in the bigwig)."""
    fa = ga.io.Fasta(FASTA)
    out = {}
    for c in CHROMS:
        seq_d = cp.asarray(fpmod._LUT[fa.chrom(c)])
        out_reg = tracks.layouts["corrected"].intervals[c]
        override = {s: bias64[f"{c}__{s}"] for s in ("forward", "reverse")}
        tr, _ = fpmod._correct_chrom(out_reg, cuts[gi][c], seq_d, tracks.bias_models[g],
                                     float(tracks.stats.loc[g, "correction_factor"]), keep=("corrected",),
                                     bias_override=override)
        out[c] = cp.asnumpy(tr["corrected"].astype(cp.float32))
    return out


def _exact_footprints(tracks, tobias_corrected):
    """Footprint scores from GATAC's scorer given TOBIAS's corrected track."""
    out = {}
    lay = tracks.layouts["corrected"]
    for c in CHROMS:
        dense = cp.zeros(tracks.chrom_sizes[c], dtype=cp.float32)
        dense[cp.asarray(lay.positions(c))] = cp.asarray(tobias_corrected[c])
        out[c] = cp.asnumpy(fpmod._footprints_chrom(tracks.layouts["footprint"].intervals[c], dense))
    return out


def _compare_results(ref, res):
    """Fraction of equal values per column of the BINDetect results table."""
    gpu = res.results
    eq = {}
    for col in ref.columns:
        a, b = ref[col], gpu[col]
        if col.endswith("_pvalue"):
            a = a.to_numpy(float)
            b = np.array([float(f"{p:.5E}") if np.isfinite(p) else np.nan for p in b])
            e = (a == b) | (np.isnan(a) & np.isnan(b))
        elif a.dtype.kind in "fi":
            a, b = a.to_numpy(float), b.to_numpy(float)
            e = (a == b) | (np.isnan(a) & np.isnan(b))
        else:
            e = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        eq[col] = float(e.mean())
    return eq


def _deviation(corr, fp, table, dump, ref, groups, comparisons):
    """Deviation of one end-to-end run from TOBIAS (tracks per group, results table)."""
    out = {}
    for g in groups:
        b = np.concatenate([dump[f"{g}_corrected__{c}"] for c in CHROMS])
        fb = np.concatenate([dump[f"{g}_footprints__{c}"] for c in CHROMS])
        out[f"corrected 1-Pearson {g}"] = 1 - float(np.corrcoef(corr[g], b)[0, 1])
        out[f"footprint 1-Pearson {g}"] = 1 - float(np.corrcoef(fp[g], fb)[0, 1])
    t = table.set_index("output_prefix").loc[ref.output_prefix]
    r = ref.set_index("output_prefix")
    out["TFBS count differences"] = int(np.abs(t.total_tfbs.to_numpy() - r.total_tfbs.to_numpy()).sum())
    for g in groups:
        out[f"bound sites |diff| {g}"] = float(np.abs(t[f"{g}_bound"].to_numpy() - r[f"{g}_bound"].to_numpy()).sum()
                                               / r[f"{g}_bound"].sum())
    for g1, g2 in comparisons:
        col = f"{g1}_{g2}_change"
        out[f"change 1-Spearman {g1}/{g2}"] = 1 - float(scipy.stats.spearmanr(t[col], r[col]).statistic)
        top_t = set(t[col].abs().sort_values().index[-50:])
        top_r = set(r[col].abs().sort_values().index[-50:])
        out[f"top-50 |change| missed {g1}/{g2}"] = 50 - len(top_t & top_r)
    return out


def test_footprinting(run_gatac_only=False, rerun_tobias=False):
    adata = ad.read_h5ad(H5AD)
    adata = adata[adata.obs["cell_type"].isin(list(GROUPS))].copy()
    adata.obs["group"] = pd.Categorical(adata.obs["cell_type"].map(GROUPS).astype(str),
                                        categories=list(GROUPS.values()))
    groups = list(GROUPS.values())
    peaks = pd.Series(adata.var_names.astype(str)).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    peaks = peaks[peaks["chrom"].isin(CHROMS)].astype({"start": np.int64, "end": np.int64})
    # wipe before peaks.bed goes in: TOBIAS reads it from the cache
    if rerun_tobias and not run_gatac_only and os.path.isdir(CACHE):
        shutil.rmtree(CACHE)
    os.makedirs(CACHE, exist_ok=True)
    peaks_bed = os.path.join(CACHE, "peaks.bed")
    peaks.to_csv(peaks_bed, sep="\t", header=False, index=False)

    # ---- GATAC ----
    ga.io.Fasta(FASTA).close()                          # build the .fai outside the timing
    def run_gatac():
        t0 = time.perf_counter()
        tracks = ga.tobias.footprint(adata, "group", PARQUET, FASTA, groups=groups, peaks=peaks,
                                     chromosomes=CHROMS, verbose=False)
        t_fp = time.perf_counter() - t0
        t0 = time.perf_counter()
        sites = ga.tobias.bindetect_sites(tracks, MEME, verbose=False)     # reused for the TOBIAS-footprint run
        res = ga.tobias.bindetect(tracks, sites=sites, verbose=False)
        return tracks, sites, res, t_fp, time.perf_counter() - t0

    (tracks, sites, res, t_fp, t_bd), gatac_cold, _ = cold_warm(run_gatac)
    gatac_time = t_fp + t_bd

    results = [
        "=== Footprinting (TOBIAS ATACorrect + ScoreBigwig + BINDetect) Benchmark ===",
        f"Data: PBMC 5k, {', '.join(f'{k} ({(adata.obs.group == v).sum()} cells)' for k, v in GROUPS.items())}",
        f"Peaks: {len(peaks):,} on {', '.join(CHROMS)}; motifs: {len(res.results):,} (cisBP)",
        "",
        f"GATAC footprint:\t{t_fp:.2f}s",
        f"GATAC bindetect:\t{t_bd:.2f}s",
        f"GATAC total:\t{gatac_time:.2f}s",
        f"GATAC cold start:\t{gatac_cold:.2f}s",
        f"Sites: {len(res.sites):,}; bound threshold {res.threshold}",
    ]
    checks = {}

    if not run_gatac_only:
        timings = _run_tobias(adata, peaks_bed, tracks.layouts)
        tobias_time = timings["run"]
        dump = np.load(os.path.join(CACHE, "dump.npz"))
        results += ["", f"TOBIAS total ({timings.get('cores', 6)} cores):\t{tobias_time:.2f}s",
                    f"Speedup: {tobias_time / gatac_time:.1f}x",
                    f"Speedup (cold start): {tobias_time / gatac_cold:.1f}x", "", "--- Stage-isolated (exact) ---"]

        # 1. bias model and normalisation
        ok_model = True
        for g in groups:
            for s in ("forward", "reverse"):
                for k in ("counts", "bg_counts", "bias_pwm_log", "bias_dwm_log", "bg_pwm_log", "bg_dwm_log", "pssm"):
                    ok_model &= np.array_equal(tracks.bias_models[g][s][k], dump[f"{g}__{s}__{k}"])
            ok_model &= float(tracks.stats.loc[g, "correction_factor"]) == float(dump[f"{g}__correction_factor"])
        checks["bias_model"] = ok_model
        results.append(f"Bias model, read counts, correction factor identical:\t{ok_model}")

        # 2. correction given TOBIAS's fp64 bias
        cuts = fpmod._load_cuts(PARQUET, {b: i for i, g in enumerate(groups)
                                          for b in adata.obs_names[adata.obs.group == g]},
                                tracks.chrom_sizes, 50, True)
        eq_corr = []
        for gi, g in enumerate(groups):
            mine = _exact_correction(tracks, g, gi, np.load(os.path.join(CACHE, f"{g}_bias64.npz")), cuts)
            eq_corr += [np.mean(mine[c] == dump[f"{g}_corrected__{c}"]) for c in CHROMS]
        checks["correction"] = min(eq_corr)
        results.append(f"Corrected track given TOBIAS bias, equal positions:\t{min(eq_corr) * 100:.4f}%")

        # 3. footprints given TOBIAS's corrected track
        eq_fp = []
        for g in groups:
            mine = _exact_footprints(tracks, {c: dump[f"{g}_corrected__{c}"] for c in CHROMS})
            eq_fp += [np.mean(mine[c] == dump[f"{g}_footprints__{c}"]) for c in CHROMS]
        checks["footprints"] = min(eq_fp)
        results.append(f"Footprints given TOBIAS corrected, equal positions:\t{min(eq_fp) * 100:.4f}%")

        # 4. BINDetect given TOBIAS's footprints
        tob = copy.copy(tracks)
        tob.values = {"footprint": {g: tracks.values["footprint"][g].copy() for g in groups}}
        flay = tracks.layouts["footprint"]
        for g in groups:
            for c in CHROMS:
                tob.values["footprint"][g][flay.chrom_slice(c)] = dump[f"{g}_footprints__{c}"]
        res_t = ga.tobias.bindetect(tob, sites=sites, verbose=False)
        ref = pd.read_csv(os.path.join(CACHE, "bindetect", "bindetect_results.txt"), sep="\t",
                          keep_default_na=False, na_values=["NA"], float_precision="round_trip")
        same_rows = list(ref.output_prefix) == list(res_t.results.output_prefix)
        same_cols = list(ref.columns) == list(res_t.results.columns)
        eq = _compare_results(ref, res_t) if same_rows and same_cols else {}
        dist_f = os.path.join(CACHE, "bindetect", "bindetect_distances.txt")
        D = np.loadtxt(dist_f, skiprows=1)
        G = np.array([[float("%.4f" % v) for v in row] for row in res_t.distances.to_numpy()])
        dist_ok = open(dist_f).readline().lstrip("# ").rstrip("\n").split("\t") == list(res_t.distances.index) \
            and np.array_equal(D, G)
        checks["bindetect"] = same_rows and same_cols and all(v == 1.0 for v in eq.values())
        checks["distances"] = dist_ok
        results.append(f"BINDetect given TOBIAS footprints: rows/columns identical {same_rows and same_cols}; "
                       f"min column agreement {min(eq.values()) * 100 if eq else 0:.4f}%")
        results += [f"  {k}: {v * 100:.4f}%" for k, v in eq.items() if v < 1.0]
        results.append(f"Motif distance matrix identical:\t{dist_ok}")

        # 5. end to end, against the TOBIAS noise control
        lay = tracks.layouts["corrected"]
        mine = _deviation(
            {g: np.concatenate([tracks.values["corrected"][g][lay.chrom_slice(c)] for c in CHROMS]) for g in groups},
            {g: np.concatenate([tracks.values["footprint"][g][flay.chrom_slice(c)] for c in CHROMS]) for g in groups},
            res.results, dump, ref, groups, res.comparisons)
        controls = []
        for seed in CONTROL_SEEDS:
            ctrl_ref = pd.read_csv(os.path.join(_control_dir(seed), "bindetect", "bindetect_results.txt"), sep="\t",
                                   keep_default_na=False, na_values=["NA"], float_precision="round_trip")
            controls.append(_deviation(
                {g: np.concatenate([dump[f"control{seed}_{g}_corrected__{c}"] for c in CHROMS]) for g in groups},
                {g: np.concatenate([dump[f"control{seed}_{g}_footprints__{c}"] for c in CHROMS]) for g in groups},
                ctrl_ref, dump, ref, groups, res.comparisons))
        ctrl = {k: max(c[k] for c in controls) for k in mine}
        results += ["", f"--- End to end: deviation from TOBIAS (GATAC vs the largest of {len(controls)} "
                        "TOBIAS ±1e-13 bias noise controls) ---"]
        for k in mine:
            spread = ", ".join(f"{c[k]:.3g}" for c in controls)
            results.append(f"{k}:\t{mine[k]:.3g}\t(controls {spread})")
        checks["e2e"] = {k: (mine[k], ctrl[k]) for k in mine}

    log_path = os.path.join(HERE, "footprinting.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in results:
            print(line)
            f.write(line + "\n")

    assert len(res.sites) > 0 and res.results.total_tfbs.sum() == len(res.sites)
    if not run_gatac_only:
        assert checks["bias_model"], "bias model differs from TOBIAS"
        assert checks["correction"] == 1.0, f"corrected track given TOBIAS bias: {checks['correction']:.6f} equal"
        assert checks["footprints"] == 1.0, f"footprints given TOBIAS corrected: {checks['footprints']:.6f} equal"
        assert checks["bindetect"], "BINDetect results given TOBIAS footprints differ"
        assert checks["distances"], "motif distance matrix differs"
        e2e = dict(checks["e2e"])
        assert e2e.pop("TFBS count differences")[0] == 0, "TFBS counts depend on sequence only and must match"
        for k, (m, c) in e2e.items():
            # within twice TOBIAS's own sensitivity (integer counts: twice the control's value + 1)
            limit = 2 * c if isinstance(c, float) else 2 * c + 1
            assert m <= limit, f"end to end {k}: {m:.3g} exceeds 2x the largest TOBIAS noise control ({c:.3g})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC footprinting vs TOBIAS 0.17.5")
    parser.add_argument("--run-gatac-only", action="store_true", help="Skip the TOBIAS run and comparison")
    parser.add_argument("--rerun-tobias", action="store_true", help="Ignore cached TOBIAS outputs")
    args = parser.parse_args()
    test_footprinting(run_gatac_only=args.run_gatac_only, rerun_tobias=args.rerun_tobias)
