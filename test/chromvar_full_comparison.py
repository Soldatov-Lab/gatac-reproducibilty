#!/usr/bin/env python3
"""
GATAC Reproducibility Test - Comprehensive ChromVAR Comparison Script

This script performs THREE comparison tests:

1. **scPrinter Motif Matching vs GATAC**
   - Compares GATAC motif scanning (motifmatchr mode) against scPrinter-style MOODS scanning
   - Tests motif matching accuracy and performance

2. **Full ChromVAR R Pipeline vs GATAC**
   - Compares GATAC's full chromVAR pipeline against R chromVAR output
   - Uses R's motif matches, background peaks, and deviation scores
   - Tests end-to-end chromVAR reproducibility

3. **ChromVAR Dev Scores - Same Motifs Comparison**
   - Uses GATAC motif matches but R background peaks
   - Shows that deviation scores are similar when using the same motif inputs
   - Isolates the deviation calculation from motif matching differences
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
import logging
import sys
import time
import tempfile
from pyfaidx import Fasta

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import GATAC
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GATAC"))
import gatac as ga

# Import MOODS for scPrinter-style scanning
try:
    import MOODS
    import MOODS.parsers
    import MOODS.scan
    import MOODS.tools
    HAS_MOODS = True
except ImportError:
    HAS_MOODS = False
    print("WARNING: MOODS not available, scPrinter comparison will be skipped")


# =============================================================================
# scPrinter-style motif scanning (replicated from scprinter/motifs.py)
# =============================================================================

class PFM:
    """Simple wrapper for PFM matrices compatible with MOODS."""
    def __init__(self, name, counts):
        self.name = name
        self.counts = counts
        self.length = len(counts["A"])


def parse_jaspar(file_path):
    """Parse JASPAR format motifs."""
    records = []
    record = None
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if record:
                    records.append(PFM(record["name"], record["weights"]))
                record = {
                    "name": line[1:].split(" ")[-1],
                    "weights": {"A": [], "C": [], "G": [], "T": []},
                }
            elif len(line) > 0 and record:
                nucleotide, values_str = line.split(" ", 1)
                values = list(map(float, values_str.strip(" ").strip("[]").split()))
                record["weights"][nucleotide] = values
        
        if record:
            records.append(PFM(record["name"], record["weights"]))
    
    return records


def jaspar_to_moods_matrix(jaspar_motif, bg, pseudocount, mode="motifmatchr"):
    """Convert a JASPAR motif to a MOODS matrix."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pfm', delete=False) as fn:
        for base in "ACGT":
            line = " ".join(str(x) for x in jaspar_motif.counts[base])
            fn.write(line + "\n")
        fn.flush()
        
        if mode != "motifmatchr":
            m = MOODS.parsers.pfm_to_log_odds(fn.name, bg, pseudocount, 2)
        else:
            # Consistent with motifmatchr
            even = [0.25, 0.25, 0.25, 0.25]
            m = MOODS.parsers.pfm_to_log_odds(fn.name, even, pseudocount, 2)
            m = [tuple(m[i] - (np.log2(0.25) - np.log2(bg[i]))) for i in range(len(bg))]
        
        import os
        os.unlink(fn.name)
        return m


def prepare_moods_settings(motifs, bg, pseudocount, pvalue):
    """Prepare MOODS scanner settings."""
    matrices_p = []
    thresholds_p = []
    matrices_m = []
    thresholds_m = []
    
    for motif in motifs:
        m = jaspar_to_moods_matrix(motif, bg, pseudocount, mode="motifmatchr")
        m_rc = MOODS.tools.reverse_complement(m)
        
        threshold = MOODS.tools.threshold_from_p(m, bg, pvalue)
        threshold_rc = MOODS.tools.threshold_from_p(m_rc, bg, pvalue)
        
        matrices_p.append(m)
        thresholds_p.append(threshold)
        matrices_m.append(m_rc)
        thresholds_m.append(threshold_rc)
    
    return matrices_p, thresholds_p, matrices_m, thresholds_m


def scprinter_scan_motifs(adata, motif_file, genome_fasta, pvalue=5e-5, n_jobs=8):
    """
    Scan motifs using scPrinter-style MOODS scanning.
    
    Replicates scPrinter's chromvar_scan function.
    """
    # Parse motifs
    all_motifs = parse_jaspar(motif_file)
    print(f"    Parsed {len(all_motifs)} motifs from JASPAR file")
    
    # Background frequencies
    bg = [0.25, 0.25, 0.25, 0.25]
    pseudocount = 0.8
    
    # Prepare MOODS settings
    matrices_p, thresholds_p, matrices_m, thresholds_m = prepare_moods_settings(
        all_motifs, bg, pseudocount, pvalue
    )
    
    # Create scanner
    matrices = matrices_p + matrices_m
    thresholds = thresholds_p + thresholds_m
    
    scanner = MOODS.scan.Scanner(7)  # window size 7
    scanner.set_motifs(matrices, bg, thresholds)
    
    # Handle gzipped fasta - decompress first
    import gzip
    import shutil
    
    if genome_fasta.endswith('.gz'):
        print("    Decompressing genome for pyfaidx...")
        decompressed_path = tempfile.NamedTemporaryFile(suffix='.fa', delete=False).name
        with gzip.open(genome_fasta, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        genome_fasta = decompressed_path
        cleanup_genome = True
    else:
        cleanup_genome = False
    
    # Open genome
    genome = Fasta(genome_fasta)
    
    # Parse peak regions
    peaks = []
    for var_name in adata.var_names:
        chrom, coords = var_name.split(":")
        start, end = coords.split("-")
        peaks.append((chrom, int(start), int(end)))
    
    # Scan each peak
    n_peaks = len(peaks)
    n_motifs = len(all_motifs)
    motif_matches = np.zeros((n_peaks, n_motifs), dtype=bool)
    
    from tqdm.auto import tqdm
    for peak_idx, (chrom, start, end) in enumerate(tqdm(peaks, desc="    Scanning peaks")):
        try:
            seq = genome[chrom][start:end].seq.upper()
            results = scanner.scan(seq)
            
            # Results has 2*n_motifs entries (forward + reverse for each motif)
            for motif_idx in range(n_motifs):
                # Forward strand
                if len(results[motif_idx]) > 0:
                    motif_matches[peak_idx, motif_idx] = True
                # Reverse strand
                if len(results[motif_idx + n_motifs]) > 0:
                    motif_matches[peak_idx, motif_idx] = True
        except Exception as e:
            # Skip problematic regions
            pass
    
    # Cleanup decompressed file
    if cleanup_genome:
        import os
        os.unlink(genome_fasta)
    
    # Store results
    adata.varm["motif_match"] = motif_matches
    adata.uns["motif_name"] = [m.name for m in all_motifs]
    
    return motif_matches


def load_motifs_from_csv(path):
    """Load motifs from R-exported PWM CSV."""
    df = pd.read_csv(path)
    motifs = []
    for motif_id in df['motif'].unique():
        motif_df = df[df['motif'] == motif_id].sort_values('pos')
        pwm = motif_df[['A', 'C', 'G', 'T']].values.astype(np.float64)
        # PWMs from R are already normalized, but let's be safe
        pwm = pwm / pwm.sum(axis=1, keepdims=True)
        motifs.append(ga.tl.DNAMotif(id=motif_id, pwm=pwm, name=motif_id))
    return motifs


def create_jaspar_file(motifs_df, out_path, motif_ids=None):
    """Create a JASPAR format file from a PWM DataFrame."""
    if motif_ids is None:
        motif_ids = motifs_df['motif'].unique()
    
    with open(out_path, 'w') as f:
        for motif_id in motif_ids:
            motif_df = motifs_df[motifs_df['motif'] == motif_id].sort_values('pos')
            pwm_raw = motif_df[['A', 'C', 'G', 'T']].values.astype(np.float64)
            
            # Convert to counts for JASPAR format
            row_sums = pwm_raw.sum(axis=1, keepdims=True)
            if np.allclose(row_sums, 1.0):
                # Already normalized - treat as frequencies with assumed count of 100
                counts = pwm_raw * 100.0
            else:
                counts = pwm_raw
            
            f.write(f">{motif_id} {motif_id}\n")
            for base, idx in [('A', 0), ('C', 1), ('G', 2), ('T', 3)]:
                values = ' '.join(f"{int(c)}" for c in counts[:, idx])
                f.write(f"{base} [{values}]\n")
    
    return out_path


# =============================================================================
# TEST 1: scPrinter Motif Matching vs GATAC
# =============================================================================

def test_scprinter_vs_gatac(adata, motifs, motifs_df, genome_path, results_dict):
    """
    Test 1: Compare GATAC motif scanning against scPrinter-style MOODS scanning.
    """
    print("\n" + "="*70)
    print("TEST 1: scPrinter Motif Matching vs GATAC")
    print("="*70)
    
    if not HAS_MOODS:
        print("SKIPPED: MOODS module not available")
        results_dict["scprinter_vs_gatac"] = {"status": "SKIPPED", "reason": "MOODS not available"}
        return
    
    # Create JASPAR file for scPrinter
    jaspar_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pfm', delete=False).name
    create_jaspar_file(motifs_df, jaspar_file)
    
    # === GATAC Motif Scanning ===
    print("\n  [GATAC] Scanning motifs (motifmatchr mode)...")
    adata_gatac = adata.copy()
    
    gatac_start = time.time()
    ga.tl.scan_motifs(
        adata_gatac,
        motifs,
        str(genome_path),
        pvalue=5e-5,
        mode="motifmatchr",
        coordinate_system="1-based",  # R uses 1-based
    )
    gatac_time = time.time() - gatac_start
    
    gatac_matches = adata_gatac.varm["motif_match"]
    if sp.issparse(gatac_matches):
        gatac_matches = gatac_matches.toarray()
    gatac_motif_names = adata_gatac.uns["motif_name"]
    
    print(f"    Time: {gatac_time:.2f}s")
    print(f"    Total matches: {np.sum(gatac_matches):,}")
    
    # === scPrinter Motif Scanning ===
    print("\n  [scPrinter] Scanning motifs (MOODS)...")
    adata_scp = adata.copy()
    
    scp_start = time.time()
    scprinter_scan_motifs(
        adata_scp,
        jaspar_file,
        str(genome_path),
        pvalue=5e-5,
        n_jobs=8
    )
    scp_time = time.time() - scp_start
    
    scp_matches = adata_scp.varm["motif_match"]
    if sp.issparse(scp_matches):
        scp_matches = scp_matches.toarray()
    scp_matches_bool = (scp_matches > 0).astype(bool)
    scp_motif_names = adata_scp.uns["motif_name"]
    
    print(f"    Time: {scp_time:.2f}s")
    print(f"    Total matches: {np.sum(scp_matches_bool):,}")
    
    # === Comparison ===
    print("\n  [Comparison]")
    
    # Align motif names
    gatac_motif_set = set(gatac_motif_names)
    scp_motif_set = set(scp_motif_names)
    common_motifs = sorted(gatac_motif_set & scp_motif_set)
    
    if len(common_motifs) == 0:
        # Match by index
        n_compare = min(len(gatac_motif_names), len(scp_motif_names))
        gatac_idx = list(range(n_compare))
        scp_idx = list(range(n_compare))
    else:
        gatac_motif_map = {m: i for i, m in enumerate(gatac_motif_names)}
        scp_motif_map = {m: i for i, m in enumerate(scp_motif_names)}
        gatac_idx = [gatac_motif_map[m] for m in common_motifs]
        scp_idx = [scp_motif_map[m] for m in common_motifs]
    
    gatac_aligned = gatac_matches[:, gatac_idx].astype(bool)
    scp_aligned = scp_matches_bool[:, scp_idx].astype(bool)
    
    # Compute metrics
    gatac_flat = gatac_aligned.flatten().astype(float)
    scp_flat = scp_aligned.flatten().astype(float)
    correlation = np.corrcoef(gatac_flat, scp_flat)[0, 1]
    
    both_match = np.sum(gatac_aligned & scp_aligned)
    gatac_only = np.sum(gatac_aligned & ~scp_aligned)
    scp_only = np.sum(~gatac_aligned & scp_aligned)
    neither = np.sum(~gatac_aligned & ~scp_aligned)
    
    total = gatac_aligned.size
    agreement = (both_match + neither) / total
    jaccard = both_match / (both_match + gatac_only + scp_only) if (both_match + gatac_only + scp_only) > 0 else 0
    
    print(f"    Match Correlation: {correlation:.4f}")
    print(f"    Agreement Rate: {agreement:.4f} ({100*agreement:.1f}%)")
    print(f"    Jaccard Index: {jaccard:.4f}")
    print(f"    GATAC Time: {gatac_time:.2f}s, scPrinter Time: {scp_time:.2f}s")
    
    # Cleanup
    import os
    os.unlink(jaspar_file)
    
    results_dict["scprinter_vs_gatac"] = {
        "correlation": correlation,
        "agreement": agreement,
        "jaccard": jaccard,
        "gatac_matches": int(np.sum(gatac_matches)),
        "scprinter_matches": int(np.sum(scp_matches_bool)),
        "gatac_time": gatac_time,
        "scprinter_time": scp_time,
    }


# =============================================================================
# TEST 2: Full ChromVAR R Pipeline vs GATAC
# =============================================================================

def test_chromvar_full_pipeline(adata, motifs, r_output_dir, genome_path, results_dict):
    """
    Test 2: Compare GATAC's full chromVAR pipeline against R chromVAR output.
    Uses R's motif matches and background peaks, compares deviation scores.
    """
    print("\n" + "="*70)
    print("TEST 2: Full ChromVAR R Pipeline vs GATAC")
    print("="*70)
    
    # Load R outputs
    r_matches = pd.read_csv(r_output_dir / "motif_matches.csv", index_col=0)
    if r_matches.iloc[0, 0] in ["TRUE", "FALSE"]:
        r_matches = (r_matches == "TRUE")
    r_gc = pd.read_csv(r_output_dir / "gc_content.csv", index_col=0)
    r_bg = pd.read_csv(r_output_dir / "background_peaks.csv", index_col=0)
    r_scores = pd.read_csv(r_output_dir / "deviation_scores.csv", index_col=0).T
    
    # === GC Content Comparison ===
    print("\n  [GC Content] Comparing GC bias computation...")
    peak_seqs_df = pd.read_csv(r_output_dir / "peak_sequences.csv")
    
    def compute_gc(seq):
        seq = seq.upper()
        if len(seq) == 0: return 0.0
        return (seq.count('G') + seq.count('C')) / len(seq)
    
    peak_seqs_df['python_gc'] = peak_seqs_df['sequence'].apply(compute_gc)
    peak_seqs_df.set_index('peak', inplace=True)
    gc_comp = r_gc.join(peak_seqs_df[['python_gc']])
    gc_corr = gc_comp.iloc[:, 0].corr(gc_comp['python_gc'])
    print(f"    GC Correlation (R vs Python): {gc_corr:.6f}")
    
    # === GATAC Motif Scanning ===
    print("\n  [GATAC] Scanning motifs (motifmatchr mode)...")
    adata_gatac = adata.copy()
    
    gatac_start = time.time()
    ga.tl.scan_motifs(
        adata_gatac,
        motifs,
        str(genome_path),
        pvalue=5e-5,
        mode="motifmatchr",
        coordinate_system="1-based",
    )
    gatac_time = time.time() - gatac_start
    
    gatac_matches = adata_gatac.varm["motif_match"].toarray()
    
    # Compare motif matches
    print("\n  [Motif Match Comparison]")
    try:
        r_matches_aligned = r_matches.reindex(
            index=adata_gatac.var_names, 
            columns=adata_gatac.uns["motif_name"]
        ).fillna(False).values.astype(bool)
        
        match_corr = np.corrcoef(r_matches_aligned.flatten(), gatac_matches.flatten().astype(bool))[0, 1]
        print(f"    GATAC matches: {np.sum(gatac_matches):,}")
        print(f"    R matches:     {np.sum(r_matches_aligned):,}")
        print(f"    Match Correlation: {match_corr:.6f}")
    except Exception as e:
        print(f"    Alignment error: {e}")
        match_corr = 0.0
    
    # === ChromVAR Deviations ===
    print("\n  [ChromVAR Deviations]")
    
    # Use R background peaks
    r_bg_indices = r_bg.values - 1  # R is 1-indexed
    adata_gatac.varm["bg_peaks"] = r_bg_indices.astype(np.int32)
    
    dev_start = time.time()
    dev_adata = ga.tl.chromvar(adata_gatac)
    dev_time = time.time() - dev_start
    
    # Compare deviation scores
    common_cells = list(set(r_scores.index) & set(dev_adata.obs_names))
    common_motifs = list(set(r_scores.columns) & set(dev_adata.var_names))
    print(f"    Common cells: {len(common_cells)}, Common motifs: {len(common_motifs)}")
    
    if common_motifs:
        r_scores_aligned = r_scores.loc[common_cells, common_motifs]
        gatac_scores_aligned = pd.DataFrame(
            dev_adata.X, index=dev_adata.obs_names, columns=dev_adata.var_names
        ).loc[common_cells, common_motifs]
        
        overall_corr = np.corrcoef(r_scores_aligned.values.flatten(), gatac_scores_aligned.values.flatten())[0, 1]
        print(f"    GATAC scores mean: {np.mean(dev_adata.X):.4f}")
        print(f"    R scores mean:     {np.mean(r_scores.values):.4f}")
        print(f"    Deviation Score Correlation: {overall_corr:.6f}")
    else:
        overall_corr = 0.0
    
    print(f"\n    Timing: Motif scan {gatac_time:.2f}s, ChromVAR {dev_time:.2f}s")
    
    results_dict["chromvar_full_pipeline"] = {
        "gc_correlation": gc_corr,
        "match_correlation": match_corr,
        "deviation_correlation": overall_corr,
        "gatac_matches": int(np.sum(gatac_matches)),
        "r_matches": int(np.sum(r_matches_aligned)) if 'r_matches_aligned' in dir() else 0,
        "motif_scan_time": gatac_time,
        "chromvar_time": dev_time,
    }


# =============================================================================
# TEST 3: ChromVAR Dev Scores - Same Motifs Comparison
# =============================================================================

def test_chromvar_same_motifs(adata, r_output_dir, results_dict):
    """
    Test 3: Compare ChromVAR deviation scores using IDENTICAL motif matches.
    
    This test uses R's motif matches directly in GATAC, showing that deviation
    scores are similar when using the same motif inputs (isolating the deviation
    calculation from motif matching differences).
    """
    print("\n" + "="*70)
    print("TEST 3: ChromVAR Dev Scores - Same Motifs Comparison")
    print("="*70)
    print("  Using R's motif matches in GATAC to isolate deviation calculation")
    
    # Load R outputs
    r_matches = pd.read_csv(r_output_dir / "motif_matches.csv", index_col=0)
    if r_matches.iloc[0, 0] in ["TRUE", "FALSE"]:
        r_matches = (r_matches == "TRUE")
    r_gc = pd.read_csv(r_output_dir / "gc_content.csv", index_col=0)
    r_bg = pd.read_csv(r_output_dir / "background_peaks.csv", index_col=0)
    r_scores = pd.read_csv(r_output_dir / "deviation_scores.csv", index_col=0).T
    
    # Create adata with R's motif matches
    adata_same = adata.copy()
    
    # Set R's GC content
    adata_same.var["gc_content"] = r_gc.loc[adata_same.var_names].values.flatten()
    
    # Set R's motif matches directly
    motif_names = list(r_matches.columns)
    r_matches_matrix = r_matches.reindex(index=adata_same.var_names).fillna(False).values.astype(bool)
    
    # Store as sparse matrix
    adata_same.varm["motif_match"] = sp.csr_matrix(r_matches_matrix)
    adata_same.uns["motif_name"] = motif_names
    
    print(f"\n  Using R's {len(motif_names)} motifs")
    print(f"  R motif matches: {np.sum(r_matches_matrix):,}")
    
    # Use R background peaks
    r_bg_indices = r_bg.values - 1  # R is 1-indexed
    adata_same.varm["bg_peaks"] = r_bg_indices.astype(np.int32)
    
    # Run GATAC chromVAR
    print("\n  [GATAC ChromVAR] Running with R's motif matches...")
    dev_start = time.time()
    dev_adata = ga.tl.chromvar(adata_same)
    dev_time = time.time() - dev_start
    
    # Compare deviation scores
    common_cells = list(set(r_scores.index) & set(dev_adata.obs_names))
    common_motifs = list(set(r_scores.columns) & set(dev_adata.var_names))
    print(f"    Common cells: {len(common_cells)}, Common motifs: {len(common_motifs)}")
    
    if common_motifs:
        r_scores_aligned = r_scores.loc[common_cells, common_motifs]
        gatac_scores_aligned = pd.DataFrame(
            dev_adata.X, index=dev_adata.obs_names, columns=dev_adata.var_names
        ).loc[common_cells, common_motifs]
        
        overall_corr = np.corrcoef(r_scores_aligned.values.flatten(), gatac_scores_aligned.values.flatten())[0, 1]
        
        # Per-motif correlations
        per_motif_corrs = []
        for motif in common_motifs:
            r_col = r_scores_aligned[motif].values
            g_col = gatac_scores_aligned[motif].values
            if np.std(r_col) > 0 and np.std(g_col) > 0:
                corr = np.corrcoef(r_col, g_col)[0, 1]
                per_motif_corrs.append(corr)
        
        mean_per_motif_corr = np.mean(per_motif_corrs) if per_motif_corrs else 0.0
        
        print(f"\n    GATAC scores mean: {np.mean(dev_adata.X):.4f}, std: {np.std(dev_adata.X):.4f}")
        print(f"    R scores mean:     {np.mean(r_scores.values):.4f}, std: {np.std(r_scores.values):.4f}")
        print(f"\n    Overall Deviation Correlation: {overall_corr:.6f}")
        print(f"    Mean Per-Motif Correlation:    {mean_per_motif_corr:.6f}")
    else:
        overall_corr = 0.0
        mean_per_motif_corr = 0.0
    
    print(f"\n    ChromVAR Time: {dev_time:.2f}s")
    
    results_dict["chromvar_same_motifs"] = {
        "deviation_correlation": overall_corr,
        "mean_per_motif_correlation": mean_per_motif_corr,
        "chromvar_time": dev_time,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    start_time = time.time()
    results = {}
    
    # Paths
    r_output_dir = Path("data/chromvar_r_output")
    genome_path = Path("data/hg19.fa.gz")
    
    # Check prerequisites
    if not r_output_dir.exists():
        print(f"ERROR: R output directory not found: {r_output_dir}")
        print("Run chromvar_comparison_R.R first to generate reference data")
        sys.exit(1)
    
    if not genome_path.exists():
        print(f"ERROR: Genome file not found: {genome_path}")
        sys.exit(1)
    
    # Load data
    print("Loading data...")
    count_mat = pd.read_csv(r_output_dir / "count_matrix.csv", index_col=0)
    r_gc = pd.read_csv(r_output_dir / "gc_content.csv", index_col=0)
    motifs_df = pd.read_csv(r_output_dir / "motifs_pwm.csv")
    
    # Create AnnData
    adata = ad.AnnData(
        X=count_mat.T.values.astype(np.float32),
        obs=pd.DataFrame(index=count_mat.columns),
        var=pd.DataFrame(index=count_mat.index)
    )
    adata.var["gc_content"] = r_gc.loc[adata.var_names].values.flatten()
    
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} peaks")
    
    # Load motifs
    motifs = load_motifs_from_csv(r_output_dir / "motifs_pwm.csv")
    print(f"  Loaded: {len(motifs)} motifs")
    
    # Run tests
    test_scprinter_vs_gatac(adata, motifs, motifs_df, genome_path, results)
    test_chromvar_full_pipeline(adata, motifs, r_output_dir, genome_path, results)
    test_chromvar_same_motifs(adata, r_output_dir, results)
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    log_lines = []
    
    # Test 1 Summary
    if "scprinter_vs_gatac" in results:
        r = results["scprinter_vs_gatac"]
        if r.get("status") == "SKIPPED":
            log_lines.append(f"Test 1 (scPrinter vs GATAC):\tSKIPPED ({r.get('reason', 'unknown')})")
        else:
            log_lines.append(f"Test 1 (scPrinter vs GATAC) Correlation:\t{r['correlation']:.6f}")
            log_lines.append(f"Test 1 (scPrinter vs GATAC) Agreement:\t{r['agreement']:.6f}")
    
    # Test 2 Summary
    if "chromvar_full_pipeline" in results:
        r = results["chromvar_full_pipeline"]
        log_lines.append(f"Test 2 (Full R Pipeline) GC Correlation:\t{r['gc_correlation']:.6f}")
        log_lines.append(f"Test 2 (Full R Pipeline) Match Correlation:\t{r['match_correlation']:.6f}")
        log_lines.append(f"Test 2 (Full R Pipeline) Deviation Correlation:\t{r['deviation_correlation']:.6f}")
    
    # Test 3 Summary
    if "chromvar_same_motifs" in results:
        r = results["chromvar_same_motifs"]
        log_lines.append(f"Test 3 (Same Motifs) Deviation Correlation:\t{r['deviation_correlation']:.6f}")
        log_lines.append(f"Test 3 (Same Motifs) Per-Motif Correlation:\t{r['mean_per_motif_correlation']:.6f}")
    
    log_lines.append(f"Total Time:\t{total_time:.2f}s")
    
    for line in log_lines:
        print(line)
    
    # Save results
    log_file = Path(__file__).stem + ".log"
    with open(log_file, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')
    print(f"\nResults saved to {log_file}")


if __name__ == '__main__':
    main()
