"""
TOBIAS side of the footprinting test (runs in the ``tobias`` pixi env).

Subcommands, all driven by ``test/footprinting.py``:

* ``bam``     write a pseudobulk BAM from the fragments parquet, so TOBIAS sees
              exactly the cut sites GATAC reads (forward mate starts at
              ``start - 4``, reverse mate ends at ``end + 4``, 50-bp reads,
              one read pair per distinct fragment);
* ``run``     TOBIAS ATACorrect + ScoreBigwig per group, then BINDetect;
* ``bias64``  TOBIAS's fp64 DWM bias scores over the ATACorrect output regions
              (the bias bigwig is float32, too coarse to inject);
* ``dump``    bigwig values over a set of intervals, and the bias model of each
              group, as ``.npz``.

``run --perturb`` is the noise control: TOBIAS with its DWM bias scores perturbed by
uniform noise in ±1e-13, the size of floating-point summation-order differences.
"""
import argparse
import os
import pickle
import subprocess
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pysam
import sys

READ_LEN = 50


def cmd_bam(a):
    barcodes = [l.strip() for l in open(a.barcodes) if l.strip()]
    lengths = dict(l.split("\t")[:2] for l in open(a.fasta + ".fai"))
    lengths = {c: int(n) for c, n in lengths.items()}
    chroms = a.chroms.split(",")
    t = pq.read_table(a.parquet, columns=["chrom", "start", "end", "barcode"],
                      filters=[("chrom", "in", chroms), ("barcode", "in", barcodes)])
    t = t.group_by(["chrom", "start", "end"]).aggregate([])        # one pair per distinct fragment
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": c, "LN": lengths[c]} for c in chroms]}
    tmp = a.out + ".unsorted.bam"
    qual = pysam.qualitystring_to_array("I" * READ_LEN)
    seq = "N" * READ_LEN
    n = 0
    with pysam.AlignmentFile(tmp, "wb", header=header) as out:
        tid = {c: i for i, c in enumerate(chroms)}
        for c in chroms:
            sub = t.filter(pc.equal(t["chrom"], c))
            L = lengths[c]
            for s, e in zip(sub["start"].to_numpy(), sub["end"].to_numpy()):
                fwd, rev_end = int(s) - 4, int(e) + 4
                rev = rev_end - READ_LEN
                if fwd < 0 or rev < 0 or rev_end > L or fwd + READ_LEN > L:
                    continue
                for is_rev in (False, True):
                    r = pysam.AlignedSegment()
                    r.query_name = f"f{n}"
                    r.query_sequence = seq
                    r.flag = 1 | 2 | (16 if is_rev else 32) | (128 if is_rev else 64)
                    r.reference_id = r.next_reference_id = tid[c]
                    r.reference_start = rev if is_rev else fwd
                    r.next_reference_start = fwd if is_rev else rev
                    r.mapping_quality = 60
                    r.cigarstring = f"{READ_LEN}M"
                    r.template_length = -(rev_end - fwd) if is_rev else rev_end - fwd
                    r.query_qualities = qual
                    out.write(r)
                n += 1
    pysam.sort("-@", str(a.cores), "-o", a.out, tmp)
    pysam.index(a.out)
    os.remove(tmp)
    print(f"{n:,} read pairs -> {a.out}")


def _tobias(args, log, perturb=False):
    cmd = [sys.executable, os.path.abspath(__file__), "tobias-perturbed", *args] if perturb else ["TOBIAS", *args]
    with open(log, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def tobias_perturbed(args):
    """Stock TOBIAS with DiNucleotideMatrix.score_sequence perturbed by U(-1e-13, 1e-13)."""
    import tobias.utils.sequences as seqmod
    from tobias.TOBIAS import main

    orig = seqmod.DiNucleotideMatrix.score_sequence
    rng = np.random.default_rng(0)

    def noisy(self, sequence):
        s = orig(self, sequence)
        return s + rng.uniform(-1e-13, 1e-13, size=s.shape)

    seqmod.DiNucleotideMatrix.score_sequence = noisy
    sys.argv = ["TOBIAS", *args]
    main()


def cmd_run(a):
    groups = a.groups.split(",")
    sigs = []
    bamdir = a.bamdir or a.outdir
    os.makedirs(a.outdir, exist_ok=True)
    for g in groups:
        od = os.path.join(a.outdir, g)
        _tobias(["ATACorrect", "--bam", os.path.join(bamdir, f"{g}.bam"), "--genome", a.fasta,
                 "--peaks", a.peaks, "--outdir", od, "--prefix", g, "--cores", str(a.cores)],
                os.path.join(a.outdir, f"{g}.atacorrect.log"), perturb=a.perturb)
        _tobias(["ScoreBigwig", "--signal", os.path.join(od, f"{g}_corrected.bw"), "--regions", a.peaks,
                 "--output", os.path.join(od, f"{g}_footprints.bw"), "--cores", str(a.cores)],
                os.path.join(a.outdir, f"{g}.score.log"))
        sigs.append(os.path.join(od, f"{g}_footprints.bw"))
    _tobias(["BINDetect", "--motifs", a.motifs, "--signals", *sigs, "--cond-names", *groups,
             "--genome", a.fasta, "--peaks", a.peaks, "--outdir", os.path.join(a.outdir, "bindetect"),
             "--cores", str(a.cores), "--skip-excel"], os.path.join(a.outdir, "bindetect.log"))


def _bias64_work(args):
    from tobias.utils.regions import OneRegion
    from tobias.utils.sequences import GenomicSequence

    pkl, fasta, c, chunk = args
    bo = pickle.load(open(pkl, "rb"))
    for s in ("forward", "reverse"):
        bo.bias[s].prepare_mat()
    fa = pysam.FastaFile(fasta)
    fw, rv = [], []
    for s0, e0 in chunk:        # output regions extended by k_flank + window/2 = 62
        seq = GenomicSequence(OneRegion([c, int(s0) - 62, int(e0) + 62])).from_fasta(fa)
        fw.append(np.nan_to_num(bo.bias["forward"].score_sequence(seq.sequence)))
        rv.append(np.nan_to_num(bo.bias["reverse"].score_sequence(seq.revcomp)[::-1]))
    return np.concatenate(fw), np.concatenate(rv)


def cmd_bias64(a):
    reg = np.load(a.regions)
    pkl = os.path.join(a.outdir, a.group, f"{a.group}_AtacBias.pickle")
    out = {}
    with Pool(a.cores) as pool:
        for c in reg.files:
            if len(reg[c]) == 0:
                continue
            res = pool.map(_bias64_work, [(pkl, a.fasta, c, ch) for ch in np.array_split(reg[c], 4 * a.cores)
                                          if len(ch)])
            out[f"{c}__forward"] = np.concatenate([r[0] for r in res])
            out[f"{c}__reverse"] = np.concatenate([r[1] for r in res])
    np.savez(a.out, **out)


def cmd_dump(a):
    import pyBigWig

    out = {}
    for spec in a.bigwig:                     # key=path=intervals.npz
        key, path, ivf = spec.split("=")
        iv = np.load(ivf)
        bw = pyBigWig.open(path)
        for c in iv.files:
            vals = [np.nan_to_num(np.array(bw.values(c, int(s), int(e)), dtype=np.float64)) for s, e in iv[c]]
            out[f"{key}__{c}"] = np.concatenate(vals).astype(np.float32) if vals else np.zeros(0, np.float32)
    for g in a.groups.split(",") if a.groups else []:
        bo = pickle.load(open(os.path.join(a.outdir, g, f"{g}_AtacBias.pickle"), "rb"))
        out[f"{g}__correction_factor"] = np.array(bo.correction_factor)
        out[f"{g}__no_reads"] = np.array(bo.no_reads)
        for s in ("forward", "reverse"):
            m = bo.bias[s]
            m.prepare_mat()
            for k in ("counts", "bg_counts", "bias_pwm_log", "bias_dwm_log", "bg_pwm_log", "bg_dwm_log", "pssm"):
                out[f"{g}__{s}__{k}"] = getattr(m, k)
    np.savez(a.out, **out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bam")
    b.add_argument("parquet"); b.add_argument("barcodes"); b.add_argument("out")
    b.add_argument("--chroms", required=True); b.add_argument("--fasta", required=True)
    b.add_argument("--cores", type=int, default=4)
    r = sp.add_parser("run")
    r.add_argument("outdir"); r.add_argument("--groups", required=True); r.add_argument("--peaks", required=True)
    r.add_argument("--fasta", required=True); r.add_argument("--motifs", required=True)
    r.add_argument("--cores", type=int, default=6)
    r.add_argument("--bamdir", default=None, help="directory of the BAMs (default: outdir)")
    r.add_argument("--perturb", action="store_true", help="noise control: perturb the DWM bias scores")
    x = sp.add_parser("bias64")
    x.add_argument("outdir"); x.add_argument("group"); x.add_argument("regions"); x.add_argument("out")
    x.add_argument("--fasta", required=True); x.add_argument("--cores", type=int, default=6)
    d = sp.add_parser("dump")
    d.add_argument("outdir"); d.add_argument("out"); d.add_argument("--bigwig", nargs="*", default=[])
    d.add_argument("--groups", default="")
    if len(sys.argv) > 1 and sys.argv[1] == "tobias-perturbed":
        tobias_perturbed(sys.argv[2:])
        sys.exit(0)
    a = p.parse_args()
    {"bam": cmd_bam, "run": cmd_run, "bias64": cmd_bias64, "dump": cmd_dump}[a.cmd](a)
