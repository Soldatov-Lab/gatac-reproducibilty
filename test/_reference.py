"""
Cached outputs of the reference tools (SnapATAC2, gseapy, ...).

A test passes its reference tool's inputs as the cache key: while they stay
the same, later runs load the stored result and its run time instead of
running the tool again, and still make every comparison. The run time is the
one measured when the result was stored, with the cores the tool had then.
``--rerun-reference`` (``run_suite.py --rerun-references``) runs the tool
again, e.g. after updating it.

Results are pickled under data/references/<name>.pkl.
"""

import hashlib
import os
import pickle
import time

from _paths import data_path
from _timing import N_CPUS


def key(*parts) -> str:
    """Hash of the inputs a reference result depends on (anything picklable)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(pickle.dumps(p, protocol=5))
    return h.hexdigest()


def file_key(path) -> str:
    """Hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def cached(name: str, inputs: str, run, rerun: bool = False):
    """``run()``'s result and seconds, from the cache while ``inputs`` (a :func:`key`)
    is the one it was stored with; otherwise run, time and store it."""
    path = data_path("references", f"{name}.pkl")
    if not rerun and os.path.exists(path):
        with open(path, "rb") as f:
            entry = pickle.load(f)
        if entry["inputs"] == inputs:
            print(f"{name}: cached reference result ({entry['seconds']:.2f}s with {entry['cores']} cores)")
            return entry["value"], entry["seconds"]
        print(f"{name}: inputs changed, rerunning the reference")
    t0 = time.perf_counter()
    value = run()
    seconds = time.perf_counter() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"inputs": inputs, "value": value, "seconds": seconds, "cores": N_CPUS}, f, protocol=5)
    os.replace(path + ".tmp", path)
    return value, seconds
