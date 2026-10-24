"""
How the tests time GATAC.

The first GPU calls of a process pay a one-off start-up: creating the CUDA
context, loading CUDA libraries (cuBLAS, cuSPARSE, ...) on first use and, for
clustering, importing cuML and cuGraph. It does not depend on the data. Most of
it is paid once per process, by whichever GPU function runs first; a smaller
part by each function the first time it runs. On an L40S, chromVAR's first call
in a fresh process took 0.48 s, 0.24 s after TSS enrichment had run, and 0.14 s
once warm.

The reference tools are timed with their libraries already loaded, so the
speedups use a second, warm call. The first call is reported next to it as the
cold start: with each test file run on its own, as the README runs them, that
is GATAC's first GPU work in a fresh process, the worst case.
"""

import os
import time

#: Cores this process may use: the SLURM allocation, which os.cpu_count()
#: ignores. Every reference tool that can use several cores is given all of
#: them (R scripts as NTHREADS).
N_CPUS = len(os.sched_getaffinity(0))


def cold_warm(fn):
    """Call ``fn`` twice; return the second result, the first call's
    seconds and the second call's seconds."""
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = fn()
    return result, cold, time.perf_counter() - t0
