#!/usr/bin/env python3
"""
GPU memory line-level tracer for Python scripts.

Automatically injects NVTX markers at Python function boundaries so that
nsys profiling can correlate GPU memory events to specific Python source lines.
Also optionally samples GPU memory at each function call/return to identify
which Python function caused the largest GPU memory increase.

Usage:
    # With nsys (recommended — gives full GPU memory timeline with Python context):
    nsys profile --trace=cuda,nvtx ... pixi run python -m test.gpu_line_tracer test/peak_calling.py --skip-snapatac2

    # Standalone (fast summary of GPU memory per function):
    pixi run python -m test.gpu_line_tracer test/peak_calling.py --skip-snapatac2

    # Control what gets traced:
    GPU_TRACE_MODULES=gatac,cudf pixi run python -m test.gpu_line_tracer test/peak_calling.py

Environment variables:
    GPU_TRACE_MODULES   Comma-separated list of module name prefixes to trace.
                        Default: "gatac,test/,test." (traces gatac and test scripts).
    GPU_TRACE_LINES     If set to "1", also emit NVTX markers per source line
                        (very slow, use only for narrow debugging).
    GPU_TRACE_MEMORY    If set to "0", disable pynvml memory sampling (NVTX only).
                        Default: "1" (enabled).

Does NOT require any modification to the target script.
"""

import sys
import os
import time
import runpy
import atexit
import threading
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# NVTX helpers (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import nvtx as _nvtx

    def _nvtx_push(msg: str):
        _nvtx.push_range(msg)

    def _nvtx_pop():
        _nvtx.pop_range()

except ImportError:
    try:
        import cupy.cuda.nvtx as _cnvtx

        def _nvtx_push(msg: str):
            _cnvtx.RangePush(msg)

        def _nvtx_pop():
            _cnvtx.RangePop()

    except ImportError:
        # No NVTX available — silently do nothing
        def _nvtx_push(msg: str):
            pass

        def _nvtx_pop():
            pass


# ---------------------------------------------------------------------------
# pynvml helpers
# ---------------------------------------------------------------------------
_nvml_handle = None


def _init_nvml():
    global _nvml_handle
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _nvml_handle = None


def _gpu_mem_used_bytes() -> int:
    """Return current GPU memory used (bytes) via NVML, or -1 if unavailable."""
    if _nvml_handle is None:
        return -1
    try:
        import pynvml
        info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return info.used
    except Exception:
        return -1


class GPUMemPoller(threading.Thread):
    """Background thread that polls GPU memory to capture peaks between trace events."""

    def __init__(self, tracer, interval: float = 0.01):
        super().__init__(daemon=True)
        self.tracer = tracer
        self.interval = interval
        self._stop_event = threading.Event()
        self.global_peak = 0

    def run(self):
        while not self._stop_event.is_set():
            used = _gpu_mem_used_bytes()
            if used >= 0:
                if used > self.global_peak:
                    self.global_peak = used

                # Update all active peaks in the call stack
                with self.tracer._lock:
                    for i in range(len(self.tracer._active_peaks)):
                        if used > self.tracer._active_peaks[i]:
                            self.tracer._active_peaks[i] = used
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Tracer core
# ---------------------------------------------------------------------------
class GPULineTracer:
    """Traces Python function calls, emits NVTX markers and samples GPU memory."""

    def __init__(
        self,
        trace_prefixes: list[str] | None = None,
        trace_lines: bool = False,
        sample_memory: bool = True,
    ):
        # Module prefixes to trace (matched against absolute file paths)
        if trace_prefixes is None:
            env = os.environ.get("GPU_TRACE_MODULES", "gatac")
            trace_prefixes = [p.strip() for p in env.split(",") if p.strip()]
        self.trace_prefixes = trace_prefixes

        self.trace_lines = trace_lines or os.environ.get("GPU_TRACE_LINES") == "1"
        self.sample_memory = sample_memory and os.environ.get("GPU_TRACE_MEMORY", "1") != "0"

        # Book-keeping for memory sampling
        self._depth = 0
        self._call_stack: list[tuple[str, int, float]] = []  # (label, mem_before, t_start)
        self._active_peaks: list[int] = []  # Tracks max memory seen while each level is active
        self._poller: GPUMemPoller | None = None

        # Aggregated stats per (file, lineno, funcname)
        #   key -> {'calls': int, 'mem_delta_sum': int, 'mem_peak_increase': int,
        #           'mem_peak_abs': int, 'time_sum': float, 'file': str,
        #           'lineno': int, 'func': str}
        self.stats: dict[str, dict] = defaultdict(lambda: {
            "calls": 0,
            "mem_delta_sum": 0,
            "mem_peak_increase": 0,
            "mem_peak_abs": 0,
            "time_sum": 0.0,
            "file": "",
            "lineno": 0,
            "func": "",
        })
        self._lock = threading.Lock()
        self._start_wall = time.monotonic()

    # ---- filtering --------------------------------------------------------

    def _should_trace(self, filename: str) -> bool:
        """Return True if *filename* belongs to a module we want to trace."""
        if not self.trace_prefixes:
            return True
        for prefix in self.trace_prefixes:
            if prefix in filename:
                return True
        return False

    # ---- sys.settrace callback --------------------------------------------

    def _trace_calls(self, frame, event, arg):
        co = frame.f_code
        filename = co.co_filename

        if not self._should_trace(filename):
            return None  # don't trace locals in this frame

        if event == "call":
            short = _short_path(filename)
            label = f"{short}:{frame.f_lineno}:{co.co_name}"
            _nvtx_push(label)

            mem_now = _gpu_mem_used_bytes() if self.sample_memory else -1
            self._call_stack.append((label, mem_now, time.monotonic()))
            if self.sample_memory:
                with self._lock:
                    self._active_peaks.append(mem_now)
            self._depth += 1
            return self._trace_calls

        if event == "return":
            if self._depth > 0:
                _nvtx_pop()
                self._depth -= 1

                if self._call_stack:
                    label, mem_before, t_start = self._call_stack.pop()
                    mem_after = _gpu_mem_used_bytes() if self.sample_memory else -1
                    elapsed = time.monotonic() - t_start

                    peak_during = -1
                    if self.sample_memory:
                        with self._lock:
                            peak_during = self._active_peaks.pop()
                            # Ensure we at least captured the return value
                            if mem_after > peak_during:
                                peak_during = mem_after

                    if mem_before >= 0 and mem_after >= 0:
                        delta = mem_after - mem_before
                    else:
                        delta = 0

                    with self._lock:
                        s = self.stats[label]
                        s["calls"] += 1
                        s["mem_delta_sum"] += delta
                        if delta > s["mem_peak_increase"]:
                            s["mem_peak_increase"] = delta
                        if peak_during > s["mem_peak_abs"]:
                            s["mem_peak_abs"] = peak_during
                        s["time_sum"] += elapsed
                        if not s["file"]:
                            parts = label.rsplit(":", 2)
                            s["file"] = parts[0] if len(parts) == 3 else label
                            s["lineno"] = int(parts[1]) if len(parts) == 3 else 0
                            s["func"] = parts[2] if len(parts) == 3 else ""

            return self._trace_calls

        if event == "line" and self.trace_lines:
            short = _short_path(filename)
            label = f"L {short}:{frame.f_lineno}"
            _nvtx_push(label)
            _nvtx_pop()
            return self._trace_calls

        return self._trace_calls

    # ---- enable / disable -------------------------------------------------

    def enable(self):
        """Activate tracing."""
        if self.sample_memory:
            _init_nvml()
            self._poller = GPUMemPoller(self)
            self._poller.start()
        sys.settrace(self._trace_calls)
        threading.settrace(self._trace_calls)

    def disable(self):
        """Deactivate tracing."""
        sys.settrace(None)
        threading.settrace(None)
        if self._poller:
            self._poller.stop()
            self._poller.join(timeout=1.0)

    # ---- reporting --------------------------------------------------------

    def report(self, top_n: int = 30):
        """Print a summary of functions sorted by peak GPU memory increase."""
        if not self.stats:
            print("\n[gpu_line_tracer] No traced function calls recorded.")
            return

        wall = time.monotonic() - self._start_wall
        global_peak_mb = (self._poller.global_peak / (1024 * 1024)) if self._poller else 0

        print("\n" + "=" * 90)
        print("GPU Line Tracer — Function-level GPU Memory Report")
        print("=" * 90)
        print(f"Total wall time: {wall:.2f}s")
        print(f"Traced functions: {len(self.stats)}")
        if global_peak_mb:
            print(f"Overall peak GPU memory: {global_peak_mb:.2f} MB")
        print()

        # Sort by peak single-call memory increase (descending)
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: s["mem_peak_increase"],
            reverse=True,
        )

        # --- Top by peak increase ---
        print(f"{'Top functions by peak GPU memory increase (single call)':^90}")
        print("-" * 90)
        header = f"{'Function':<40} {'Calls':>6} {'Peak Δ':>10} {'Peak Res':>10} {'Cumul Δ':>10} {'Time':>8}"
        print(header)
        print("-" * 90)

        for s in sorted_stats[:top_n]:
            label = f"{s['file']}:{s['lineno']}:{s['func']}"
            if len(label) > 38:
                label = "…" + label[-(37):]
            peak_mb = s["mem_peak_increase"] / (1024 * 1024)
            res_mb = s["mem_peak_abs"] / (1024 * 1024)
            cumul_mb = s["mem_delta_sum"] / (1024 * 1024)
            print(
                f"{label:<40} {s['calls']:>6} {peak_mb:>10.1f} {res_mb:>10.1f} {cumul_mb:>10.1f} {s['time_sum']:>8.2f}"
            )

        # --- Top by cumulative increase ---
        sorted_cumul = sorted(
            self.stats.values(),
            key=lambda s: s["mem_delta_sum"],
            reverse=True,
        )

        print()
        print(f"{'Top functions by cumulative GPU memory delta':^90}")
        print("-" * 90)
        print(header)
        print("-" * 90)

        for s in sorted_cumul[:top_n]:
            label = f"{s['file']}:{s['lineno']}:{s['func']}"
            if len(label) > 38:
                label = "…" + label[-(37):]
            peak_mb = s["mem_peak_increase"] / (1024 * 1024)
            res_mb = s["mem_peak_abs"] / (1024 * 1024)
            cumul_mb = s["mem_delta_sum"] / (1024 * 1024)
            print(
                f"{label:<40} {s['calls']:>6} {peak_mb:>10.1f} {res_mb:>10.1f} {cumul_mb:>10.1f} {s['time_sum']:>8.2f}"
            )

        print("=" * 90)
        print()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _short_path(filepath: str) -> str:
    """Shorten a file path for display: keep the last 2-3 components."""
    parts = Path(filepath).parts
    if len(parts) <= 3:
        return str(Path(*parts))
    return str(Path(*parts[-3:]))


# ---------------------------------------------------------------------------
# __main__ entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a Python script with GPU line-level tracing (NVTX + memory sampling).",
        usage="python -m gpu_line_tracer.py [tracer options] <script.py> [script args ...]",
    )
    parser.add_argument(
        "--trace-modules",
        default="gatac,cudf,cupy",
        help="Comma-separated module prefixes to trace (default: $GPU_TRACE_MODULES or 'gatac').",
    )
    parser.add_argument(
        "--trace-lines",
        action="store_true",
        default=False,
        help="Also emit per-line NVTX markers (very slow).",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="Disable pynvml memory sampling (NVTX markers only).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top functions to show in report (default: 30).",
    )
    parser.add_argument(
        "script",
        help="Path to the Python script to run.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the script.",
    )

    args = parser.parse_args()

    # Resolve trace_modules
    prefixes = None
    if args.trace_modules:
        prefixes = [p.strip() for p in args.trace_modules.split(",")]

    tracer = GPULineTracer(
        trace_prefixes=prefixes,
        trace_lines=args.trace_lines,
        sample_memory=not args.no_memory,
    )

    # Prepare sys.argv for the target script
    script_path = os.path.abspath(args.script)
    sys.argv = [script_path] + args.script_args

    # Add script's directory to sys.path (like `python script.py` would)
    script_dir = os.path.dirname(script_path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Register report at exit
    atexit.register(tracer.report, top_n=args.top)

    print(f"[gpu_line_tracer] Tracing modules matching: {tracer.trace_prefixes}")
    print(f"[gpu_line_tracer] NVTX markers: enabled | Memory sampling: {tracer.sample_memory}")
    print(f"[gpu_line_tracer] Line-level tracing: {tracer.trace_lines}")
    print()

    # Enable tracing and run the user's script
    tracer.enable()
    try:
        # Use runpy to execute the script as __main__
        runpy.run_path(script_path, run_name="__main__")
    finally:
        tracer.disable()


if __name__ == "__main__":
    main()
