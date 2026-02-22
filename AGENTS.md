use pixi to run code and manage environment. always run test from the base directory:

```bash
pixi run python test/{name_of_test}.py
```

## Designing a new test file

Every test in `test/` follows the same structure. Use the existing files as
reference (e.g. `feature_selection.py`, `spectral_embedding.py`).

### 1. Function signature

Wrap the test body in a single `test_<name>(run_gatac_only=False)` function so
pytest can collect it:

```python
def test_foo(run_gatac_only=False):
    ...
```

### 2. `--run-gatac-only` flag

Always add an `argparse` block so the test can run GATAC-only (useful when
SnapATAC2 is unavailable or slow):

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATAC <feature>")
    parser.add_argument(
        "--run-gatac-only",
        action="store_true",
        help="Run GATAC only, skip SnapATAC2 run and comparison",
    )
    args = parser.parse_args()
    test_foo(run_gatac_only=args.run_gatac_only)
```

### 3. Timing

Wrap every tool call with `time.perf_counter()`:

```python
t0 = time.perf_counter()
ga.tl.foo(adata)
gatac_time = time.perf_counter() - t0
```

Always include `Speedup: {snap_time / gatac_time:.1f}x` in the log when both
tools are run.

### 4. Log file

Collect all result strings into a `results` list and write them to
`test/<name>.log` (next to the test file). Always do this in **both** the
comparison branch and the `run_gatac_only` branch:

```python
results = [
    "=== <Feature> Benchmark ===",
    f"Matrix: {n_cells:,} cells × {n_features:,} features",
    "",
    f"SnapATAC2:\t{snap_time:.2f}s",
    f"GATAC:\t{gatac_time:.2f}s",
    ...
]

log_path = os.path.join(os.path.dirname(__file__), "<name>.log")
with open(log_path, "w", encoding="utf-8") as f:
    for line in results:
        print(line)
        f.write(line + "\n")
```

### 5. Assertions

Place `assert` statements **after** the log is written so the log is always
produced even when a check fails in a future run. Assertions should cover:

- **Correctness** – correlation, overlap ratio, Jaccard, etc. vs SnapATAC2
  output (only when `not run_gatac_only`)
- **Thresholds** – use tight but realistic values derived from an initial
  passing run (e.g. correlation > 0.99, overlap > 99.5 %)

```python
assert metric > threshold, f"<metric> too low: {metric:.4f} (expected > {threshold})"
```

### Minimal skeleton

```python
import os, time, argparse
import snapatac2 as snap
import gatac as ga

def test_foo(run_gatac_only=False):
    # 1. load data
    # 2. run GATAC + time it
    # 3. if not run_gatac_only: run SnapATAC2 + time it, compute metrics
    # 4. build results list
    # 5. write log
    # 6. assert (only when not run_gatac_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-gatac-only", action="store_true")
    args = parser.parse_args()
    test_foo(run_gatac_only=args.run_gatac_only)
```