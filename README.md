Full test suite for comparing gatac with exisitng tools, mainly snapatac2, but also macs3 and chromvar.

to be run in that order:

```bash
pixi run python test/fragment_loading.py
pixi run python test/tss_enrichment.py
pixi run python test/tile_matrix.py
pixi run python test/feature_selection.py
pixi run python test/peak_calling.py
pixi run python test/make_peak_matrix.py
pixi run python test/motif_enrichment.py
pixi run python test/chromvar_full_comparison.py
```

## Results Summary

| Test | Speedup | Result | Notes |
|------|---------|--------|-------|
| Fragment Loading | x3.5 | ✅ Full Match | Identical cell barcodes and fragment counts |
| TSS Enrichment | x1.2 | ✅ Correlation: 1.000 | Cell count match, perfect TSSe correlation |
| Tile Matrix | x4.5 | ✅ Full Match | Sum=123,061,807 for both tools |
| Feature Selection | x1.1 | ⚠️ Overlap: 99.8% | 80 features differ due to tie-breaking at boundary count (272 features at count=204) |
| Peak Calling | x4.1 | ⚠️ Jaccard: 0.963 | Different algorithms (gmacs vs MACS2); Recall snap=97.3%, Recall gatac=98.9% |
| Peak Matrix | x4.8 | ✅ Full Match | Shape match, Peak/Cell correlation=1.0 |
| Motif Enrichment | x7.5 | ⚠️ Avg Corr: 0.984 | Minor numerical differences in p-value calculation |


