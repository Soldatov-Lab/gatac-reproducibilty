Full test suite for comparing gatac with exisitng tools, mainly snapatac2, but also macs3 and chromvar.

to be run in that order:

```bash
pixi run python test/fragment_loading.py
pixi run python test/tile_matrix.py
pixi run python test/peak_calling.py
pixi run python test/make_peak_matrix.py
pixi run python test/motif_enrichment.py
```