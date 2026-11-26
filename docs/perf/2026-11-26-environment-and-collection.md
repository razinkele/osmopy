# Follow-ups to the 2026-11-25 three-engine assessment — 2026-11-26

## 1. Why the production environment (`shiny`) is slower and heavier than `.venv`

**Memory (+~200 MB) and start-up (+0.9 s): fixed.** `xr.open_dataset` with no engine (or an engine
named by string) makes xarray list every installed backend plugin; `shiny` has one more,
`rioxarray`, which imports rasterio/GDAL on the first open. Opening with the netCDF4 backend CLASS
(`osmose/netcdf_io.py`) skips the scan. EEC config preparation in `shiny`: 365 MB / 1.37 s before,
160 MB / 0.46 s after (`.venv`: 162 MB / 0.43 s). Guarded by `tests/test_netcdf_io.py`.

**Simulation (+25% Python, +~50% in the compiled mortality kernels): Numba >= 0.66.** EEC 5 years,
one thread, time inside `mortality()` (the Numba kernels), throwaway venvs on the `shiny` Python:

| Python | Numba (llvmlite) | NumPy | mortality kernels |
|---|---|---|---|
| 3.12.3 | 0.65.1 (0.47.0) | 2.4.6 | 2.30 s |
| 3.13.9 | 0.65.1 (0.47.0) | 2.4.6 | 2.29 s; 2.43–2.54 s with a fresh cache |
| 3.13.9 | 0.66.0 (0.48.0) | 2.4.6 | 3.53 s |
| 3.13.9 | 0.67.0 (0.49.0) | 2.4.6 | 3.49 s |
| 3.13.9 | 0.68.1 (0.50.0) | 2.4.6 | 3.66 s |
| 3.13.9 | 0.68.1 (0.50.0) | 2.5.5 | 3.46–3.76 s |

Neither Python 3.13 nor NumPy 2.5 is the cause. `NUMBA_OPT=2/3`, `NUMBA_SLP_VECTORIZE=1` and
`NUMBA_LOOP_VECTORIZE=0` (fresh caches) did not recover it; SVML is off in both. Not yet localised to
a kernel construct. Remedy available: pin `numba<0.66` (owner decision; the env is shared).

## 2. Per-step collection cost of the outputs added since 2026-11-22

EEC 70 years, C# at 14 threads, all new families on vs off: +~3 s, of which outputs phase
+1.9 s (staged diet collection 1.2 s, Java-rule distributions 0.3 s), mortality +0.75 s, life
cycle +0.2 s. Tried on the staged diet collector, all bit-identical: skipping zero columns (no
gain), splitting the accumulation by predator species (1.2 -> 0.8 s), by column block (slower).
The cost is memory traffic over the dense per-school staged rows; a real cut needs sparse rows in
both engines' kernels. Nothing shipped for this part.
