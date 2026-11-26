# Java, Python and C# on the same model — 2026-11-26 (v1.20.1)

Same harness, model, seed and protocol as [2026-11-25](2026-11-25-three-engines-eec_full.md), at
git `64d658bd`, after the two follow-ups ([findings](2026-11-26-environment-and-collection.md)):
NetCDF opened without xarray's backend-plugin scan, and Numba pinned below 0.66 (the production
`shiny` environment now has Python 3.13.9, NumPy 2.4.6, Numba 0.65.1).

Raw records: [70 years](2026-11-26-three-engines-eec_full.json),
[5 years](2026-11-26-three-engines-eec_full-5yr.json).

## 70 years (median of five; minimum–maximum)

| Engine | Threads | Wall time | vs Java | Peak memory | vs 2026-11-25 |
|---|---|---|---|---|---|
| Java 4.4.1 | none | 209.3 s (186.9–230.3) | 1.0× | 588 MB | −1% (noise) |
| Python | 1 | 94.9 s (92.3–95.5) | 2.2× | 571 MB | **−26%** |
| Python | 14 | 48.6 s (46.1–50.0) | 4.3× | 578 MB | −1% |
| C# | 1 | 32.8 s (32.4–33.3) | 6.4× | 485 MB | +1% |
| C# | 14 | 13.4 s (12.9–13.6) | 16× | 493 MB | −4% |

## 5 years

| Engine | Threads | Wall time | vs Java | Peak memory | vs 2026-11-25 |
|---|---|---|---|---|---|
| Java 4.4.1 | none | 7.8 s (7.6–8.3) | 1.0× | 475 MB | −1% |
| Python | 1 | 5.6 s (5.6–5.9) | 1.4× | 522 MB | **−20%** |
| Python | 14 | 3.7 s (3.6–3.9) | 2.1× | 523 MB | −7% |
| C# | 1 | 4.8 s (4.7–5.3) | 1.6× | 450 MB | **−19%** |
| C# | 14 | 4.3 s (4.3–4.5) | 1.8× | 448 MB | **−21%** |

## Reading it

- **Numba below 0.66** brings the Python engine's one-thread 70-year run from 128.5 s to 94.9 s.
  On 14 threads the kernels are a smaller share of the run and it barely moves.
- **The NetCDF fix** saves about 0.9 s of start-up in every Python and C# process: it shows on
  the short runs (−19 to −21% at 5 years) and is lost in the noise of the long ones.
- **Peak memory** is still ~150 MB above the 2026-11-22 `.venv` figures. It is not the plugin
  scan any more (no scan, `rioxarray` never loaded) and not threads or Numba compilation; it
  appears during the simulation and follows the Python 3.13 library stack (a 1-year Python run:
  `.venv` 163 -> 291 MB, `shiny` 163 -> 439 MB). Not investigated further.
