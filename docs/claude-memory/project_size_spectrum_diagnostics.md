---
name: project_size_spectrum_diagnostics
description: Community size-spectrum diagnostics (Sheldon-style slope, LFI, mean size from *DistribBySize output) — pure-analysis feature, SHIPPED to origin/master 2026-06-04. Copernicus forcing is the queued next item.
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Community **size-spectrum diagnostics** (first of two science extensions the user picked; **Copernicus forcing integration is the queued NEXT item**). Pure-analysis feature like the ICES/delta/fisheries validators. Merged fast-forward to master + **pushed to origin/master 2026-06-04** (`32347e4..d7003de`, branch `feature/size-spectrum-diagnostics` deleted, origin synced). 18 tests; 6 impl commits.

## What shipped
- `osmose/size_spectrum.py`: `compute_size_spectrum(output_dir, *, metric, prefix, window_years, lfi_threshold_cm, min_size_cm) -> SizeSpectrum` (16-field frozen dataclass: spectrum curve, log-log slope+intercept+R²+n_bins_fit, LFI, mean_size, peak_size_cm, ...); `size_spectrum_timeseries` (per-step slope/LFI/mean); `format_size_spectrum_report` (markdown); `spectrum_plot_df`; helpers `_read_community_by_size`/`_community_long`/`_window_by_time`/`_infer_bin_width`/`_large_fish_indicator`/`_mean_size`/`_fit_slope`.
- `osmose/plotting.py`: NEW `make_size_indicator_timeseries` (trend chart). REUSES existing `make_size_spectrum_plot` (log-log scatter+regression) — not duplicated.
- `scripts/compute_size_spectrum.py` CLI; `tests/test_size_spectrum.py` (18); CHANGELOG `### Added`.
- EEC validated: biomass slope −1.90 (R²0.67), LFI@40 0.073, mean 20.8cm, peak 15cm; min_size_cm=10 → −2.77.

## Reuse (the review's key save) + hard-won data facts
- **REUSES** `osmose.analysis.size_spectrum_slope` (df cols `size`,`abundance`; raises ValueError if <2 positive → `_fit_slope` wraps →None) and `osmose.plotting.make_size_spectrum_plot`. The first spec draft was UNAWARE both already existed (round-1 review caught the duplication).
- **The `OsmoseResults.*_by_size()` accessors do NOT read the community `*DistribBySize` files** — they glob `*BySize` (not `*DistribBySize`), search only root+Mortality/Bioen (not `Indicators/`), and `_read_2d_output` mis-melts the `Time,Size,<species cols>` community layout (→ species="all", edges in `bin`). So the feature **reads the file DIRECTLY**: rglob `{prefix}_{metric}DistribBySize*.csv` + `osmose.results._read_output_csv` (preamble-safe; private cross-module import, acknowledged) → wide `Time,Size,<species>`; sum species per (Time,Size); window by Time. NO OsmoseResults/engine change.
- Real community file: `data/eec_full/output/Indicators/eec_{biomass,abundance}DistribBySize_Simu0.csv` (21 bins, 10cm width, 0–200, 14 spp, 70 yr). **Baltic's committed `*DistribBySize` are 0 bytes** (flag `output.biomass.bysize.enabled;true` IS on in `data/baltic/baltic_param-output.csv`; the committed run just didn't persist them → a re-run produces them). EEC is the validated substrate; Baltic is BYO-run-output (out of scope; a Baltic run would also reflect the cod ×17-48/percid ×100 overshoots → misleading showcase).
- 0-byte file → `_read_output_csv` raises `pandas.errors.EmptyDataError` (NOT FileNotFoundError); CLI catches both → exit 1.
- It is a **length–biomass spectrum over linear cm bins, for trend/comparison — NOT the canonical Sheldon normalized-by-body-mass exponent** (biomass vs abundance slope gap is allometric ~3, not the Sheldon ~1; the first spec wrongly claimed ~1). Small bins (recruits) dominate an all-bins fit → `min_size_cm` cutoff (compared to bin MIDPOINTS, not edges) + reported `peak_size_cm`/`n_bins_fit`/R² so the user fits the descending limb. LFI uses bin LOWER edge ≥ threshold (default 40cm OSPAR).

## Process (full superpowers flow, 2 in-loop review rounds on BOTH spec and plan)
brainstorm → spec → **spec round 1** (3 executing reviewers vs live EEC data: BLOCKER readers-don't-read-the-file + methodology gaps + existing-fns-to-reuse) → revise → **spec round 2** (prototyped corrected chain end-to-end, CLEAN) → plan → **plan round 1** (2 reviewers prototyped all test math: BLOCKER ruff E402 from mid-file appended test imports — NOT auto-fixed by ruff format; + NaN-JSON) → fix → subagent-driven build T1–T6 (per-task spec+quality review; implementer caught a real plan-test bug: `pandas.Series == pytest.approx` always False → use `.tolist()`) → final whole-impl review (READY TO MERGE). 18 + 128 (analysis|plotting|size_spectrum) tests pass, ruff clean.

## Gotchas carried
- **ruff E402** ("module import not at top") fires on imports appended after functions and is NOT fixed by `ruff format`/`--fix` → test files must keep ALL module-level imports in the top block (edit it per task; append only functions). Function-local imports are exempt.
- selectize/pandas-approx/etc. — see the per-feature memories.
- **Next: build the Copernicus forcing integration** (the 2nd science extension: MCP `generate_osmose_ltl`/`generate_osmose_physics` + engine `physical_data.py`/`temp_function.py`/`resources.py` both already exist → it's an end-to-end integration/demo, externally-coupled: CMEMS creds + network + a model run). See [[project_compare_runs_decouple]], [[project_result_delta_tracking]] for the analysis-feature pattern.
