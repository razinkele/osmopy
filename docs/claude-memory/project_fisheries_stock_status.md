---
name: project-fisheries-stock-status
description: 2026-06-25 indicative fisheries stock-status page (Kobe/B-Bmsy/F-Fmsy) shipped + the SSB-engine-output + cadence + scientific-reframe facts
metadata:
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

Shipped to local master 2026-06-25 (merge `ede0a77`, --no-ff; 10 deliverable commits `a9fe0ec..4a0e7d0` + spec/plan). A new **Fisheries** Shiny page (`ui/pages/fisheries.py`): indicative Kobe (B/Bmsy vs F/Fmsy) + B/Bmsy & F/Fmsy time-series + the existing-but-never-surfaced F/M bars. Library: `osmose/validation/{stock_status,fisheries_reference}.py` + `fisheries.annual_by_year` + `plotting.make_kobe_plot/make_ratio_timeseries`. Output-only, parity-safe (14/14 EEC atol=0 + 8/8 BoB unchanged). Unblocks the 2026-06-03 fisheries-diagnostics deferral. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-25-fisheries-stock-status*`.

## KEY durable facts (non-obvious)
- **SSB is now a real engine output** (wired the dormant `output.ssb.enabled` → `output_ssb`/`output_ssb_netcdf` config flags + `StepOutput.ssb` + `_collect_ssb` + CSV/in-memory/NetCDF + `results.ssb()` — the yieldN/meanSize pattern). `_collect_ssb` uses the engine's OWN maturity conjunction from `reproduction.py`: `length>=maturity_size AND age_dt>=maturity_age_dt AND abundance>0`, **mean**-aggregated across the record window (NOT summed). **Why a new output:** SSB CANNOT be reconstructed from the marginal `biomass_by_age`/`biomass_by_size` outputs (a species with BOTH size- and age-at-maturity needs the JOINT mask), and `maturity_age_dt` defaults to 0 → an age-only reconstruction silently gives SSB==total biomass.
- **CADENCE GOTCHA (was a CRITICAL plan bug):** outputs (mortalityRate, SSB, biomass) are saved every `output.recordfrequency.ndt` steps — Baltic = `ndtperyear` = once/year, and the engine SUMS mortality over the window. So annual aggregation must NOT assume one-row-per-step. Use `fisheries.annual_by_year(values, time, how=)` = **groupby `int(floor(Time))`** (absolute sim-year): SSB `how="mean"`, F `how="sum"`. Correct for ANY record frequency; labels both axes by absolute year (never positional).
- **Realized F**: single exploited stage (the fished stage `F>_FISHED_TOL`, Eggs excluded, with the LARGEST annual F; caveat when >1). Years from the mortalityRate Time column (`df.iloc[:,0]`).
- **App state has NO EngineConfig** — `state.config` is a flat `dict[str,str]`; the page builds `EngineConfig.from_dict(state.config.get())`. `prefix` from `OsmoseResults`. ICES snapshot dir resolved per-ecosystem (`data/<eco>/reference/ices_snapshots`).
- Reference sidecar at `data/<ecosystem>/reference/fisheries_reference_points.json` (NOT a per-run output dir — runs are fresh mkdtemp). `save` persists only USER fmsy, never the ICES auto-fill.

## Scientific reframe (deep scite + ICES-MCP literature review drove it)
The page is **INDICATIVE, not a formal assessment** (soft Kobe shading + disclaimer). Grounded findings: no precedent for OSMOSE-on-a-single-stock-Kobe — OSMOSE/EwE derive reference points INTERNALLY + compare biomass relatively (Travers-Trolet 2020, Mackinson 2018, Bănaru 2019); ICES publishes NO Bmsy (`msy_btrigger`=`Bpa`<Bmsy — verified sprat 541000=541000); summing MSY-Btrigger across a species' sub-stocks MASKS depleted components (Eero 2014 cod, Forrest 2023 herring). → **B-axis = user-supplied Bmsy ONLY** (SSB numerator); **ICES auto-fills Fmsy ONLY** (deterministic primary tonnes-stock = largest Btrigger; herring→her.27.3031/0.218); data-limited stocks (eastern Baltic cod = null fmsy) first-class. **Deferred v2 = model-internal reference points** (Fmsy from an OSMOSE yield-vs-F sweep). Key DOIs: 10.1111/faf.12591, 10.1093/icesjms/fsu060, 10.1139/cjfas-2022-0168, 10.3389/fmars.2020.568232, 10.1371/journal.pone.0190015.

## Process note
brainstorm → spec (3 in-loop workflow reviews: code-grounding, **deep scite+ICES literature** (~25 papers, reframed the whole feature), multi-angle UX/Shiml/integration/adversarial) → plan (apply-and-run workflow review: an agent applied all 7 tasks in an isolated worktree → caught the SSB-mean-aggregation + cadence bugs) → subagent-driven TDD (7 tasks, each reviewed; 2 fix loops). Related: [[project-python-engine-yieldn-meansize]] (the SSB output reused its pattern), [[reference-engine-mortality-dispatch]].
