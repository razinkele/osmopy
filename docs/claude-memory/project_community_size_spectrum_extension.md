---
name: project-community-size-spectrum-extension
description: Sheldon NBSS mass spectrum + MTL/MTI + ABC W-statistic community diagnostics — SHIPPED 2026-06-17 PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

Community ecosystem-state diagnostics — SHIPPED 2026-06-17, master `9d176d3` (PR #73; changelog PR #74 `b172c43`); prod-verified live. Extends the length-only [[size_spectrum_diagnostics]] with the canonical **Sheldon (body-mass) NBSS spectrum** + community indicators.

`osmose/community_metrics.py` (pure; reuses `size_spectrum._read_community_by_size`/`_infer_bin_width`/`_window_by_time` + `analysis.size_spectrum_slope` + `OsmoseResults`):
- **Sheldon NBSS**: per-species length→mass via config `W=a·L^b` (`species.length2weight.condition.factor`/`allometric.power.spN`), equal log₂ (octave) mass bins, NBSS = bin biomass / lower-edge, log-log slope. Canonical normalized-biomass slope ≈ **−1** (NOT 0 — that's un-normalized per-octave; review fix). **Size diversity** = Shannon evenness over RAW per-octave biomass, NOT NBSS density (review fix — w_ref-alignment dependence).
- **MTL / Marine Trophic Index** (biomass-weighted standing-stock analogue of catch-based Pauly & Watson, TL≥3.25) from `meanTL`.
- **Warwick ABC W-statistic** = Σ(Bi−Ai)/(50(S−1)) over cumulative dominance curves (ranked separately desc); W>0 undisturbed, W<0 disturbed, ∈[−1,1].
- `community_report` orchestrator: each unit degrades to `None`+note on missing file (catches only FileNotFoundError); `format_community_report` markdown.
- Plots `make_sheldon_spectrum_plot`/`make_abc_plot` in `plotting.py`; Diagnostics page: 2 chart rtypes (`sheldon_spectrum`,`abc_curve`, branch BYPASSES `_get_result_data`, must stay above the catch-all) + a "Community Metrics" nav_panel.
- Also fixed `OsmoseResults._ENGINE_SUBDIRS` (+Trophic/Indicators/SizeIndicators/AgeIndicators) so it finds Java outputs in subdirs (guarded by `_matches_output_type`).

Process: brainstorm→spec→plan→**4-angle in-loop review** (sci/API/test-math/arch + verify round; caught 1 blocker+5 majors PRE-code)→**subagent-driven TDD** (8 tasks, spec+quality review each)→final review. Spec/plan in `docs/superpowers/{specs,plans}/2026-06-17-community-size-spectrum-extension*`.

**KNOWN GAP (not a regression):** on **Python-engine** Baltic runs, Sheldon + Trophic show "unavailable" — the Python engine does NOT persist `biomassDistribBySize` or 1D `meanTL` (existing Size-Spectrum diagnostic equally empty). ABC works (biomass+abundance ARE written). Validated end-to-end vs Java EEC output. Follow-up logged in [[project_feature_improvements_backlog]].
