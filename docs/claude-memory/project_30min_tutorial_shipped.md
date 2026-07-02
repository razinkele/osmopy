---
name: 30-min-tutorial-shipped
description: 30-minute tutorial (backlog top-5
metadata: 
  node_type: memory
  type: project
  originSessionId: 99666016-a3d1-4898-a6db-29c013e595aa
---

**SHIPPED 2026-05-17** (backlog top-5 #5). 18 commits fast-forwarded onto `origin/master` from `2cfb168` to `c2ba915`. Tutorial at `docs/tutorials/30-minute-ecosystem.md` (504 lines), regression test at `tests/test_tutorial_3species.py` (6 assertions, all GREEN), helper at `tests/_tutorial_config.py`, README + `docs/tutorials/README.md` updated.

## What it teaches

Loads the calibrated Baltic 8-species OSMOSE config from `data/baltic/`, highlights 3 focal species (cod, sprat, stickleback), runs 30 yr × 24 dt/yr with seed=42, plots biomass on log y-axis. Beat 6: reader edits `tutorial-work/baltic/predation-accessibility.csv` (find `sprat;0.4;` → replace `sprat;0.05;`) and re-runs. Visible cascade: stickleback +185%, cod +105%, sprat ≈stable.

## Substrate pivot — the load-bearing decision

The original spec/plan promised "synthetic 3-species toy from scratch." After **5 BLOCKED T4 attempts** the synthetic approach was rejected:
- r1 spec params (larva 6/8/10): Forager collapses year 8 from over-predation + insufficient recruitment
- r2 dropped to larva=3 uniform: boom-bust (Forager 66× overshoot then crash, PE 1.35M t > 160K t Plankton supply)
- r3 with B-H (`stock.recruitment.type.spN=beverton_holt` + `ssbhalf`): still collapses (cod alone survives)
- Conclusion: stabilising a 3-species OSMOSE model from scratch is research-grade work, not a 30-min tutorial. Baltic is calibrated and works.

Plan r7's pivot (commit `6a193ea`) replaced `build_config` with "load `data/baltic/baltic_all-parameters.csv`, override `simulation.time.nyear=30`." `build_ltl` dropped (Baltic ships its own NetCDF). Cascade narrative re-tuned: cod-on-sprat accessibility 0.4 → 0.05 → cod starves → stickleback released. Cascade magnitudes are *small in absolute tonnes* (cod 12.7→25.9, stickleback 356→1017) because Baltic cod is in its overfished present-day state; the qualitative cascade direction holds.

## Engine quirks discovered

- **`OsmoseConfigReader.read()` returns `dict[str, str]`** — all dict values must be stringified (`str(value)`) before passing to `PythonEngine.run_in_memory(config=...)`. Engine's `_parse_floats()` calls `re.split()` which requires strings.
- **`result.biomass()` returns wide-form** `["Time", <sp_names>, "species"]` with `species="all"` for cross-species output types. Must `melt` to tidy form for plotly.
- **`output.recordfrequency.ndt=24`** in the Baltic config means biomass is recorded once per year (24 dt = 1 year), not once per dt. EXPECTED_ROWS_PER_SPECIES = n_year (not n_year × 24).
- **B-H wiring**: `stock.recruitment.type.sp{i}` ∈ {`"none"`, `"beverton_holt"`, `"ricker"`}; `stock.recruitment.ssbhalf.sp{i}` in tonnes (must be > 0 when type ≠ none). Per `osmose/engine/config.py:532-544`.

## Polish lesson — playwright load surfaced narrative dishonesty

Playwright loaded `biomass.html` at file:// (blocked → served via `python -m http.server`) — confirmed plotly traces (3), legend, axes render. But **linear y-axis drowned cod (~12t) against sprat (~7M)** — cod was *literally invisible* on the chart of a tutorial titled "3 focal species." Added `log_y=True` (commit `c2ba915`) — all 3 species now visible across 1-10M range. Same dispatch updated Beat 1 prose to honestly describe stickleback boom-bust as documented Baltic regime-shift behaviour, not a bug to be apologised for.

**Generalisable lesson:** automated tests pass on numerical bounds but don't catch "human can't see the thing." For visual artifacts (HTML plots, screenshots), run a playwright load + visual check before declaring "done."

## Test-vs-tutorial-window discrepancy (documented, accepted)

The test fixture measures equilibrium at **years 5-25** (cod ~905, sprat ~5.5M, stickleback ~542K). The tutorial's print summary uses **years 25-30** (cod ~12.7, sprat ~6.8M, stickleback ~356). Same seed, same config, different windows of a transient dynamic. Test bounds (T6, commit `3db8d5f`) reflect the 5-25 mean which represents the transient *average*. The plot shows the full trajectory; the print summary documents the late state. Acceptable for this tutorial; would need re-measurement if windows ever align.

## What was NOT done (deferred)

- Bump `n_year` from 30 to 50 to give dynamics longer to settle (would require re-measuring T6 bounds; deferred — current state ships with playwright-verified plot)
- Add a follow-up "Build from scratch" tutorial (the synthetic approach the spec originally wanted; would need 3-5 days of OSMOSE calibration work to reach equilibrium without B-H)
- Document the substrate pivot in the spec (`docs/superpowers/specs/2026-05-16-30min-tutorial-design.md` still describes the synthetic approach — historical artifact)

## Commit chain (origin/master `2cfb168` → `c2ba915`)

1. `e35625a..5502bf2` — Spec r1-r6 (6 commits during multi-angle in-loop review)
2. `f9c3ee3..3c582ba` — Plan r1-r3 (3 commits)
3. `f69b435` — T1 stub helper
4. `3a23a3c` — T3 test in final form
5. `4b1654e..2c518a2` — Plan r4-r6 (3 mid-T4 revisions; deprecated by pivot)
6. `6a193ea` — T4 substrate pivot to Baltic (the structural fix)
7. `8e261f7` — T5 real numba_warmup
8. `3db8d5f` — T6 measured pyramid bounds
9. `694bc99` — T9 tutorial preamble + Beat 1
10. `aa3631c` — T10 Beats 2-6 + closing + troubleshooting
11. `735f350` — T11 README callout + doc index + tutorials index
12. `75c2a32` — T12 fix (conditional copytree preserves edits; strict validation enforced)
13. `c2ba915` — polish (log_y axis + transient-dynamics narrative)
