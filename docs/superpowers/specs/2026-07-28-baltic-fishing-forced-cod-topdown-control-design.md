# Fishing-forced cod + calibrated top-down control — design

**Date:** 2026-07-28
**Status:** design (pre-plan)
**Depends on / supersedes:** builds on the aggregate-cod 5/8 baseline; motivated by the
cod E/W disaggregation failure (`docs/baltic_cod_ew_phase1_report_2026-07-25.md`) and the
percid missing-removals finding (`docs/baltic_findings_summary_2026-07-28.docx`).

## 1. Goal

Reproduce a realistically **depressed cod stock** by forcing its fishing mortality from the
ICES history, and use **calibrated seal/cormorant predation + realistic percid removals** to
hold the prey field in its ICES envelopes — closing the gap the aggregate baseline (5/8
in-envelope, 2/8 stable over 50 yr) leaves, **without** the apex-predator-release trap that
sank the disaggregation.

The disaggregation showed that suppressing cod releases the prey field 10–80× over envelope
because the aggregate baseline's prey params were tuned to rely on cod's top-down predation.
This design keeps cod a **live focal predator** (just smaller, via forced fishing) and
supplies the *missing, scientifically-documented* top-down controls the model currently
under-represents:

- **Recreational / small-scale coastal fishing** of perch and pikeperch — poorly reported,
  not analytically assessed by ICES; recreational catch can equal or exceed commercial in
  Lithuania (Curonian Lagoon) and the Archipelago Sea.
- **Seal and cormorant predation** — for coastal fish, birds + seals consume 2–3× the
  fishery (perch specifically ~2×); total Baltic removals ≈ 7×10⁵ t fisheries, 1×10⁵ t
  seals, 1×10⁵ t birds (Hansson et al. 2018).

## 2. Base configuration and isolation

- **Base:** the 8-species aggregate-cod 5/8 baseline (`calibration_results/phase13_equilibrium.json`,
  obj 2.33). NOT the disaggregated experimental config currently on `master`.
- **Isolation:** a new git branch off the pre-disaggregation commit `646a36d` (which is the
  aggregate 5/8 baseline). `master` keeps the disaggregation experiment untouched. All work
  in this design happens on that branch.

## 3. Engine feasibility (verified, from the code trace)

These constraints are load-bearing and were confirmed by reading the Python engine:

- **Background predators DO impose predation on focal fish** (GreySeal sp15, Cormorant sp16),
  scaled by prescribed biomass × `predation.ingestion.rate.max`, gated by a predator/prey
  size-ratio window. They are absent from the accessibility matrix, so they currently predate
  at the default coefficient **1.0**; adding a matrix column lets that coefficient be tuned
  *down* per prey. Levers: prescribed biomass, ingestion rate, size-ratio window, (new) matrix
  column. A regression test asserts a background predator reduces focal-prey abundance.
- **Multi-year BIOMASS forcing is NOT supported** — background/resource biomass is a
  within-year seasonal cycle repeated identically every year (indexed `step % ndtperyear`);
  focal species have no biomass-forcing at all. So seal/cormorant biomass is a **scalar level
  lever**, not a trajectory.
- **Per-year FISHING forcing IS supported** — `mortality.fishing.rate.byYear` and
  `mortality.fishing.catches.byYear` feed an annual time series via
  `osmose/engine/timeseries.py` (`ByYearTimeSeries`). This is the route to drive cod down.

## 4. Components

### 4.1 Component 1 — Forced cod fishing mortality

**What:** feed the ICES **cod.27.24-32** (eastern Baltic, the dominant stock) fishing-mortality
series into `mortality.fishing.rate.byYear` for cod (sp0). Data retrieved via the ICES data
service (F ≈ 0.4 in 1946, rising to ~0.9 in the 1960s–80s, 0.6–0.75 in the 2000s, falling to
0.27/0.16/0.04/0.02/0.015 in 2018–2022 as the moratorium took hold).

**Interface (config keys, verified):**
- `mortality.fishing.rate.byYear.file.sp0` = a tracked CSV forcing file holding the per-year F
  series; the engine loads it via `osmose/engine/timeseries.py` (`config.py:1554`,
  `_load_fishing_rate_by_year`).
- Horizon mapping: align the model's **final decade** with the recent (post-2010) collapsed
  period so the equilibrium-target window reflects the depressed state; hold the last series
  value beyond its end.

**Why forced-F + calibrated-M together:** the real F *fell* post-2014 while the stock stayed
collapsed (elevated natural mortality + recruitment failure). Forced F alone will therefore
not depress cod enough. So cod's **additional mortality M stays a calibration free param**
(Component 4); the DE tunes M so forced-F + M reproduces the ~70–80 kt depressed cod biomass
in the final decade.

**Depends on:** the ICES cod F data (Section 5).

### 4.2 Component 2 — Realistic percid removal (both levers)

The calibrated baseline's percid F (perch 0.029, pikeperch 0.0095) is implausibly low — the
optimizer chose it because percid assessment weight is 0.2, so simply widening the F bound
does NOT help (the bound already allows more; the objective just doesn't engage it).

**Lever A — fixed elevated percid F (fishing side).** Set perch (fsh4) and pikeperch (fsh5)
`fisheries.rate.base` to **fixed** literature-grounded elevated values representing total
(commercial + recreational) removal — roughly ≥2× a defensible commercial baseline, since
recreational ≈ or > commercial. Fixed (not a free param) so the realistic removal is *imposed*
rather than optimized away. Exact values set in the plan from the best coastal-fishery
statistics for the Baltic/Curonian region; documented with provenance.

**Lever B — cormorant predation on percids (predation side).** Strengthen cormorant predation
reaching perch and *young* pikeperch: `species.biomass.multiplier.sp16` (scales the NetCDF
standing biomass; `background.py:235`) and `predation.ingestion.rate.max.sp16` become free
params; confirm the size-ratio window (2.5–8×, 70–85 cm bird) covers young pikeperch (9–34 cm
prey); optionally add a tunable cormorant column to the accessibility matrix to shape which
prey it takes.

### 4.3 Component 3 — Seal/cormorant top-down control on forage fish

Seals (sp15) eat herring/sprat/cod/flounder (size ratio 3–12×). To absorb the forage fish that
lower cod releases, make `species.biomass.multiplier.sp15` (NetCDF standing-biomass scale) and
`predation.ingestion.rate.max.sp15` free params, bounded to keep total seal consumption near
the documented ~1×10⁵ t/yr. Same for cormorant on the forage-fish side (already in 4.2).

### 4.4 Component 4 — Full 8-species re-calibration

Joint differential-evolution re-calibration over:
- the existing phase-13 free params (per-species larval M, adult M, fishing F, ssb_half, shape β),
- **plus** the new levers: seal biomass-multiplier + ingestion, cormorant biomass-multiplier +
  ingestion (and matrix column if added),
- with **cod F forcing active** and **percid F fixed** (Lever A).

Warm-started from `phase13_equilibrium.json`. Threading fix (OMP/NUMBA=1 per worker),
isolated-eval + sim-timeout, checkpointing, wall-clock cap (~4–8 h).

**Acceptance bar (honest, structural — not 8/8):**
- cod at its depressed level (~70–80 kt final-decade mean, tracking the forced decline);
- the high-weight prey (herring, sprat, flounder) **no worse than the 5/8 baseline**;
- percids **improved toward** their envelopes (reduced overshoot) via the grounded removals,
  with any residual accepted as the coarse-grid structural limit;
- no unintended collapse; long-horizon stability compared to the 2/8 baseline.

## 5. Data requirements

- **ICES cod.27.24-32 fishing-mortality series** — via the ICES data service (already
  accessed this cycle). Stored as a tracked forcing CSV under `data/baltic/forcing/` with a
  provenance header (stock, retrieval date, ICES assessment year).
- **Percid removal values** — a short provenance note citing the recreational/coastal-fishery
  literature for the chosen perch/pikeperch F.
- **Seal/cormorant biomass** — the existing `baltic_predator_biomass.nc` (per-cell seasonal);
  the calibration tunes scalar multipliers, not the NetCDF.

## 6. Testing

- **Forcing loader test:** the cod F byYear series loads and applies the expected per-year F
  (deterministic unit test on `timeseries.py` / config parse).
- **Percid-F config test:** the elevated fixed percid F is present and read.
- **Seal/cormorant lever test:** biomass-multiplier + ingestion params parse and feed the
  background predation (extend `tests/test_engine_background.py`).
- **Smoke run:** the forced-cod config runs end-to-end; cod tracks down; both predators active.
- **Calibration pipeline test:** the extended free-param set builds (bounds/x0 lengths
  consistent) and the objective evaluates finite at the warm-start x0.
- **Post-calibration certification:** `baltic_stability_certify.py` (aggregate-cod ENVELOPE)
  against the acceptance bar + the 2/8 baseline.

## 7. Risks and mitigations

- **Percid mortality may destabilize** — prior work found percid mortality levers
  destabilizing. This uses *grounded* magnitudes, not arbitrary tuning, but must be verified
  empirically. Mitigation: fixed percid F (bounded, documented) + certify stability; accept
  residual overshoot rather than over-crank.
- **Transient vs equilibrium** — forced time-varying F makes the run transient; the
  calibration targets the final-decade mean aligned with the recent collapsed period.
- **Cormorant reaches only young pikeperch** (size ratio) — adult pikeperch ×217 may not fully
  close; the design targets *improvement*, not envelope-exact, and is honest that the residual
  is the coarse-grid structural limit.
- **Seal/cormorant level is a scalar, not a trajectory** (engine limit) — acceptable; we tune
  the standing level, not a multi-year history.
- **Forced-F not depressing cod enough** — mitigated by keeping cod M a free param so forced-F
  + M reaches the target; fallback is a sustained elevated F rather than the declining tail.

## 8. Out of scope / future

- Herring/flounder disaggregation (separate effort).
- Finer/nested coastal grid for percid habitat (the "proper" structural fix; large engine
  effort — this design instead adds the missing *removals*).
- Multi-year seal/cormorant biomass trajectories (needs an engine change to the seasonal-wrap
  biomass indexing).

## 9. References

See `docs/baltic_findings_summary_2026-07-28.docx` §6 for the full, verification-tagged list.
Key: Hansson et al. (2018) *ICES JMS* 75(3):999 (removals by fisheries/seals/birds); the
Baltic pikeperch status reviews (recreational ≥ commercial); Heikinheimo et al. (2021) and
Östman et al. (2013) (cormorant predation on perch); ICES cod.27.24-32 assessment (cod F).
