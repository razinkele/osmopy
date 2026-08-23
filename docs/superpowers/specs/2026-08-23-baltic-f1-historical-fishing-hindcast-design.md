# Baltic F1 — historical fishing forcing + hindcast validation (Stage 1 of B1)

**Date:** 2026-08-23
**Status:** approved (design), pending implementation plan
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 2
(B1 item d / F1). **Staging decision (user, 2026-08-23): F1-first thin slice** — historical F is
config + data with zero engine risk, while B1's interannual-LTL/physics work is the expensive
part; this stage's hindcast verdict decides how much of that work is validation-motivated.
**Related:** `docs/baltic_c2b_blocked_by_forcing_2026-08-10.md` (Phase 2 ordering),
`docs/diagnostics/2026-07-15-ssb-f-hindcast-spike.md` (the July NO-GO this supersedes),
`docs/superpowers/specs/2026-07-28-baltic-fishing-forced-cod-topdown-control-design.md` (§5.3
byYear-vs-horizon warning), `docs/superpowers/specs/2026-07-11-baltic-rv-interannual-hindcast-design.md`
(harness discipline: shared-everything A/B, usable-window honesty).

## Goal

Force the certified 9-species Baltic config with the **observed ICES fishing-mortality history
(1993–2023)** and measure whether it moves modeled SSB trajectories toward observed ICES SSB,
against a constant-F baseline. This is the spec's Phase 2 hindcast-interpretability prerequisite:
without historical F, any interannual-forcing hindcast is confounded by fishing history.

## What already exists (verified 2026-08-23, file:line audited)

* **The engine lever is fully wired.** `mortality.fishing.rate.byyear.file.sp{i}` loads via
  `_load_fishing_rate_by_year` (`osmose/engine/config.py:461-466`, called at `:2285`) — headerless
  `np.loadtxt` (so `#` comment lines are skipped — provenance headers are safe), one F per line,
  sim-year 0 first. The override is applied on the production Numba path
  (`osmose/engine/processes/mortality.py:739-751`) and the reference path
  (`processes/fishing.py:36-45`), **before** selectivity/seasonality/effort — and it overrides the
  v4 fisheries base rate, because `fisheries.rate.base.fsh{i}` populates the same
  `config.fishing_rate` array (`config.py:328-330`). Historical F is therefore a
  **config + data change**; the engine work in this stage is hardening only.
* **The ICES data are already cached offline.** `data/baltic/reference/ices_snapshots/` carries
  per-year `f`/`catches`/`ssb` for all assessed stocks, 1993–2023 (31 F years; values are strings).
  One gap: **cod.27.22-24 F ends 2021** (category-3 downgrade; snapshot falls back to the 2022
  assessment).
* **The July spike's NO-GO does not transfer.** `2026-07-15-ssb-f-hindcast-spike.md` found F
  forcing washed out by the intrinsic attractor — but on the 8-species config with **no RV gate**
  and aggregate cod. Today's config has the E/W split and a per-calendar-year recruitment driver
  on cod_east. Two staleness traps if reusing spike code: it maps sp0↔`cod.27.24-32` (on master
  sp0 is **cod_west** ↔ `cod.27.22-24`), and assumes `nyear=30` against a config default of 15.
* **Known defects in the by-year path** (found in the B1 audit, fixed by this stage):
  the config reader lowercases all keys (`osmose/config/reader.py:173`) while `config.py:2296`
  (`byDt.byAge`/`byDt.bySize`) and `:2319` (`catches.byYear`) look up camelCase — three of the
  four time-varying fishing scenarios are dead code for any real config file; and a by-year
  series **shorter than the run silently falls back to base F** mid-run
  (`fishing.py:44`, `mortality.py:750`) — the same silent-degradation class B1 exists to kill.

## Decisions (recorded)

1. **Staging: F1-first** (user, 2026-08-23). Stages 2 (unified time policy + interannual
   bottom-O₂) and 3 (interannual LTL via proxy) are gated on this stage's verdict.
2. **F scaling: relative, anchored 2018–2022** (user, 2026-08-23).
   `F_model(y) = base_F × F_ices(y) / mean(F_ices, 2018–2022)`. The anchor window matches the
   biomass-envelope targets (2018–22 means), so the run's final years sit at factor ≈1 —
   consistent with the certified equilibrium. Absolute ICES F would be a regime change
   (calibrated cod rates are 10–25× below ICES F; the model's mortality budget carries the rest
   as predation) and is out of scope, at most a labelled sensitivity arm later.
3. **Herring aggregation: catch-weighted mean F over the four ICES stocks**
   (`her.27.25-2932`, `her.27.28`, `her.27.3031`, `her.27.20-24`):
   `F̄(y) = Σ C_s(y)·F_s(y) / Σ C_s(y)`. Rationale: the model species' own targets are the
   4-stock complex (`biomass_targets.csv:19,30,39` — biomass "aggregate across all Baltic herring
   management units", catch summed over the same four), catches are in comparable tonnes while
   SSB units are mixed ("index" scale for `her.27.25-2932`), and relative scaling keeps only the
   pattern anyway. Sensitivity to the weighting is reported, not swept.
4. **Non-assessed species (perch, pikeperch, smelt, stickleback) keep constant F.** Their per-year
   removals would need a separate provenance pipeline (coastal statistics, cormorants) — out of
   scope, consistent with the acceptance criterion scoring assessed stocks only.

## Non-goals (YAGNI)

* No production-config change: `data/baltic/` stays byte-identical; certification stays
  climatological (`nyear=15`, constant F). The hindcast is an opt-in overlay.
* No recalibration against the hindcast; no tuning of any parameter to improve skill.
* No calendar-convention engine feature (`year`-column CSV + `start.year` + offset for the F
  path) — deferred to Stage 2's unified time policy. Stage 1 encodes the calendar by row order
  and documents it.
* No Java-engine support: by-year keys never enter the production config, so the Java
  cross-check arm is unaffected. The hindcast is a Python-engine exercise.

## Design

### 1. Data derivation — `scripts/build_baltic_f_byyear.py` (offline, no network)

Reads `ices_snapshots/*.assessment.json`; writes `data/baltic/reference/f_byyear_sp{i}.csv`
(headerless values + `#` provenance header: stock keys, anchor window, base F, generation date)
for the five assessed species:

| sp | species | ICES source | note |
|---|---|---|---|
| sp0 | cod_west | cod.27.22-24 | F ends 2021 → hold-last for 2022–23; anchor = mean over the available anchor-window years (2018–2021) |
| sp1 | herring | catch-weighted mean over the 4 stocks | decision 3 |
| sp2 | sprat | spr.27.22-32 | |
| sp3 | flounder | fle.27.2223 | |
| sp8 | cod_east | cod.27.24-32 | |

Each file carries **50 rows**: 19 spin-up rows at base_F (factor 1), then 31 rows
`base_F × factor(1993..2023)` — see §3 for why 19. Base F is read from the live
`data/baltic/baltic_param-fishing.csv` (`fisheries.rate.base.fsh{i}`) at generation time and
recorded in the header; regenerating after any recalibration is one script run.

### 2. Engine hardening (load-path only)

a. **Case fix:** lowercase the lookups at `config.py:2296` and `:2319` (and the
   `_FISHING_SCENARIOS` table, cosmetic) so `byDt.byAge`/`byDt.bySize`/`catches.byYear` resolve
   for reader-produced configs. Regression tests go through the real reader, not hand-built
   camelCase dicts (the existing tests' blind spot).
b. **Fail-fast on short series:** at `EngineConfig.from_dict`, if any by-year F series has
   `len(arr) < n_year`, raise `ValueError` naming the key, series length, and run length
   (mirrors the RV-spatial guard, `config.py:1219-1226`). Longer-than-run is allowed (extra
   years ignored). This intentionally diverges from Java's silent base-rate fallback; noted in
   the docstring, irrelevant to parity in practice (no production config uses the key).
c. **Schema:** one `OsmoseField` for `mortality.fishing.rate.byyear.file.sp{idx}`
   (`ParamType.FILE_PATH`, indexed, optional) beside the other `mortality.fishing.*` fields in
   `osmose/schema/species.py` — kills the unknown-key warning, surfaces the param in the UI.

### 3. Hindcast harness — `scripts/baltic_f_hindcast.py`

Two arms × 5 house seeds [42, 123, 7, 999, 2024] × **50 yr**, Python engine, in-memory:

* **Arm A (baseline):** production config, constant F, `nyear=50`.
* **Arm B (fhist):** identical + the five `mortality.fishing.rate.byyear.file.sp{i}` keys.

Calendar: **sim-year 19 = 1993.** The RV gate's default mapping (offset 0 → sim-year 0 = series
year 1974) already runs 1974–2020 over sim-years 0–46 and clamps 2021–23 at the 2020 terminal
value — byte-identical to certified behavior, shared by both arms. The 19 spin-up years absorb
the seeding bootstrap (~12 yr) inside a genuinely historical pre-period; F is constant there in
both arms (arm B's prepended rows are base_F). The arms therefore share seeding transient, RV
history, horizon, and seeds — their difference isolates the ICES F pattern.

Outputs per arm/seed: annual SSB per species (maturity-based, like-for-like with the July
harness), plus realized yield for a forcing sanity check (arm B's cod yields must visibly track
the F pattern — the "verify the instrument can see it" lesson).

### 4. Validation — the parent spec's acceptance criterion, made concrete

Per assessed stock, over 1993–2023 (sim-years 19–49), 5-seed mean trajectories, both series
z-scored over the window (model biomass is on its own absolute scale; three ICES SSB series are
index-scaled anyway):

* **Trend test:** decadal-trend sign (1993–2002, 2003–2012, 2013–2023) must match observed ICES
  SSB in the majority of the three decades.
* **Skill test:** normalized RMSE (on the z-scored series) of arm B must beat arm A.

**Cod stocks are reported but excluded from pass/fail** (parent spec: until C2(b) ships —
cod_east's trajectory is partly prescribed by the RV narrative series; additionally cod_east's
calibrated base F is a placeholder-level 0.01/yr, so its relative-scaled by-year forcing is
nearly inert by construction). Pass/fail therefore rides on **herring, sprat, flounder**. Headline verdict: arm B beats arm A on ≥2 of those 3.

Output: a dated results doc `docs/baltic_f_hindcast_YYYY-MM-DD.md` + figure (trajectories vs
observed, per stock).
A null result is legitimate and must be reported as such: it **demotes Stages 2–3 from
validation-motivated to capability-motivated** (the model's equilibria don't track forcing
history at decadal scale → prioritize scenario-track work like C1 over hindcast realism), and
that decision rule is written down here in advance.

### 5. Certification guard

After the §2 engine changes, one standard 50-yr climatological certification run must come back
identical (the changes are load-path only; the case fix touches keys absent from every shipped
config; the guard is cheap insurance). Production `data/baltic/` is untouched by construction.

## Testing

* **CI-safe units:** case-fix resolution through the real `OsmoseConfigReader` (fixture config
  file, not a hand-built dict); the short-series fail-fast (raises, names key and lengths;
  longer-than-run passes); derivation math on tiny fixtures (anchor-window factor, catch-weighted
  herring F, cod_west hold-last, 19+31 row layout); by-year override still composes with
  selectivity/seasonality (existing coverage, extended through the reader path).
* **NOT a CI gate:** the hindcast outcome (emergent, seed- and machine-sensitive — house rule).

## Success criteria

1. `f_byyear_sp{0,1,2,3,8}.csv` generated reproducibly from the cached snapshots with recorded
   provenance; regeneration is one script run.
2. Engine hardening lands with tests; the previously dead by-year/by-dt/catches scenarios
   resolve through the real reader; short series fail fast at load.
3. Certification guard: unchanged verdict on the standard climatological run.
4. The harness runs 2 arms × 5 seeds × 50 yr; arm B's realized cod yields visibly track the F
   pattern (instrument check) — if they don't, stop and debug before interpreting SSB.
5. A dated results doc reports the trend and skill tests per stock, cod reported-only, with an
   honest verdict either way, and applies the pre-registered Stage 2/3 gating rule.

## Stage 2/3 preview (not designed here)

* **Stage 2 — unified out-of-range time policy + interannual bottom-O₂.** One documented policy
  per mechanism class (fail-fast for interannual fields, per the RV-spatial precedent; explicit
  `start.year` convention ported to the F path), fixing the audit's findings: LTL stride-collapse,
  the oxygen loader's frames==24 lock, the RV-gate mean-preserving self-inconsistency, the dead
  temperature NetCDF branch. Data for interannual bottom-O₂ (and salinity) already on disk
  (54 GB per-year `so`/`o2`, 1993–2021).
* **Stage 3 — interannual LTL via proxy.** The multi-year reanalysis has no
  phyto/zoo/benthos biomass; requires a science decision (scale the 6-group climatology by a
  `chl`/`nppv` index vs hold groups climatological) — to be brainstormed with the outcome of
  Stages 1–2 in hand.
