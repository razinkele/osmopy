# Baltic cod reproductive-volume recruitment gate — design

**Date:** 2026-07-02
**Status:** approved design, revised after in-loop review, pre-implementation
**Author:** brainstormed with the user
**Related:** `docs/baltic-fish-lifecycle.md:386-406` (the RV gap), `scripts/baltic_rv_overshoot_diagnostic.py`, `scripts/download_baltic_rv_forcing.py`, `docs/diagnostics/baltic_rv_overshoot.png`

## 1. Motivation

The OSMOSE Baltic model reproduces eastern-Baltic cod as an unconstrained
boom-and-decline: a spin-up to ~2930 t by model year 7, then a monotone decline
to ~1150 t at year 15 with no equilibrium. The RV-vs-overshoot diagnostic
confirmed the cause: the model has **zero salinity/oxygen coupling**, so cod
recruitment has no environmental brake. In reality eastern-Baltic-cod
recruitment is gated by the **reproductive volume** — deep-basin water that is
simultaneously saline enough (≥ ~11 PSU, egg neutral buoyancy) and oxygenated
enough (≥ ~2 mL/L, egg survival). Using 29 years of CMEMS reanalysis (1993–2021,
full depth), the diagnostic measured this RV directly: the **spawning-season
(Mar–Aug) mean is ~8%** of cod-basin cells, and it pulses with Major Baltic
Inflows (2004–05 after the 2003 MBI; 2016–18 after the 2014 MBI), with troughs
in 1993/2001/2002/2012/2013. This interannual variability is the negative
feedback the model lacks.

This spec adds an RV-driven recruitment gate so cod recruitment responds to that
signal.

## 2. Goals and non-goals

**Goals**
- Add a per-model-year multiplier on cod egg production driven by a precomputed
  spawning-season RV series, so recruitment rises after inflows and falls in
  stagnation.
- Support two modes via config: a mean-normalised *variability test*
  (`mean_preserving`) and a raw *environmental cap* (`raw_cap`).
- Keep the feature config-gated and **inert by default** — all existing configs
  (Baltic, EEC, Bay of Biscay) produce bit-identical output unless the gate is
  explicitly enabled.
- Quantify the effect by re-running the diagnostic with the gate on vs off.

**Non-goals**
- No in-engine salinity/oxygen state or 3D forcing fields (the gate consumes a
  precomputed scalar series; spatial egg placement is unchanged).
- No hindcast / calendar-anchored validation. The model is a 15-year spin-up
  with no real-year anchor; this tests the *mechanism's stabilizing effect*, not
  a specific year's cod stock.
- No recalibration of `ssb_half` / larval mortality in this change (the raw_cap
  mode will likely need it later; that is called out, not done here).
- No gate for other species (percids are freshwater — no RV mechanism).
- No effect under bioenergetics mode (see §10) — Baltic runs bioen off, so this
  is a documented limitation, not a gap.

## 3. Design overview

The gate is a per-species, **per-model-year** scalar multiplier `m(year)` applied
to cod's `n_eggs` in `osmose/engine/processes/reproduction.py`, immediately after
`apply_stock_recruitment(...)` and immediately before the egg-school creation
loop. The multiplier is **constant within a model year** and is driven by that
year's spawning-season RV.

```
n_eggs = apply_stock_recruitment(...)               # existing
for sp in gated species (e.g. cod):
    if not seeded_this_step[sp]:                     # see §3.1
        n_eggs[sp] *= m(sp, step)                    # NEW
```

### Why annual, not per-step

Cod eggs are produced only during the spawning season (`season_factor ≈ 0`
outside Mar–Aug, `reproduction.py:115-118`). The reproductive-volume hypothesis
is that **that season's** RV sets **that year's** recruitment. Using a single
per-year spawning-season RV value (constant across the year's steps) therefore:
- matches the biology (season-integrated RV → year-class strength);
- avoids seasonal aliasing (a per-*step* RV series, normalised by an all-step
  mean, would apply a season-weighted mean ≠ 1 because deep-basin O₂ is itself
  strongly seasonal — the in-loop review flagged this as a real bug);
- uses the same `~8%` spawning-season metric the diagnostic already reports,
  rather than an all-step mean of a different value;
- removes monthly→biweekly resampling, seam-wrapping, and month/step phase
  alignment entirely.

### 3.1 Seeding-window interaction (must-fix)

`reproduction.py:122-126` overrides `ssb[sp] = seeding_biomass[sp]` whenever a
species' real SSB is zero and `step < seeding_max_step[sp]`; `seeding_max_step`
defaults to the full lifespan in steps (`config.py:529` — multi-year for cod).
Applying the gate to these *seeded bootstrap* eggs would let a low-RV year (or
`raw_cap` with `rv≈0`) drive cod eggs to zero during establishment and prevent
the population from ever bootstrapping. **The gate must not apply on any step
where the seeding override supplied the SSB.** Implementation: `reproduction()`
tracks a per-species `seeded_this_step` boolean (true when mature-derived SSB was
0 and seeding fired); the gate multiply is skipped for those species/steps.

### 3.2 Mode formulas

Let the run sample model years `y = 0 … nyear-1`, each mapping to a series index
`idx(y)` (§4), and let `rv_{idx(y)}` be that year's spawning-season RV.

- **mean_preserving** (default): `m(y) = rv_{idx(y)} / D`, where the denominator
  `D = mean over model years y'=0..nyear-1 of rv_{idx(y')}` — a **multiset mean
  with repeats**, NOT the mean over the unique set of sampled indices (the
  distinction matters only when a run is longer than the series and years wrap,
  in which case a repeated year must be counted each time it is used).
  The **annual egg-multiplier has mean 1 over the run window by construction.**
  This does **not** guarantee mean-preserved *recruitment*: recruits feed back
  into future SSB, and eggs `= c·SSB·BH(SSB)·m` with `BH(SSB)=1/(1+SSB/ssb_half)`
  concave/decreasing in SSB, so injecting mean-1 multiplicative variance can
  still shift the realised mean recruitment (Jensen + spawner–recruit
  covariance). Realised mean is therefore an **empirical outcome, verified by
  success criterion §10.2**, not a construction guarantee. (Note: at *fixed* SSB
  the egg→recruit map here is linear — B-H depends only on SSB, larval mortality
  is density-independent, `natural.py:139-140` — so the only source of mean shift
  is the closed-loop feedback, which is why §10.2 measures it.)
- **raw_cap**: `m(y) = clip(rv_{idx(y)} / rv_ref, 0, 1)`.
  Literal environmental survival cap; lowers the mean too. **This shifts the
  equilibrium and will need `ssb_half` / larval-mortality recalibration to avoid
  cod collapse — out of scope here.** raw_cap is provided for experimentation and
  is validated in this deliverable by unit tests and qualitative behaviour only,
  **not** by the damping success metric (§10.2).

Optional low-end floor `reproduction.rv.gate.floor` (default `0.0`) applies
`m = max(m, floor)` after the mode formula. The linear-through-origin form sends
`rv→0 ⇒ m→0` (full recruitment shutoff); the floor exists to sensitivity-test
whether hard zeros drive stabilization artifactually. Default keeps the pure
linear form.

Order of application (`m` after B-H) is mathematically equivalent to applying it
to `linear_eggs` before B-H, because every recruitment type here is a pure
function of SSB (multiplication commutes; no density-dependence double-count).

`m = 1.0` (exact) for every disabled species and for every step when the master
switch is off. When the master switch is off the factor array is not constructed
or applied at all (belt-and-suspenders against a non-exact 1.0).

## 4. The RV series and model-year mapping

**Source & builder.** A new `build_rv_gate_series(...)` function in
`scripts/baltic_rv_overshoot_diagnostic.py` (exposed via `--emit-gate-series`)
writes the per-year spawning-season RV to
`data/baltic/forcing/baltic_rv_gate_series.csv`. Format:

```
year,spawning_rv
1993,0.00
1994,0.07
...
2021,0.06
```

Rows are one per year, **contiguous and strictly ascending**; `first_year` is
derived from the first data row (no header metadata to keep in sync).
`spawning_rv` is the Mar–Aug mean computed by `annual_rv(..., months=SPAWNING_MONTHS)`.
The builder **requires both criteria** (salinity + oxygen, `both_criteria=True`)
and raises if `annual_rv` returns `None` (non-calendar time axis) or produces a
NaN year — it must never silently emit the oxygen-only proxy or a header-only file.

**Mapping.** Model year `y = step // n_dt_per_year`. Real year =
`start_year + y`. Series index `idx(y) = (start_year - first_year + y) mod n_years`,
where `n_years` = number of rows. Indexing is **positional** — row `i` is assumed
to be year `first_year + i`; the load-time contiguity check (§8) guarantees this.
With defaults (`start_year=1993=first_year`) a 15-year run samples model years
0–14 → real 1993–2007 (a trough → 2003-MBI-pulse cycle), no wrap. Wrapping only
occurs for runs longer than the series. A negative `start_year - first_year` is
allowed and wraps via `mod` (documented).

**mean_preserving denominator** `D` (§3.2) is the multiset mean over the sampled
model years `y = 0 … nyear-1` (with repeats when wrapping), computed at config
load from `nyear`, `start_year`, and the series — so the applied annual
multiplier has mean 1 over the years the run actually uses, not over the full
29-year series.

**Note on 1-year series.** Under annual gating a single-row series makes
`mean_preserving` degenerate (`m ≡ rv/rv = 1.0`, silently inert); a constant
non-unit gate is only meaningful under `raw_cap`. There is therefore no
"climatology" mode here — the series is the real per-year sequence. The `annual_rv`
helper returns `(None, None)` for a single-year span, so a 1-row file is
hand-authored (used only in unit tests), not builder-emitted.

## 5. Configuration keys

New keys (schema `OsmoseField`s in `osmose/schema/species.py`,
`category="reproduction"`; lowercase dot-separated):

| key | type | default | meaning |
|---|---|---|---|
| `reproduction.rv.gate.enabled` | bool | `false` | master switch |
| `reproduction.rv.gate.mode` | str | `mean_preserving` | `mean_preserving` \| `raw_cap` |
| `reproduction.rv.gate.series.file` | path | `""` | per-year spawning-RV CSV (required iff enabled) |
| `reproduction.rv.gate.ref` | float | `0.20` | `rv_ref` for raw_cap (~95th pctile of spawning RV) |
| `reproduction.rv.gate.floor` | float | `0.0` | optional low-end floor on `m` |
| `reproduction.rv.gate.start.year` | int | `1993` | model-year-0 → real year |
| `reproduction.rv.gate.species.enabled.sp{idx}` | bool | `false` | per-species enable (cod sp0 on when used) |

Per-species key is `…species.enabled.sp{idx}` (not `…enabled.sp{idx}`) so the
master switch `reproduction.rv.gate.enabled` is **not a strict prefix** of the
per-species family — avoiding the prefix-filter / UI-input-ID collisions
`CLAUDE.md` warns about. `start.year` is always written fully-qualified in prose
to avoid confusion with the existing `output.start.year`.

The new fields are added to `SPECIES_FIELDS` in `osmose/schema/species.py`, which
`build_registry()` already iterates via `osmose/schema/__init__.py` — no
`registry.py` edit is needed.
`EngineConfig.from_dict` validation (`config_validation.py`) auto-captures keys
read via `cfg.get(...)` in `config.py`; the RV loader lives in `config.py`, so no
`_SUPPLEMENTARY_ALLOWLIST` entry is needed. The pure helper (§6) must read only
from `EngineConfig` fields, never `cfg.get` (it is not in `_EXTRA_ENGINE_SOURCES`).

## 6. Components and files

**New**
- `build_rv_gate_series(...)` in `scripts/baltic_rv_overshoot_diagnostic.py`
  (`--emit-gate-series`) → `data/baltic/forcing/baltic_rv_gate_series.csv`.
- `osmose/engine/processes/recruitment_gate.py`: pure, engine-state-free.
  `rv_gate_factor(config, step) -> NDArray[n_species]` returns `1.0` for disabled
  species and the mode factor (constant within a model year) otherwise. Reads
  only precomputed `EngineConfig` fields.

**Modified**
- `osmose/engine/config.py`: at build, load the per-year series, precompute the
  mode-factor-by-index 1-D array (see Representation), the per-species enable
  mask, and the mapping offset; store on `EngineConfig` (fields analogous to
  `spawning_season`). Perform all fail-fast validation here (§8).
- `osmose/engine/processes/reproduction.py`: apply the factor to `n_eggs` with
  the seeding-skip guard (§3.1).
- `osmose/schema/species.py` + registry: the new fields (append to
  `SPECIES_FIELDS`, which `build_registry()` already iterates via
  `osmose/schema/__init__.py` — no `registry.py` edit needed).

**Representation.** Store a 1-D `rv_gate_factor_by_index[·]` of **length
`n_years`** (= number of series rows), **indexed by the series index `idx(y)`**
(§4), NOT by model year — because `idx ∈ [0, n_years)` always, whereas model year
`y` can exceed `n_years` on a wrapping run, and a length-`nyear` array would be
mis-indexed for any non-default `start.year`. The mode formula is already applied
(for `mean_preserving`, the denominator `D` is the run-window multiset mean from
§3.2/§4, baked in at load). Also store a per-species boolean `rv_gate_enabled[·]`
mask. The helper computes `idx(step)` and returns
`rv_gate_factor_by_index[idx]` for enabled species, `1.0` otherwise — no
redundant `[n_species, n_years]` array. Because the factor is baked at load from
this run's `nyear`/`start_year`, a different run length gets a fresh array (no
stale reuse; `EngineConfig` is rebuilt per run).

## 7. Data flow (end to end)

```
download_baltic_rv_forcing.py  → CMEMS so/o2 reanalysis (done, 26 GB)
baltic_rv_overshoot_diagnostic.py → monthly RV(t) 1993-2021 (done)
   └─ build_rv_gate_series (--emit-gate-series)
        → data/baltic/forcing/baltic_rv_gate_series.csv  (29 rows: year, spawning_rv)
config (gate keys) → EngineConfig: factor-by-index array + enable mask + offset (+ validation)
reproduction(step) → for cod, if not seeded: n_eggs[cod] *= factor_by_index[idx(step)]
re-run diagnostic (gate on vs off) → docs/diagnostics/ comparison
```

## 8. Validation on load (fail-fast)

At config build, when `reproduction.rv.gate.enabled` is true, raise a clear error
(never silently fall back to `1.0`) if:
- `series.file` is empty (`""`), missing, or unreadable;
- the series has zero data rows;
- the `year` column is not contiguous and strictly ascending (positional
  indexing assumes row `i` = `first_year + i` — §4);
- any `spawning_rv` value is NaN or negative;
- `mode` ∉ {`mean_preserving`, `raw_cap`};
- `D == 0` (mean_preserving, the run-window multiset mean) or `ref <= 0`
  (raw_cap) — division by zero / nonsensical scale;
- `floor ∉ [0, 1]`;
- no species is enabled (`…species.enabled.sp{idx}` all false) while master is on.

When the master switch is off, none of these are evaluated and no field is built.

## 9. Testing

**Unit (pure factor helper + loader)**
- disabled species / master off → factor exactly `1.0` (bitwise), all steps.
- `mean_preserving`: `mean over run-window years of factor == 1.0 ± 1e-9`;
  low-RV years < 1, high-RV years ≥ 1.
- `raw_cap`: factor ∈ [0, 1]; `rv ≥ rv_ref` → 1.0; `rv == 0` → 0.0; floor applied.
- year mapping: `idx = (start_year - first_year + step // n_dt_per_year) mod
  n_years`; correct for a non-zero `start.year` and for runs longer than the
  series (wrap → repeated indices). For a hand-authored 1-year series,
  `mean_preserving` yields `m ≡ 1.0` (inert) and `raw_cap` yields a constant.
- fail-fast: each §8 condition raises at build.

**Integration (reproduction)**
- gate on for cod → cod `n_eggs` scaled by the expected annual factor at a given
  step; all other focal species and background species unchanged.
- seeding-skip: on a step where cod SSB is seeded (real SSB 0), the gate does
  **not** scale cod eggs.
- gate off → `reproduction()` output identical to pre-change.

**Regression / parity**
- Gate disabled (default): Baltic + EEC + BoB engine outputs bit-identical to
  pre-change. Run the cross-engine/parity suite (`migration-check` skill).
- Determinism: a gate-on run is reproducible under `simulation.rng.fixed`.

## 10. Success criteria (quantified)

1. **Inert by default:** Baltic/EEC/BoB outputs bit-identical with the gate off;
   parity suite green.
2. **mean_preserving effect** (all metrics use **annual cod biomass** from
   `OsmoseResults.biomass("cod")` — the diagnostic already reads it; model-year
   numbering is 0-based, so the measurement window is model years **3–14**,
   post-spin-up):
   - the cod **boom/bust ratio** = `max / min` of annual cod biomass over model
     years 3–14 is **reduced by ≥ 25%** vs gate-off;
   - the **mean** annual cod biomass over years 3–14 stays **within ±10%** of
     gate-off (the band absorbs the closed-loop mean shift of §3.2; a larger
     drift is a finding to report, not an automatic failure).
   Measuring this needs a small window-slice helper in the diagnostic
   (`characterise_instability` currently spans the whole series); an optional
   age-0 recruitment metric (`abundance_by_age`) can be added later for a
   stricter recruitment-level check, but biomass is the shipped criterion.
3. All unit + integration tests pass; ruff + pyright clean.
4. The gate series is regenerable from the diagnostic; the mechanism is
   documented (docstrings + a short `docs/` note).

raw_cap has **no** quantitative success target in this deliverable (it needs
recalibration); it must only pass its unit tests and behave monotonically
(lower RV → fewer recruits).

## 11. Risks and flags

- **mean_preserving is mean-neutral only for the egg multiplier, not realised
  recruitment.** The closed-loop concave SSB→egg feedback can shift the mean;
  §10.2 measures it and the ±10% band tolerates a modest shift. If the drift is
  large, fit a single normalisation constant `k` in `m = rv/k` to match gate-off
  mean (follow-up, not this change).
- **raw_cap shifts the equilibrium** (mean spawning RV ~8%); needs `ssb_half` /
  `mortality.additional.larva.rate.sp0` (=360) recalibration — separate task.
- **No calendar anchor** — mechanism test, not a hindcast. Stated in code + spec.
- **Bioen no-op.** `reproduction()` (and thus the gate) runs only when
  `config.bioen_enabled` is false; the bioen egg path (`simulate.py:518+`) is
  separate. Baltic runs bioen off, so the target is unaffected; enabling the gate
  under bioen has no effect and the code/docs must say so.
- **Allowlist.** New engine-read keys must pass `EngineConfig.from_dict`
  validation; keeping the loader in `config.py` and the helper field-only
  satisfies this (verified against `config_validation.py`).

## 12. Out of scope (future)

- Recalibrating cod SR under raw_cap; fitting the mean_preserving `k`.
- Generalizing the gate to a full salinity/oxygen forcing field with spatial
  egg-survival, or to the bioen reproduction path.
- Applying an analogous mechanism to other stocks.
