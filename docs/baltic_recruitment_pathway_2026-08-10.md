# How recruitment actually works in this engine — a factual map

**Date:** 2026-08-10
**Why this exists:** three specs were written and withdrawn on 2026-08-10, and the common cause was
designing against an assumed model of the code rather than the code. Two of the three asserted
mechanisms that do not exist (`docs/superpowers/specs/2026-08-10-baltic-computed-rv-design.md`,
`…-bornholm-spawning-fix-design.md`). This note is the antidote: what the recruitment path does,
read off `osmose/engine/processes/reproduction.py` and its loaders. **No design, no proposal.**

## The chain, in execution order

`reproduction()` runs once per timestep and produces egg schools. Every step below is applied in
this order, and every gate after step 2 is a **scalar per-species multiplier** on the egg count.

| # | stage | source | notes |
|---|---|---|---|
| 0 | **SSB** | mature biomass per species | seeding substitutes `seeding_biomass` while SSB == 0 and `step < seeding_max_step` |
| 1 | **linear eggs** | `sex_ratio · relative_fecundity · SSB · season_factor · 1e6` | `reproduction.py:141-147`. The 1e6 is tonnes→grams. **No spatial term.** |
| 2 | **stock–recruitment** | `apply_stock_recruitment` | `none / beverton_holt / ricker / hockey_stick / shepherd`; shepherd is `linear / (1 + (SSB/SSB½)^β)` |
| 3 | **RV gate** | `rv_gate_factor` | per **year**, cod_east only on Baltic; skipped on seeded steps |
| 4 | **recruitment ceiling** | `recruitment_ceiling_by_season` | per-season absolute cap |
| 5 | **thermal gate** | `thermal_gate_factor` | per **year**, from a temperature series — **exists, disabled on Baltic** (see below) |
| 6 | **depensation gate** | `depensation_factor` | SSB-dependent Allee term, not time-dependent |
| 7 | **school creation** | — | eggs split into `n_schools`; created at `cell_x/cell_y = -1` |

## Three facts that invalidated earlier assumptions

1. **Egg production is spatially blind.** Nothing in stages 0–6 reads a map, a cell, or a spatial
   field. Schools are created *unlocated* (`cell_x/cell_y = -1`, `reproduction.py:245-247`) and are
   placed on the following step by whichever movement map covers `age_dt 0` — the **juvenile** map,
   not the spawning map. Consequence: changing a spawning map cannot change recruitment volume. It
   relocates adults during the spawning window, nothing more.
2. **Every environmental gate is a per-year scalar.** RV and thermal both index a *year*, not a
   cell — `idx = min(offset + year, n_years - 1)`, clamped past the series end
   (`recruitment_gate.py:37`). So "make the gate physical" means supplying a better annual series,
   never a field, and a series shorter than the run leaves the tail clamped: with a 50-yr run, a
   29-row series makes the scored final decade a single constant.
3. **The gates compose multiplicatively and all skip seeded steps.** Enabling a second gate on a
   species that already has one multiplies their factors. Any new gate must state what it does
   alongside the existing ones rather than assuming it acts alone.

## The thermal gate already exists

`osmose/engine/processes/thermal_gate.py` + `osmose/engine/config.py:_load_thermal_gate`, wired at
`reproduction.py:190-197`. It is **implemented, tested, and switched off** — no
`reproduction.thermal.gate.*` key appears anywhere in `data/baltic/`.

* **Config surface:** `reproduction.thermal.gate.enabled`,
  `reproduction.thermal.gate.series.file` (a per-year CSV, contiguous ascending years, same shape
  as the RV series), `reproduction.thermal.gate.species.enabled.sp{N}`, plus mode/ref/floor.
* **Response:** `logistic_response(temp, t50, slope) = 1/(1+exp(-(temp-t50)/slope))` — saturating,
  0.5 at `t50`.
* **Normalisation:** `thermal_cap` = `clip(r/r_ref, 0, 1)` (mean-reducing, like the RV gate's
  `raw_cap`), or `mean_preserving` = `r / mean(r)` over the run window. A nonzero `floor` is
  **rejected** under `mean_preserving` because it destroys the unit-mean property.
* **Provenance:** built for **percid** year-class strength (Pekcan-Hekim et al. 2011, *Ambio*;
  Olin et al. 2019, *Hydrobiologia*), which is why its defaults are percid-shaped.

**Bearing on spec C1** (temperature-dependent stock–recruitment for cod and herring, motivated by
Voss & Quaas 2026, doi:10.1093/icesjms/fsag033): the mechanism is largely present. C1 is closer to
"derive an SST series, choose `t50`/`slope`/mode per species, enable for the right species, gate
it" than to building a new pathway. Whether the *logistic* shape suits cod/herring — as opposed to
the percid case it was designed for — is an open question for that work, not something this note
settles.

## What this note deliberately does not do

It proposes nothing. Its only claim is that the seven stages above are what the code executes, with
file and line references so the next design can be checked against them cheaply — before it is
written, not after it is withdrawn.
