# Computed reproductive volume for the cod recruitment gate (spec Phase 2, C2b)

**Date:** 2026-08-10
**Status:** WITHDRAWN as written, 2026-08-10 — falsified by measurement before implementation.
> See `docs/baltic_computed_rv_divergence_2026-08-10.md`. Three defects: (a) the machinery it
> specifies already exists (`osmose/forcing/reproductive_volume.py`, `data/baltic_rv/baltic_rv_field_interannual.nc`,
> 80 cached CMEMS files) — a search failure on my part; (b) the proposed domain excludes the
> Bornholm Basin, the eastern stock's principal spawning ground; (c) with the domain corrected,
> the computed series still shows NO positive rank agreement with the prescribed one
> (Bornholm-only rho=+0.04, CV 0.14 vs 0.60; wider domains rho<0), and a 29-row series leaves the
> certification-scored decade 100%% clamped. The acceptance criteria below did their job: they
> falsified the design cheaply. Retained in full as the record of what was proposed and why it
> did not survive contact with the data.
**Parent spec:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 2, C2(b)
**Prerequisite status:** none. C2(b) is an **offline data derivation**, not an engine change —
corrected 2026-08-10, see `docs/baltic_c2b_blocked_by_forcing_2026-08-10.md`.

## 1. What this replaces, and why

`data/baltic/reference/baltic_cod_reproductive_volume.csv` is a **prescribed** 47-year series
(1974–2020, `spawning_rv` 48–380, units nominally km³) driving the cod_east recruitment gate:

```
factor(y) = clip(rv[y] / ref, 0, 1)     # mode raw_cap, ref = 150
```

applied to egg production for sp8 only (`reproduction.rv.gate.species.enabled.sp8=true`),
constant within a model year, clamped (not wrapped) past the series end
(`osmose/engine/processes/recruitment_gate.py`).

Two reasons to replace it:

1. **Defensibility.** The series is a hand-curated external input with no in-repo derivation. The
   gate is simultaneously the *dominant* control on cod_east — turning it off puts the stock at
   137,302 t, 1.61× over its ceiling (`docs/baltic_rv_gate_rederivation_2026-08-09.md`). A
   load-bearing input with no reproducible provenance is the weakest link in the configuration.
2. **Scenario capability.** A prescribed historical series cannot be projected. An RV *computed
   from salinity and oxygen* can be recomputed under any forcing, which is the prerequisite for
   the climate-scenario track (spec C1/B2).

## 2. Scientific definition

Baltic cod reproductive volume is the standard measure of habitat available for successful egg
development: the **water volume simultaneously satisfying** salinity high enough for egg
neutral buoyancy and oxygen high enough for egg survival. The canonical thresholds are
**S ≥ 11 PSU** and **O₂ ≥ 2 ml/L** (≈ 89 mmol m⁻³), applied over the deep basins where cod spawn.

This is a genuine **volume integral over depth**, not a bottom-area proxy: cod eggs are neutrally
buoyant near the halocline, so the qualifying layer sits *between* the depth where salinity rises
above 11 and the depth where oxygen falls below the threshold. A 2-D bottom field cannot express
that layer — it collapses to "is the seabed habitable", which is the wrong question. The design
therefore uses depth-resolved fields.

## 3. Data

CMEMS Baltic multiyear reanalysis (verified against the live catalogue 2026-08-10; the repo's
copy in `mcp_servers/copernicus/server.py` is incomplete and must be corrected as part of this
work):

| product | dataset id | variables used |
|---|---|---|
| PHY reanalysis | `cmems_mod_bal_phy_my_P1M-m` | `so` (salinity, depth-resolved) |
| BGC reanalysis | `cmems_mod_bal_bgc_my_P1M-m` | `o2` (dissolved oxygen, depth-resolved) |

Bottom variants (`sob`, `o2b`) exist and are **not** used for the volume integral, for the reason
in §2; they are useful only as a sanity cross-check.

Coverage: the Baltic reanalysis begins ~1993, so the computed series overlaps the prescribed
series over **1993–2020 (28 years)** — enough for in-sample validation, and it spans both the
2003 and 2014 Major Baltic Inflows plus the intervening stagnation, i.e. the dynamic range the
gate exists to represent.

**Domain:** the spawning basins, not the whole model domain. Use the existing spawning mask
(`data/baltic/maps/cod_spawning.csv`, already read by `_load_rv_spatial`) as the horizontal
extent so the computed series is commensurate with where the model actually spawns cod.

**Season:** the prescribed series is a *spawning* RV. Restrict the integral to the cod spawning
months and state which months, rather than annual-averaging.

## 4. Design

### 4.1 Offline derivation (no engine change)

The gate consumes a per-year CSV via `reproduction.rv.gate.series.file`. So the entire feature is:

1. A generator script producing `data/baltic/reference/baltic_cod_reproductive_volume_computed.csv`
   with the same schema (`year,spawning_rv`, contiguous ascending years — enforced by
   `_load_rv_gate`).
2. A config change repointing `reproduction.rv.gate.series.file` at it, plus a re-derived `ref`.

Nothing in `ResourceState`, `PhysicalData`, `simulate.py` or the time-policy code is touched.

### 4.2 Units and `ref` are coupled — re-derive, never carry over

`ref = 150` is calibrated to the prescribed series' scale (48–380). A computed volume in km³ over
the spawning basins will have its **own** scale, which there is no reason to expect matches. The
factor is `clip(rv/ref, 0, 1)`, so an unrescaled swap silently rescales recruitment by
`ref_old/ref_new`.

**Rule: `ref` is re-derived for the computed series, and the swap is validated on the resulting
factor trajectory, not on raw RV values.** The comparable quantity between old and new is the
**factor series** `clip(rv/ref,0,1)`, which is dimensionless.

### 4.3 Acceptance criteria

The trap this design must not fall into: a computed series that is effectively **constant**
degrades the gate to a fixed multiplier on cod_east egg production. That would still pass
certification, because the constant is a free parameter tunable to reproduce current biomass —
the model would look unchanged, be less realistic, and carry a physics-derived label. Hence:

* **A1 — variance (hard gate, checked before the series touches any config).** The computed
  series' coefficient of variation over 1993–2020 must be at least half the prescribed series' CV
  over the same years. A series flatter than that is rejected outright, no certification run.
* **A2 — in-sample agreement.** Over the 28 overlap years, the computed *factor* series must
  correlate with the prescribed factor series (Spearman ρ, sign and rank agreement — not exact
  magnitudes, which are not expected to match). The direction that matters: both must fall
  through the stagnation period and rise at the 2003/2014 inflows.
* **A3 — inflow signal.** The computed series must show local maxima at the 2003 and 2014 MBI
  years. This is the single most falsifiable check that the computation is physically right, and
  it is independent of the prescribed series.
* **A4 — certification.** 50 yr × 5 seeds, identity-pinned gate (5 assessed + perch +
  stickleback). **Gate on the LOW side**: post-adoption cod_east sits 8.0% above its 60 kt floor
  and only the floor edge is tight (admissible `ref` band ~115–161 with production at 150 — 7.2%
  from the floor edge, `docs/baltic_rv_ref_sweep_2026-08-09.md`). A physically-computed RV over
  the stagnation period will tend to run *lower* than the prescribed series, which is the
  breaching direction. This inverts the pre-adoption risk statement in
  `docs/baltic_rv_gate_mechanism_ab_2026-08-02.md`, which is void for the adopted config.

A1–A3 are evaluated offline and cost minutes. Only a series passing all three earns a
certification run.

### 4.4 Clamping and the series tail

`rv_gate_factor` clamps past the series end rather than wrapping, deliberately: the series ends in
the low-RV stagnation regime and post-series years stay low. A computed series ending in 2020–2024
inherits that behaviour correctly. **The tail value matters more than any other single point**,
because a 50-year certification run spends its scored final decade clamped to it. The plan must
report the tail value explicitly and compare it to the prescribed series' tail (48).

## 5. Out of scope

* **B1 interannual forcing of the running model.** Still wanted for hindcast validation, still a
  separate, larger piece. This design deliberately does not need it.
* **Spatial RV** (`reproduction.rv.spatial.*`, already implemented but not enabled on the Baltic
  config) — a per-cell egg-survival field is a different mechanism from the scalar gate; combining
  them is untested and out of scope.
* **Applying the gate to cod_west (sp0).** Currently disabled; leave it.
* **Re-tuning anything other than `ref`.** If the computed series cannot be made to certify by
  `ref` alone within the admissible band, that is a reportable negative result, not a licence to
  tune mortality or accessibility.

## 6. Deliverables

1. `scripts/make_baltic_computed_rv.py` — downloads (cached), computes, writes the series, and
   prints the A1–A3 diagnostics.
2. `data/baltic/reference/baltic_cod_reproductive_volume_computed.csv`.
3. `tests/test_baltic_computed_rv.py` — schema, contiguity, variance floor (A1), inflow maxima
   (A3), and that the file loads through `_load_rv_gate` without error.
4. A validation note documenting A1–A3 outcomes, the derived `ref`, and the factor-series
   comparison — committed whether or not the swap proceeds.
5. Corrected `mcp_servers/copernicus/server.py` catalogue entries (`o2b` on BGC reanalysis, `sob`
   and the other missing PHY reanalysis variables) — a defect this work uncovered.
6. On A4 PASS only: the config swap plus certification record; on FAIL, the negative result.
