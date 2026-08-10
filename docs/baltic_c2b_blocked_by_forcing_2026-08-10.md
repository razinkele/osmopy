# C2(b) computed reproductive volume is blocked on interannual forcing

**Date:** 2026-08-10
**Scope:** spec `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 2, item C2(b)
**Verdict:** C2(b) cannot be implemented on the current forcing. B1 (interannual forcing) is a
hard prerequisite, not a sibling item.

## The dependency

C2(b) replaces the prescribed RV series with a reproductive volume **computed from bottom
salinity and oxygen** — RV being, by definition, the water volume where both are sufficient for
cod egg survival (S ≳ 11 PSU, O₂ ≳ 2 ml/L).

The gate consumes that as a *per-year* factor:

```
factor(y) = clip(rv[y] / ref, 0, 1)      # mode raw_cap, ref = 150
```

applied to cod_east egg production, constant within a model year
(`osmose/engine/processes/recruitment_gate.py`, `osmose/engine/config.py:_load_rv_gate`).

**Both candidate input fields are single-year climatologies:**

| field | file | shape | interannual variation |
|---|---|---|---|
| bottom salinity | `data/baltic/baltic_salinity_bottom_climatology.nc` | `(time=24, lat=40, lon=50)` | none |
| bottom oxygen | `data/baltic/baltic_oxygen_bottom.nc` | `(time=24, lat=40, lon=50)` | none |

24 frames = one year at the Baltic's 24 steps/yr (months duplicated ×2). Neither carries a year
axis.

**The prescribed series it would replace is fundamentally interannual:**
`data/baltic/reference/baltic_cod_reproductive_volume.csv`, 47 annual values 1974–2020, ranging
**48–380 km³ — an ~8× swing** — declining into the post-2014 stagnation tail (last five values
70, 60, 55, 50, 48).

## Why this is a trap, not merely a gap

A computed RV over climatological fields returns **one value, repeated every year**. The gate
would collapse from a time-varying recruitment modulator into a constant multiplier on cod_east
egg production.

That failure is silent and would likely *pass certification*: the constant is effectively a free
parameter, so it can be set to reproduce the current final-decade cod_east biomass and satisfy the
identity-pinned gate — while having discarded the entire mechanism. The model would look
unchanged and be strictly less realistic, with a physics-derived label attached. The spec's own
in-sample agreement criterion would not catch it either, since agreement is scored on the
final-decade window where the prescribed series is already flat in its stagnation tail.

## Consequence for the roadmap

Revised order for spec Phase 2:

1. ~~RV-gate re-derivation~~ — **done** (`docs/baltic_rv_gate_rederivation_2026-08-09.md`,
   `docs/baltic_rv_ref_sweep_2026-08-09.md`): the gate is still load-bearing post-adoption
   (gate-off → cod_east 137.3 kt, 1.61× over ceiling) and the admissible `ref` band is ~115–161,
   with production at 150 sitting only **7.2% from the floor-side edge**.
2. **B1 — interannual forcing** — now a prerequisite for C2(b), not a parallel item. Needs
   multi-year bottom salinity and oxygen (and LTL) from the CMEMS reanalysis over a window that
   overlaps the RV series (1974–2020 is not reachable; the Baltic reanalysis products start
   ~1993, which covers the stagnation period and the 2003/2014 inflows — enough to validate
   in-sample against the last ~28 series years).
3. **C2(b) — computed RV**, once (2) exists. Note the risk direction has **inverted** since the
   spec was written: pre-adoption the danger was a computed RV running *higher* than the series
   (cod_east was pinned 2.4% under its ceiling); post-adoption the tight edge is the floor, so a
   computed RV running *lower* — the likely direction for a physically-derived RV over the
   stagnation period — is what breaches. Gate on the low side.
4. **F1 — historical fishing**, required to make B1's hindcast interpretable (established in the
   spec's own review); not required for C2(b) itself.

## CORRECTION (same day, after querying the live CMEMS catalogue)

**The conclusion above is wrong in its central claim: C2(b) does NOT require B1.** Two errors:

1. **The repo's catalogue is incomplete.** `mcp_servers/copernicus/server.py` lists
   `bgc_monthly_reanalysis` without `o2b`, and `phy_monthly_reanalysis` without `sob`. Querying
   `copernicusmarine.describe` directly returns:
   * `cmems_mod_bal_bgc_my_P1M-m` → `chl, nh4, no3, nppv, o2, **o2b**, ph, po4, spco2, zsd`
   * `cmems_mod_bal_phy_my_P1M-m` → `bottomT, mlotst, siconc, sithick, sla, **so**, **sob**,
     thetao, uo, vo`
   So multi-year bottom oxygen AND bottom salinity are both directly available, as are the
   depth-resolved `so` / `o2` needed for a true volume integral.
2. **I conflated "the model needs interannual forcing" with "the RV series needs interannual
   inputs".** The gate does not consume a field — it consumes a **per-year CSV**
   (`reproduction.rv.gate.series.file`, columns `year,spawning_rv`). Computing RV is therefore an
   **offline derivation**: integrate the qualifying volume per year from multi-year reanalysis
   fields, write a series, repoint the config key. The running model still uses climatological
   forcing; nothing in `ResourceState`, `PhysicalData`, or the time-policy code is touched.

**Revised verdict: C2(b) is unblocked and is a data task, not an engine task.** B1 remains
required for the *hindcast validation* ambition (running the model under interannual forcing), but
that is a separate goal from replacing the prescribed RV series.

What survives from the original analysis, and still matters:

* The climatology fields *already in the repo* cannot produce an interannual RV — that reasoning
  was correct, it just pointed at the wrong source. The fix is new multi-year downloads, not B1.
* The trap it identified is real and now the acceptance criterion: a computed RV that comes out
  effectively constant would silently degrade the gate to a fixed multiplier and could still pass
  certification. **The plan must assert interannual variance in the computed series before the
  series is allowed anywhere near the config.**
* The risk direction is still inverted (gate on the low side, 7.2% floor margin).

## Cheap interim option, if C2(b) is wanted before B1

Compute RV **spatially** from the climatology to obtain a domain RV *shape* (which cells qualify,
and the seasonal cycle), then drive its interannual amplitude from an observed scalar index
(e.g. Major Baltic Inflow years, or a bottom-salinity index at a reference station). That is a
hybrid, not a computed RV, and should be labelled as such — it would validate the spatial
computation while leaving the time axis prescribed. Recorded as an option, not a recommendation.
