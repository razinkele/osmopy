# Baltic stability — habitat / reproductive-volume follow-up (post SP-A/SP-B/salinity-spawning)

**Date:** 2026-07-02
**Status:** recommendation, informed by three completed negative experiments.

## What we've now ruled out — the overshoot is NOT spatial

The Baltic ecosystem collapses/explodes over long horizons (SP-A gate: 0/8 species stay in their ICES
envelopes; cod/percids overshoot ×44-68). Three levers have now been tested and **all three fail**:

| lever | experiment | result |
|---|---|---|
| **Parameters** (mortality + fishing + recruitment `ssbhalf`/`shape`) | SP-A ε-constraint sweep, 39 params | min instability 8.11 ≫ targets; 0/8 in-envelope. Params can't stabilise. |
| **Cell density** (finer grid, more cells per footprint) | SP-B spike, 2× upsample (80×100) | percid/cod overshoot unmoved (×38-96); 2/8 gain was forage fish, not the target. |
| **Spawning location** (salinity-correct spawning *areas*) | this run — cod → saline deeps (145→81 cells), percids → freshwater (pikeperch 19→3) | **0/8 in-envelope; cod overshoot UNCHANGED (63.6→63.7×)**, pikeperch slightly worse. |

**The decisive insight:** a spawning *map* controls only **where** eggs are placed, not **how many**
recruits the Beverton-Holt stock-recruitment produces. The overshoot is a **population-level quantity**
(recruitment magnitude × mortality balance), so no *spatial* refinement — parameters within the SR/mortality
space, grid cells, or spawning footprints — can change it. That is why all three failed identically.

## The one lever with a real mechanism: a *dynamic* reproductive-volume recruitment gate

`docs/baltic-fish-lifecycle.md:386-406` describes the actual controlling process, which the model **does
not implement**: eastern-Baltic-cod recruitment is governed by the **reproductive volume** — deep-basin
water simultaneously **saline enough (≳11 PSU)** for egg buoyancy and **oxygenated enough (≳2 mL·L⁻¹)** for
egg survival, replenished only by sporadic **Major Baltic Inflows** of North Sea water. Between inflows the
reproductive volume **shrinks toward zero**, and cod recruitment fails regardless of spawner biomass.

This is a **recruitment *limiter*, not a spawning *area***. It would cap cod recruits in low-volume years —
directly attacking the population-level overshoot that spatial levers can't. This is the mechanistically-
correct follow-up, and the negative salinity-spawning result is precisely what points to it.

### Design sketch (a real modeling addition, ~weeks, not a map edit)

1. **Forcing:** extend the CMEMS pipeline (`osmose/forcing/`) to emit two new per-cell, per-step NetCDFs
   for the Baltic — **bottom salinity** (CMEMS `so` at depth, product `BALTICSEA_MULTIYEAR_PHY_003_011`)
   and **bottom oxygen** (`o2b`, `..._BGC_003_012`). The pipeline is `GridSpec`-driven and already handles
   `so` at a selectable depth (`osmose/forcing/physics.py`); it needs a bottom-field selection + the O₂ var.
2. **Reproductive-volume metric:** per timestep, `RV = Σ cell_volume where (bottom_salinity ≥ 11 & bottom_O2 ≥ 2)`
   over the deep basins — a scalar time series (or a per-cell suitability field).
3. **Recruitment gate:** in `osmose/engine/processes/reproduction.py`, multiply cod's B-H recruitment by a
   `RV / RV_ref` factor (clipped to [0,1]) so recruits collapse when the volume shrinks. Cod-only initially.
4. **Validate:** does gating cod recruitment on RV pull cod from ×64 toward its ICES envelope, and does the
   released trophic pressure help the rest? Re-run the SP-A cert.

### Percids (perch/pikeperch) are a *different* problem

They are freshwater coastal/lagoon spawners with **no reproductive-volume mechanism** — a salinity/oxygen
gate does not apply. Their overshoot is either (a) coarse-grid habitat under-resolution (SP-B: their
estuary/lagoon habitat isn't resolved at 25 km — but the spike showed finer cells alone don't fix it), or
(b) a fundamental SR/carrying-capacity mismatch at basin scale (the long-standing "percids are structural"
finding, [[project_percid_overshoot_diagnostic]]). Neither is a habitat-map problem. **Recommendation:
accept the percids as the residual** (they are ICES weight 0.2 — poorly assessed at basin scale anyway) and
judge success on the assessed stocks (cod + the 1.0-weight pelagics).

## Recommendation

1. **Do NOT invest further in spatial/habitat work** (grid rebuild, real habitat maps, spawning-area edits)
   — three experiments show the overshoot is not spatial.
2. **If pursuing a fix:** build the **dynamic cod reproductive-volume recruitment gate** above — it is the
   only mechanism with a real chance, and it closes a documented model gap. Budget it as a forcing +
   reproduction-process addition (weeks), cod-first, percids accepted as residual.
3. **Pragmatic alternative (recommended default):** accept that Baltic is a **short/medium-horizon
   (~15-yr) Python-engine config** — stable and useful over the horizon it was calibrated for (default
   `nyear` already lowered to 15, `7d77862`) — and stop investing in long-horizon equilibrium, which three
   levers now show is not reachable by tuning, resolution, or habitat placement. Escalate to the
   reproductive-volume gate only if a specific long-horizon cod study requires it.

All experiments (SP-A, SP-B spike, salinity-spawning) are banked on isolated branches; `data/baltic` on
master is untouched except the `nyear` 50→15 default.
