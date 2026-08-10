# Eastern Baltic cod reproductive-volume series

`baltic_cod_reproductive_volume.csv` — drives the reproductive-volume (RV) recruitment
gate (`reproduction.rv.gate.*`; see
`docs/superpowers/plans/2026-07-24-baltic-rv-recruitment-gate.md`).

## What it is — read this first

**This series is a recruitment-failure index, not a measured reproductive volume.** It is
labelled RV and drives a gate named `reproduction.rv.*`, but its organising feature — the
sustained post-2003 decline — encodes the eastern-cod *recruitment collapse* (the outcome), not
the hydrographic driver. Measurement, 2026-08-10: a reproductive volume computed from CMEMS
reanalysis over the Bornholm Basin shows **no decline at all** across 1993–2021 (oscillating
~300, record high in 2021) while this series falls 84% (300 → 48); Spearman rho between them is
+0.04. The computed result is the one consistent with the literature (Koster et al. 2005: Bornholm
kept sustaining egg development while Gdansk and Gotland lost RV). See
`docs/baltic_rv_divergence_explained_2026-08-10.md`.

Consequences: (a) do not describe this input as physical RV forcing in papers or configs; (b) the
"production refinement" proposed below was attempted and **withdrawn** — swapping in the computed
series removes the decline that holds cod_east in envelope and degenerates the gate toward a
constant (`docs/baltic_computed_rv_divergence_2026-08-10.md`); (c) the scenario ambition needs a
mechanism whose driver the physics can supply — temperature-dependent stock-recruitment — not a
projected RV.

## What it is

`spawning_rv` is the eastern-Baltic-cod **reproductive volume** by `year`: the volume of
deep-basin water simultaneously meeting the minimum requirements for successful cod egg
development — **salinity ≥ ~11 PSU** (egg neutral buoyancy; below it eggs sink into the
anoxic deep and die) **and dissolved oxygen ≥ ~2 ml/l** — concentrated in the Bornholm,
Gdańsk, and Gotland deeps. It is replenished only by sporadic **Major Baltic Inflows
(MBIs)** of North Sea water; between inflows it shrinks and cod recruitment fails
regardless of spawner biomass (Plikshs et al. 1993; MacKenzie / Köster / Wieland; the
ICES WGBFAS reproductive-volume indicator).

## This file (PoC status — literature-informed, not a data product)

The values encode the documented **relative interannual pattern**, NOT a calibrated data
product: high and variable through the MBI-rich 1970s–80s (spikes at the 1976/1981
inflows), the long 1983–1993 stagnation decline, the large 1993 and 2003 inflow spikes, a
moderate 2014 inflow, and the sustained low after 2003 that underlies the eastern-cod
recruitment failure. Under the gate's default `mean_preserving` mode the **absolute scale
cancels** (the factor is `rv / mean(rv over the run window)`), so only the shape matters.

**Production refinement (not blocking):** replace with a CMEMS-derived RV — extend
`osmose/forcing/` to emit **bottom salinity** (CMEMS `so` at depth) and **bottom oxygen**
(`o2b`), compute `RV = Σ deep-basin cell_volume where (bottom_salinity ≥ 11 & bottom_O2 ≥
2)` per year, and regenerate this CSV. The gate code and config consume this file
unchanged.
