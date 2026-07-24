# Eastern Baltic cod reproductive-volume series

`baltic_cod_reproductive_volume.csv` — drives the reproductive-volume (RV) recruitment
gate (`reproduction.rv.gate.*`; see
`docs/superpowers/plans/2026-07-24-baltic-rv-recruitment-gate.md`).

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
