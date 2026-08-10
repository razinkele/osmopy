# Config defect: the Bornholm Deep is assigned to cod_west, and cod_east cannot spawn there

**Date:** 2026-08-10
**Status:** verified, unfixed — fixing it is a model change requiring the standard gate protocol.
**Found via:** the C2(b) computed-RV review (`docs/baltic_rv_divergence_explained_2026-08-10.md`);
independent of that work and unaffected by its outcome.

## The defect

The Bornholm Basin is the eastern Baltic cod stock's principal spawning ground — and since the
late 1980s effectively the only one producing viable eggs (MacKenzie, Hinrichsen & Plikshs 2000,
*MEPS* 193:143–156, doi:10.3354/meps193143; Köster et al. 2005, *ICES JMS* 62:1408–1425,
doi:10.1016/j.icesjms.2005.05.004). In the disaggregated Baltic config it belongs to the **wrong
stock**.

Grid: 50 × 40, 0.3° lat × 0.4° lon, centres 54.15–65.85 °N / 10.2–29.8 °E. Maps are stored
south-first and flipped on load (`osmose/engine/config.py:_load_spatial_csv`). Cells in the
Bornholm Deep band (55.05–55.85 °N, 14.5–17.0 °E):

| map | cells in the Deep band | note |
|---|---|---|
| `maps/cod_east_spawning.csv` | **0** | its only nearby cells are 5 at 55.95 °N — the northern rim / Hanö Bight |
| `maps/cod_west_spawning.csv` | **12** | 55.05/55.35/55.65 °N × 14.6–15.8 °E — the Deep proper |
| `maps/cod_spawning.csv` (pre-split aggregate) | **0** | the root cause, see below |

This contradicts the stock definition the config itself cites
(`data/baltic/reference/biomass_targets.csv`): `cod_east` = **cod.27.24-32**, which includes
SD 25 (Bornholm); `cod_west` = cod.27.22-24 (Belt/Arkona).

## Root cause: two different source maps, one of which never had the Deep

`scripts/build_cod_ew_maps.py` builds the two spawning maps from **different sources**, by design
and documented in its own docstring:

```
cod_west_spawning = western adult footprint (western cod spawn in
                    SD22-24; the aggregate spawning map is eastern-biased)
cod_east_spawning = aggregate cod SPAWNING map masked east
```

The column split is `WEST_COLS = 0..14`, `EAST_COLS = 13..49` (SD 24 deliberately shared).

So:

* `cod_east_spawning` inherits whatever the aggregate `cod_spawning.csv` contained — and that map
  has **no cells in the Bornholm Deep band at all**. The masking is not at fault; the gap is
  upstream, in the aggregate map.
* `cod_west_spawning` was built from the *adult* distribution instead, which does cover the Deep —
  so the Deep ended up west purely as a side effect of using a different source.

The builder's docstring says the eastern stock covers "SD25-32 deep basins Bornholm/Gdansk/Gotland".
The stated intent is right; the data does not implement it.

## Why it matters

1. **cod_east cannot spawn in the basin where the real stock spawns.** Its spawning footprint is
   the Gotland complex plus a Hanö-Bight rim, i.e. exactly the basins the literature reports as
   having lost reproductive volume since the late 1980s.
2. **It interacts with the RV gate.** The gate (`reproduction.rv.gate.*`) is enabled for **sp8
   (cod_east) only** and is the dominant control on that stock (gate-off → 137,302 t, 1.61× over
   ceiling, `docs/baltic_rv_gate_rederivation_2026-08-09.md`). A gate representing Bornholm-driven
   reproductive success is being applied to a stock that does not spawn in Bornholm.
3. **It distorts any spatial RV work.** `reproduction.rv.spatial.*` is implemented but unused; if
   enabled, it would read `maps/cod_spawning.csv` — the map missing the Deep — as its extent
   (`osmose/engine/config.py:_load_rv_spatial`).

## Why it is not fixed here

Moving the Bornholm Deep from `cod_west` to `cod_east` (or into both, as SD 24 already is) changes
where two calibrated stocks reproduce. That is a model change, not a data correction: it must go
through the same protocol as every other change in this series — A/B against the identity-pinned
gate (5 assessed + perch + stickleback), 50 yr × 5 seeds, with adoption only on PASS. cod_east
currently sits 8.0% above its floor, so a change to its spawning grounds is not obviously safe in
either direction.

## Suggested fix, when someone takes it

1. Repair the **aggregate** map first — `maps/cod_spawning.csv` should contain the Bornholm Deep;
   everything downstream inherits from it.
2. Rebuild both spawning maps from a single, documented source rather than one from the spawning
   map and one from the adult map, so the west/east split is a partition of one footprint.
3. Decide explicitly whether the Deep is eastern-only or shared like SD 24, and record the reason.
4. A/B and certify. Report cod_east and cod_west deltas; expect the RV gate's effect on cod_east to
   change, since the gate and the spawning ground would finally refer to the same water.
