# BoB forcing-change characterization (A4.3) — 365-subsample vs 24-bin-average + regrid

2026-07-06. Script: `scripts/bob_forcing_characterization.py`. This is a **report, not a gate** —
the forcing change it measures is a deliberate, intended correction (see framing below), and the
script asserts nothing.

Run: `PYTHONPATH=. .venv/bin/python scripts/bob_forcing_characterization.py`
(Python engine, both arms fixed-seed `seed=42`, `nyear=5`, comparing
`data/examples_433_orig` (pre-migration snapshot) against `data/examples` (migrated native 4.4.1).)

## Result

```
metric          max|rel|   median|rel|
abundance          0.000         0.000
biomass            0.000         0.000
yield              0.000         0.000
```

Divergence on all three metrics (biomass/abundance/yield, 120 timesteps × 9 species = 1080 values
each) is **exactly zero** — the two arms produce bit-identical Python-engine population output,
not merely "small." This was independently checked: the two LTL forcing files genuinely differ
(`roms_n2p2z2d2_biscay.nc`: 365×20×30, mean `SmallPhyto`≈1.4053; `roms_n2p2z2d2_biscay_24step.nc`:
24×20×20, mean `SmallPhyto`≈1.4075), and the resource-predation code path (`_predation_on_resources`
in `osmose/engine/processes/predation.py`) does deduct eaten biomass from the resource pool each
step — so the forcing is genuinely read and consumed on both arms, it just doesn't move the fish
population outputs at BoB's resource-abundance level. That is consistent with the resource pool
being non-limiting for this config: predators reach their ration cap regardless of which forcing
file backs it, so small (sub-1%) shifts in the LTL time/space discretization don't propagate to
biomass, abundance, or yield.

## Framing: this is a deliberate change, not a regression

The forcing "divergence" characterized here has **two intended parts**, both landed earlier in
this migration and both now shared by the Python and 4.4.1-Java engines identically going forward:

1. **365-day → 24-step bin-average resample** (Task 2, `scripts/resample_bob_forcing.py`). The old
   Python engine read the legacy `ltl.*` config and subsampled the LTL NetCDF — one calendar day
   picked per model step out of the 365-day source. The new native 4.4.1 config points every
   `species.file.sp8`–`sp13` resource species at a pre-computed 24-step file where each step is the
   **bin-average (window-mean)** of the days it covers, per the 4.4.x native resource-forcing
   convention. Bin-averaging conserves each window's mean and only discards sub-window
   (intra-step) variability — it does not shift the seasonal signal.
2. **20×30 → 20×20 regrid onto the true model grid** (Task 5/6, `scripts/regrid_bob_forcing_to_grid.py`).
   BoB's original ROMS forcing was delivered on a 20×30 lat/lon grid, but the OSMOSE model grid
   itself is 20×20 (`grid.nlon;20`). The old Python engine nearest-neighbor-regridded the 20×30
   data onto the 20×20 model grid at runtime, silently, inside the load path. The 4.4.1 jar has no
   such runtime regrid and requires the forcing file to already match the grid, so the file was
   pre-interpolated (linear) to 20×20 once, ahead of time, and committed.

Both changes were necessary so that the Python engine and the 4.4.1 Java jar read **identical**
forcing bytes from here on — prior to this migration the Python engine's implicit runtime
subsample + nearest-neighbor regrid had no Java-side equivalent, which is precisely what blocked
BoB from running on 4.4.1 at all (Task 5). The characterization above shows the price of making
that correction is, for this config, indistinguishable from zero at the population-output level:
not a regression, and considerably better than merely "small."
