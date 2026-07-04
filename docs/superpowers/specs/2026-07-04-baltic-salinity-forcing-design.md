# Real CMEMS salinity forcing for the salinity gate — design (BLOCKED on data)

**Date:** 2026-07-04
**Status:** design settled, **BLOCKED on a user action** (see §2). Not yet planned/implemented — resume at writing-plans once the full-depth `so` data is downloaded.
**Author:** brainstormed with the user
**Related:** `docs/superpowers/specs/2026-07-04-salinity-gated-cod-occupancy-design.md` (the gate this feeds); `docs/superpowers/specs/2026-07-04-salinity-gate-numba-path-design.md` (gate now on both movement paths); `osmose/forcing/physics.py` + `osmose/forcing/grid.py` (regrid/resample primitives to reuse); `scripts/download_baltic_rv_forcing.py` (the download tool).

## 1. Goal

Replace the gate's synthetic/constant salinity field with a **real, Baltic-grid bottom-salinity climatology** derived from CMEMS `so`, wire it into the Baltic config, and run the Baltic A/B (gate off vs on) to measure the cod–percid effect.

## 2. BLOCKER (must clear first — user action)

The local CMEMS `so` (`data/cmems_cache/cmems_downloads/baltic_phy_monthly_reanalysis_so_*.nc`, 29 files, 1993–2021) is **deep-only**: depth levels 41 m → 245 m (shallow levels were deliberately dropped for the RV/reproductive-volume work — see that script's docstring). The gate needs salinity in the **shallow coastal cells (<41 m)** — exactly where percids live and low salinity should exclude cod — and those cells are all-NaN in this data. Building on it would degenerate the gate into a ">41 m depth mask," not a salinity ramp.

**To unblock** (both steps are user actions; the download needs the CMEMS credential):
1. **Rotate the CMEMS credential** at Copernicus Marine (the standing owed security item) and set the fresh value in `.env` (`CMEMS_USERNAME`/`CMEMS_PASSWORD`).
2. **Re-download full-depth `so`** (surface→bottom). The existing script is parameterized — the only change from the deep-only pull is `--depth-min 0`:
   ```bash
   .venv/bin/python scripts/download_baltic_rv_forcing.py --vars so --depth-min 0 --depth-max 260 --start 1993 --end 2021
   ```
   **Caveat:** this overwrites the deep-only `so` files in place (same filenames). To preserve the RV work's deep-only inputs, download the full-depth `so` to a *separate* directory/filename instead (a small tweak to the script's `OUT_DIR`/filename, or copy the deep-only files aside first). Decide at resume time.

Once full-depth `so` is present, this design proceeds unchanged.

## 3. Settled design decisions (from brainstorming)

- **Depth reduction: bottom salinity** — deepest valid (non-NaN) level per cell (`so.ffill(dim="depth").isel(depth=-1)`, or a numpy deepest-valid without the `bottleneck` dep). Demersal-correct for cod: deep basins are saline (cod retained), shallow coastal is brackish (cod excluded). *This is the step `phy_to_physics` does NOT do (it takes a fixed depth) — the new code.*
- **Time: seasonal climatology, 24 steps, cycled.** Average the CMEMS months across years → 12-month climatology → `resample_to_24` → 24 steps. `PhysicalData.get_grid(step)` cycles via `step % n_time`, so 24 steps cover all 360 Baltic steps (ndt=24 × nyear=15). A real-year series is rejected: the Baltic run has no calendar anchor, so year-mapping would be arbitrary (same objection that dogged the RV gate).
- **Regrid + orientation:** reuse `regrid`/`resample_to_24`/`target_coords`/`get_coords`/`get_var` from `osmose/forcing/grid.py`, and the Baltic `GridSpec`, so the output (time=24, latitude, longitude) matches the movement-map grid orientation (latitude descending, per `phy_to_physics`). **Load-bearing correctness point:** the salinity `(ny,nx)` must align with `cell_y/cell_x` exactly as the movement maps do — test by comparing the field against the Baltic grid mask / a known basin (high salinity in the deep Bornholm/Gotland basins, low toward the northern/coastal cells).
- **NaN / gap handling:** after bottom-extraction + regrid, every OSMOSE **ocean** cell that cod's map can occupy must have a finite salinity. Fill any residual NaN ocean cells (nearest-valid or basin-mean) so the gate never spuriously excludes cod from an ocean cell for missing data. Land cells (cod map = 0) can stay NaN.
- **Output:** `data/baltic/baltic_salinity_bottom_climatology.nc`, varname `salinity`, shape (24, ny, nx). Loaded by the gate via `movement.salinity.field.file` + `movement.salinity.field.varname=salinity`.

## 4. Components (for the eventual plan)

1. **Builder** — `scripts/build_baltic_salinity_forcing.py` (or a function in `osmose/forcing/`): open the full-depth `so` files → bottom-extract → monthly climatology → regrid to Baltic grid → `resample_to_24` → gap-fill ocean NaN → write `baltic_salinity_bottom_climatology.nc`. Reuses the forcing primitives; the bottom-extraction is the only genuinely new logic.
2. **Config wiring** — add `movement.salinity.field.file` + `movement.salinity.field.varname=salinity` to the Baltic config, **gate `enabled=false` by default (stays inert)**. The A/B enables it.
3. **Smoke test** — the Baltic config with the gate on loads the real field (`_load_salinity_gate` → `PhysicalData` of shape (24, ny, nx)); `_movement_salinity_weight` returns a *graded* grid (not all-0/all-1); a short engine run completes; the field's spatial pattern matches the grid (basins saline, coast/north low).
4. **Baltic A/B diagnostic** — `scripts/baltic_salinity_gate_diagnostic.py`: run Baltic gate-off vs gate-on (real field, cod sp0 enabled), report cod–percid (perch sp4, pikeperch sp5) spatial overlap and percid biomass off vs on. Mirror the RV-gate / recruitment-ceiling diagnostics.

## 5. Honest expectation (from the percid refuge lit review)

Gating cod out of low-salinity cells **reduces cod predation on percids there → raises percid biomass**. This is a **spatial-realism** correction, NOT a percid-overshoot fix — if anything it *worsens* the ×38–96 overshoot. The A/B measures the spatial effect (cod–percid overlap ↓, percid biomass ↑ in coastal cells); report it faithfully with this framing, not as an overshoot cure. Also note the documented side effect: gating cod occupancy also cuts cod off from its real prey (herring/sprat) in the same coastal cells.

## 6. Resume point

When full-depth `so` is downloaded: invoke `writing-plans` from this spec → subagent-driven build (4 components in §4) → smoke test → Baltic A/B → report. No re-brainstorming needed; the decisions in §3 are settled.
