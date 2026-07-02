---
name: project_temperature_forcing_diagnostic
description: CMEMS temperature forcing (2nd science extension) CLOSED via diagnostic — no engine bug (Java-source-verified), blocked behind a phiT-calibration prerequisite. No feature built. SHIPPED to origin/master 2026-06-04.
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

The 2nd science extension (climate-driven **temperature** forcing via Copernicus CMEMS) was **CLOSED via diagnostic — NO feature built** (percid/HOLAS-3 pattern). Merged fast-forward to master + **pushed to origin/master 2026-06-04** (`d7003de..34c558b`, branch `feature/cmems-temperature-forcing` deleted, origin synced, docs only). Doc: `docs/baltic_temperature_forcing_diagnostic_2026-06-04.md`; spec kept as provenance w/ CLOSED banner.

## Why closed (in-loop review of 3 reviewers + an Arrhenius diagnostic)
The imagined "pull CMEMS temperature → run baltic → compare" is blocked, and there is **NO engine bug**:
1. **LTL forcing already CMEMS-wired** (done 2026-04-16; `baltic_param-ltl.csv` + `baltic_ltl_biomass.nc`, consumed by `resources.py`). The plankton half is complete.
2. **The Python engine never loads a temperature NetCDF** — `simulate.py:1287-1292` reads only the constant `temperature.value` when `bioen_enabled`; `PhysicalData.from_netcdf` is dormant (never called). The consumption side (`simulate.py:361-363`: `temp_data.get_grid(step)[cell_y,cell_x]`→Arrhenius) ALREADY handles spatial temp, so the loader is the only code gap (~80 lines, parity-safe; the 3 `temperature.*` keys are already allowlisted at `config_validation.py:224-227`).
3. **Temperature has ~zero effect on `baltic_ev`** (the ONLY bioen config: `simulation.bioen.enabled;true`, but `phit.enabled;false`). With phiT off, temp enters only via Arrhenius MAINTENANCE = bare `exp(-e_m/k_B·T)` ≈ 4e-12 at 15°C; at `c_m=0.001` maintenance is ~1e-12 of intake → `compute_energy_budget` reproduced gives **dw bit-identical 4°C↔18°C**. A constant-vs-CMEMS demo = numerical noise (the HOLAS-3 marquee trap). `baltic` (non-ev) has bioen off entirely.
4. **DECISIVE — no engine bug (Java source read at `/home/razinka/osmose/osmose-master/java/.../process/bioen/`):** `TempFunction.java:204 get_Arrhenius` = `Math.exp(-e_m/(k*(temp+273.15)))` and `EnergyBudget.java:205 getMaintenance` = `c_m*pow(weight*1e6,beta)*get_Arrhenius` are **byte-for-byte the Python formulas**. Bare Arrhenius is intentional (metabolic-theory standard; `c_m` carries the absolute scale). Growth-temp is driven by the NORMALIZED `getPhiT` (toggled by `phit.enabled`). My "missing reference-temp normalization = latent bug" hypothesis was REFUTED. Do NOT normalize the maintenance Arrhenius — it would break Java parity.

## Decision + don't-re-investigate
Climate temperature forcing is blocked behind a **calibration prerequisite** (a `phiT`-enabled + `e_M`/`e_D`/`T_p`-calibrated bioen config, and/or a maintenance-significant `c_m`) — a recalibration project, out of scope. **Don't:** normalize the Arrhenius (parity break); ship the NetCDF loader in isolation (inert, no consumer); build a constant-vs-CMEMS baltic_ev demo (noise).

## Actionable-later (recon de-risked the eventual build; in the diagnostic doc)
If a temperature-sensitive bioen config is ever pursued: (1) a phiT-calibrated bioen config is the GATING work; (2) then the thin loader change at `simulate.py:1287-1292` (netcdf-first via `resolve_data_path`+`from_netcdf`, constant fallback; fix stale `config_validation.py:133-134` comment); (3) forcing-gen verified feasible — CMEMS creds in `.env` valid, dataset `cmems_mod_bal_phy_my_P1M-m`/`thetao` reachable, MCP `generate_osmose_physics` writes var **`temperature`** (NOT `temp` → set `temperature.varname;temperature`), `(24,40,50)` lat-descending, **0.0 on land (NOT NaN)** → validate ocean temps finite+plausible (−2..30°C); cap `download_field(depth_max≈10)` (full BGC pull was 2.9GB); grid ny=nlat=40/nx=nlon=50, no transpose, `step%24` biweekly.

## Both science extensions now resolved
Size-spectrum diagnostics SHIPPED ([[project_size_spectrum_diagnostics]]); temperature forcing CLOSED via diagnostic. **Next: pick a fresh backlog item** (NOT temperature forcing, NOT a HOLAS-3/percid re-do). See [[project_feature_improvements_backlog]] (open: size-spectrum is done; remaining = Copernicus real-time(LTL done), sensitivity explorer UI, scenario diff, config presets, parser errors, `__slots__`/mutable-SchoolState perf, property-based tests).
