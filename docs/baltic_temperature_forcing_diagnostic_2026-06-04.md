# CMEMS temperature forcing for the Baltic model — diagnostic

**Date:** 2026-06-04
**Status:** Closed via diagnostic — no feature built. Spec retained as provenance
(`docs/superpowers/specs/2026-06-04-cmems-temperature-forcing-design.md`).

## Question

Can we drive the OSMOSE Baltic model with real CMEMS temperature and see it matter — i.e. the
second "climate-driven physical forcing" science extension (LTL/plankton being the first)?

## Answer: not now — blocked behind calibration, not code. And there is NO engine bug.

The investigation (executed against the engine, the configs, and the **Java OSMOSE source**)
established four things:

### Finding 1 — LTL forcing is already CMEMS-wired (done)

`data/baltic/baltic_param-ltl.csv` + `baltic_ltl_biomass.nc` were generated from CMEMS via
`mcp_servers/copernicus` (`generate_osmose_ltl`, 2026-04-16) and are consumed by
`osmose/engine/resources.py`. The plankton half of "climate forcing" is complete.

### Finding 2 — the engine never loads a temperature NetCDF (the loader gap)

`simulate.py:1287-1292`: when `config.bioen_enabled`, the engine reads only the **constant**
`temperature.value` (`PhysicalData.from_constant`). It never reads `temperature.filename` and never
calls the existing, dormant `PhysicalData.from_netcdf`. The *consumption* side already handles a
spatial field (`simulate.py:361-363`: `temp_data.get_grid(step)[cell_y, cell_x]` → Arrhenius
maintenance), so the only gap is loading. Wiring it is ~80 lines and parity-safe (no config sets
`temperature.filename`; the three `temperature.*` keys are already in the
`config_validation.py:224-227` allowlist). But — see Finding 3 — it would be **inert**.

### Finding 3 — temperature has ~zero effect on the only bioen config (`baltic_ev`)

`baltic_ev` is the only config with `simulation.bioen.enabled;true`. It sets
`simulation.bioen.phit.enabled;false`, so temperature enters **only** via Arrhenius maintenance.
Reproducing `compute_energy_budget` at `baltic_ev`'s real parameters (`c_m=0.001`, default
`e_maint=0.65 eV`): the bare Arrhenius factor at 15 °C is `exp(-0.65/(8.62e-5·288)) ≈ 4e-12`, so
maintenance is **~1e-12 of gross intake** → `e_net ≈ e_gross` and `dw` is **bit-identical between
4 °C and 18 °C**. The constant-vs-CMEMS demo would show numerical noise, not a climate signal — a
misleading marquee (the HOLAS-3 / percid trap). `baltic` (non-ev) has bioen off entirely, so
temperature does nothing there either.

### Finding 4 (decisive) — NO engine bug: the Python bioenergetics matches Java OSMOSE byte-for-byte

The hypothesis that the maintenance Arrhenius was missing a reference-temperature normalization
(making temperature structurally invisible) was **refuted by reading the Java source**
(`/home/razinka/osmose/osmose-master/java/.../process/bioen/`):

- `TempFunction.java:204` — `return Math.exp(-this.e_m[i] / (k * (temp + 273.15)));` (k = 8.62e-5)
  → identical to Python `arrhenius`: `np.exp(-e_m / (K_B * t_k))`. **Bare Arrhenius, no T_ref
  normalization, in BOTH.**
- `EnergyBudget.getMaintenance` (`:205`) — `c_m[i] * Math.pow(weight*1e6, beta) *
  get_Arrhenius(school)` → identical to Python `c_m * w_grams^beta * arrhenius(T) / n_dt`.
- Growth-temperature is driven through `getPhiT` (the **normalized** Johnson curve, `=1.0` at
  `T_P`) in both — and that is what `phit.enabled` toggles.

So the bare Arrhenius is **intentional and parity-correct** (standard metabolic-theory form; the
`c_m` coefficient carries the absolute scale). There is nothing to fix in the engine.

## Conclusion / decision

Climate temperature forcing is blocked behind a **calibration prerequisite**, not code: for
temperature to visibly drive a config, that config must enable `phiT` with calibrated
`e_M`/`e_D`/`T_p` (the canonical thermal-performance lever), and/or calibrate `c_m` so maintenance
is a non-negligible fraction of intake. `baltic_ev` (a FIE/genetics demo) has neither. Building a
such-config is a recalibration project, out of scope and not currently justified.

**No feature built:** no engine change, no CMEMS download, no inert NetCDF loader. The diagnostic
preserves the finding; the design spec is kept as provenance with a CLOSED banner.

## Don't re-investigate

- The maintenance Arrhenius is **NOT a bug** — it is byte-for-byte the Java OSMOSE form
  (`TempFunction.get_Arrhenius` + `EnergyBudget.getMaintenance`). Do not "normalize" it; that would
  break Java parity.
- Do not ship the NetCDF-temperature loader in isolation — it is inert until a phiT-enabled /
  maintenance-calibrated bioen config exists.
- Do not build a constant-vs-CMEMS `baltic_ev` demo — the delta is numerical noise (Finding 3).

## Actionable-later (if a temperature-sensitive bioen config is pursued)

The recon de-risked the eventual build; carry these forward:
1. **A phiT-calibrated bioen config is the prerequisite** (enable `simulation.bioen.phit.enabled`
   with per-species `e_M`/`e_D`/`T_p`, or scale `c_m` so maintenance bites). This is the gating
   work.
2. **Engine loader** (then a thin, parity-safe change): in `simulate.py:1287-1292`, prefer
   `temperature.filename` → `PhysicalData.from_netcdf(path, varname=temperature.varname,
   nsteps_year=temperature.nsteps.year, factor, offset)` via `resolve_data_path(.., config_dir)`,
   else fall back to the constant. Default (no-filename) path stays bit-exact; the keys are already
   allowlisted; update the stale `config_validation.py:133-134` comment.
3. **Forcing generation** (verified feasible): CMEMS creds in `.env` are valid; dataset
   `cmems_mod_bal_phy_my_P1M-m` (var `thetao`) is reachable; the MCP `generate_osmose_physics`
   writes var **`temperature`** (NOT `temp` — set `temperature.varname;temperature`), shape
   `(24, 40, 50)`, lat-descending, **0.0 on land (NOT NaN)** — so validate ocean-cell temps are
   finite AND in a plausible band (−2…30 °C), guarding against silent 0 °C at CMEMS coastal gaps.
   Cap `download_field(depth_max≈10)` so the one-time `thetao` pull stays small (the BGC pull was
   2.9 GB at full depth). Grid alignment confirmed (ny=nlat=40, nx=nlon=50; no transpose;
   `step % 24` maps biweekly).
