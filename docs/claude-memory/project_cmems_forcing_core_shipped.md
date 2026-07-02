---
name: project-cmems-forcing-core-shipped
description: 2026-06-21 CMEMS→OSMOSE forcing conversion CORE (sub-project A) — shipped; B/C/D deferred
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**CMEMS→OSMOSE forcing conversion core (sub-project A) — SHIPPED 2026-06-21, merged master `9d39da0` (local merge + push, no PR), prod clone advanced** (library/CLI/MCP-refactor only — no UI wiring yet, so prod runtime behavior unchanged).

Backlog item "Copernicus LTL forcing" (user chose it). Decomposed into 4 sub-projects; **A is the foundation**, B/C/D deferred to their own cycles.

**What A shipped:** lifted the LTL + physics conversion logic OUT of `mcp_servers/copernicus/server.py` into a pure, grid-general **`osmose/forcing/`** package (numpy/xarray/scipy only — clean-CI safe, NEVER imports copernicusmarine/fastmcp), added a convert-only CLI, and refactored the MCP `generate_osmose_*` tools to DELEGATE to the core (parity-guarded).
- `osmose/forcing/grid.py` — `target_coords`/`regrid`/`resample_to_24`/`cell_volume_m3`/`get_var`/`load_ocean_mask`/`apply_land_mask`, all parameterized by `GridSpec` (reused from `osmose/maps/builder.py`); `regrid` warns when the target grid exceeds the source extent (silent edge-extrapolation guard).
- `osmose/forcing/ltl.py` — `LtlParams` (Baltic-calibrated defaults, provenance in docstrings + a `calibration` output attr) + `bgc_to_ltl` (Mode A direct phyc/zooc / Mode B chl-derived, 6 groups, depth-slice guard).
- `osmose/forcing/physics.py` — `phy_to_physics` (temperature/salinity).
- `osmose/forcing/io.py` — `write_ltl`/`write_physics` (overwrite clobber guard; `write_physics` returns `{name: path}`).
- `scripts/convert_cmems_forcing.py` — convert-only CLI (BYO downloaded NetCDF; dir-or-master `--config`; `--force`).
- MCP `generate_osmose_ltl/physics` now thin wrappers over the core; 8 helpers + `OSMOSE_GRID` deleted; anti-drift parity test (`tests/test_forcing_mcp_parity.py`, find_spec-guarded).

**GOTCHAS:** faithful extraction (parity test guards the math); `osmose/forcing/__init__.py` created EMPTY in task 1 so setuptools `packages.find` (not find_namespace) ships the subpackage in a wheel; physics output targets the JAVA engine (Python engine reads only scalar `temperature.value`, no salinity — sub-project C must not assume Python-engine wiring); CLI is dev-tree-only (not in wheel, no `[project.scripts]` entry).

**Deferred (own specs/plans/cycles):** B = live CMEMS download (creds-gated; **blocked on the owed `.env` credential rotation**); C = config scaffolding/repointing (wire forcing into a config; resolves the "consumed unchanged" LTL species-name contract); D = "CMEMS Forcing" Shiny page. Spec+plan `docs/superpowers/{specs,plans}/2026-06-20-cmems-forcing-core*`. Hardened via 3 in-loop rounds + 1 workflow review; verified full suite + parity + clean-venv guard.
