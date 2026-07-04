# Salinity-gated cod occupancy — prototype design (spike)

**Date:** 2026-07-04
**Status:** approved design, pre-implementation. **This is a PROTOTYPE / proof-of-concept spike** — minimal, config-gated, inert-by-default, demonstrable in isolation. Production integration (Numba movement path + real CMEMS salinity + full Baltic run) is explicitly deferred to follow-up.
**Author:** brainstormed with the user
**Related:** `docs/baltic_percid_low_salinity_refuge_literature_review_2026-07-04.md` (the mechanism + caveat); `osmose/engine/processes/movement.py` (`_map_move_school`); `osmose/engine/movement_maps.py` (`MovementMapSet`); `osmose/engine/physical_data.py` (`PhysicalData`).

## 1. Motivation

The 2026-07-04 literature review found that Baltic percids (perch, pikeperch) are sheltered from marine predation mainly because cod's **distribution** is salinity/depth-structured — cod occupies the deep saline basins, not the oligohaline coastal cells where percids concentrate — and that adult cod are *not* foraging-excluded by salinity per se (Bergström et al. 2025). The recommended, most-defensible representation is therefore to make cod's **spatial occupancy** salinity-dependent (Ecospace habitat-capacity precedent; Neuenfeldt & Beyer 2003 show cod's realized prey overlap is already salinity/oxygen-gated in nature), rather than a local per-cell accessibility block.

This spike prototypes that mechanism: weight cod's movement-map occupancy by a per-cell salinity factor so cod schools avoid low-salinity cells; the reduction in cod–percid spatial overlap (and thus cod predation on percids) is **emergent** from where cod ends up.

## 2. Goals and non-goals

**Goals**
- A pure, tested salinity→occupancy-weight function and its application to a species' movement map.
- Wire it into the reference movement path (`_map_move_school`) so a gated predator's placement is weighted by the per-cell salinity field.
- Config-gated and **inert by default** — bit-identical output for every existing config unless explicitly enabled.
- Demonstrate the mechanism in isolation: with a synthetic salinity grid, a gated cod school lands only in high-salinity cells.

**Non-goals (deferred to follow-up)**
- The Numba movement batch path (`_map_move_batch_numba`) — production runs use it; the spike hooks only the Python reference path (mirrors the predation-kernel Numba situation). A full Baltic run therefore requires the Numba hookup as a follow-up.
- Real CMEMS `so` salinity forcing (26 GB, local, gitignored) — the spike uses a synthetic/constant salinity field.
- Measuring the actual percid-biomass / diet-overlap change in a full Baltic run.
- Any change to the predation kernel, the accessibility matrix, or other species' movement.
- Curing percid overshoot (see §7).

## 3. Concept & data flow

Cod's movement map assigns a per-cell occupancy weight `map[y,x] ≥ 0`. The placement sampler in `_map_move_school` draws cells with probability ∝ `map[y,x]` (rejection sampling, self-normalizing). Multiplying the map by a per-cell salinity weight `w(S[y,x]) ∈ [0,1]` makes occupancy come out ∝ `map · w` — cells below the low-salinity threshold get weight 0 (cod excluded), cells above the high threshold keep full weight, with a linear ramp between. No renormalization is needed because the rejection sampler self-normalizes.

```
per step, for a gated predator species:
  S = salinity_field.get_grid(step)              # (ny, nx) psu
  w = clip((S - s_low) / (s_high - s_low), 0, 1) # (ny, nx) in [0,1]
_map_move_school(..., salinity_weight_grid = w): # None when not gated -> inert
  wmap = current_map * w
  if wmap has no positive finite cell:  use current_map   # all-zero guard
  else:                                 use wmap (+ its nanmax) for placement AND random-walk
```

## 4. Components

### 4.1 Pure salinity-weight helpers (new module `osmose/engine/processes/salinity_gate.py`)
```python
def salinity_weight(salinity, s_low, s_high):   # scalar or ndarray -> [0,1]
    # clip((S - s_low) / (s_high - s_low), 0, 1); requires s_high > s_low
def salinity_weighted_map(map2d, salinity_grid, s_low, s_high):
    # returns map2d * salinity_weight(salinity_grid, ...);
    # if the result has no positive finite cell, returns map2d unchanged (guard).
```
Both are engine-state-free and fully unit-tested independent of any run.

### 4.2 `_map_move_school` change (`osmose/engine/processes/movement.py`)
Add an optional keyword parameter `salinity_weight_grid: NDArray[np.float64] | None = None`. When `None` (default), behavior is **unchanged** (bit-identical). When provided:
- Compute `wmap = current_map * salinity_weight_grid` once at the top (after the `current_map is None` check).
- If `wmap` has no positive, non-NaN cell, fall back to `current_map` (cod is never annihilated).
- Use `wmap` (and `np.nanmax(wmap)` as the acceptance normalizer) in place of `current_map`/`max_p` in **both** Step 3a (placement) and Step 3b (random walk).

### 4.3 Movement caller wiring (`movement()` Python path)
In the Python-path school loop (the branch that calls `_map_move_school`), before the loop compute the per-step weight grid once when the gate is enabled. Handle both a constant and a gridded salinity field:
```python
sal_w = None
if config.salinity_gate_enabled and config.salinity_field is not None:
    if config.salinity_field.is_constant():
        S = np.full((grid.ny, grid.nx), config.salinity_field.get_scalar())
    else:
        S = config.salinity_field.get_grid(step)         # (ny, nx)
    sal_w = np.clip((S - config.salinity_gate_s_low) /
                    (config.salinity_gate_s_high - config.salinity_gate_s_low), 0.0, 1.0)
```
Then pass `salinity_weight_grid = sal_w if config.salinity_gate_species[sp_id] else None` to `_map_move_school`. (Numba path unchanged — deferred.)

### 4.4 Config loading (`osmose/engine/config.py`)
New `EngineConfig` fields (all inert/None when the master switch is off):
- `salinity_gate_enabled: bool`
- `salinity_gate_species: NDArray[np.bool_] | None` (per-species predator mask)
- `salinity_gate_s_low: float`, `salinity_gate_s_high: float`
- `salinity_field: PhysicalData | None` (constant or NetCDF `so`)

A loader `_load_salinity_gate(cfg, n_species)` mirrors the RV-gate loader pattern: returns all-off when `movement.salinity.gate.enabled != true`; else validates `s_high > s_low`, at least one gated species, and a resolvable salinity field; fail-fast on bad config.

## 5. Config keys
```
movement.salinity.gate.enabled          = false        # master switch (inert default)
movement.salinity.gate.species.sp{idx}  = true|false   # gated predators (Baltic: cod sp0)
movement.salinity.gate.s.low            = 5.0           # psu; weight 0 at/below
movement.salinity.gate.s.high           = 11.0         # psu; weight 1 at/above
movement.salinity.field.constant        = <psu>        # OR the NetCDF pair below
movement.salinity.field.file            = <path.nc>
movement.salinity.field.varname         = so
```
New keys added as schema `OsmoseField`s and captured by the `config_validation` allowlist (AST walker, like the RV-gate keys).

## 6. Testing / demonstration

- **Unit (TDD):**
  - `salinity_weight`: 0 at/below `s_low`, 1 at/above `s_high`, linear between; array form; `s_high <= s_low` raises.
  - `salinity_weighted_map`: zeros low-salinity cells, leaves high-salinity cells, linear ramp mid-range; all-zero-result guard returns the original map.
- **Placement-level (demonstration):** call `_map_move_school` directly with a synthetic map (uniform over ocean) + a synthetic salinity grid (low in one region, high in another); assert a gated school is placed only in high-salinity cells over many draws, and is placed across the whole map when `salinity_weight_grid=None`.
- **Parity (inert-by-default):** with the master switch off, a short engine run is bit-identical to master (movement untouched — `salinity_weight_grid` stays `None`).
- **Loader:** `_load_salinity_gate` fail-fast cases (bad thresholds, no species, missing field).

## 7. The honest caveat (carried from the review, stated in the spike output)

This is a **spatial-realism** correction, not a percid-overshoot fix. Sheltering percids from cod *raises* percid biomass, which would if anything **worsen** the ×38–96 percid overshoot, not cure it. The spike demonstrates the mechanism and its spatial effect on cod placement; it is not expected to reduce overshoot, and the follow-up full-run analysis should be read in that light. (The review further suggests that a faithful refuge also needs the non-fish predators — cormorant/seal — represented *inside* the refuge so percid mortality there is not spuriously zeroed.)

## 8. Deliverables

- `osmose/engine/processes/salinity_gate.py` — the two pure helpers.
- `osmose/engine/processes/movement.py` — optional `salinity_weight_grid` on `_map_move_school` + the Python-path caller wiring.
- `osmose/engine/config.py` — `_load_salinity_gate` + new `EngineConfig` fields, wired in `from_dict`.
- `osmose/schema/species.py` — the new config-key `OsmoseField`s, placed alongside the existing `movement.*` field definitions (same registry as `movement.species.map{idx}` / `movement.randomseed.fixed`; grep for where those are defined and add there).
- `osmose/engine/config_validation.py` — allowlist entries if the AST walker does not auto-capture.
- Tests covering §6.
- A short demonstration script (optional) building a synthetic Baltic salinity grid and showing cod placement shift on vs off.
