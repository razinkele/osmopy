# Salinity-gated cod occupancy — prototype design (spike)

**Date:** 2026-07-04
**Status:** approved design, pre-implementation. **This is a PROTOTYPE / proof-of-concept spike** — minimal, config-gated, inert-by-default, demonstrable in isolation. Production integration (Numba movement path + real CMEMS salinity + full Baltic run) is explicitly deferred to follow-up.
**Author:** brainstormed with the user
**Related:** `docs/baltic_percid_low_salinity_refuge_literature_review_2026-07-04.md` (the mechanism + caveat); `osmose/engine/processes/movement.py` (`_map_move_school`); `osmose/engine/movement_maps.py` (`MovementMapSet`); `osmose/engine/physical_data.py` (`PhysicalData`).

## 1. Motivation

The 2026-07-04 literature review found that Baltic percids (perch, pikeperch) are sheltered from marine predation mainly because cod's **distribution** is salinity/depth-structured — cod occupies the deep saline basins, not the oligohaline coastal cells where percids concentrate — and that adult cod are *not* foraging-excluded by salinity per se (Bergström et al. 2025). The recommended, most-defensible representation is therefore to make cod's **spatial occupancy** salinity-dependent (Ecospace habitat-capacity precedent; Neuenfeldt & Beyer 2003 show cod's realized prey overlap is already salinity/oxygen-gated in nature), rather than a local per-cell accessibility block.

This spike prototypes that mechanism: weight cod's movement-map occupancy by a per-cell salinity factor so cod schools avoid low-salinity cells. The *hypothesis* — to be tested in the follow-up full run, not asserted here — is that reduced cod occupancy in oligohaline cells lowers cod–percid spatial overlap (and thus cod predation on percids) as an **emergent** effect (with the important side effect, see §7, that it also lowers cod predation on *all* other prey in those cells). Whether percids actually concentrate in those cells in the Baltic config, and whether the overlap drop materializes, is verified only once the real salinity field is wired.

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
  if wmap has no positive finite cell:  use current_map   # all-zero guard (bit-identical to ungated)
  else:                                 # occupancy ∝ wmap, via:
    # placement (Step 3a): rejection sampling with nanmax(wmap) normalizer
    # random-walk (Step 3b): weighted selection ∝ wmap  (NOT uniform)  — see §4.2
```

## 4. Components

### 4.1 Pure salinity-weight helpers (new module `osmose/engine/processes/salinity_gate.py`)
```python
def salinity_weight(salinity, s_low, s_high):   # scalar or ndarray -> [0,1]
    # clip((S - s_low) / (s_high - s_low), 0, 1); raises ValueError if s_high <= s_low
def salinity_weighted_map(map2d, salinity_grid, s_low, s_high):
    # returns map2d * salinity_weight(salinity_grid, ...);
    # if the result has no positive finite cell, returns map2d unchanged (guard).
    # "positive finite" == > 0 and not NaN/inf; same guard phrasing as §4.2.
```
Both are engine-state-free and fully unit-tested independent of any run.

### 4.2 `_map_move_school` change (`osmose/engine/processes/movement.py`)
Add an optional keyword parameter `salinity_weight_grid: NDArray[np.float64] | None = None`. When `None` (default), behavior is **unchanged** (bit-identical). When provided:
- Compute `wmap = current_map * salinity_weight_grid` once at the top (after the `current_map is None` check, where `current_map` is guaranteed non-None).
- **All-zero guard:** if `wmap` has no positive finite cell (`> 0`, not NaN/inf), fall back to the *original* `current_map` **and the original cached normalizer** `map_set.max_proba[index_map]` (i.e. behave exactly as ungated for this school) so cod is never annihilated and the fallback is bit-identical to ungated placement.
- Otherwise use `wmap` in **both** Step 3a and Step 3b, but note the two steps have **different sampling mechanics** (verified against the code, review R4):
  - **Step 3a (new placement, lines 80–92)** is rejection sampling with an acceptance normalizer. Use `wmap` with `np.nanmax(wmap)` in place of `current_map`/`max_p`; occupancy comes out ∝ `wmap`.
  - **Step 3b (random walk, lines 94–103)** does **not** use `max_p` — the current code builds an `accessible` list of cells where `current_map > 0` and picks one **uniformly** (`rng.integers(0, len(accessible))`). A literal `wmap > 0` substitution here would make the gate a *hard binary cutoff* during walk (no difference between a 4 psu cell, w=0.33, and an 8 psu cell, w=1.0), collapsing the graded ramp exactly where schools spend most steps. **Fix:** when gated, admit cells with `wmap > 0` and select **weighted by `wmap[cell]`** (e.g. cumulative-weight draw with a single `rng.random()`, or `rng.choice(p=weights/weights.sum())`), so 3b occupancy is also ∝ `wmap`. When *ungated* (`salinity_weight_grid is None`), 3b keeps the exact original uniform `rng.integers` draw so master runs stay bit-identical.
  - Both placement tests in §6 must assert the graded effect (mid-range weight yields intermediate occupancy), not just a binary present/absent split, so this 3b path is actually covered.

**Important semantics of the existing `max_p` (from review R1).** In the current code `max_p == 0.0` is **not** a "no valid cells" sentinel — `MovementMapSet` sets `max_proba = 0.0` deliberately for **presence/absence maps** (raw values ≥ 1.0), which makes the acceptance test `max_p == 0.0 or proba >= rng.random()*max_p` short-circuit to *uniform* acceptance among present cells. **Every bundled movement-map CSV in this repo is binary `{-99, 0, 1}`** (verified), so with the gate on, `wmap = 1·w = w`, `np.nanmax(wmap) = max(w)`, and the acceptance test `w >= rng.random()*max(w)` samples occupancy ∝ `w` among present cells — exactly the intended salinity weighting. **Assumption (in-scope):** this spike assumes binary presence/absence maps. A future *graded*-probability map would make `wmap = map·w` reintroduce the raw-magnitude weighting that the `max_p == 0.0` flag deliberately suppresses; handling graded maps is out of spike scope and must be flagged if such a map is ever added.

### 4.3 Movement caller wiring (`movement()` Python path)
In the Python-path school loop (the branch that calls `_map_move_school`), before the loop compute the per-step weight grid once when the gate is enabled. Handle both a constant and a gridded salinity field:
```python
sal_w = None
if config.salinity_gate_enabled and config.salinity_field is not None:
    if config.salinity_field.is_constant:   # property, NOT a method — no parens
        S = np.full((grid.ny, grid.nx), config.salinity_field.get_scalar())
    else:
        S = config.salinity_field.get_grid(step)         # (ny, nx)
    sal_w = np.clip((S - config.salinity_gate_s_low) /
                    (config.salinity_gate_s_high - config.salinity_gate_s_low), 0.0, 1.0)
```
Then pass `salinity_weight_grid = sal_w if config.salinity_gate_species[sp_id] else None` to `_map_move_school`. (Numba path unchanged — deferred.) `sal_w` is computed once per step (not per school); `wmap = current_map * sal_w` is recomputed per gated school inside `_map_move_school` because `current_map` varies by the school's age/step map — accepted O(schools) redundancy, by design for the Python-only spike (perf/Numba deferred).

### 4.4 Config loading (`osmose/engine/config.py`)
New `EngineConfig` fields. When the master switch is off the *feature* is inert (the movement caller only computes `sal_w` when `salinity_gate_enabled and salinity_field is not None`), so the scalar fields still hold their schema defaults but are never read:
- `salinity_gate_enabled: bool` — `False` when off.
- `salinity_gate_species: NDArray[np.bool_] | None` — `(n_species,)` per-species predator mask; `None` when off.
- `salinity_gate_s_low: float`, `salinity_gate_s_high: float` — always the schema defaults (`3.0`/`6.0`) unless overridden; unused when `salinity_gate_enabled` is `False`.
- `salinity_field: PhysicalData | None` — constant or NetCDF `so`; `None` when off.

The inert guarantee comes from `salinity_field is None` / `salinity_gate_enabled is False` gating the caller, not from the scalar thresholds — mirroring how `rv_gate_enabled`/`rv_gate_factor_by_index` are `None` when off while other scalars keep defaults.

A loader `_load_salinity_gate(cfg, n_species)` mirrors the RV-gate loader pattern. **Return contract** (populates the five §4.4 fields, in order):
```python
def _load_salinity_gate(cfg, n_species) -> tuple[
    bool,                       # salinity_gate_enabled
    NDArray[np.bool_] | None,   # salinity_gate_species  (n_species,)
    float,                      # salinity_gate_s_low
    float,                      # salinity_gate_s_high
    PhysicalData | None,        # salinity_field
]: ...
```
When `movement.salinity.gate.enabled != "true"` it returns `(False, None, 3.0, 6.0, None)` (scalars are the schema defaults, unused). When on, it parses/validates and returns `(True, mask, s_low, s_high, field)`. Validation (fail-fast, `ValueError`/`FileNotFoundError`): `s_high > s_low`, at least one gated species, and a resolvable salinity field (constant or NetCDF).

**Config-validation capture (from review R1).** The `config_validation` AST walker only auto-captures keys read as **literal** `cfg.get("...")` / f-string `cfg.get(f"...sp{sp}")` calls inside `config.py` — exactly how `_load_rv_gate` reads its keys (proven captured, no allowlist entry). Write `_load_salinity_gate` the same way. It does **not** capture keys assembled via `key.startswith(...)` string-matching — the codebase has a documented counter-example (`movement.species.map{idx}`, manually listed in `_SUPPLEMENTARY_ALLOWLIST`). After implementing, run `test_from_dict_warn_mode_clean_on_example_configs`; if it flags any new `movement.salinity.*` key, add that key pattern to `_SUPPLEMENTARY_ALLOWLIST` (fallback only).

## 5. Config keys
```
movement.salinity.gate.enabled          = false        # master switch (inert default)
movement.salinity.gate.species.enabled.sp{idx} = true|false   # gated predators (Baltic: cod sp0)
movement.salinity.gate.s.low            = 3.0           # psu; weight 0 at/below
movement.salinity.gate.s.high           = 6.0           # psu; weight 1 at/above
movement.salinity.field.constant        = <psu>        # OR the NetCDF pair below
movement.salinity.field.file            = <path.nc>
movement.salinity.field.varname         = so
```
New keys added as schema `OsmoseField`s and captured by the `config_validation` allowlist (AST walker, like the RV-gate keys).

**Threshold calibration (from review R2).** The defaults `s_low = 3.0`, `s_high = 6.0` are chosen to respect the review's explicit guidance (§8): *act on very-low salinity (<~5 psu) nearshore, but be modest across the 7–11 psu band where adult cod forage unimpeded (Bergström et al. 2025)*. With this ramp the weight is `clip((S-3)/3, 0, 1)`: **1.0 at ≥6 psu** (cod untouched at 7–7.5 psu and above — no re-import of the ~11 psu reproductive floor), **0.67 at 5 psu** (a mild ~33% reduction, *not* exclusion), **0.33 at 4 psu**, and **0 at ≤3 psu**. So cod is only *strongly* excluded in the oligohaline <~4 psu cells (e.g. Bothnian Bay ~2.7 psu, inner Gulf of Riga, Curonian-Lagoon mouth) and merely *nudged* across 4–6 psu. This is deliberately gentler than the review's "act on <~5 psu" phrasing (which the ramp treats as the *onset* of a mild reduction, not a cutoff), erring toward the review's stronger "modest, not a hard cutoff" constraint. An earlier `5/11` default cut cod ~60% at 7 psu and is rejected as too aggressive. **The thresholds are the primary calibration knob** and should be re-tuned against the real salinity field in the follow-up Baltic run.

**Stage-awareness (from review R2).** The gate is a *species-level* switch (`...species.enabled.sp{idx}`) applied uniformly to whichever cod life-stage map is active. This is acceptable for the spike precisely because the feeding-calibrated ramp (weight 1 by 6 psu) does **not** re-impose cod's ~11–12 psu *reproductive* floor on any stage — it models cod's *feeding* occupancy in oligohaline cells only. A stage-aware variant (a tighter gate on the `cod_spawning` map, looser/none on `cod_adult`/`cod_juvenile`) is a sensible refinement but is out of spike scope and noted here rather than built.

## 6. Testing / demonstration

**Shared test fixture (pin this so the two placement tests agree).** Use a `Grid.from_dimensions(ny=5, nx=6)` (the convention in `tests/test_engine_map_movement.py`), all ocean. Synthetic salinity with **three bands** so the graded ramp (and Step 3b weighting) is actually exercised, not just a binary split — with defaults `s_low=3`, `s_high=6`:
- columns 0–1 = `2.0` psu → weight `0.0` (excluded),
- columns 2–3 = `4.5` psu → weight `0.5` (mid, retained at half occupancy),
- columns 4–5 = `8.0` psu → weight `1.0` (full).

Cod map = presence (`1.0`) over all cells. Expected with the gate on, over many draws: ~0 schools in cols 0–1, and cols 4–5 hold ≈2× the occupancy of cols 2–3 (the graded effect); gate off: uniform across all columns. **Force the Python (non-Numba) path** by calling `movement(...)` with `flat_map_data=None` (branch condition `_HAS_NUMBA and flat_map_data is not None`, `movement.py:264`), as `TestMovementOrchestrator` already does. Use a fixed-seed RNG and a statistical tolerance on the ~2× ratio.

- **Unit (TDD):**
  - `salinity_weight`: 0 at/below `s_low`, 1 at/above `s_high`, linear between; array form; `s_high <= s_low` raises.
  - `salinity_weighted_map`: zeros low-salinity cells, leaves high-salinity cells, linear ramp mid-range; all-zero-result guard returns the original map.
- **Placement-level (demonstration):** call `_map_move_school` directly with a synthetic map (uniform over ocean) + a synthetic salinity grid (low in one region, high in another); assert a gated school is placed only in high-salinity cells over many draws, and is placed across the whole map when `salinity_weight_grid=None`.
- **Caller-wiring, gate ON (from review R2):** run the Python-path `movement()` end-to-end (force the non-Numba branch) with the gate enabled for cod and a constant/synthetic salinity field, asserting cod schools shift toward high-salinity cells vs the gate-off run. This exercises the §4.3 glue — the `is_constant` branch, `salinity_gate_species[sp_id]` indexing, per-step `sal_w` — which the bare `_map_move_school` test does not cover.
- **Parity (inert-by-default):** with the master switch off, a short engine run is bit-identical to master (movement untouched — `salinity_weight_grid` stays `None`).
- **Loader:** `_load_salinity_gate` fail-fast cases (bad thresholds `s_high <= s_low`, no gated species, missing field) + the constant-field happy path.

**Salinity-field scope (from review R2).** The loader supports both a constant field and a NetCDF `so` field (via `PhysicalData.from_netcdf`), but the spike's tests exercise only the **constant** path and the fail-fast cases. The gridded-NetCDF path is wired (cheap reuse of `PhysicalData`) but its behavior is first genuinely exercised in the follow-up real-CMEMS Baltic run, not in the spike — stated here so it is not mistaken for tested-in-spike.

## 7. The honest caveat (carried from the review, stated in the spike output)

Three caveats, all carried from the review (§4/§8), that the spike output must state:

1. **Not an overshoot fix.** This is a *spatial-realism* correction. Sheltering percids from cod *raises* percid biomass, which would if anything **worsen** the ×38–96 percid overshoot, not cure it. The spike demonstrates the mechanism and its spatial effect on cod placement; it is not expected to reduce overshoot, and the follow-up full-run analysis should be read in that light.
2. **Side effect on cod's own foraging (from review R2, data-confirmed).** Gating cod *occupancy* suppresses cod predation on **all** prey in the gated cells — and cod's actual dominant Baltic prey (herring, sprat, round goby) occupies the *same* oligohaline coastal cells as the percids (verified in the bundled Baltic maps: e.g. `herring_adult` overlaps 100% of `perch_juvenile` cells). So excluding cod from those cells also cuts cod off from its real food, with bioenergetic/population consequences for cod itself. The feeding-calibrated `3/6` ramp limits this to the <~4 psu cells, but the effect is real, not hypothetical, and must be reported (and quantified in the follow-up run — e.g. change in cod's clupeid/goby diet fraction).
3. **Modest, not a hard cutoff.** Per review §8, the gate must act on very-low salinity and stay *modest* across 7–11 psu (adult cod feed fine at 7–7.5 psu). The `3/6` default satisfies this (weight 1 by 6 psu); do not tighten it into the 7–11 band. A faithful refuge would also need the non-fish predators (cormorant/seal) represented *inside* the refuge so percid mortality there is not spuriously zeroed.

## 8. Deliverables

- `osmose/engine/processes/salinity_gate.py` — the two pure helpers.
- `osmose/engine/processes/movement.py` — optional `salinity_weight_grid` on `_map_move_school` + the Python-path caller wiring.
- `osmose/engine/config.py` — `_load_salinity_gate` + new `EngineConfig` fields, wired in `from_dict`.
- `osmose/schema/movement.py` — the new config-key `OsmoseField`s, placed alongside the existing `movement.*` field definitions (`movement.species.map{idx}` / `movement.randomseed.fixed` are defined here, verified).
- `osmose/engine/config_validation.py` — allowlist entries if the AST walker does not auto-capture.
- Tests covering §6.
- A short demonstration script (optional) building a synthetic Baltic salinity grid and showing cod placement shift on vs off.
