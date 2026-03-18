# Python Engine — Complete Implementation Plan (Remaining Gaps)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all remaining gaps between the Python and Java OSMOSE engines to achieve ecosystem-level parity on the Bay of Biscay example configuration.

**Current state:** 1313 tests passing, ~40% feature coverage, processes match Java formulas exactly but ecosystem dynamics diverge due to missing structural features.

**Spec:** `docs/superpowers/specs/2026-03-18-python-engine-design.md`

---

## Priority-Ordered Task Groups

### Group A: Ecosystem-Critical (must fix for ecosystem to self-sustain)

| Task | Feature | Impact | Effort | Why Critical |
|------|---------|--------|--------|-------------|
| A1 | Interleaved mortality ordering | High | Medium | Java shuffles causes per school; Python applies in fixed order. Changes competitive dynamics. |
| A2 | Egg retain/release | Medium | Low | Java releases eggs progressively; Python exposes all immediately. Affects egg survival rate. |
| A3 | Trophic level computation | Medium | Low | Needed for output comparison and diet tracking. |

### Group B: Quantitative Accuracy

| Task | Feature | Impact | Effort |
|------|---------|--------|--------|
| B1 | Map-based movement | High | Medium |
| B2 | Feeding stages (per-stage size ratios) | Medium | Medium |
| B3 | Fishing selectivity (knife-edge, sigmoid) | Medium | Low |

### Group C: Output & Validation

| Task | Feature | Impact | Effort |
|------|---------|--------|--------|
| C1 | Mortality rate CSV output | Medium | Low |
| C2 | Diet matrix output | Medium | Low |
| C3 | Validation pipeline script | High | Medium |

### Group D: Performance & Extensions

| Task | Feature | Impact | Effort |
|------|---------|--------|--------|
| D1 | JAX/GPU backend | Performance | High |
| D2 | Background species | Medium | Medium |
| D3 | Incoming flux (migration) | Low | Low |

---

## Task A1: Interleaved Mortality Ordering

**Files:**
- Create: `osmose/engine/processes/mortality.py` (orchestrator)
- Modify: `osmose/engine/simulate.py` (replace `_mortality`)
- Create: `tests/test_engine_mortality_loop.py`

**Algorithm (matching Java MortalityProcess.java lines 464-716):**

```python
def mortality_loop(state, resources, config, rng, grid):
    """Full mortality orchestrator matching Java's double-loop structure."""
    n_subdt = config.mortality_subdt
    n_schools = len(state)

    # Pre-pass: larva mortality on eggs
    state = larva_mortality(state, config)

    for sub in range(n_subdt):
        # Each cause gets its own random school ordering
        seq_pred = rng.permutation(n_schools).astype(np.int32)
        seq_fish = rng.permutation(n_schools).astype(np.int32)
        seq_starv = rng.permutation(n_schools).astype(np.int32)
        seq_nat = rng.permutation(n_schools).astype(np.int32)

        # Per-school cause shuffling
        for i in range(n_schools):
            causes = rng.permutation(4)  # [PRED, STARV, FISH, ADDITIONAL]
            for cause in causes:
                if cause == 0:  # PREDATION
                    idx = seq_pred[i]
                    apply_predation_for_school(idx, state, resources, config, rng, n_subdt, grid)
                elif cause == 1:  # STARVATION
                    idx = seq_starv[i]
                    apply_starvation_for_school(idx, state, config, n_subdt)
                elif cause == 2:  # FISHING
                    idx = seq_fish[i]
                    apply_fishing_for_school(idx, state, config, n_subdt)
                elif cause == 3:  # ADDITIONAL
                    idx = seq_nat[i]
                    apply_additional_for_school(idx, state, config, n_subdt)

    # Post-loop
    state = out_mortality(state, config)
    state = update_starvation_rate(state, config)
    state = update_trophic_level(state)
    return state
```

**Key change:** Individual mortality functions need per-school variants (not batch).

- [ ] **Step 1:** Write per-school mortality functions in `mortality.py`
- [ ] **Step 2:** Write tests verifying cause order is randomized per school
- [ ] **Step 3:** Wire into simulate.py
- [ ] **Step 4:** Verify all existing tests still pass
- [ ] **Step 5:** Commit

**Note:** This is the most complex structural change. The per-school predation call means predation can no longer be batched per cell — each school's predation happens at a different point in the sequence. This requires restructuring the predation to work on individual predators rather than cells. The Numba inner loop already works per-predator within a cell; the change is to call it from the mortality orchestrator rather than the cell-grouping outer loop.

---

## Task A2: Egg Retain/Release

**Files:**
- Modify: `osmose/engine/state.py` (add `egg_retained` field)
- Modify: `osmose/engine/processes/mortality.py` (add retain/release)
- Create: `tests/test_engine_egg_handling.py`

**Algorithm (matching Java School.java):**

```python
def retain_eggs(state):
    """Called before mortality loop. Eggs are withheld from prey pool."""
    egg_mask = state.is_egg
    egg_retained = np.zeros(len(state))
    egg_retained[egg_mask] = state.abundance[egg_mask]
    return state.replace(egg_retained=egg_retained)

def release_eggs(state, n_subdt):
    """Called each sub-timestep. Release 1/n_subdt of retained eggs."""
    egg_mask = state.is_egg & (state.egg_retained > 0)
    release = state.abundance[egg_mask] / n_subdt
    new_retained = state.egg_retained.copy()
    new_retained[egg_mask] = np.maximum(0, new_retained[egg_mask] - release)
    return state.replace(egg_retained=new_retained)
```

**Predation change:** Use `abundance - egg_retained` as available prey biomass for egg schools.

- [ ] **Step 1:** Add `egg_retained` field to SchoolState
- [ ] **Step 2:** Implement retain/release functions
- [ ] **Step 3:** Integrate into mortality loop
- [ ] **Step 4:** Test: eggs released progressively, not all at once
- [ ] **Step 5:** Commit

---

## Task A3: Trophic Level Computation

**Files:**
- Modify: `osmose/engine/state.py` (add diet tracking)
- Create: `osmose/engine/processes/trophic.py`
- Modify: `osmose/engine/processes/predation.py` (track prey TL)
- Create: `tests/test_engine_trophic.py`

**Algorithm (matching Java MortalityProcess.java post-loop):**

```python
def update_trophic_level(state, diet_record):
    """Compute weighted-average trophic level from consumed prey.

    TL = 1 + sum(prey_TL * biomass_eaten) / total_preyed_biomass
    """
    new_tl = state.trophic_level.copy()
    for i in range(len(state)):
        if state.preyed_biomass[i] > 0 and i in diet_record:
            weighted_tl = sum(
                prey_tl * bio for prey_tl, bio in diet_record[i]
            )
            new_tl[i] = 1.0 + weighted_tl / state.preyed_biomass[i]
    return state.replace(trophic_level=new_tl)
```

**Diet tracking:** Add a lightweight per-timestep dict `{predator_idx: [(prey_tl, biomass), ...]}` populated during predation.

- [ ] **Step 1:** Add diet tracking structure
- [ ] **Step 2:** Record prey TL during predation
- [ ] **Step 3:** Implement `update_trophic_level`
- [ ] **Step 4:** Test: TL = 1 + prey_TL for single prey, weighted average for multiple
- [ ] **Step 5:** Commit

---

## Task B1: Map-Based Movement

**Files:**
- Create: `osmose/engine/processes/movement_maps.py`
- Modify: `osmose/engine/processes/movement.py` (dispatch by method)
- Modify: `osmose/engine/config.py` (add map loading config)
- Create: `tests/test_engine_movement_maps.py`

**Algorithm (matching Java MapDistribution.java):**

```python
def map_distribution(state, grid, config, step, rng):
    """Distribute schools on probability maps per species/age/season."""
    for sp in range(config.n_species):
        if config.movement_method[sp] != "maps":
            continue
        sp_mask = state.species_id == sp
        if not sp_mask.any():
            continue

        # Get map for this species/age/timestep
        map_2d = load_movement_map(config, sp, step)
        if map_2d is None:
            # Mark as out-of-domain
            state.is_out[sp_mask] = True
            continue

        # Probability-weighted cell selection
        flat_probs = map_2d.flatten()
        flat_probs = np.maximum(0, flat_probs)
        total = flat_probs.sum()
        if total <= 0:
            continue
        flat_probs /= total

        n = sp_mask.sum()
        cells = rng.choice(len(flat_probs), size=n, p=flat_probs)
        state.cell_y[sp_mask] = cells // grid.nx
        state.cell_x[sp_mask] = cells % grid.nx

    return state
```

- [ ] **Step 1:** Create movement map loader (NetCDF per species/age/season)
- [ ] **Step 2:** Implement probability-weighted cell selection
- [ ] **Step 3:** Add out-of-domain detection (null map → `is_out=True`)
- [ ] **Step 4:** Wire into movement orchestrator
- [ ] **Step 5:** Test: schools distributed according to map probabilities
- [ ] **Step 6:** Commit

---

## Task B2: Feeding Stages

**Files:**
- Modify: `osmose/engine/config.py` (per-stage size ratio arrays)
- Modify: `osmose/engine/processes/predation.py` (stage-indexed ratios)
- Modify: `tests/test_engine_predation.py`

**Current:** Single `size_ratio_min/max` per species.
**Needed:** `size_ratio_min[species][stage]`, `size_ratio_max[species][stage]` where stage is determined by age or size class.

**Config keys:**
```
predation.predPrey.stage.structure ; age  (or "size")
predation.predPrey.stage.threshold.sp0 ; 0;1;2  (age boundaries in years)
predation.predPrey.sizeRatio.min.sp0 ; 1.0;1.5;2.0  (per-stage min)
predation.predPrey.sizeRatio.max.sp0 ; 3.5;3.0;4.0  (per-stage max)
```

- [ ] **Step 1:** Parse per-stage ratio arrays from config
- [ ] **Step 2:** Compute feeding stage from age_dt
- [ ] **Step 3:** Index ratios by stage in predation
- [ ] **Step 4:** Test: different ratios at different ages
- [ ] **Step 5:** Commit

---

## Task B3: Fishing Selectivity

**Files:**
- Create: `osmose/engine/processes/selectivity.py`
- Modify: `osmose/engine/processes/fishing.py`
- Create: `tests/test_engine_selectivity.py`

**Types:**
- Knife-edge: `S(L) = 1 if L >= L50, else 0`
- Sigmoid: `S(L) = 1 / (1 + exp(-slope * (L - L50)))`

```python
def knife_edge(length, l50):
    return np.where(length >= l50, 1.0, 0.0)

def sigmoid(length, l50, slope):
    return 1.0 / (1.0 + np.exp(-slope * (length - l50)))
```

Apply as multiplier to fishing mortality: `F_effective = F_base * selectivity(length)`

- [ ] **Step 1:** Implement selectivity functions
- [ ] **Step 2:** Wire into fishing_mortality
- [ ] **Step 3:** Test: no catch below L50, full catch above
- [ ] **Step 4:** Commit

---

## Task C1: Mortality Rate CSV Output

**Files:**
- Modify: `osmose/engine/output.py`
- Modify: `osmose/engine/simulate.py` (add mortality tracking to StepOutput)
- Modify: `tests/test_engine_output.py`

**Format (matching Java):**
```
"Mortality rates per time step..."
"Time","Mpred","Mstarv","Madd","F","Zout","Mage"
0.041666668,0.0,0.0,0.0,0.0,0.0,0.0
```

**Implementation:** Add `mortality_by_cause: NDArray` to `StepOutput`, aggregate from `state.n_dead` per species.

- [ ] **Step 1:** Extend StepOutput with mortality data
- [ ] **Step 2:** Write mortality CSV matching Java format
- [ ] **Step 3:** Test: correct headers, correct values
- [ ] **Step 4:** Commit

---

## Task C2: Diet Matrix Output

**Files:**
- Modify: `osmose/engine/output.py`
- Modify: `osmose/engine/simulate.py`

**Format:** `{prefix}_dietMatrix_Simu0.csv` with predator rows × prey columns.

- [ ] **Step 1:** Aggregate diet from predation tracking
- [ ] **Step 2:** Write diet CSV
- [ ] **Step 3:** Test
- [ ] **Step 4:** Commit

---

## Task C3: Validation Pipeline Script

**Files:**
- Create: `scripts/validate_engines.py`

**What it does:**
1. Run Java engine on Bay of Biscay (30 years)
2. Run Python engine on same config
3. Compare: biomass trajectories, mortality by cause, diet matrices
4. Generate HTML report with plots
5. Print pass/fail per metric

- [ ] **Step 1:** Create script skeleton
- [ ] **Step 2:** Add Java runner integration
- [ ] **Step 3:** Add comparison metrics (KS test, Frobenius norm, etc.)
- [ ] **Step 4:** Generate HTML report
- [ ] **Step 5:** Commit

---

## Task D1: JAX/GPU Backend

**Files:**
- Modify: `osmose/engine/processes/predation.py` (JAX-compatible inner loop)
- Modify: `osmose/engine/processes/growth.py` (use `xp` pattern)
- Create: `osmose/engine/backend.py`

**Approach:**
- Fixed-capacity buffer with `alive` mask for JIT compatibility
- `jax.vmap` over padded cell batches
- `jax.lax.fori_loop` for sub-timestep loop

- [ ] **Step 1:** Create backend abstraction (`xp` module pattern)
- [ ] **Step 2:** Convert growth to use `xp`
- [ ] **Step 3:** Create JAX-compatible predation kernel
- [ ] **Step 4:** Benchmark: Java vs NumPy vs JAX
- [ ] **Step 5:** Commit

---

## Execution Order

```
Phase 1: A1 (mortality ordering) + A2 (egg handling)  — ecosystem structure
Phase 2: A3 (trophic level) + C1 (mortality output)   — observability
Phase 3: B1 (map movement) + B2 (feeding stages)      — spatial accuracy
Phase 4: B3 (selectivity) + C2 (diet output)           — fisheries
Phase 5: C3 (validation pipeline)                       — verification
Phase 6: D1 (JAX backend)                               — performance
```

Each phase is independently testable and deployable.

---

## Success Criteria

**Ecosystem parity achieved when:**
1. Bay of Biscay 30-year run produces biomass within **1 order of magnitude** of Java for all 8 species
2. Mortality-by-cause proportions match Java within **20%** per species
3. Diet matrix Frobenius norm < 0.1
4. At least **6/8 species** persist (same as Java)
5. Shannon diversity within **15%** of Java

**Performance target:**
- Python NumPy: <= 2x Java speed
- Python Numba: <= 0.5x Java speed (already achieved: 0.3s vs 0.57s)
- Python JAX/GPU: <= 0.1x Java speed
