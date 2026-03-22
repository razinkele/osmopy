# Python Engine Performance Optimization Design

> **Date:** 2026-03-22
> **Current state:** Python ~12-165x slower than Java depending on run length (school population size)
> **Goal:** Reduce gap to <5x for typical configs, <10x for large configs

---

## 1. Profiling Summary

### 1.1 Benchmark Configurations

| Config | Species | Grid | n_dt/yr | subdt | Schools/sp |
|--------|---------|------|---------|-------|------------|
| Bay of Biscay (BoB) | 8 focal + 6 rsc | 10x12 (120 cells) | 24 | 4 | 20-40 |
| Eastern English Channel (EEC) | 14 focal | 22x45 (990 cells) | 24 | 10 | 14-58 |

### 1.2 Current Performance

| Config | Java | Python | Ratio |
|--------|------|--------|-------|
| BoB 1-year | 0.80s | 33.3s | **42x** |
| BoB 5-year | 2.31s (0.46s/yr) | 381.7s (76s/yr) | **165x** |
| EEC 1-year | 2.53s | 29.3s | **12x** |

The BoB 5-year ratio (165x) is worse than 1-year (42x) because the school population
grows over time — more schools means O(n^2) predation cost escalates. EEC has more
species but fewer schools per species and a larger grid that dilutes per-cell density.

### 1.3 Where Time Is Spent (BoB 1-year, 33.3s)

```
mortality()                          32.8s  (98.5%)
  _mortality_in_cell()               32.7s  (98.2%)
    _apply_predation_for_school()    28.2s  (84.7%)
      _inst_abundance()              18.9s  (56.7%)  ← 7.67M calls
        numpy.ufunc.reduce           10.2s  (30.6%)  ← .sum() on 8-elem arrays
      prey scanning loop              9.6s  (28.8%)
    _apply_fishing_for_school()       1.1s  ( 3.3%)
    _apply_additional_for_school()    0.9s  ( 2.7%)
    _apply_starvation_for_school()    0.9s  ( 2.7%)
growth()                              0.1s  ( 0.3%)
movement()                            0.1s  ( 0.3%)
reproduction()                        0.05s ( 0.2%)
other                                 0.2s  ( 0.6%)
```

### 1.4 Call Frequency Analysis

| Function | Calls | Avg Time | Total |
|----------|-------|----------|-------|
| `_inst_abundance` | 7,670,000 | 2.5μs | 18.9s |
| `_apply_predation_for_school` | 182,630 | 154μs | 28.2s |
| `_mortality_in_cell` | 17,380 | 1.88ms | 32.7s |
| `_apply_fishing_for_school` | 182,630 | 6.3μs | 1.1s |
| `_apply_starvation_for_school` | 182,630 | 4.8μs | 0.9s |
| `_apply_additional_for_school` | 182,630 | 5.0μs | 0.9s |

---

## 2. Root Cause Analysis

### 2.1 Why Python Is Slow

The bottleneck is **not** algorithmic — both Java and Python use the same O(n^2)
per-cell predation algorithm. The difference is **per-operation overhead**:

| Operation | Java | Python (current) | Ratio |
|-----------|------|-------------------|-------|
| Read school abundance | ~1ns (field access) | ~2.5μs (`_inst_abundance` + `.sum()`) | 2500x |
| Check size ratio | ~1ns (comparison) | ~50ns (Python float ops) | 50x |
| Update n_dead | ~1ns (array write) | ~100ns (NumPy index + assign) | 100x |

The Python engine pays Python interpreter + NumPy dispatch overhead on every
per-school, per-prey operation. With 7.67M calls to `_inst_abundance` alone,
this overhead completely dominates.

### 2.2 Why Java Is Fast

Java's `MortalityProcess.computeMortality()` operates on `School` objects with
direct field access. No boxing, no dispatch overhead. Each school's
`getInstantaneousAbundance()` is a simple `abundance - nDead_sum` computed
via a cached field. The JIT compiler inlines everything.

### 2.3 Architectural Mismatch

The Python engine uses a **per-school Python function call** pattern that mimics
Java's OOP approach:

```
for cell in cells:                  # 17K calls
  for i in range(n_local):          # ~10 schools/cell
    for cause in shuffled_causes:   # 4 causes
      _apply_X_for_school(idx)      # 183K calls each
        _inst_abundance(idx)        # 7.67M calls total
```

This is the worst possible pattern for Python: millions of scalar Python function
calls with tiny NumPy operations inside. Each call pays ~2-5μs in interpreter
overhead for operations that should take nanoseconds.

---

## 3. Optimization Strategy — Three Tiers

### Tier 1: Cache Instantaneous Abundance (Target: 2x speedup)

**Problem:** `_inst_abundance()` called 7.67M times, each computing
`abundance[idx] - n_dead[idx].sum()` at 2.5μs/call = 18.9s total.

**Solution:** Maintain a pre-computed `inst_abd` array that is updated in-place
when deaths are applied. Replace all `_inst_abundance()` calls with direct
array indexing at 0.18μs/call.

**Microbenchmark validation:**
```
np.sum()       : 1.26 μs/call
cache lookup   : 0.18 μs/call  (7x faster)
```

**Expected savings:** 18.9s → ~1.4s = **17.5s saved** (52% of total)

**Implementation:**
1. In `mortality()`, before the sub-timestep loop, initialize from abundance
   (n_dead is guaranteed zero at this point — reset on line 515):
   ```python
   inst_abd = state.abundance.copy()  # n_dead is zeros here
   ```
2. Pass `inst_abd` to `_mortality_in_cell()` and all per-school helpers
3. Replace all `_inst_abundance(state, idx)` calls with `inst_abd[idx]`
4. When applying deaths, update cache in-place:
   ```python
   state.n_dead[idx, cause] += n_dead_val
   inst_abd[idx] -= n_dead_val
   ```
5. Remove the `_inst_abundance()` function entirely

**Risk:** LOW — pure refactor, no behavioral change. The array tracks the same
value as before, just avoids recomputing it 7.67M times.

**Files changed:** `mortality.py` only

### Tier 2: Numba-ify Per-Cell Mortality Loop (Target: 10-15x speedup)

**Problem:** After Tier 1 eliminates the `_inst_abundance` overhead, the
remaining bottleneck is ~14s of Python interpretation in `_mortality_in_cell()`
and `_apply_predation_for_school()` — scanning prey lists, checking size ratios,
computing accessibility, updating n_dead. These are tight scalar loops that
Java JIT-compiles but Python interprets.

**Solution:** Compile `_mortality_in_cell()` and all four per-school mortality
helpers into a single `@njit`-decorated function. This eliminates Python
interpreter overhead entirely for the inner loop.

**Implementation approach:**

Extract all per-cell mortality data into flat NumPy arrays that Numba can work
with (no Python objects, no dicts, no lists of tuples):

```python
@njit(cache=True)
def _mortality_in_cell_jit(
    # School data (flat arrays, cell-local indices)
    abundance,          # float64[n_local]
    weight,             # float64[n_local]
    length,             # float64[n_local]
    age_dt,             # int32[n_local]
    species_id,         # int32[n_local]
    is_background,      # bool[n_local]
    feeding_stage,      # int32[n_local]
    starvation_rate,    # float64[n_local]
    pred_success_rate,  # float64[n_local] (output)
    n_dead,             # float64[n_local, 8] (output)

    # Config (flat arrays indexed by species)
    size_ratio_min,     # float64[n_species, n_stages]
    size_ratio_max,     # float64[n_species, n_stages]
    ingestion_rate,     # float64[n_species]
    additional_rate,    # float64[n_species]
    fishing_rate,       # float64[n_species]
    # ... other config ...

    # Resource data for this cell
    rsc_biomass,        # float64[n_resources]
    rsc_size_min,       # float64[n_resources]
    rsc_size_max,       # float64[n_resources]
    rsc_access,         # float64[n_resources]

    # Control
    n_subdt, n_dt_per_year,
    rng_seed,           # int64 for Numba-compatible RNG
):
    # Entire interleaved mortality loop in compiled code
    ...
```

**Key Numba constraints and solutions:**

- **No Python objects** — all data must be NumPy arrays or scalars
- **No dict lookups** — pre-flatten accessibility matrix into index arrays
- **No list-of-tuples** — the current `all_prey: list[tuple[str, int, float]]`
  must be replaced with parallel arrays:
  ```python
  prey_is_rsc: bool[max_prey]       # True = resource, False = school
  prey_idx: int32[max_prey]          # school index or resource index
  prey_eligible: float64[max_prey]   # accessible biomass
  n_prey_actual: int                 # count of valid entries
  ```
  Pre-allocate to `n_local + n_resources` (max possible prey count).
- **No module-level globals** — `_tl_weighted_sum` and `_diet_matrix` are
  currently written as globals from `_apply_predation_for_school()`. These
  must be passed in as explicit output arrays:
  ```python
  tl_weighted_sum: float64[n_schools]           # TL tracking accumulator
  diet_matrix: float64[n_schools, n_species+n_rsc]  # diet tracking (or dummy)
  diet_enabled: bool                            # skip diet writes if False
  ```
  The global pattern in `mortality.py` must be refactored to pass these
  arrays through the call chain before Tier 2 can begin.
- **Resource cell_id** — `resources.grid.nx` is a Python object attribute.
  Pass `grid_nx: int` as an explicit parameter; compute `cell_id = cell_y * grid_nx + cell_x`
  inside the JIT function.
- **RNG strategy** — Pre-generate all random permutations and cause shuffles
  in Python before calling the JIT function, passing them as pre-computed
  index arrays. This preserves RNG parity with the pure-Python path (same
  `numpy.random.Generator` sequences) while avoiding Numba's incompatible
  built-in RNG. Cost: O(n_local) array allocation per cell, negligible vs
  the O(n_local^2) predation computation inside.
  ```python
  # Python side (before JIT call):
  seq_pred = rng.permutation(n_local).astype(np.int32)
  seq_starv = rng.permutation(n_local).astype(np.int32)
  seq_fish = rng.permutation(n_local).astype(np.int32)
  seq_nat = rng.permutation(n_local).astype(np.int32)
  cause_orders = np.array([rng.permutation(4) for _ in range(n_local)], dtype=np.int32)
  # Pass all of these into the @njit function
  ```

**Expected savings:** After Tier 1, ~14s remains in Python interpretation.
Numba compilation should reduce this to ~1-2s (10-15x speedup on the inner loop).

**Combined Tier 1+2 expected:** 33.3s → ~3-4s for BoB 1-year (~8-10x speedup)

**Risk:** MEDIUM — Numba has strict type requirements. The existing
`_predation_in_cell_numba()` in `predation.py` provides a template, but
this is a larger scope (4 mortality causes + resource predation + accessibility
matrix + fishing selectivity). Requires careful data marshaling.

**Files changed:** `mortality.py` (major rewrite of inner loop), `predation.py`
(diet tracking API — `_diet_matrix` and `_diet_tracking_enabled` globals live
in `predation.py` and are imported by `mortality.py`; the `enable_diet_tracking()`/
`disable_diet_tracking()` API must be adapted to support the Numba path)

### Tier 3: Structural Optimizations (Target: additional 2-3x)

These are independent optimizations that stack on top of Tiers 1-2.

#### 3a. Vectorized Cell Batching

Instead of iterating cells in Python, process all cells with the same
school count together using vectorized NumPy operations. Many cells
contain 0-2 schools and can be handled without the full interleaved
loop.

- **Empty cells:** Skip entirely (already done)
- **Single-school cells:** Only additional/starvation/fishing apply (no predation)
- **Two-school cells:** Simplified predation with 1 predator-prey pair

#### 3b. Compact State Before Mortality

After `state.compact()` removes dead schools, the state arrays are dense.
But during mortality, schools with `inst_abd <= 0` are still scanned.
Pre-filter to active schools before entering the inner loop.

#### 3c. Reduce Resource Scanning

Each predator scans all `n_resources` to check size overlap. Pre-compute
a per-species eligible resource mask once, then reuse for all schools
of that species.

**Risk:** LOW — each is an independent optimization with clear rollback.

**Expected savings:** 2-3x additional on top of Tiers 1-2.

---

## 4. Expected Performance After Each Tier

| Tier | BoB 1yr | BoB 5yr | EEC 1yr | BoB Ratio vs Java |
|------|---------|---------|---------|-------------------|
| Current | 33.3s | 381.7s | 29.3s | 42-165x |
| **Tier 1** (cache inst_abd) | ~16s | ~190s | ~15s | ~20-80x |
| **Tier 2** (Numba mortality) | ~3-4s | ~30-50s | ~4-5s | ~4-10x |
| **Tier 3** (structural) | ~2-3s | ~15-25s | ~2-3s | ~2-5x |
| Java reference | 0.80s | 2.31s | 2.53s | 1x |

The 5-year BoB will always be slower than Java due to school population
growth creating O(n^2) predation in dense cells. The per-year ratio
should stabilize at ~3-5x after all tiers.

---

## 5. Java vs Python Architectural Comparison

### 5.1 Data Representation

| Aspect | Java | Python |
|--------|------|--------|
| School data | `School` objects with fields | Structure-of-Arrays (flat NumPy) |
| Access pattern | `school.getAbundance()` (direct field) | `state.abundance[idx]` (array index) |
| Mortality tracking | `school.getNdead(cause)` (field) | `state.n_dead[idx, cause]` (2D array) |
| Instantaneous biomass | Cached field, updated on death | Recomputed per access (7.67M times) |

### 5.2 Execution Model

| Aspect | Java | Python |
|--------|------|--------|
| Compilation | JIT (HotSpot) | Interpreted (CPython) + Numba for predation |
| Inner loop | Compiled to native code | Python bytecode interpreter |
| Function calls | Inlined by JIT | ~2μs overhead per call |
| Array access | Direct memory (no bounds check after JIT) | NumPy dispatch + bounds check |
| RNG | `XSRandom` (fast custom PRNG) | `numpy.random.Generator` |

### 5.3 Parallelism

| Aspect | Java | Python |
|--------|------|--------|
| Multi-threading | `MortalityWorker` threads per cell batch | None (single-threaded) |
| CPU utilization | Multi-core via `CountDownLatch` | Single core |

Java's `computeMortalityParallel()` divides cells across available CPUs.
Python is single-threaded. However, parallelism is a lower priority than
eliminating interpreter overhead — even single-threaded compiled code
(Numba) would be 10-20x faster than current Python.

---

## 6. Implementation Priorities

### Phase 0: Test Infrastructure (1 hour)

1. Create `scripts/save_parity_baseline.py`
2. Create `scripts/benchmark_engine.py`
3. Create `tests/test_engine_parity.py` with self-parity and Java cross-parity tests
4. Save pre-optimization baselines for BoB 1yr, BoB 5yr, EEC 1yr
5. Run parity tests to confirm green baseline

### Phase 1: Tier 1 — Cache inst_abundance (1-2 hours)

1. Verify baseline from Phase 0 exists; re-save only if re-entering after other changes
2. Add `inst_abd` float64 array to mortality function
3. Initialize from `abundance.copy()` before sub-timestep loop (n_dead is zero at this point)
4. Pass to all per-school helpers
5. Update in-place when deaths applied
6. Remove `_inst_abundance()` function
7. Run `test_engine_parity.py::TestSelfParity` — must be bit-identical
8. Run full test suite + Java parity validation
9. Run benchmark, compare against baseline

### Phase 2: Tier 2 — Numba inner loop (4-6 hours)

1. Refactor globals: pass `tl_weighted_sum` and `diet_matrix` explicitly through call chain
2. Design flat-array data interface for `@njit` function
3. Extract accessibility lookups into pre-computed index arrays
4. Implement `_mortality_in_cell_jit()` with all 4 causes
5. Marshal data at cell boundaries (Python → Numba → Python)
6. Maintain pure-Python fallback for debugging (activated by env var)
7. Run `test_engine_parity.py::TestJavaParity` — must maintain BoB 8/8, EEC 14/14
8. Run full test suite + benchmark

### Phase 3: Tier 3 — Structural optimizations (2-3 hours)

1. Single-school cell fast path
2. Pre-compute resource eligibility per species
3. Pre-filter dead schools before inner loop
4. Run `test_engine_parity.py` after each optimization
5. Run benchmarks to measure each optimization independently

---

## 7. Testing Strategy

### 7.1 Correctness Guards

- All existing tests must pass (1705 as of v0.5.0) after each tier
- Bay of Biscay validation: 8/8 within 1 OoM
- EEC validation: 14/14 within 1 OoM
- Numerical parity:
  - **Tier 1:** output CSVs must be **identical** (same RNG seed) before
    and after — Tier 1 is a pure refactor with no RNG changes
  - **Tier 2:** output CSVs must be **statistically equivalent** — the Numba
    path uses pre-generated shuffles from the same `numpy.random.Generator`,
    so sequences are preserved. If implementation requires Numba's built-in
    RNG, accept different-but-valid sequences and validate within tolerance

### 7.2 Performance Benchmarks

Add `scripts/benchmark_engine.py` that:
- Runs BoB 1-year and EEC 1-year with timing
- Reports per-process breakdown via `time.perf_counter()` instrumentation
- Compares against stored baseline times
- Runs as part of CI (non-blocking, informational)

### 7.3 Parity Tests (New)

Add `tests/test_engine_parity.py` — automated tests that run both Python and
Java engines on the same configuration and compare outputs. These are the
definitive correctness gate for performance work.

#### 7.3.1 Bit-Identical Self-Parity (Tier 1 gate)

Tier 1 is a pure refactor. Verify output CSVs are **byte-identical** before
and after:

```python
class TestSelfParity:
    """Output must be identical before and after Tier 1 optimization."""

    def test_bob_1yr_identical(self):
        """BoB 1-year output CSVs must match pre-optimization baseline."""
        # 1. Load pre-computed baseline (saved before Tier 1 changes)
        # 2. Run Python engine with same seed
        # 3. Assert all output CSVs are byte-identical
        baseline = load_baseline("bob_1yr_baseline.npz")
        result = run_python_engine("examples", seed=42, years=1)
        np.testing.assert_array_equal(result.biomass, baseline.biomass)
        np.testing.assert_array_equal(result.abundance, baseline.abundance)
        np.testing.assert_array_equal(result.mortality, baseline.mortality)
```

Generate baselines before starting Tier 1 work:
```bash
.venv/bin/python scripts/save_parity_baseline.py  # saves to tests/fixtures/
```

#### 7.3.2 Java Cross-Engine Parity (All tiers gate)

Compare Python vs Java final-year biomass within tolerance. These tests
require Java to be installed and the OSMOSE JAR to be present — skip
gracefully if unavailable.

```python
@pytest.mark.skipif(not JAR_PATH.exists(), reason="OSMOSE JAR not found")
class TestJavaParity:
    """Python engine output must match Java within 1 order of magnitude."""

    def test_bob_8_of_8_species(self):
        """Bay of Biscay: all 8 species within 1 OoM of Java."""
        java_bio = run_java("examples", years=1)
        python_bio = run_python("examples", seed=42, years=1)
        for species in java_bio.columns:
            if java_bio[species].iloc[-1] == 0 and python_bio[species].iloc[-1] == 0:
                continue  # both extinct = parity
            log_ratio = abs(np.log10(python_bio[species].iloc[-1] / java_bio[species].iloc[-1]))
            assert log_ratio <= 1.0, f"{species}: {log_ratio:.2f} OoM divergence"

    def test_eec_14_of_14_species(self):
        """EEC: all 14 species within 1 OoM of Java."""
        java_bio = run_java("eec_full", years=1)
        python_bio = run_python("eec_full", seed=42, years=1)
        for species in java_bio.columns:
            if java_bio[species].iloc[-1] == 0 and python_bio[species].iloc[-1] == 0:
                continue
            log_ratio = abs(np.log10(python_bio[species].iloc[-1] / java_bio[species].iloc[-1]))
            assert log_ratio <= 1.0, f"{species}: {log_ratio:.2f} OoM divergence"

    def test_bob_5yr_no_drift(self):
        """5-year run: species don't drift beyond 1 OoM over longer simulations."""
        java_bio = run_java("examples", years=5)
        python_bio = run_python("examples", seed=42, years=5)
        for species in java_bio.columns:
            j = java_bio[species].iloc[-1]
            p = python_bio[species].iloc[-1]
            if j == 0 and p == 0:
                continue
            log_ratio = abs(np.log10(p / j))
            assert log_ratio <= 1.0, f"{species}: {log_ratio:.2f} OoM after 5 years"
```

#### 7.3.3 Per-Process Parity Checks

Targeted tests that verify individual process outputs match Java at the
formula level (existing in `test_engine_java_comparison.py`, extend):

```python
class TestProcessParity:
    """Individual process formulas match Java reference values."""

    def test_von_bertalanffy_growth_curves(self):
        """VB growth at 8 example species matches Java within 0.01%."""

    def test_mortality_decay_rates(self):
        """Mortality exponential decay matches Java within 0.01%."""

    def test_predation_proportional_distribution(self):
        """Unified predation distributes eating proportionally (Java parity)."""

    def test_starvation_lagged_rate(self):
        """Starvation uses previous timestep's pred_success_rate."""

    def test_reproduction_egg_count(self):
        """Egg count formula: sex_ratio * fecundity * SSB * season * 1e6."""
```

#### 7.3.4 Performance Parity Tests

Not correctness tests — these verify that optimizations actually improve
performance, and flag regressions:

```python
class TestPerformanceParity:
    """Track that optimizations don't regress."""

    def test_bob_1yr_under_threshold(self):
        """BoB 1-year must complete within the current tier's target time."""
        t = time_python_engine("examples", seed=42, years=1)
        # Updated per tier: Tier 1 = 20s, Tier 2 = 5s, Tier 3 = 3s
        assert t < CURRENT_TIER_THRESHOLD

    def test_mortality_is_dominant_cost(self):
        """Mortality should remain >90% of total time (validates profiling)."""
        breakdown = profile_python_engine("examples", seed=42, years=1)
        assert breakdown["mortality"] / breakdown["total"] > 0.90
```

#### 7.3.5 Helper Functions and Scripts

**Test helpers** (in `tests/test_engine_parity.py`):

```python
from types import SimpleNamespace

def run_python_engine(config_name: str, seed: int, years: int) -> SimpleNamespace:
    """Run Python engine, return SimpleNamespace with .biomass, .abundance, .mortality DataFrames."""
    # Loads config from data/{config_name}/, runs PythonEngine, reads output CSVs
    ...
    return SimpleNamespace(biomass=bio_df, abundance=abd_df, mortality=mort_df)

def run_java(config_name: str, years: int) -> pd.DataFrame:
    """Run Java engine, return biomass DataFrame. Wraps validate_engines.run_java()."""
    ...

def load_baseline(name: str) -> SimpleNamespace:
    """Load .npz baseline from tests/fixtures/parity_baselines/{name}."""
    data = np.load(path)
    return SimpleNamespace(biomass=data["biomass"], abundance=data["abundance"], ...)

# Performance threshold: updated per tier, module-level constant
CURRENT_TIER_THRESHOLD = 20.0  # seconds; Tier 1 = 20s, Tier 2 = 5s, Tier 3 = 3s
```

**`scripts/save_parity_baseline.py`** — Run Python engine on BoB and EEC,
save output arrays to `tests/fixtures/parity_baselines/` as `.npz` files.
Keys: `biomass` (2D: timesteps × species), `abundance` (2D), `mortality`
(3D: timesteps × species × causes). Run once before starting each tier.

**`scripts/benchmark_engine.py`** — Run both engines on BoB and EEC with
timing, print comparison table, optionally save results to JSON for CI
tracking.

### 7.4 Regression Detection

Before each optimization:
1. Run `scripts/save_parity_baseline.py` to save current output
2. Run `scripts/benchmark_engine.py` to save current timing
3. Implement optimization
4. Run full test suite including `test_engine_parity.py`
5. Run benchmark and compare timing
6. Require: no regression in correctness, measurable speedup

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Numba compilation time (~30s first run) | UX on first launch | Use `cache=True`, pre-compile on install |
| Numba type errors at runtime | Crashes | Maintain pure-Python fallback, activated by env var |
| Numerical differences from Numba float ops | False test failures | Use `rtol=1e-10` for Numba path comparisons |
| Increased code complexity | Maintainability | Keep pure-Python reference alongside Numba, clear documentation |
| RNG divergence in Numba | Different random sequences | Pre-generate shuffles in Python (preserving Generator sequences), pass as arrays to JIT. Only fall back to Numba RNG if marshaling cost is prohibitive |

---

## 9. Non-Goals

- **GPU acceleration** — OSMOSE's per-cell interleaved mortality is inherently
  sequential within a cell; GPU parallelism doesn't map well
- **Multiprocessing** — Cells can be processed in parallel (Java does this), but
  adds complexity for diminishing returns after Tier 2
- **Cython rewrite** — Numba achieves similar speedups without a compilation step
  or C code maintenance burden
- **Algorithmic changes** — The O(n^2) predation is fundamental to OSMOSE's
  stochastic interleaved design; changing it would break parity

---

## 10. Success Criteria

1. **BoB 1-year:** < 5s (currently 33.3s) — **6.7x speedup**
2. **BoB 5-year:** < 50s (currently 381.7s) — **7.6x speedup**
3. **EEC 1-year:** < 5s (currently 29.3s) — **5.9x speedup**
4. **All existing tests pass** (1705 as of v0.5.0) with zero failures
5. **Both validation configs** maintain full parity (BoB 8/8, EEC 14/14)
6. **No new dependencies** beyond NumPy and Numba (already installed)
7. **Tier 1 produces bit-identical output** to pre-optimization (same seed)
