# Performance Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 5x performance gap on BoB 5yr (11.5s → ~2s) by batching the per-cell Python→Numba dispatch into a single call and adding prange parallelism.

**Architecture:** Two phases, each independently shippable. Phase A replaces 250 per-cell Python→Numba dispatches per sub-timestep with one batch call, pre-generating all RNG in Python. Phase B adds `prange` to the batch loop for multi-core execution. Statistical equivalence accepted (RNG consumption order changes).

**Tech Stack:** NumPy, Numba (`njit`, `prange`), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `osmose/engine/processes/mortality.py` | Modify | Add `_pre_generate_cell_rng()`, `_mortality_all_cells_numba()`, `_mortality_all_cells_parallel()`, update `mortality()` orchestration |
| `tests/test_engine_parity.py` | Modify | Add `TestStatisticalParity` class |
| `scripts/save_parity_baseline.py` | Modify | Add `--statistical` mode for multi-seed baseline |
| `scripts/benchmark_engine.py` | Modify | Add `--years 5` support validation (already works, just document) |
| `tests/baselines/` | Add files | `statistical_baseline_bob_1yr_10seeds.npz` |

---

## Phase A: Batch Cell Loop

### Task 1: Pre-save benchmark baseline (5yr)

**Files:**
- Read: `scripts/benchmark_engine.py`
- Output: `tests/baselines/benchmark_5yr_pre_phase_a.json`

This captures the "before" timing so we can measure Phase A speedup.

- [ ] **Step 1: Run 5yr benchmark and save**

Run: `.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3 --output tests/baselines/benchmark_5yr_pre_phase_a.json`
Expected: JSON file with median ~11-12s

- [ ] **Step 2: Commit**

```bash
git add tests/baselines/benchmark_5yr_pre_phase_a.json
git commit -m "bench: save 5yr pre-Phase-A timing baseline"
```

---

### Task 2: Add `--statistical` mode to save_parity_baseline.py

**Files:**
- Modify: `scripts/save_parity_baseline.py`

Before changing RNG order, we need to capture multi-seed statistical baselines under the current code.

- [ ] **Step 1: Write the test**

Create a quick manual test: run `_run_engine` with 3 seeds and check shapes. We'll validate this through the script itself rather than a unit test — the real validation is `TestStatisticalParity` in Task 3.

- [ ] **Step 2: Add `--statistical` and `--seeds` arguments**

In `scripts/save_parity_baseline.py`, add the statistical baseline mode. The function should:

1. Accept `--statistical` flag and `--seeds N` (default 10)
2. Run engine N times with seeds `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`
3. Compute mean and std of final-step biomass per species
4. Save to `tests/baselines/statistical_baseline_bob_{years}yr_{n}seeds.npz`

Add this function after `save_baseline()`:

```python
STATISTICAL_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def save_statistical_baseline(n_years: int, n_seeds: int) -> Path:
    """Run engine with multiple seeds and save mean/std biomass as statistical baseline."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    reader = OsmoseConfigReader()
    raw = reader.read(EXAMPLES_CONFIG)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            PROJECT_DIR / "data" / "examples" / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    seeds = STATISTICAL_SEEDS[:n_seeds]
    final_biomasses = []

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{n_seeds})...")
        rng = np.random.default_rng(seed)
        outputs = simulate(cfg, grid, rng)
        final_biomasses.append(outputs[-1].biomass)

    all_bio = np.array(final_biomasses)  # (n_seeds, n_species)
    mean_bio = np.mean(all_bio, axis=0)
    std_bio = np.std(all_bio, axis=0)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"statistical_baseline_bob_{n_years}yr_{n_seeds}seeds.npz"
    out_path = BASELINE_DIR / filename

    np.savez_compressed(
        out_path,
        mean_biomass=mean_bio,
        std_biomass=std_bio,
        seeds=np.array(seeds),
        species_names=np.array(cfg.species_names),
        n_years=np.array(n_years),
    )

    print(f"Statistical baseline saved: {out_path}")
    print(f"  Seeds: {n_seeds}, Species: {len(mean_bio)}")
    return out_path
```

Update `main()` to handle the new arguments:

```python
parser.add_argument("--statistical", action="store_true", help="Save multi-seed statistical baseline")
parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for statistical baseline")
```

And in the body:

```python
if args.statistical:
    save_statistical_baseline(args.years, args.seeds)
else:
    save_baseline(args.years, args.seed)
```

- [ ] **Step 3: Generate statistical baseline**

Run: `.venv/bin/python scripts/save_parity_baseline.py --statistical --seeds 10 --years 1`
Expected: `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz` created

- [ ] **Step 4: Commit**

```bash
git add scripts/save_parity_baseline.py tests/baselines/statistical_baseline_bob_1yr_10seeds.npz
git commit -m "feat: add --statistical mode to save_parity_baseline.py"
```

---

### Task 3: Add TestStatisticalParity to test_engine_parity.py

**Files:**
- Modify: `tests/test_engine_parity.py`

This test validates that after RNG order changes (Phase A), mean biomass across 10 seeds stays within 5%.

- [ ] **Step 1: Write the test class**

Add after `TestSanityChecks` in `tests/test_engine_parity.py`:

```python
# ---------------------------------------------------------------------------
# Statistical parity tests (cross-version — tolerate RNG order changes)
# ---------------------------------------------------------------------------

STATISTICAL_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def _statistical_baseline_path(n_years: int = DEFAULT_YEARS, n_seeds: int = 10) -> Path:
    return BASELINE_DIR / f"statistical_baseline_bob_{n_years}yr_{n_seeds}seeds.npz"


class TestStatisticalParity:
    """Mean biomass across 10 seeds must stay within 5% of baseline.

    This test tolerates RNG consumption order changes (e.g., Phase A batch
    cell loop) that produce different per-seed results but statistically
    equivalent distributions.
    """

    @pytest.fixture(scope="class")
    def statistical_data(self):
        path = _statistical_baseline_path()
        if not path.exists():
            pytest.skip(
                f"No statistical baseline: {path}. "
                "Run: scripts/save_parity_baseline.py --statistical"
            )
        if not EXAMPLES_CONFIG.exists():
            pytest.skip(f"No example config: {EXAMPLES_CONFIG}")

        data = np.load(path)
        baseline_means = data["mean_biomass"]

        # Run engine with same seeds
        final_biomasses = []
        for seed in STATISTICAL_SEEDS:
            bio, _, _ = _run_engine(DEFAULT_YEARS, seed)
            final_biomasses.append(bio[-1])

        current_means = np.mean(final_biomasses, axis=0)
        return {"baseline_means": baseline_means, "current_means": current_means}

    def test_multi_seed_biomass_within_tolerance(self, statistical_data):
        """10 seeds, mean final biomass per species within 5% of baseline."""
        np.testing.assert_allclose(
            statistical_data["current_means"],
            statistical_data["baseline_means"],
            rtol=0.05,
            err_msg="Statistical parity failed — mean biomass drifted >5% from baseline",
        )
```

- [ ] **Step 2: Run the test to verify it passes (pre-Phase-A, exact match expected)**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestStatisticalParity -v`
Expected: PASS (current code matches its own baseline exactly)

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_parity.py
git commit -m "test: add TestStatisticalParity for cross-version RNG tolerance"
```

---

### Task 4: Implement `_pre_generate_cell_rng()`

**Files:**
- Modify: `osmose/engine/processes/mortality.py`

This function pre-generates ALL random data for all cells in one Python pass before the Numba call.

- [ ] **Step 1: Write a unit test for the RNG pre-generation**

Create the test in a new file `tests/test_mortality_rng.py`:

```python
"""Tests for mortality RNG pre-generation."""

from __future__ import annotations

import numpy as np
import pytest


def test_pre_generate_cell_rng_shapes():
    """Verify output shapes match cell structure."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    # 3 cells with 2, 0, 3 schools respectively
    boundaries = np.array([0, 2, 2, 5], dtype=np.int64)
    n_cells = 3

    seq_bufs, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells)

    assert len(seq_bufs) == 4
    for buf in seq_bufs:
        assert buf.shape == (5,)
        assert buf.dtype == np.int32
    assert cause_orders_buf.shape == (5, 4)
    assert cause_orders_buf.dtype == np.int32


def test_pre_generate_cell_rng_local_indices():
    """Verify permutation values are in [0, n_local) for each cell."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 4, 4, 7], dtype=np.int64)
    n_cells = 3

    seq_bufs, _ = _pre_generate_cell_rng(rng, boundaries, n_cells)

    # Cell 0: 4 schools → values in [0, 4)
    for buf in seq_bufs:
        cell0 = buf[0:4]
        assert np.all(cell0 >= 0) and np.all(cell0 < 4)
        assert sorted(cell0) == [0, 1, 2, 3]  # permutation

    # Cell 1: 0 schools → nothing to check

    # Cell 2: 3 schools → values in [0, 3)
    for buf in seq_bufs:
        cell2 = buf[4:7]
        assert np.all(cell2 >= 0) and np.all(cell2 < 3)
        assert sorted(cell2) == [0, 1, 2]


def test_pre_generate_cell_rng_cause_orders_valid():
    """Each cause_orders row must be a permutation of [0,1,2,3]."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 3, 6], dtype=np.int64)
    n_cells = 2

    _, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells)

    for i in range(6):
        assert sorted(cause_orders_buf[i]) == [0, 1, 2, 3]


def test_pre_generate_cell_rng_deterministic():
    """Same seed must produce identical output."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    boundaries = np.array([0, 3, 5], dtype=np.int64)

    rng1 = np.random.default_rng(99)
    s1, c1 = _pre_generate_cell_rng(rng1, boundaries, 2)

    rng2 = np.random.default_rng(99)
    s2, c2 = _pre_generate_cell_rng(rng2, boundaries, 2)

    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(c1, c2)
```

- [ ] **Step 2: Run tests to verify they fail (function doesn't exist yet)**

Run: `.venv/bin/python -m pytest tests/test_mortality_rng.py -v`
Expected: ImportError — `_pre_generate_cell_rng` not found

- [ ] **Step 3: Implement `_pre_generate_cell_rng`**

Add this function in `osmose/engine/processes/mortality.py`, after the dummy array definitions (around line 422, after `_DUMMY_DIET`), before `_precompute_resource_arrays`:

```python
def _pre_generate_cell_rng(
    rng: np.random.Generator,
    boundaries: NDArray[np.int64],
    n_cells: int,
) -> tuple[list[NDArray[np.int32]], NDArray[np.int32]]:
    """Pre-generate all random data for all cells in one Python pass.

    Returns:
        seq_bufs: list of 4 int32 arrays, each of length boundaries[n_cells].
            seq_bufs[k][start:end] is rng.permutation(n_local) for cell k.
        cause_orders_buf: int32 array of shape (total, 4).
            cause_orders_buf[start+i] is a shuffled [PRED, STARV, ADDITIONAL, FISHING].
    """
    total = int(boundaries[n_cells])
    seq_bufs = [np.empty(total, dtype=np.int32) for _ in range(4)]
    cause_orders_buf = np.empty((total, 4), dtype=np.int32)
    causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]

    for cell in range(n_cells):
        start = int(boundaries[cell])
        end = int(boundaries[cell + 1])
        n_local = end - start
        if n_local == 0:
            continue
        for k in range(4):
            seq_bufs[k][start:end] = rng.permutation(n_local).astype(np.int32)
        for i in range(n_local):
            rng.shuffle(causes)
            cause_orders_buf[start + i, 0] = causes[0]
            cause_orders_buf[start + i, 1] = causes[1]
            cause_orders_buf[start + i, 2] = causes[2]
            cause_orders_buf[start + i, 3] = causes[3]

    return seq_bufs, cause_orders_buf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mortality_rng.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_mortality_rng.py
git commit -m "feat: add _pre_generate_cell_rng for batch RNG pre-generation"
```

---

### Task 5: Implement `_mortality_all_cells_numba()`

**Files:**
- Modify: `osmose/engine/processes/mortality.py`

This is the core batch function — one Numba call processes all cells, replacing 250 per-cell dispatches.

- [ ] **Step 1: Write the batch Numba function**

Add this inside the `if _HAS_NUMBA:` block (after `_mortality_in_cell_numba`, around line 919), keeping it within the same conditional:

```python
    @njit(cache=True)
    def _mortality_all_cells_numba(
        sorted_indices, boundaries, n_cells,
        seq_pred_buf, seq_starv_buf, seq_fish_buf, seq_nat_buf,
        cause_orders_buf,
        inst_abd, n_dead,
        eff_starv, eff_additional, eff_fishing, fishing_discard,
        species_id, length, weight, age_dt,
        first_feeding_age_dt, feeding_stage, pred_success_rate,
        preyed_biomass, trophic_level,
        size_ratio_min, size_ratio_max, ingestion_rate,
        n_dt_per_year, n_subdt,
        access_matrix, has_access, use_stage_access,
        prey_access_idx, pred_access_idx,
        rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
        rsc_access_rows, n_resources, n_species,
        tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
    ):
        """Numba-compiled batch mortality for ALL cells in one call."""
        for cell in range(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue

            cell_indices = sorted_indices[start:end]
            n_local = end - start
            cell_id = cell  # flat row-major index

            seq_pred = seq_pred_buf[start:end]
            seq_starv = seq_starv_buf[start:end]
            seq_fish = seq_fish_buf[start:end]
            seq_nat = seq_nat_buf[start:end]
            cause_orders = cause_orders_buf[start:end]

            for i in range(n_local):
                for c in range(4):
                    cause = cause_orders[i, c]
                    if cause == 0:  # PREDATION
                        p_idx = cell_indices[seq_pred[i]]
                        _apply_predation_numba(
                            p_idx, cell_indices,
                            inst_abd, n_dead, species_id, length, weight,
                            age_dt, first_feeding_age_dt, feeding_stage,
                            pred_success_rate, preyed_biomass, trophic_level,
                            size_ratio_min, size_ratio_max, ingestion_rate,
                            n_dt_per_year, n_subdt,
                            access_matrix, has_access, use_stage_access,
                            prey_access_idx, pred_access_idx,
                            rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
                            rsc_access_rows, n_resources, n_species, cell_id,
                            tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
                        )
                    elif cause == 1:  # STARVATION
                        idx = cell_indices[seq_starv[i]]
                        D = eff_starv[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 1] += dead
                                inst_abd[idx] -= dead
                    elif cause == 2:  # ADDITIONAL
                        idx = cell_indices[seq_nat[i]]
                        D = eff_additional[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 2] += dead
                                inst_abd[idx] -= dead
                    elif cause == 3:  # FISHING
                        idx = cell_indices[seq_fish[i]]
                        F = eff_fishing[idx]
                        if F > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-F))
                                discard_r = fishing_discard[idx]
                                if discard_r > 0:
                                    n_dead[idx, 3] += dead * (1.0 - discard_r)
                                    n_dead[idx, 6] += dead * discard_r
                                else:
                                    n_dead[idx, 3] += dead
                                inst_abd[idx] -= dead
```

Note: The inner loop body is identical to `_mortality_in_cell_numba`. The difference is: the `for cell in range(n_cells)` loop is now INSIDE the Numba function, and RNG data comes from pre-generated buffers.

- [ ] **Step 2: Run lint check**

Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "feat: add _mortality_all_cells_numba batch function"
```

---

### Task 6: Wire batch function into `mortality()` orchestration

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (the `mortality()` function, lines 1217-1263)

Replace the per-cell dispatch loop with pre-generation + batch call.

- [ ] **Step 1: Update the sub-timestep loop**

Replace the per-cell loop (lines 1228-1263) inside `mortality()`. The current code:

```python
        # Per-cell mortality
        for cell in range(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue
            cell_indices = sorted_indices[start:end]
            cy = cell // grid.nx
            cx = cell % grid.nx

            _mortality_in_cell(
                cell_indices,
                work_state,
                config,
                resources,
                cy,
                cx,
                rng,
                n_subdt,
                access_matrix,
                has_access,
                use_stage_access,
                prey_access_idx,
                pred_access_idx,
                inst_abd=inst_abd,
                step=step,
                rsc_size_min=rsc_sm,
                rsc_size_max=rsc_sx,
                rsc_tl=rsc_tl,
                rsc_access_rows=rsc_ar,
                n_rsc=n_rsc,
                eff_starv=eff_s,
                eff_additional=eff_a,
                eff_fishing=eff_f,
                fishing_discard=f_disc,
                grid_nx=grid.nx,
            )
```

Replace with:

```python
        # Per-cell mortality
        if _HAS_NUMBA and len(valid_indices) > 0:
            # Pre-generate RNG for all cells (one Python pass)
            seq_bufs, cause_orders_buf = _pre_generate_cell_rng(
                rng, boundaries, n_cells
            )

            # Extract tracking arrays from module globals BEFORE Numba call
            rsc_bio = resources.biomass if resources is not None else _DUMMY_RSC_2D
            tl_ws = _tl_weighted_sum if _tl_weighted_sum is not None else _DUMMY_RSC_1D
            tl_track = _tl_weighted_sum is not None
            d_mat = (
                _diet_matrix
                if _diet_tracking_enabled and _diet_matrix is not None
                else _DUMMY_DIET
            )
            d_en = _diet_tracking_enabled and _diet_matrix is not None

            # Single Numba call for all cells
            _mortality_all_cells_numba(
                sorted_indices, boundaries, n_cells,
                seq_bufs[0], seq_bufs[1], seq_bufs[2], seq_bufs[3],
                cause_orders_buf,
                inst_abd, work_state.n_dead,
                eff_s, eff_a, eff_f, f_disc,
                work_state.species_id, work_state.length, work_state.weight,
                work_state.age_dt, work_state.first_feeding_age_dt,
                work_state.feeding_stage, work_state.pred_success_rate,
                work_state.preyed_biomass, work_state.trophic_level,
                config.size_ratio_min, config.size_ratio_max, config.ingestion_rate,
                config.n_dt_per_year, n_subdt,
                access_matrix, has_access, use_stage_access,
                prey_access_idx, pred_access_idx,
                rsc_bio, rsc_sm, rsc_sx, rsc_tl, rsc_ar,
                n_rsc, config.n_species,
                tl_ws, tl_track, d_mat, d_en,
            )
        else:
            # Python fallback: per-cell dispatch (unchanged)
            for cell in range(n_cells):
                start = boundaries[cell]
                end = boundaries[cell + 1]
                if end <= start:
                    continue
                cell_indices = sorted_indices[start:end]
                cy = cell // grid.nx
                cx = cell % grid.nx
                _mortality_in_cell(
                    cell_indices, work_state, config, resources,
                    cy, cx, rng, n_subdt,
                    access_matrix, has_access, use_stage_access,
                    prey_access_idx, pred_access_idx,
                    inst_abd=inst_abd, step=step,
                    rsc_size_min=rsc_sm, rsc_size_max=rsc_sx,
                    rsc_tl=rsc_tl, rsc_access_rows=rsc_ar, n_rsc=n_rsc,
                    eff_starv=eff_s, eff_additional=eff_a,
                    eff_fishing=eff_f, fishing_discard=f_disc,
                    grid_nx=grid.nx,
                )
```

**Important:** The `rsc_bio` (resources.biomass) extraction was previously done inside `_mortality_in_cell` per cell. Now it's extracted once before the batch call and passed in. The `cell_id` indexing inside the batch function uses `cell_id = cell` (the loop variable), which is correct because `boundaries` uses the same row-major grid numbering as `rsc_biomass[r, cell_id]`.

- [ ] **Step 2: Run sanity tests (determinism, invariants)**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestDeterminism tests/test_engine_parity.py::TestSanityChecks -v`
Expected: All PASS (determinism preserved, no negative values)

Note: `TestBaselineParity` will FAIL because RNG consumption order changed — this is expected and correct. We'll regenerate the baseline in Task 7.

- [ ] **Step 3: Run statistical parity test**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestStatisticalParity -v`
Expected: PASS (mean biomass within 5%)

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest --ignore=tests/test_engine_parity.py -x -q`
Expected: All pass (no regressions outside parity tests)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "feat: wire batch cell loop into mortality() orchestration (Phase A)"
```

---

### Task 7: Regenerate baselines and validate Phase A

**Files:**
- Regenerate: `tests/baselines/parity_baseline_bob_1yr_seed42.npz`
- Regenerate: `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz`
- Output: `tests/baselines/benchmark_5yr_post_phase_a.json`

After Phase A, the exact-match baseline must be regenerated (RNG order changed). The statistical baseline should also be regenerated so it reflects the new code.

- [ ] **Step 1: Regenerate exact-match baseline**

Run: `.venv/bin/python scripts/save_parity_baseline.py --years 1 --seed 42`
Expected: `parity_baseline_bob_1yr_seed42.npz` updated

- [ ] **Step 2: Regenerate statistical baseline**

Run: `.venv/bin/python scripts/save_parity_baseline.py --statistical --seeds 10 --years 1`
Expected: `statistical_baseline_bob_1yr_10seeds.npz` updated

- [ ] **Step 3: Run ALL parity tests (should all pass now)**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
Expected: ALL PASS including `TestBaselineParity`

- [ ] **Step 4: Run benchmark**

Run: `.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3 --output tests/baselines/benchmark_5yr_post_phase_a.json`
Expected: Median ~3-4s (down from ~11.5s)

- [ ] **Step 5: Compare benchmarks**

Run: `.venv/bin/python scripts/benchmark_engine.py --compare tests/baselines/benchmark_5yr_pre_phase_a.json tests/baselines/benchmark_5yr_post_phase_a.json`
Expected: 3-4x speedup reported

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add tests/baselines/
git commit -m "bench: regenerate baselines after Phase A batch cell loop"
```

---

### Task 8: Clean up — remove Numba cache files

Phase A changes the Numba function signatures, so stale `.nbi`/`.nbc` cache files may cause issues.

- [ ] **Step 1: Delete Numba cache**

Run: `find /home/razinka/osmose/osmose-python -type d -name "__pycache__" -path "*/engine/*" -exec rm -rf {} + 2>/dev/null; echo "Caches cleared"`

Note: This is safe — Numba regenerates caches on next run. Only needed if Numba raises signature mismatch errors.

- [ ] **Step 2: Verify clean run**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestDeterminism -v`
Expected: PASS (may take longer on first run due to Numba recompilation)

---

## Phase B: Parallel Cells (`prange`)

### Task 9: Add `prange` import

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (import section, line 36)

- [ ] **Step 1: Update Numba import**

Change the Numba import block from:

```python
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

To:

```python
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py`
Expected: No errors (prange used in Task 10)

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "refactor: import prange from numba for Phase B"
```

---

### Task 10: Implement `_mortality_all_cells_parallel()`

**Files:**
- Modify: `osmose/engine/processes/mortality.py`

This is the parallel version — identical to `_mortality_all_cells_numba` except `range(n_cells)` → `prange(n_cells)` and `parallel=True`.

- [ ] **Step 1: Add the parallel function**

Add immediately after `_mortality_all_cells_numba` (still inside `if _HAS_NUMBA:` block):

```python
    @njit(cache=True, parallel=True)
    def _mortality_all_cells_parallel(
        sorted_indices, boundaries, n_cells,
        seq_pred_buf, seq_starv_buf, seq_fish_buf, seq_nat_buf,
        cause_orders_buf,
        inst_abd, n_dead,
        eff_starv, eff_additional, eff_fishing, fishing_discard,
        species_id, length, weight, age_dt,
        first_feeding_age_dt, feeding_stage, pred_success_rate,
        preyed_biomass, trophic_level,
        size_ratio_min, size_ratio_max, ingestion_rate,
        n_dt_per_year, n_subdt,
        access_matrix, has_access, use_stage_access,
        prey_access_idx, pred_access_idx,
        rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
        rsc_access_rows, n_resources, n_species,
        tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
    ):
        """Parallel batch mortality — prange over cells for multi-core execution.

        Safe because all school-level mutations are cell-local: each cell's
        index range [start, end) is disjoint, so no two threads write to
        the same array element.
        """
        for cell in prange(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue

            cell_indices = sorted_indices[start:end]
            n_local = end - start
            cell_id = cell

            seq_pred = seq_pred_buf[start:end]
            seq_starv = seq_starv_buf[start:end]
            seq_fish = seq_fish_buf[start:end]
            seq_nat = seq_nat_buf[start:end]
            cause_orders = cause_orders_buf[start:end]

            for i in range(n_local):
                for c in range(4):
                    cause = cause_orders[i, c]
                    if cause == 0:  # PREDATION
                        p_idx = cell_indices[seq_pred[i]]
                        _apply_predation_numba(
                            p_idx, cell_indices,
                            inst_abd, n_dead, species_id, length, weight,
                            age_dt, first_feeding_age_dt, feeding_stage,
                            pred_success_rate, preyed_biomass, trophic_level,
                            size_ratio_min, size_ratio_max, ingestion_rate,
                            n_dt_per_year, n_subdt,
                            access_matrix, has_access, use_stage_access,
                            prey_access_idx, pred_access_idx,
                            rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
                            rsc_access_rows, n_resources, n_species, cell_id,
                            tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
                        )
                    elif cause == 1:  # STARVATION
                        idx = cell_indices[seq_starv[i]]
                        D = eff_starv[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 1] += dead
                                inst_abd[idx] -= dead
                    elif cause == 2:  # ADDITIONAL
                        idx = cell_indices[seq_nat[i]]
                        D = eff_additional[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 2] += dead
                                inst_abd[idx] -= dead
                    elif cause == 3:  # FISHING
                        idx = cell_indices[seq_fish[i]]
                        F = eff_fishing[idx]
                        if F > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-F))
                                discard_r = fishing_discard[idx]
                                if discard_r > 0:
                                    n_dead[idx, 3] += dead * (1.0 - discard_r)
                                    n_dead[idx, 6] += dead * discard_r
                                else:
                                    n_dead[idx, 3] += dead
                                inst_abd[idx] -= dead
```

Note: The body is byte-for-byte identical to `_mortality_all_cells_numba`. The only differences are: `@njit(cache=True, parallel=True)` and `prange(n_cells)`.

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "feat: add _mortality_all_cells_parallel with prange"
```

---

### Task 11: Add `parallel` argument to `mortality()` and wire dispatch

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (the `mortality()` function signature + dispatch)

- [ ] **Step 1: Update `mortality()` signature**

Change the function signature (line 1101) from:

```python
def mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
    step: int = 0,
    species_rngs: list[np.random.Generator] | None = None,
) -> SchoolState:
```

To:

```python
def mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
    step: int = 0,
    species_rngs: list[np.random.Generator] | None = None,
    parallel: bool = True,
) -> SchoolState:
```

- [ ] **Step 2: Update the dispatch inside the sub-timestep loop**

In the batch dispatch section (from Task 6), change the Numba call from:

```python
            _mortality_all_cells_numba(
```

To:

```python
            _batch_fn = (
                _mortality_all_cells_parallel
                if parallel
                else _mortality_all_cells_numba
            )
            _batch_fn(
```

- [ ] **Step 3: Run determinism test**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestDeterminism -v`
Expected: PASS (prange with disjoint writes is deterministic)

- [ ] **Step 4: Run full parity suite**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
Expected: ALL PASS (parallel produces identical results to sequential)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "feat: add parallel flag to mortality() for prange dispatch (Phase B)"
```

---

### Task 12: Verify caller compatibility

**Files:**
- Read: `osmose/engine/simulate.py` (to check how `mortality()` is called)

The new `parallel` parameter defaults to `True`, so existing callers don't need changes. But we should verify no callers pass positional arguments that would conflict.

- [ ] **Step 1: Check all callers of `mortality()`**

Search for: `mortality(` in `osmose/engine/` and `tests/`
Expected: All calls use keyword arguments or pass fewer than 8 positional args (the new `parallel` parameter is 8th, after `species_rngs`)

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 3: Commit (if any changes needed)**

Only commit if callers needed updates. Otherwise, skip.

---

### Task 13: Final benchmark and baseline regeneration

**Files:**
- Regenerate: `tests/baselines/parity_baseline_bob_1yr_seed42.npz` (only if Phase B changed results — it should NOT)
- Output: `tests/baselines/benchmark_5yr_post_phase_b.json`

- [ ] **Step 1: Verify Phase B doesn't change results**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestBaselineParity -v`
Expected: PASS (Phase B should produce bit-identical results to Phase A — prange with disjoint writes preserves result)

If FAIL: investigate — `prange` should not change results when writes are disjoint. If FP non-determinism due to thread scheduling, we may need to accept statistical equivalence.

- [ ] **Step 2: Run 5yr benchmark**

Run: `.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3 --output tests/baselines/benchmark_5yr_post_phase_b.json`
Expected: Median ~1.5-2.5s

- [ ] **Step 3: Compare Phase A → Phase B**

Run: `.venv/bin/python scripts/benchmark_engine.py --compare tests/baselines/benchmark_5yr_post_phase_a.json tests/baselines/benchmark_5yr_post_phase_b.json`
Expected: 1.5-2.5x additional speedup from parallelism

- [ ] **Step 4: Compare original → Phase B**

Run: `.venv/bin/python scripts/benchmark_engine.py --compare tests/baselines/benchmark_5yr_pre_phase_a.json tests/baselines/benchmark_5yr_post_phase_b.json`
Expected: 5-8x total speedup (11.5s → 1.5-2.5s)

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/baselines/
git commit -m "bench: final Phase B benchmarks — BoB 5yr at Java parity"
```

---

## Contingency: Phase B Determinism

If `prange` introduces FP non-determinism (unlikely given disjoint writes, but possible in edge cases):

1. Regenerate exact-match baseline under parallel mode
2. If determinism test fails across runs, add `parallel=False` to `TestDeterminism` fixture
3. The statistical parity test (5% tolerance) should still pass regardless

## Rollback

Each phase is independently shippable. If Phase B causes issues, revert to Phase A (sequential batch) which still provides 3-4x speedup. The `parallel=False` default would achieve this without code changes.
