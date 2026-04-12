# M-7: Movement Map Strict Coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in `strict` mode to `MovementMapSet` that raises `ValueError` on uncovered `(age_dt, step)` slots instead of just warning.

**Architecture:** Phase A builds a test fixture producing known uncovered slots. Phase B adds the `strict` parameter, config key, and tests. Default behavior unchanged.

**Tech Stack:** Python 3.12, NumPy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-12-movement-strict-coverage-design.md`

**Baseline:** 2148 tests passed, 15 skipped, 0 failed. Ruff clean. Parity 12/12 bit-exact.

---

## File structure

**Production files:**
- Modify: `osmose/engine/movement_maps.py` — add `strict` parameter
- Modify: `osmose/engine/config.py` — add config key parsing

**Test files:**
- Modify: `tests/test_engine_map_movement.py` — add fixture and strict tests

---

## Phase A — Fixture Spike

### Task 1: Build uncovered-slot fixture

**Files:**
- Modify: `tests/test_engine_map_movement.py`

**Context:** `MovementMapSet.__init__` fills `index_maps[age_dt, global_step]` per map config. To get uncovered slots, define a species with lifespan covering more age_dt values than any map's age range.

- [ ] **Step 1: Read MovementMapSet.__init__ to understand the constructor signature**

Run: `grep -n "class MovementMapSet" osmose/engine/movement_maps.py`
Then read the `__init__` signature. It should be approximately:
```python
def __init__(self, config, species_name, n_dt_per_year, n_years, lifespan_dt, ny, nx, config_dir)
```
Record exact parameter names.

- [ ] **Step 2: Write the fixture test**

In `tests/test_engine_map_movement.py`, add a new test class:

```python
class TestUncoveredSlotFixture:
    """Fixture spike: build a MovementMapSet with known uncovered (age_dt, step) slots.

    Species: lifespan=4yr, n_dt=12 → 48 age_dt values.
    Sim: 1 year → 12 global steps. Total slots: 48 × 12 = 576.
    Map0: covers ages 0-1 only → 24 age_dt × 12 steps = 288 covered.
    Uncovered: ages 2-3 → 24 age_dt × 12 steps = 288 uncovered.
    """

    def _make_config_and_map(self, tmp_path):
        """Create a 3×3 CSV map and config dict for a single map covering ages 0-1."""
        import numpy as np

        # Write a valid 3×3 movement map CSV
        map_path = tmp_path / "map0.csv"
        grid = np.full((3, 3), 0.5, dtype=np.float64)
        np.savetxt(map_path, grid, delimiter=",")

        config = {
            "movement.species.map0": "TestFish",
            "movement.file.map0": str(map_path),
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            # All steps covered for map0
        }
        return config

    def test_fixture_produces_known_uncovered_slots(self, tmp_path):
        from osmose.engine.movement_maps import MovementMapSet

        config = self._make_config_and_map(tmp_path)
        ms = MovementMapSet(
            config=config,
            species_name="TestFish",
            n_dt_per_year=12,
            n_years=1,
            lifespan_dt=48,  # 4 years × 12
            ny=3,
            nx=3,
            config_dir=str(tmp_path),
        )
        uncovered = int((ms.index_maps == -1).sum())
        total = 48 * 12
        covered = total - uncovered
        assert covered == 24 * 12, f"Expected 288 covered slots, got {covered}"
        assert uncovered == 24 * 12, f"Expected 288 uncovered slots, got {uncovered}"
```

**IMPORTANT:** Read the actual `MovementMapSet.__init__` signature before writing. The parameter names may differ (e.g., `config_dir` might be a separate parameter or derived from `config["_osmose.config.dir"]`). Adapt accordingly. Check existing tests in the file for the fixture pattern — `_make_full_config_with_maps()` at the bottom of the file is a good template.

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py::TestUncoveredSlotFixture -v`
Expected: PASS.

- [ ] **Step 4: Full suite + ruff**

Run: `.venv/bin/python -m pytest tests/ -q` → 2148 + 1 = 2149 passed
Run: `.venv/bin/ruff check tests/test_engine_map_movement.py`

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine_map_movement.py
git commit -m "test: add uncovered-slot fixture for MovementMapSet (M-7 spike)"
```

---

## Phase B — Implementation

### Task 2: Add `strict` parameter to MovementMapSet

**Files:**
- Modify: `osmose/engine/movement_maps.py`

- [ ] **Step 1: Add `strict` parameter to `__init__`**

Find `def __init__` in `MovementMapSet`. Add `strict: bool = False` as the last parameter.

- [ ] **Step 2: Replace the warning block**

Find the uncovered validation block (grep for `"slots have no movement map assigned"`). Replace:

```python
        # --- Validate: warn about uncovered (age, step) slots ---
        uncovered = int((self.index_maps == -1).sum())
        if uncovered > 0:
            logger.warning(
                "Species %r: %d of %d (age_dt, step) slots have no movement map assigned",
                species_name,
                uncovered,
                lifespan_dt * n_total_steps,
            )
```

with:

```python
        # --- Validate: warn or raise about uncovered (age, step) slots ---
        uncovered = int((self.index_maps == -1).sum())
        if uncovered > 0:
            msg = (
                f"Species {species_name!r}: {uncovered} of {lifespan_dt * n_total_steps} "
                f"(age_dt, step) slots have no movement map assigned"
            )
            if strict:
                raise ValueError(msg)
            logger.warning("%s", msg)
```

- [ ] **Step 3: Run existing tests**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -q`
Expected: all pass (default `strict=False` preserves existing behavior).

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/movement_maps.py
git commit -m "feat(engine): add strict parameter to MovementMapSet (M-7)"
```

---

### Task 3: Add strict coverage tests

**Files:**
- Modify: `tests/test_engine_map_movement.py`

- [ ] **Step 1: Add strict=True test (raises)**

```python
class TestStrictCoverage:
    """Tests for the strict coverage mode of MovementMapSet."""

    def _make_partial_config(self, tmp_path):
        """Config with map0 covering ages 0-1 of a 4-year species."""
        import numpy as np

        map_path = tmp_path / "map0.csv"
        np.savetxt(map_path, np.full((3, 3), 0.5), delimiter=",")
        return {
            "movement.species.map0": "TestFish",
            "movement.file.map0": str(map_path),
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
        }

    def _make_full_config(self, tmp_path):
        """Config with map0 covering ALL ages of a 4-year species."""
        import numpy as np

        map_path = tmp_path / "map0.csv"
        np.savetxt(map_path, np.full((3, 3), 0.5), delimiter=",")
        return {
            "movement.species.map0": "TestFish",
            "movement.file.map0": str(map_path),
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3.99",
        }

    def test_strict_raises_on_uncovered_slots(self, tmp_path):
        from osmose.engine.movement_maps import MovementMapSet
        import pytest

        config = self._make_partial_config(tmp_path)
        with pytest.raises(ValueError, match="slots have no movement map assigned"):
            MovementMapSet(
                config=config,
                species_name="TestFish",
                n_dt_per_year=12,
                n_years=1,
                lifespan_dt=48,
                ny=3, nx=3,
                config_dir=str(tmp_path),
                strict=True,
            )

    def test_strict_false_warns_on_uncovered_slots(self, tmp_path, caplog):
        from osmose.engine.movement_maps import MovementMapSet
        import logging

        config = self._make_partial_config(tmp_path)
        with caplog.at_level(logging.WARNING):
            MovementMapSet(
                config=config,
                species_name="TestFish",
                n_dt_per_year=12,
                n_years=1,
                lifespan_dt=48,
                ny=3, nx=3,
                config_dir=str(tmp_path),
                strict=False,
            )
        assert "slots have no movement map assigned" in caplog.text

    def test_strict_no_error_when_fully_covered(self, tmp_path):
        from osmose.engine.movement_maps import MovementMapSet

        config = self._make_full_config(tmp_path)
        ms = MovementMapSet(
            config=config,
            species_name="TestFish",
            n_dt_per_year=12,
            n_years=1,
            lifespan_dt=48,
            ny=3, nx=3,
            config_dir=str(tmp_path),
            strict=True,
        )
        assert (ms.index_maps >= 0).all()
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py::TestStrictCoverage -v`
Expected: 3 passed.

- [ ] **Step 3: Full suite + ruff + commit**

Run: `.venv/bin/python -m pytest tests/ -q` → 2149 + 3 = 2152 passed
Run: `.venv/bin/ruff check tests/test_engine_map_movement.py osmose/engine/movement_maps.py`

```bash
git add tests/test_engine_map_movement.py osmose/engine/movement_maps.py
git commit -m "test: add strict coverage mode tests for MovementMapSet (M-7)"
```

---

### Task 4: Add config key and plumb through from_dict

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** Add `movement.map.strict.coverage` config key, parse it in `from_dict`, store on `EngineConfig`, and pass through to `MovementMapSet` construction in `simulate.py` (or wherever MapSets are built).

- [ ] **Step 1: Find where MovementMapSet is constructed**

Run: `grep -rn "MovementMapSet(" osmose/`
Record the file and line. It's likely in `osmose/engine/movement.py` or `osmose/engine/simulate.py`.

- [ ] **Step 2: Add field to EngineConfig**

In `osmose/engine/config.py`, find the `EngineConfig` dataclass fields. Add:

```python
    movement_strict_coverage: bool = False
```

- [ ] **Step 3: Parse in from_dict**

In `from_dict`, near the movement params section, add:

```python
        movement_strict = cfg.get("movement.map.strict.coverage", "false").lower() == "true"
```

And in the return statement, add:

```python
            movement_strict_coverage=movement_strict,
```

- [ ] **Step 4: Pass to MovementMapSet construction**

At the `MovementMapSet(...)` call site found in Step 1, add `strict=config.movement_strict_coverage` to the constructor arguments.

- [ ] **Step 5: Parity + full suite + ruff + commit**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` → 12 passed
Run: `.venv/bin/python -m pytest tests/ -q` → 2152 passed
Run: `.venv/bin/ruff check osmose/`

```bash
git add osmose/engine/config.py osmose/engine/movement.py
git commit -m "feat(engine): add movement.map.strict.coverage config key (M-7)"
```

---

## Final gate

- [ ] **Full suite**: `.venv/bin/python -m pytest tests/ -q` → ≥2152 passed
- [ ] **Ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
- [ ] **Parity**: 12 passed, bit-exact
- [ ] Invoke `superpowers:finishing-a-development-branch`
