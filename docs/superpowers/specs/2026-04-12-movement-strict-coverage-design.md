# M-7: Movement Map Strict Coverage — Design Spec

> Deep review v3 finding M-7. Deferred from the remediation plan because building a
> minimal MovementMapSet fixture with known uncovered slots requires a non-trivial
> fixture-discovery spike.

## Problem

`MovementMapSet.__init__` logs an aggregated warning when `(age_dt, step)` slots have
no movement map assigned (`index_maps == -1`). This is the only signal — no way to
make uncovered slots a hard error. Users running misconfigured movement maps get silent
degradation instead of early failure.

## Goal

Add an opt-in `strict` mode to `MovementMapSet` that raises `ValueError` on uncovered
slots instead of warning. Default behavior unchanged.

## Approach

### Phase A — Fixture Spike

Build a test fixture that produces a *known* number of uncovered slots:

**Strategy:**
- Species with `lifespan=4` years, `n_dt_per_year=12` → 48 age_dt values
- 1-year simulation → 12 global steps
- Total slots: 48 × 12 = 576
- Define `map0` covering ages 0-1 only (24 age_dt × 12 steps = 288 covered)
- Ages 2-3 (24 age_dt × 12 steps = 288 slots) remain uncovered

**Fixture needs:**
- A `tmp_path` directory with a valid CSV movement map file (3×3 grid, positive values)
- Config dict with: `movement.species.map0`, `movement.file.map0`,
  `movement.initialage.map0=0`, `movement.lastage.map0=1`, `movement.steps.map0` (all steps)
- Assertion: `(map_set.index_maps == -1).sum() == 288`

**Deliverable:** Working test `test_fixture_produces_known_uncovered_slots` in
`tests/test_engine_map_movement.py`.

### Phase B — Implementation

1. Add parameter `strict: bool = False` to `MovementMapSet.__init__`
2. At the existing warning site (lines 267-275 of `movement_maps.py`):
   - If `strict` and `uncovered > 0`: raise `ValueError` with species name + count
   - Else: existing `logger.warning` (unchanged)
3. Add config key `movement.map.strict.coverage` (boolean, default false)
4. Parse in `EngineConfig.from_dict`, pass through to `MovementMapSet` construction
5. Tests:
   - Spike fixture with `strict=True` → raises `ValueError`
   - Spike fixture with `strict=False` → warns (caplog check)
   - Full coverage fixture (all slots covered) with `strict=True` → no error

## Out of scope

- Changing the existing warning behavior (stays as default)
- Per-species strict/lenient toggle (global flag is sufficient)
- Fixing the underlying coverage gaps in user configs (that's user responsibility)
