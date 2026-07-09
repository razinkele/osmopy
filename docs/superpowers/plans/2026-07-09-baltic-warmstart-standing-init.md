# Warm-start standing-stock initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Context / why:** Chunk 0 (`docs/baltic_chunk0_results_2026-07-08.md`) proved the deployed Baltic model is *monostable* under the larval-mortality driver and that the accessibility A1 test can't be resolved on a ~15-y horizon — both blocked by the same gap: **the Python engine has no standing-stock initialization** (`simulate.py:1188` `initialize()` returns `SchoolState.create(n_schools=0)`; every school is born from egg-seeding). This plan builds the enabler: an age-structured standing-stock initializer so a genuine adult population (including a clupeid-dominated / sprat-dominated state) can exist at t=0 and evolve under fixed parameters — the prerequisite for a *definitive* bistability/hysteresis test and a clean accessibility A/B.

**Goal:** Add an opt-in, INERT-by-default primitive that, given a per-species initial biomass, seeds an age-structured standing adult population at t=0.

> **STATUS: IMPLEMENTED (2026-07-09), with the review's two high fixes folded in.** After a four-lens
> review (2 high, both real): (H1) the builder now reads the **canonical** flag
> `module.population.initialisation.enabled` (the deprecated `population.initialization.relativebiomass.enabled`
> is a legacy fallback) — the deprecated key is renamed away by config canonicalization, so reading it
> alone left the feature silently inert; (H2) warm-start now **disables egg-seeding** (`config._load_reproduction`
> sets `seeding_max_step=0` under the flag) so a suppressed species isn't continuously re-injected, which
> would otherwise confound a cod-absent basin. Also: age-decay uses a `max(additional_mortality, 1.5·K, 0.05)`
> proxy (documented coarse start), `floor(lifespan)` age classes with an age-bound clamp, the smoke uses
> `PythonEngine()._resolve_grid`, and the payoff is reframed as a **reciprocal-invasion** test (a monostable
> model stays monostable — see the warm-start prerequisite doc). Delivered: `osmose/engine/initialization.py`,
> `tests/test_engine_initialization.py` (8 tests incl. a real-config activation test), `config.py`/`simulate.py`
> wiring, `scripts/warmstart_smoke.py`. Parity: 174 engine tests unchanged; `ruff` clean.

**Architecture:** A new pure module `osmose/engine/initialization.py` mirroring `osmose/engine/incoming_flux.py` (which already turns a per-class biomass into age-structured schools via Von Bertalanffy length, allometric weight, biomass→abundance, and random ocean-cell placement). `initialize()` delegates to it. Behind the existing `population.initialization.relativebiomass.enabled` flag (default false), so with the flag off `initialize()` returns an empty `SchoolState` exactly as today (bit-identical parity). Init biomass reuses the already-typed `config.seeding_biomass[sp]` (`population.seeding.biomass.sp{i}`), so Chunk 0's `cod_rich_seeding`/`cod_poor_seeding` overrides become genuine standing-stock initial conditions with no new config key.

**Tech Stack:** Python 3, numpy, pytest; `osmose.engine.state.SchoolState`, `EngineConfig` (fields `linf`, `k`, `t0`, `condition_factor`, `allometric_power`, `additional_mortality_rate`, `lifespan_dt`, `n_dt_per_year`, `n_schools`, `seeding_biomass`, `raw_config`), `osmose.engine.grid.Grid` (`ocean_mask`).

## Global Constraints

- **Parity is the #1 constraint.** Default OFF: with `population.initialization.relativebiomass.enabled` absent or `false`, `initialize()` MUST return `SchoolState.create(n_schools=0)` — byte-identical to current behaviour. Every existing engine test must still pass unchanged. This is Python-engine-only; the Java engine is untouched.
- **Biomass conservation:** for each species, `sum(abundance * weight)` of the seeded schools MUST equal the target biomass within floating-point rounding.
- **Reuse, don't reinvent:** follow `incoming_flux.py` exactly for age→length (`linf*(1-exp(-k*(age-t0)))`), length→weight (`cf*L^ap*1e-6`, tonnes), biomass→abundance, and random ocean-cell placement.
- **`SchoolState` invariant:** `__post_init__`/validation requires `biomass == abundance*weight` for live schools — set `biomass = abundance*weight` and `is_egg=False`, `length_start = length`.
- **Init biomass source:** `config.seeding_biomass[sp]` (tonnes). A species with `seeding_biomass[sp] <= 0` seeds no schools (so a clupeid-dominated IC = high herring/sprat seeding biomass, ~0 cod).
- **Config keys** (already known to the alias/validation layer): master switch `population.initialization.relativebiomass.enabled` (aliased from `module.population.initialisation.enabled`). No new per-species key is introduced.

---

## File Structure

- **Create `osmose/engine/initialization.py`** — pure builder: `age_structured_population(...)` (age-structure math) + `build_initial_population(config, grid, rng) -> SchoolState`.
- **Modify `osmose/engine/simulate.py`** — `initialize()` (line ~1188) delegates to `build_initial_population`.
- **Create `tests/test_engine_initialization.py`** — unit tests (age-structure math + builder with a fake config/grid) and a parity test. CI-safe (no full sim).

---

### Task 1: Age-structure math (pure, biomass-conserving)

**Files:** Create `osmose/engine/initialization.py`; Test `tests/test_engine_initialization.py`.

**Interfaces (Produces):** `age_structured_population(target_biomass, linf, k, t0, cf, ap, mortality, lifespan_years, n_dt_per_year, min_length=0.001) -> tuple[ages_dt, lengths, weights, abundances]` (four aligned 1-D arrays; empty arrays when `target_biomass<=0` or `lifespan_years<=0`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_initialization.py
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from osmose.engine.initialization import age_structured_population  # noqa: E402


def test_age_structure_conserves_biomass_and_declines():
    ages_dt, lengths, weights, abund = age_structured_population(
        target_biomass=1000.0, linf=100.0, k=0.2, t0=-0.2, cf=0.01, ap=3.0,
        mortality=0.3, lifespan_years=10, n_dt_per_year=24,
    )
    assert len(ages_dt) == 10
    assert abs(float((abund * weights).sum()) - 1000.0) < 1e-6      # biomass conserved
    assert np.all(abund[:-1] >= abund[1:])                          # numbers decline with age
    assert np.all(lengths[:-1] <= lengths[1:])                      # length grows with age
    assert np.all(ages_dt >= 0)


def test_age_structure_empty_when_no_biomass():
    for ad in (age_structured_population(0.0, 100, 0.2, -0.2, 0.01, 3.0, 0.3, 10, 24)[0],
               age_structured_population(1000.0, 100, 0.2, -0.2, 0.01, 3.0, 0.3, 0, 24)[0]):
        assert len(ad) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_initialization.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.initialization'`.

- [ ] **Step 3: Write minimal implementation**

```python
# osmose/engine/initialization.py
"""Warm-start standing-stock initialization (opt-in, inert by default).

Given a per-species initial biomass, seed an age-structured standing adult
population at t=0 (numbers-at-age ~ exp(-M*age), Von Bertalanffy length,
allometric weight), so a genuine adult community — including a clupeid-dominated
alternative state — can exist at t=0 for a definitive bistability / hysteresis
test. Mirrors osmose/engine/incoming_flux.py. Gated by
`population.initialization.relativebiomass.enabled` (default false => empty init,
byte-identical to the current Java-convention empty population).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.state import SchoolState

_MIN_DECAY_M = 0.05  # floor on the age-decay rate so the structure isn't flat when M==0


def age_structured_population(
    target_biomass: float,
    linf: float,
    k: float,
    t0: float,
    cf: float,
    ap: float,
    mortality: float,
    lifespan_years: float,
    n_dt_per_year: int,
    min_length: float = 0.001,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Distribute target_biomass (tonnes) across integer age classes 0..lifespan.

    Returns (ages_dt, lengths_cm, weights_tonnes, abundances) aligned per age class,
    with sum(abundances*weights) == target_biomass. Empty arrays if there is nothing
    to seed.
    """
    empty = (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )
    if target_biomass <= 0.0 or lifespan_years <= 0.0:
        return empty
    n_ages = max(1, int(round(lifespan_years)))
    ages_years = np.arange(n_ages, dtype=np.float64) + 0.5   # mid-year, avoids exact age 0
    lengths = linf * (1.0 - np.exp(-k * (ages_years - t0)))
    lengths = np.maximum(lengths, min_length)
    weights = cf * lengths**ap * 1e-6                        # grams -> tonnes
    m = max(mortality, _MIN_DECAY_M)
    numbers = np.exp(-m * ages_years)
    rel_biomass = numbers * weights
    total = float(rel_biomass.sum())
    if total <= 0.0:
        return empty
    abundances = (target_biomass / total) * numbers
    ages_dt = np.maximum(0, np.round(ages_years * n_dt_per_year)).astype(np.int32)
    return ages_dt, lengths, weights, abundances
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_initialization.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/initialization.py tests/test_engine_initialization.py
git commit -m "feat(engine): age-structured biomass distribution for warm-start init"
```

---

### Task 2: build_initial_population (gated builder)

**Files:** Modify `osmose/engine/initialization.py`; Test same.

**Interfaces (Produces):** `build_initial_population(config, grid, rng) -> SchoolState`. Reads `config.raw_config["population.initialization.relativebiomass.enabled"]` (default false → empty); per species uses `config.seeding_biomass[sp]` as target and `config.{linf,k,t0,condition_factor,allometric_power,additional_mortality_rate,lifespan_dt,n_dt_per_year,n_schools}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine_initialization.py
from osmose.engine.initialization import build_initial_population  # noqa: E402


def _fake_config(enabled, seeding_biomass):
    n = len(seeding_biomass)
    arr = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    return SimpleNamespace(
        n_species=n,
        raw_config={"population.initialization.relativebiomass.enabled": "true" if enabled else "false"},
        seeding_biomass=np.array(seeding_biomass, dtype=np.float64),
        linf=arr(100.0), k=arr(0.2), t0=arr(-0.2),
        condition_factor=arr(0.01), allometric_power=arr(3.0),
        additional_mortality_rate=arr(0.3),
        lifespan_dt=arr(10 * 24), n_dt_per_year=24, n_schools=np.full(n, 10, dtype=np.int32),
    )


def _fake_grid():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True  # 6 ocean cells
    return SimpleNamespace(ocean_mask=mask)


def test_disabled_returns_empty():
    st = build_initial_population(_fake_config(False, [1000.0, 2000.0]), _fake_grid(), np.random.default_rng(0))
    assert len(st) == 0


def test_enabled_seeds_conserved_standing_stock():
    cfg = _fake_config(True, [1000.0, 0.0, 2000.0])  # sp1 has zero biomass -> no schools
    st = build_initial_population(cfg, _fake_grid(), np.random.default_rng(0))
    assert len(st) > 0
    assert not st.is_egg.any()                       # standing adults, not eggs
    assert (st.abundance > 0).all()
    # per-species biomass conserved
    for sp, target in ((0, 1000.0), (2, 2000.0)):
        m = st.species_id == sp
        assert abs(float((st.abundance[m] * st.weight[m]).sum()) - target) < 1.0
    assert not (st.species_id == 1).any()            # zero-biomass species seeds nothing
    # schools placed in ocean cells
    ys, xs = np.where(_fake_grid().ocean_mask)
    for cx, cy in zip(st.cell_x, st.cell_y):
        assert _fake_grid().ocean_mask[cy, cx]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_initialization.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_initial_population'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to osmose/engine/initialization.py

def build_initial_population(config, grid, rng) -> SchoolState:
    """Age-structured standing population at t=0, or empty if the flag is off."""
    empty = SchoolState.create(n_schools=0)
    raw = getattr(config, "raw_config", {}) or {}
    if raw.get("population.initialization.relativebiomass.enabled", "false").lower() != "true":
        return empty
    ys, xs = np.where(grid.ocean_mask)
    if len(ys) == 0:
        return empty
    ys = ys.astype(np.int32)
    xs = xs.astype(np.int32)
    parts: list[SchoolState] = []
    for sp in range(config.n_species):
        target = float(config.seeding_biomass[sp])
        if target <= 0.0:
            continue
        ages_dt, lengths, weights, abund = age_structured_population(
            target, float(config.linf[sp]), float(config.k[sp]), float(config.t0[sp]),
            float(config.condition_factor[sp]), float(config.allometric_power[sp]),
            float(config.additional_mortality_rate[sp]),
            float(config.lifespan_dt[sp]) / config.n_dt_per_year, config.n_dt_per_year,
        )
        n_schools_sp = int(config.n_schools[sp])
        for c in range(len(ages_dt)):
            if abund[c] <= 0.0 or weights[c] <= 0.0:
                continue
            n_new = n_schools_sp if (abund[c] >= n_schools_sp and n_schools_sp > 0) else 1
            abund_per = abund[c] / n_new
            idx = rng.integers(0, len(ys), size=n_new)
            new = SchoolState.create(n_schools=n_new, species_id=np.full(n_new, sp, dtype=np.int32))
            new = new.replace(
                abundance=np.full(n_new, abund_per, dtype=np.float64),
                biomass=np.full(n_new, abund_per * weights[c], dtype=np.float64),
                length=np.full(n_new, lengths[c], dtype=np.float64),
                length_start=np.full(n_new, lengths[c], dtype=np.float64),
                weight=np.full(n_new, weights[c], dtype=np.float64),
                age_dt=np.full(n_new, ages_dt[c], dtype=np.int32),
                cell_x=xs[idx],
                cell_y=ys[idx],
                is_egg=np.zeros(n_new, dtype=np.bool_),
            )
            parts.append(new)
    if not parts:
        return empty
    result = parts[0]
    for p in parts[1:]:
        result = result.append(p)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_initialization.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/initialization.py tests/test_engine_initialization.py
git commit -m "feat(engine): build_initial_population — gated age-structured standing stock"
```

---

### Task 3: Wire into initialize() (parity-preserving)

**Files:** Modify `osmose/engine/simulate.py` (`initialize()` ~line 1188); Test `tests/test_engine_initialization.py`.

**Interfaces:** Consumes `build_initial_population` (Task 2). `initialize(config, grid, rng)` unchanged signature.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine_initialization.py
from osmose.engine.simulate import initialize  # noqa: E402


def test_initialize_parity_off_and_populated_on():
    grid = _fake_grid()
    off = initialize(_fake_config(False, [1000.0]), grid, np.random.default_rng(0))
    assert len(off) == 0                                   # parity: empty when flag off
    on = initialize(_fake_config(True, [1000.0]), grid, np.random.default_rng(0))
    assert len(on) > 0 and not on.is_egg.any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_initialization.py::test_initialize_parity_off_and_populated_on -q`
Expected: FAIL — `initialize()` returns empty even when the flag is on (still the old body).

- [ ] **Step 3: Change `initialize()` in `osmose/engine/simulate.py`**

Replace the body:

```python
def initialize(config: EngineConfig, grid: Grid, rng: np.random.Generator) -> SchoolState:
    """Create the initial population.

    Default (Java convention): zero initial schools — all schools are created by
    reproduction's seeding mechanism. If ``population.initialization.relativebiomass.enabled``
    is true, seed an age-structured standing stock from ``seeding_biomass`` instead
    (warm-start; see osmose/engine/initialization.py).
    """
    from osmose.engine.initialization import build_initial_population

    return build_initial_population(config, grid, rng)
```

(`build_initial_population` returns the empty `SchoolState.create(n_schools=0)` when the flag is off, so the default path is unchanged.)

- [ ] **Step 4: Run tests to verify pass + parity**

Run: `.venv/bin/python -m pytest tests/test_engine_initialization.py -q`
Expected: PASS (5 tests).
Run (PARITY GATE — existing engine behaviour unchanged): `.venv/bin/python -m pytest tests/ -q -k "engine or parity or simulate" 2>&1 | tail -15`
Expected: no new failures (the flag is absent in all existing configs, so `initialize()` still returns empty).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_initialization.py
git commit -m "feat(engine): initialize() delegates to warm-start builder (inert by default)"
```

---

### Task 4: End-to-end smoke on Baltic (manual, not CI)

Confirms the standing stock actually materializes at t=0 in a real run and that two distinct initial conditions (cod-dominated vs clupeid-dominated) produce distinct standing stocks — the whole point of the primitive. Real-engine run, so it is a documented manual check, not a CI test (per `feedback-ci-fragile-emergent-tests`).

- [ ] **Step 1: Write a tiny smoke script** `scripts/warmstart_smoke.py`:

```python
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/razinka/osmopy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_baltic import BALTIC_CONFIG  # noqa: E402
from osmose.config.reader import OsmoseConfigReader  # noqa: E402
from osmose.engine.config import EngineConfig  # noqa: E402
from osmose.engine.grid import Grid  # noqa: E402
from osmose.engine.initialization import build_initial_population  # noqa: E402

cfg = OsmoseConfigReader().read(str(BALTIC_CONFIG))
cfg["population.initialization.relativebiomass.enabled"] = "true"
# cod-dominated vs clupeid-dominated standing initial conditions:
for label, overrides in (
    ("cod-dominated", {"population.seeding.biomass.sp0": "300000"}),
    ("clupeid-dominated", {"population.seeding.biomass.sp0": "1000",
                           "population.seeding.biomass.sp1": "1500000",
                           "population.seeding.biomass.sp2": "1500000"}),
):
    c = dict(cfg)
    c.update(overrides)
    ec = EngineConfig.from_dict(c)
    grid = Grid.from_config(c)  # adapt to the actual Grid constructor if different
    st = build_initial_population(ec, grid, np.random.default_rng(0))
    print(f"\n=== {label}: {len(st)} schools ===")
    for sp in range(ec.n_species):
        m = st.species_id == sp
        if m.any():
            print(f"  sp{sp}: biomass={(st.abundance[m] * st.weight[m]).sum():,.0f} t "
                  f"ages_dt={sorted(set(st.age_dt[m].tolist()))[:5]}...")
```

- [ ] **Step 2: Run it** (adapt the `Grid`/`EngineConfig` constructors to the real API if the imports differ):

Run: `PYTHONPATH=. .venv/bin/python scripts/warmstart_smoke.py`
Expected: each IC prints a non-empty school count; cod biomass ≈ 300 000 t (cod-dominated) vs ≈ 1 000 t (clupeid-dominated) with herring/sprat ≈ 1.5 Mt each; ages span multiple classes. Confirms the two standing ICs differ and biomass is conserved.

- [ ] **Step 3: Commit**

```bash
git add scripts/warmstart_smoke.py
git commit -m "chore(engine): warm-start standing-init smoke script"
```

---

### Task 5: Docs + Chunk-0 follow-on note

- [ ] **Step 1: Update `docs/baltic_chunk0_warmstart_prerequisite.md`** — add a "STATUS: BUILT (2026-07-09)" line at the top pointing to `osmose/engine/initialization.py` and the config flag, and note the definitive test is now unblocked.

- [ ] **Step 2: Note the follow-on** in the same doc: re-run the Chunk-0 bistability sweep with `population.initialization.relativebiomass.enabled=true` so the `cod_rich_seeding`/`cod_poor_seeding` overrides become genuine standing stocks, AND add a **clupeid-dominated** IC pair (high herring/sprat seeding biomass, ~0 cod) to `scripts/baltic_bistability_chunk0.py` so the sweep can construct the real cod↔sprat regime-shift basin — the thing Chunk 0's egg-only, single-cod-axis ICs could not. This is the definitive bistability test.

- [ ] **Step 3: Commit** (`docs: warm-start built; chunk-0 definitive-test follow-on`).

---

## The payoff (why this unblocks Phase 2)

With the flag on and per-species `seeding_biomass` set, the model starts from a genuine adult standing stock. This lets a follow-on Chunk-0 run:
- initialize a **cod-dominated** and a **clupeid-dominated** standing state at the *same* parameters and check whether they persist in different basins → a definitive alternative-stable-states / hysteresis test (not the conservative egg-only proxy);
- run the accessibility A/B from a settled standing stock, reducing the transient that forced the ~15-y PROVISIONAL verdicts.

Both were the open items Chunk 0 could not resolve; this primitive is their shared prerequisite.

## Self-Review

**Spec coverage.** Age-structure math (Task 1, biomass-conserving), gated builder (Task 2), parity-preserving wiring (Task 3), real-engine smoke incl. the two ICs (Task 4), docs + follow-on (Task 5). The parity constraint is enforced by Task 3's default-off test + the existing-suite gate.

**Placeholder scan.** Real code in every code step; exact commands with expected output. The one adapt-if-different note (Task 4 `Grid`/`EngineConfig` constructor) is flagged explicitly because the real constructor API should be confirmed at implementation time (the unit tests use a fake config/grid and do not depend on it).

**Type consistency.** `age_structured_population(...) -> (ages_dt:int32, lengths, weights, abundances)` consumed identically by `build_initial_population`; `SchoolState.create(...).replace(...)` matches the fields used by `incoming_flux.py`; `biomass = abundance*weight` and `is_egg=False` satisfy the `SchoolState` live-school invariant; `initialize()` keeps its `(config, grid, rng) -> SchoolState` signature.
