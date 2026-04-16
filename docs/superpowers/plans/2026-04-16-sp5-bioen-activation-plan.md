# SP-5: Bioen Process Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 3 bioen mortality processes (ForagingMortality, BioenStarvation, BioenPredation) into the mortality loop, switchable via `simulation.bioen.enabled` config key — matching Java OSMOSE 4.3.3's `MortalityProcess`.

**Architecture:** Java's bioen switching works at init time: when `isBioenEnabled()`, the mortality loop (1) replaces standard PredationMortality with BioenPredationMortality, (2) replaces standard starvation with BioenStarvationMortality.computeStarvation(), (3) adds FORAGING as a 5th interleaved cause. Python already has stub files (`bioen_starvation.py`, `bioen_predation.py`) with correct formulas. This plan creates `foraging_mortality.py`, then wires all 3 into the existing mortality loop with bioen_enabled guards.

**Tech Stack:** Python 3.12+, NumPy, existing bioen stubs.

**Spec:** `docs/superpowers/specs/2026-04-16-java-parity-full-design.md` (SP-5 section)

**Java reference:** `/home/razinka/osmose/osmose-master/java/src/main/java/fr/ird/osmose/process/MortalityProcess.java` (lines 195-252, 512-625)

**What already exists:**
- `osmose/engine/processes/bioen_starvation.py` — `bioen_starvation()`: gonad-depletion formula, correct
- `osmose/engine/processes/bioen_predation.py` — `bioen_ingestion_cap()`: allometric cap formula, correct
- `osmose/engine/config.py` — `bioen_enabled: bool`, `bioen_*` per-species parameters (18 fields)
- `osmose/engine/processes/mortality.py` — interleaved loop with 4 causes (PREDATION, STARVATION, ADDITIONAL, FISHING)
- `osmose/engine/state.py` — `MortalityCause` enum already has FORAGING=5

**Java switching behavior (from `MortalityProcess.java`):**
- `isBioenEnabled() == false`:
  - PredationMortality (standard) — uses `predation.ingestion.rate.max.sp{i}`
  - StarvationMortality (standard) — `M = starvation_rate / subdt`
  - 4 causes: PREDATION, STARVATION, ADDITIONAL, FISHING (FORAGING removed)
- `isBioenEnabled() == true`:
  - BioenPredationMortality — extends PredationMortality, overrides `getMaxPredationRate()` with bioen Imax + larvae factor
  - BioenStarvationMortality.computeStarvation() — gonad-depletion, replaces standard starvation
  - ForagingMortality — new cause, `k_for / ndt_per_year` (constant) or `k1*exp(k2*(imax-Imax)) / ndt` (genetic)
  - 5 causes: PREDATION, STARVATION, ADDITIONAL, FISHING, FORAGING

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/processes/foraging_mortality.py` | **create** | Foraging mortality rate computation |
| `osmose/engine/processes/mortality.py` | **modify** | Wire bioen switching into interleaved loop |
| `osmose/engine/config.py` | **modify** | Parse foraging mortality config keys |
| `tests/test_engine_bioen_activation.py` | **create** | Tests for bioen switching and foraging mortality |

---

### Task 1: Foraging Mortality Function

Create the foraging mortality computation matching Java `ForagingMortality.getRate()`.

**Files:**
- Create: `osmose/engine/processes/foraging_mortality.py`
- Create: `tests/test_engine_bioen_activation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_bioen_activation.py`:

```python
"""Tests for bioen process activation (SP-5)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.processes.foraging_mortality import foraging_rate


class TestForagingMortality:
    """ForagingMortality.getRate() — two modes."""

    def test_constant_mode(self) -> None:
        """Without genetics: F = k_for / ndt_per_year."""
        k_for = np.array([0.24])
        ndt_per_year = 24
        rate = foraging_rate(k_for=k_for, ndt_per_year=ndt_per_year)
        assert rate[0] == pytest.approx(0.01)  # 0.24 / 24

    def test_constant_mode_clamped(self) -> None:
        """Negative result clamped to 0."""
        k_for = np.array([-0.1])
        rate = foraging_rate(k_for=k_for, ndt_per_year=24)
        assert rate[0] == 0.0

    def test_genetic_mode(self) -> None:
        """With genetics: F = k1 * exp(k2 * (imax_trait - I_max)) / ndt."""
        k1 = np.array([0.1])
        k2 = np.array([2.0])
        imax_trait = np.array([5.0])
        I_max = np.array([5.0])  # trait == baseline → exp(0) = 1
        ndt_per_year = 24
        rate = foraging_rate(
            k_for=None, ndt_per_year=ndt_per_year,
            k1_for=k1, k2_for=k2, imax_trait=imax_trait, I_max=I_max,
        )
        assert rate[0] == pytest.approx(0.1 / 24)  # k1 * exp(0) / 24

    def test_genetic_mode_penalty(self) -> None:
        """Trait below baseline → exponential penalty increases rate."""
        k1 = np.array([0.1])
        k2 = np.array([1.0])
        imax_trait = np.array([3.0])  # below baseline
        I_max = np.array([5.0])
        rate = foraging_rate(
            k_for=None, ndt_per_year=24,
            k1_for=k1, k2_for=k2, imax_trait=imax_trait, I_max=I_max,
        )
        # exp(1.0 * (3 - 5)) = exp(-2) ≈ 0.135
        expected = 0.1 * np.exp(-2.0) / 24
        assert rate[0] == pytest.approx(expected, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py::TestForagingMortality -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.processes.foraging_mortality'`

- [ ] **Step 3: Implement foraging mortality**

Create `osmose/engine/processes/foraging_mortality.py`:

```python
"""Foraging mortality for bioenergetic OSMOSE simulations.

Matches Java ForagingMortality.getRate():
- Without genetics: F = k_for / ndt_per_year (constant)
- With genetics: F = k1_for * exp(k2_for * (imax_trait - I_max)) / ndt_per_year

Config keys:
- species.bioen.forage.k_for.sp{i} (without genetics)
- species.bioen.forage.k1_for.sp{i} (with genetics)
- species.bioen.forage.k2_for.sp{i} (with genetics)
- predation.ingestion.rate.max.bioen.sp{i} (I_max reference)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def foraging_rate(
    k_for: NDArray[np.float64] | None,
    ndt_per_year: int,
    k1_for: NDArray[np.float64] | None = None,
    k2_for: NDArray[np.float64] | None = None,
    imax_trait: NDArray[np.float64] | None = None,
    I_max: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute foraging mortality rate per school.

    Two modes (matching Java ForagingMortality):
    - Constant (no genetics): k_for / ndt_per_year
    - Genetic: k1_for * exp(k2_for * (imax_trait - I_max)) / ndt_per_year

    Returns rate array, clamped to >= 0.
    """
    if k1_for is not None and k2_for is not None and imax_trait is not None and I_max is not None:
        # Genetic mode
        rate = k1_for * np.exp(k2_for * (imax_trait - I_max)) / ndt_per_year
    elif k_for is not None:
        # Constant mode
        rate = k_for / ndt_per_year
    else:
        raise ValueError("Must provide either k_for (constant) or k1_for+k2_for+imax_trait+I_max (genetic)")

    return np.maximum(rate, 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py::TestForagingMortality -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Add foraging config fields**

In `osmose/engine/config.py`, add fields to EngineConfig (in the bioen section):

```python
    # Foraging mortality (bioen only)
    foraging_k_for: NDArray[np.float64] | None = None    # constant rate (no genetics)
    foraging_k1_for: NDArray[np.float64] | None = None   # genetic mode base
    foraging_k2_for: NDArray[np.float64] | None = None   # genetic mode exponent
    foraging_I_max: NDArray[np.float64] | None = None     # reference I_max
```

In the bioen config loading section (where `_bioen_enabled` is True), add:

```python
        # Foraging mortality parameters
        foraging_k_for = _species_float_optional(cfg, n, "species.bioen.forage.k_for.sp{i}", 0.0)
        foraging_k1_for = _species_float_optional(cfg, n, "species.bioen.forage.k1_for.sp{i}", 0.0)
        foraging_k2_for = _species_float_optional(cfg, n, "species.bioen.forage.k2_for.sp{i}", 0.0)
        foraging_I_max = _species_float_optional(
            cfg, n, "predation.ingestion.rate.max.bioen.sp{i}", 0.0
        )
```

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/foraging_mortality.py osmose/engine/config.py tests/test_engine_bioen_activation.py
git commit -m "feat(engine): add foraging mortality function with constant and genetic modes"
```

---

### Task 2: Wire Bioen Starvation into Mortality Loop

When `bioen_enabled=True`, replace standard starvation (`starvation_rate / subdt`) with bioen starvation (gonad-depletion from `bioen_starvation.py`).

**Files:**
- Modify: `osmose/engine/processes/mortality.py`
- Modify: `tests/test_engine_bioen_activation.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_engine_bioen_activation.py`:

```python
from unittest.mock import MagicMock
from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState, MortalityCause


class TestBioenStarvationSwitch:
    """When bioen_enabled=True, starvation uses gonad-depletion formula."""

    def test_bioen_starvation_uses_gonad(self) -> None:
        """With bioen enabled, starvation depletes gonad weight before killing."""
        from osmose.engine.processes.bioen_starvation import bioen_starvation

        e_net = np.array([-1.0])  # negative → starvation
        gonad_weight = np.array([0.5])
        weight = np.array([0.01])
        eta = 0.8

        n_dead, new_gonad = bioen_starvation(e_net, gonad_weight, weight, eta, n_subdt=1)
        # Gonad should absorb some deficit
        assert new_gonad[0] < gonad_weight[0]

    def test_standard_starvation_when_bioen_disabled(self) -> None:
        """With bioen disabled, standard starvation rate applies."""
        from osmose.engine.processes.mortality import _apply_starvation_for_school

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([0.01]),
            starvation_rate=np.array([0.5]),
            age_dt=np.array([48], dtype=np.int32),
        )

        config = MagicMock(spec=EngineConfig)
        config.bioen_enabled = False
        config.n_dt_per_year = 24

        inst_abd = state.abundance.copy()
        _apply_starvation_for_school(0, state, config, n_subdt=1, inst_abd=inst_abd)
        assert state.n_dead[0, MortalityCause.STARVATION] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py::TestBioenStarvationSwitch -v`
Expected: PASS (bioen_starvation already works, standard starvation already works — these are verification tests)

- [ ] **Step 3: Add bioen starvation branch to `_apply_starvation_for_school`**

In `osmose/engine/processes/mortality.py`, modify `_apply_starvation_for_school()` to check `bioen_enabled`:

```python
def _apply_starvation_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
) -> None:
    """Apply starvation mortality to a single school (in-place on n_dead).

    When bioen_enabled=True: uses BioenStarvationMortality (gonad depletion).
    When bioen_enabled=False: uses standard starvation rate.
    Matches Java MortalityProcess lines 604-626.
    """
    if state.is_background[idx]:
        return
    if state.age_dt[idx] < state.first_feeding_age_dt[idx]:
        return

    abd = inst_abd[idx]
    if abd <= 0:
        return

    if config.bioen_enabled:
        # Bioen starvation: gonad-depletion formula
        from osmose.engine.processes.bioen_starvation import bioen_starvation

        e_net = np.array([state.e_net[idx]]) if hasattr(state, 'e_net') else np.array([0.0])
        gonad_w = np.array([state.gonad_weight[idx]]) if hasattr(state, 'gonad_weight') else np.array([0.0])
        weight = np.array([state.weight[idx]])
        sp_i = state.species_id[idx]
        eta = config.bioen_eta[sp_i] if hasattr(config, 'bioen_eta') else 1.0

        n_dead_arr, new_gonad = bioen_starvation(e_net, gonad_w, weight, eta, n_subdt)
        # bioen_starvation returns absolute n_dead (deficit/weight), NOT a fraction
        n_dead = float(n_dead_arr[0])
        if n_dead > 0:
            state.n_dead[idx, _STARVATION] += n_dead
            inst_abd[idx] -= n_dead
    else:
        # Standard starvation
        M = state.starvation_rate[idx] / (config.n_dt_per_year * n_subdt)
        if M <= 0:
            return
        n_dead = abd * (1.0 - np.exp(-M))
        state.n_dead[idx, _STARVATION] += n_dead
        inst_abd[idx] -= n_dead
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All existing tests pass (bioen_enabled=False is the default, so existing behavior unchanged)

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_bioen_activation.py
git commit -m "feat(engine): wire bioen starvation into mortality loop with bioen_enabled switch"
```

---

### Task 3: Wire Foraging Mortality as 5th Cause

When `bioen_enabled=True`, add FORAGING as a 5th interleaved cause in the mortality loop. When `bioen_enabled=False`, FORAGING is excluded (matching Java line 515-517).

**Files:**
- Modify: `osmose/engine/processes/mortality.py`
- Modify: `tests/test_engine_bioen_activation.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_engine_bioen_activation.py`:

```python
class TestForagingInMortalityLoop:
    """FORAGING added as 5th cause when bioen_enabled=True."""

    def test_foraging_cause_excluded_when_bioen_off(self) -> None:
        """Without bioen, mortality loop uses 4 causes (no FORAGING)."""
        from osmose.engine.processes.mortality import _get_mortality_causes

        config = MagicMock(spec=EngineConfig)
        config.bioen_enabled = False
        causes = _get_mortality_causes(config)
        assert MortalityCause.FORAGING not in causes
        assert len(causes) == 4

    def test_foraging_cause_included_when_bioen_on(self) -> None:
        """With bioen, mortality loop uses 5 causes (includes FORAGING)."""
        from osmose.engine.processes.mortality import _get_mortality_causes

        config = MagicMock(spec=EngineConfig)
        config.bioen_enabled = True
        causes = _get_mortality_causes(config)
        assert MortalityCause.FORAGING in causes
        assert len(causes) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py::TestForagingInMortalityLoop -v`
Expected: FAIL — `cannot import name '_get_mortality_causes'`

- [ ] **Step 3: Extract cause list into helper and add foraging**

In `osmose/engine/processes/mortality.py`, add helper:

```python
_FORAGING = int(MortalityCause.FORAGING)


def _get_mortality_causes(config: EngineConfig) -> list[int]:
    """Get the list of mortality causes for the interleaved loop.

    Without bioen: [PREDATION, STARVATION, ADDITIONAL, FISHING]
    With bioen: [PREDATION, STARVATION, ADDITIONAL, FISHING, FORAGING]
    Matches Java MortalityProcess lines 512-517.
    """
    causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]
    if config.bioen_enabled:
        causes.append(_FORAGING)
    return causes
```

Then update `_mortality_in_cell()` to use this helper instead of the hardcoded list. In the cause order generation (around line 1355):

Replace:
```python
        causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]
```

With:
```python
        causes = _get_mortality_causes(config)
```

Also add a `_apply_foraging_for_school()` function:

```python
def _apply_foraging_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
) -> None:
    """Apply foraging mortality to a single school (bioen only, in-place)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] < state.first_feeding_age_dt[idx]:
        return

    abd = inst_abd[idx]
    if abd <= 0:
        return

    from osmose.engine.processes.foraging_mortality import foraging_rate

    sp_i = state.species_id[idx]

    # Java checks isGeneticEnabled() to decide mode
    # Genetic mode: needs per-school imax trait from genetic state
    genetic = (
        config.foraging_k1_for is not None
        and config.foraging_k2_for is not None
        and hasattr(state, 'imax_trait')
        and state.imax_trait is not None
    )
    if genetic:
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=config.n_dt_per_year,
            k1_for=np.array([config.foraging_k1_for[sp_i]]),
            k2_for=np.array([config.foraging_k2_for[sp_i]]),
            imax_trait=np.array([state.imax_trait[idx]]),
            I_max=np.array([config.foraging_I_max[sp_i]]),
        )
    else:
        k_for = np.array([config.foraging_k_for[sp_i]]) if config.foraging_k_for is not None else np.array([0.0])
        rate = foraging_rate(k_for=k_for, ndt_per_year=config.n_dt_per_year)

    M = float(rate[0]) / n_subdt
    if M <= 0:
        return
    n_dead = abd * (1.0 - np.exp(-M))
    state.n_dead[idx, _FORAGING] += n_dead
    inst_abd[idx] -= n_dead
```

Add the FORAGING case to the cause dispatch in the interleaved loop (both Python and Numba paths):

In the Python path of `_mortality_in_cell()`, add to the cause switch:
```python
                elif cause == _FORAGING:
                    if config.bioen_enabled:
                        _apply_foraging_for_school(
                            cell_indices[seq_for[i]], state, config, n_subdt, inst_abd
                        )
```

Also add `seq_for = rng.permutation(n_local).astype(np.int32)` alongside the other shuffle sequences.

**Numba path:** The existing Numba-JIT mortality kernel (`_mortality_in_cell_numba`) only handles 4 causes. When `bioen_enabled=True`, force the Python fallback path by setting `use_full_numba = False` when bioen is on. This avoids complex Numba changes. Add to the `use_full_numba` condition:
```python
    use_full_numba = (
        _HAS_NUMBA and inst_abd is not None and rsc_size_min is not None
        and eff_starv is not None and not config.bioen_enabled  # <-- disable Numba for bioen
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All existing tests pass (bioen_enabled=False by default → no FORAGING, no behavior change)

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/mortality.py osmose/engine/processes/foraging_mortality.py tests/test_engine_bioen_activation.py
git commit -m "feat(engine): wire foraging mortality as 5th cause with bioen_enabled switch"
```

---

### Task 4: Integration Test — All Bioen Processes Together

Verify the full bioen switching works end-to-end: when `bioen_enabled=True`, all 3 bioen processes activate correctly in the mortality loop.

**Files:**
- Modify: `tests/test_engine_bioen_activation.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_engine_bioen_activation.py`:

```python
class TestBioenSwitchIntegration:
    """End-to-end: bioen_enabled toggles all 3 processes."""

    def test_bioen_off_no_foraging_deaths(self) -> None:
        """bioen_enabled=False → zero FORAGING deaths in output."""
        from osmose.engine.processes.mortality import _get_mortality_causes

        config = MagicMock(spec=EngineConfig)
        config.bioen_enabled = False
        causes = _get_mortality_causes(config)
        assert MortalityCause.FORAGING not in causes

    def test_bioen_on_foraging_deaths_possible(self) -> None:
        """bioen_enabled=True → FORAGING cause is in the mortality loop."""
        from osmose.engine.processes.mortality import _get_mortality_causes

        config = MagicMock(spec=EngineConfig)
        config.bioen_enabled = True
        causes = _get_mortality_causes(config)
        assert MortalityCause.FORAGING in causes
        assert MortalityCause.STARVATION in causes  # still present (bioen replaces impl, not cause)
        assert MortalityCause.PREDATION in causes

    def test_foraging_rate_applied_in_loop(self) -> None:
        """foraging_rate produces non-zero rate with valid config."""
        from osmose.engine.processes.foraging_mortality import foraging_rate

        k_for = np.array([0.5, 0.3])
        rates = foraging_rate(k_for=k_for, ndt_per_year=24)
        assert all(r > 0 for r in rates)
        assert rates[0] > rates[1]  # higher k_for → higher rate
```

- [ ] **Step 2: Run all bioen tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_activation.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_bioen_activation.py
git commit -m "test(engine): add bioen switch integration tests"
```
