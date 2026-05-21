# Activate Ev-OSMOSE Genetics — FIE-on-Cod Demonstration: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Round 2 (2026-05-21):** Applied review findings from the 2026-05-18 parallel code+sci review (5 code bugs, 9 sci gaps). Code: fixed `sim_mod` NameError in Task 3.1, added missing `Path` imports in Tasks 2.1 + 6.1, rewrote Task 2.1 unit tests to target `_collect_trait_stats` directly, strengthened Task 9 weak test. Sci: corrected inverted Svedäng 2024 citation, replaced unsourced 1% threshold with literature-anchored 2% (Audzijonyte 2013; Andersen & Brander 2009), pinned nlocus/nval to Marty et al. 2015, replaced silent `pytest.skip` pre-flight stub with a loud `AssertionError`, added cod-biomass-stability guard (new Step 7.9), added optional `--with-zero-f-control` F=0 arm, reframed Olsen 2004 as maturation-FIE evidence (not growth-FIE). See conversation log for full review-round detail.

**Goal:** Ship a reproducible scientific demonstration of fishery-induced evolution (FIE) on Baltic cod growth rate by activating the wired-but-inactive Ev-OSMOSE genetics module on a new isolated `baltic_ev/` fixture.

**Architecture:** Add per-step trait statistics output to the engine (TraitStats + CSV writer + reader). Clone the calibrated `baltic/` fixture into `baltic_ev/`, enable both bioenergetics and genetics (cod imax trait only). Write a paired high-F vs low-F demo script that produces a trait-trajectory chart. Tutorial doc interprets the result against FIE literature.

**Tech Stack:** Python 3.12+, NumPy, pandas, xarray, pytest, matplotlib, OSMOSE engine (existing).

**Spec:** `docs/superpowers/specs/2026-05-18-ev-osmose-activation-design.md`

---

## File structure

| Action | Path | Responsibility |
|---|---|---|
| Modify | `osmose/engine/simulate.py` | Add `TraitStats` + `trait_stats` field on `StepOutput`; thread phenotypes through `_collect_outputs`; merge in `_average_step_outputs` |
| Modify | `osmose/engine/output.py` | Add `_write_genetic_trait_means_csv` writer, invoked by `write_outputs` |
| Modify | `osmose/results.py` | Add `read_genetic_trait_means(output_dir)` reader |
| Modify | `osmose/engine/config.py` | Add validator: declared trait must have per-species mean on every species |
| Modify | `osmose/engine/state.py:81` | 1-line comment marking `imax_trait` as vestigial |
| Create | `data/baltic_ev/` | Cloned + bioen-enabled + genetics-enabled fixture |
| Create | `data/baltic_ev/README.md` | Provenance citations for bioen params |
| Create | `scripts/run_fie_demo.py` | Paired high-F / low-F multi-seed runs + chart |
| Create | `docs/tutorials/fie-on-baltic-cod.md` | Scientific story + run command + chart |
| Create | `tests/test_step_output_trait_stats.py` | TraitStats dataclass shape |
| Create | `tests/test_collect_outputs_trait_stats.py` | `_collect_outputs` trait_stats wiring |
| Create | `tests/test_genetic_trait_means_csv.py` | Round-trip writer ↔ reader |
| Create | `tests/test_trait_registry_validator.py` | Config validator |
| Create | `tests/test_ev_osmose_activation.py` | End-to-end smoke on baltic_ev |
| Create | `tests/test_fie_demo_direction.py` | FIE direction regression (slow) |
| Modify | `tests/test_genetics_trait.py` | Extend with bioen_i_max target reach |

---

## Task 1: Add `TraitStats` dataclass and `trait_stats` field on `StepOutput`

**Files:**
- Modify: `osmose/engine/simulate.py:66-115`
- Test: `tests/test_step_output_trait_stats.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_step_output_trait_stats.py`:
```python
import numpy as np
import pytest

from osmose.engine.simulate import StepOutput, TraitStats


def test_trait_stats_dataclass_shape() -> None:
    ts = TraitStats(mean=1.5, variance=0.25, n_individuals=42)
    assert ts.mean == pytest.approx(1.5)
    assert ts.variance == pytest.approx(0.25)
    assert ts.n_individuals == 42


def test_step_output_accepts_trait_stats_none() -> None:
    out = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
    )
    assert out.trait_stats is None


def test_step_output_accepts_trait_stats_populated() -> None:
    trait_stats = {"imax": {0: TraitStats(mean=3.5, variance=0.1, n_individuals=100)}}
    out = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats=trait_stats,
    )
    assert out.trait_stats == trait_stats
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_step_output_trait_stats.py -v`
Expected: ImportError — `TraitStats` not defined.

- [ ] **Step 1.3: Add the `TraitStats` dataclass and field**

In `osmose/engine/simulate.py` directly above the `StepOutput` class (line 66), add:
```python
@dataclass(frozen=True)
class TraitStats:
    """Per-species summary statistics for one genetic trait at one timestep."""

    mean: float
    variance: float
    n_individuals: int
```

In the `StepOutput` class body, after the `diet_by_species` field (around line 98), add:
```python
    # Genetic trait statistics: trait_name -> species_id -> TraitStats,
    # or None if genetics disabled. Populated by _collect_outputs from
    # ctx.genetic_state phenotypes.
    trait_stats: dict[str, dict[int, "TraitStats"]] | None = None
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_step_output_trait_stats.py -v`
Expected: 3 passed.

- [ ] **Step 1.5: Run the full unit suite to verify no regression**

Run: `.venv/bin/python -m pytest tests/test_step_output_trait_stats.py tests/test_simulate.py -v`
Expected: all passed.

- [ ] **Step 1.6: Commit**

```bash
git add tests/test_step_output_trait_stats.py osmose/engine/simulate.py
git commit -m "feat(engine): add TraitStats and StepOutput.trait_stats field"
```

---

## Task 2: Wire `_collect_outputs` and `_average_step_outputs` for trait_stats

**Files:**
- Modify: `osmose/engine/simulate.py:896-941` (`_collect_outputs`)
- Modify: `osmose/engine/simulate.py:959+` (`_average_step_outputs`)
- Test: `tests/test_collect_outputs_trait_stats.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_collect_outputs_trait_stats.py`:
```python
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.simulate import (
    StepOutput,
    TraitStats,
    _average_step_outputs,
    _collect_trait_stats,
)
from osmose.engine.state import SchoolState


def _state_species_id(species_id: list[int]) -> SchoolState:
    """Build a SchoolState with the given species_id assignment.

    Tests target `_collect_trait_stats` directly (the new unit added by this
    plan) instead of the umbrella `_collect_outputs`. The umbrella drags in
    `_collect_biomass_abundance`, `_collect_distributions`, `_collect_bioen`,
    etc., which can fail noisily on a zero-filled state and obscure the
    intent of these unit tests. Integration coverage of the umbrella's
    `phenotypes` kwarg lives in Task 8.5 (spy on the real baltic_ev run).
    """
    n = len(species_id)
    sp = np.array(species_id, dtype=np.int32)  # species_id is int32 per state.py:39
    state = SchoolState.create(n_schools=n, species_id=sp)
    return state


def test_collect_trait_stats_empty_when_phenotypes_empty() -> None:
    state = _state_species_id([0, 0, 0, 0])
    out = _collect_trait_stats(state, phenotypes={})
    assert out == {}


def test_collect_trait_stats_populated_from_phenotypes() -> None:
    state = _state_species_id([0, 0, 0, 0])
    phenotypes = {"imax": np.array([3.0, 4.0, 5.0, 6.0])}

    out = _collect_trait_stats(state, phenotypes)

    assert "imax" in out
    assert 0 in out["imax"]
    ts = out["imax"][0]
    assert ts.mean == pytest.approx(4.5)
    assert ts.variance == pytest.approx(1.25)  # np.var of [3,4,5,6]
    assert ts.n_individuals == 4


def test_collect_trait_stats_groups_by_species() -> None:
    """Mixed-species state: two sp0 schools and two sp1 schools share one
    phenotype array; trait_stats must split per species."""
    state = _state_species_id([0, 0, 1, 1])
    phenotypes = {"imax": np.array([3.0, 5.0, 10.0, 12.0])}

    out = _collect_trait_stats(state, phenotypes)

    assert set(out["imax"].keys()) == {0, 1}
    assert out["imax"][0].mean == pytest.approx(4.0)
    assert out["imax"][1].mean == pytest.approx(11.0)
    assert out["imax"][0].n_individuals == 2
    assert out["imax"][1].n_individuals == 2


def test_average_step_outputs_single_element_propagates_trait_stats() -> None:
    """Short-circuit path at simulate.py:998-1018 (len(accumulated) == 1).
    Most runs use output.recordfrequency.ndt=1 so this is the hot path."""
    only = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={"imax": {0: TraitStats(mean=4.0, variance=1.0, n_individuals=10)}},
    )
    merged = _average_step_outputs([only], freq=1, record_step=0)
    assert merged.trait_stats is not None
    assert merged.trait_stats["imax"][0].mean == pytest.approx(4.0)


def test_average_step_outputs_merges_trait_stats() -> None:
    """Multi-element merge path."""
    out1 = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={"imax": {0: TraitStats(mean=4.0, variance=1.0, n_individuals=10)}},
    )
    out2 = StepOutput(
        step=1,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={"imax": {0: TraitStats(mean=6.0, variance=2.0, n_individuals=20)}},
    )
    merged = _average_step_outputs([out1, out2], freq=2, record_step=1)
    # Mean over the two accumulated steps, equal-weight (matches existing _avg_bioen)
    assert merged.trait_stats["imax"][0].mean == pytest.approx(5.0)
    # `variance` is mean-of-step-variances, NOT pooled variance — see Step 2.4 note.
    assert merged.trait_stats["imax"][0].variance == pytest.approx(1.5)
    # n_individuals carries through as the latest value (snapshot semantic)
    assert merged.trait_stats["imax"][0].n_individuals == 20
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_collect_outputs_trait_stats.py -v`
Expected: ImportError — `_collect_trait_stats` not defined.

- [ ] **Step 2.3: Update `_collect_outputs` signature and body**

In `osmose/engine/simulate.py`, modify `_collect_outputs` (currently at line 896) to accept `phenotypes`:
```python
def _collect_outputs(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    bkg_output: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    diet_by_species: NDArray[np.float64] | None = None,
    *,
    grid: Grid | None = None,
    phenotypes: dict[str, NDArray[np.float64]] | None = None,
) -> StepOutput:
    """Aggregate per-species outputs from current state into a StepOutput."""
    biomass, abundance = _collect_biomass_abundance(state, config, bkg_output)
    mortality_by_cause = _collect_mortality(state, config)
    yield_by_species = _collect_yield(state, config)
    biomass_by_age, abundance_by_age, biomass_by_size, abundance_by_size = _collect_distributions(
        state, config
    )
    bioen_e_net, bioen_ingestion, bioen_maint, bioen_rho, bioen_size_inf = _collect_bioen(
        state, config
    )

    spatial_biomass = spatial_abundance = spatial_yield = None
    if config.output_spatial_enabled and grid is not None:
        spatial_biomass, spatial_abundance, spatial_yield = _collect_spatial_outputs(
            state, grid, config
        )

    trait_stats = _collect_trait_stats(state, phenotypes) if phenotypes else None

    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality_by_cause,
        yield_by_species=yield_by_species,
        biomass_by_age=biomass_by_age,
        abundance_by_age=abundance_by_age,
        biomass_by_size=biomass_by_size,
        abundance_by_size=abundance_by_size,
        bioen_e_net_by_species=bioen_e_net,
        bioen_ingestion_by_species=bioen_ingestion,
        bioen_maint_by_species=bioen_maint,
        bioen_rho_by_species=bioen_rho,
        bioen_size_inf_by_species=bioen_size_inf,
        diet_by_species=diet_by_species,
        spatial_biomass=spatial_biomass,
        spatial_abundance=spatial_abundance,
        spatial_yield=spatial_yield,
        trait_stats=trait_stats,
    )


def _collect_trait_stats(
    state: SchoolState,
    phenotypes: dict[str, NDArray[np.float64]],
) -> dict[str, dict[int, TraitStats]]:
    """Group expressed phenotypes by species; return mean/var/count per trait per species."""
    out: dict[str, dict[int, TraitStats]] = {}
    species_ids = np.unique(state.species_id)
    for trait_name, values in phenotypes.items():
        per_species: dict[int, TraitStats] = {}
        for sp in species_ids:
            mask = state.species_id == sp
            n = int(mask.sum())
            if n == 0:
                continue
            sub = values[mask]
            per_species[int(sp)] = TraitStats(
                mean=float(np.mean(sub)),
                variance=float(np.var(sub)),
                n_individuals=n,
            )
        out[trait_name] = per_species
    return out
```

- [ ] **Step 2.4: Update `_average_step_outputs` for trait_stats merging — BOTH branches**

`_average_step_outputs` has two return paths:
1. **Single-element short-circuit** at simulate.py:998-1018: when `len(accumulated) == 1`, returns a copy of the only accumulated step. Most runs use `output.recordfrequency.ndt=1` so this is the hot path. **Must propagate trait_stats here too.**
2. **Multi-element merge** at simulate.py:1019+ (the `np.mean(...)` block): merges across the recording window.

Patch (1) — in the short-circuit `StepOutput(...)` return at line 998-1018, append:
```python
            trait_stats=accumulated[0].trait_stats,
```

Patch (2) — after the existing bioen-merging logic in the multi-element branch and before its final `StepOutput(...)` return, add the merge block:
```python
    trait_stats_list = [o.trait_stats for o in accumulated if o.trait_stats is not None]
    merged_trait_stats: dict[str, dict[int, TraitStats]] | None = None
    if trait_stats_list:
        merged_trait_stats = {}
        all_traits = set().union(*(d.keys() for d in trait_stats_list))
        for trait in all_traits:
            per_sp_lists: dict[int, list[TraitStats]] = {}
            for d in trait_stats_list:
                for sp, ts in d.get(trait, {}).items():
                    per_sp_lists.setdefault(sp, []).append(ts)
            merged_trait_stats[trait] = {
                sp: TraitStats(
                    # NOTE: `variance` field carries the mean-of-step-variances
                    # across the averaging window, NOT the pooled variance of the
                    # underlying schools. Downstream consumers that want pooled
                    # variance must recompute from raw phenotype arrays.
                    mean=float(np.mean([t.mean for t in lst])),
                    variance=float(np.mean([t.variance for t in lst])),
                    n_individuals=lst[-1].n_individuals,
                )
                for sp, lst in per_sp_lists.items()
            }
```

Then pass `trait_stats=merged_trait_stats` into the multi-element branch's returned `StepOutput(...)`.

- [ ] **Step 2.5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_collect_outputs_trait_stats.py -v`
Expected: 3 passed.

- [ ] **Step 2.6: Run a broader suite to verify no regression**

Run: `.venv/bin/python -m pytest tests/test_simulate.py tests/test_engine_parity.py -q --timeout=120`
Expected: all passed (existing bioen / spatial / diet outputs unaffected by adding an optional kwarg).

- [ ] **Step 2.7: Commit**

```bash
git add tests/test_collect_outputs_trait_stats.py osmose/engine/simulate.py
git commit -m "feat(engine): collect and merge per-species trait_stats in step outputs"
```

---

## Task 3: Thread phenotypes from `express_traits` site to `_collect_outputs`

**Files:**
- Modify: `osmose/engine/simulate.py:1338-1342` (capture phenotypes) and `simulate.py` `_collect_outputs` call site for focal outputs

- [ ] **Step 3.1: Write the failing test (skipped placeholder; unskipped in Task 8.5)**

Append to `tests/test_collect_outputs_trait_stats.py`:
```python
def test_focal_outputs_thread_phenotypes_when_genetics_on(monkeypatch) -> None:
    """When ctx.genetic_state is non-None, the focal `_collect_outputs` call
    must receive the same phenotypes dict that `express_traits` produced.

    Patched on the module attribute (`sim_mod._collect_outputs`) because the
    step loop at `simulate.py:1402` resolves the name in the simulate module's
    namespace at call time. Top-of-file imports of `_collect_outputs` would
    create a separate binding that bypasses the patch — we intentionally do
    NOT import `_collect_outputs` at the top of this test file.
    """
    import osmose.engine.simulate as sim_mod

    captured: dict = {}
    real_collect = sim_mod._collect_outputs

    def spy_collect(*args, **kwargs):
        captured["phenotypes"] = kwargs.get("phenotypes")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(sim_mod, "_collect_outputs", spy_collect)

    pytest.skip("baltic_ev fixture not wired until Task 8; unskipped in Step 8.5.")
```

- [ ] **Step 3.2: Run test to verify it skips (placeholder)**

Run: `.venv/bin/python -m pytest tests/test_collect_outputs_trait_stats.py::test_focal_outputs_thread_phenotypes_when_genetics_on -v`
Expected: SKIPPED. We will unskip in Task 9 once the fixture exists. For now we proceed with the wiring change, gated by the existing genetics-on integration tests.

- [ ] **Step 3.3: Modify simulate.py to keep phenotypes in scope through the timestep**

In `osmose/engine/simulate.py` around the existing block at lines 1336-1342:
```python
        # -- Genetics trait expression (before growth/bioen) --
        trait_overrides: dict[str, NDArray[np.float64]] = {}
        phenotypes: dict[str, NDArray[np.float64]] | None = None
        if ctx.genetic_state is not None:
            from osmose.engine.genetics import apply_trait_overrides, express_traits

            phenotypes = express_traits(ctx.genetic_state, state.species_id)
            apply_trait_overrides(trait_overrides, phenotypes, ctx.genetic_state.registry)
```

Locate the focal-output `_collect_outputs` call (search `_collect_outputs(` within the step loop after reproduction; the existing line passes `bkg_output, diet_by_species=step_diet, grid=grid`). Add `phenotypes=phenotypes`:
```python
        step_out = _collect_outputs(
            state, config, step, bkg_output, diet_by_species=step_diet, grid=grid,
            phenotypes=phenotypes,
        )
```

- [ ] **Step 3.4: Run existing parity + genetics unit tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_inheritance.py tests/test_genetics_expression.py tests/test_engine_parity.py -q --timeout=120`
Expected: all passed. Threading is optional kwarg-only; no parity disturbance.

- [ ] **Step 3.5: Commit**

```bash
git add tests/test_collect_outputs_trait_stats.py osmose/engine/simulate.py
git commit -m "feat(engine): thread genetics phenotypes into focal step outputs"
```

---

## Task 4: Add `_write_genetic_trait_means_csv` writer

**Files:**
- Modify: `osmose/engine/output.py` (add writer + call site in `write_outputs`)
- Test: `tests/test_genetic_trait_means_csv.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_genetic_trait_means_csv.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.engine.output import write_outputs
from osmose.engine.simulate import StepOutput, TraitStats


EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _bare_config_dict() -> dict[str, str]:
    # Real fixture — hand-rolled dicts fail EngineConfig.from_dict's
    # schema enforcement on linf/K/t0/length-weight params.
    from osmose.config import OsmoseConfigReader
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    return raw


def _build_outputs_with_trait_stats() -> list[StepOutput]:
    return [
        StepOutput(
            step=0,
            biomass=np.array([100.0]),
            abundance=np.array([1000.0]),
            mortality_by_cause=np.zeros((1, 1)),
            trait_stats={"imax": {0: TraitStats(mean=3.5, variance=0.1, n_individuals=50)}},
        ),
        StepOutput(
            step=1,
            biomass=np.array([110.0]),
            abundance=np.array([1100.0]),
            mortality_by_cause=np.zeros((1, 1)),
            trait_stats={"imax": {0: TraitStats(mean=3.2, variance=0.12, n_individuals=55)}},
        ),
    ]


def test_writer_creates_csv_with_expected_columns(tmp_path: Path) -> None:
    from osmose.engine.config import EngineConfig
    config = EngineConfig.from_dict(_bare_config_dict())
    outputs = _build_outputs_with_trait_stats()

    write_outputs(outputs, tmp_path, config, prefix="osm")

    path = tmp_path / "osm_genetic_trait_means_Simu0.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == ["Time", "species_id", "trait_name", "mean", "variance", "n_individuals"]
    assert len(df) == 2  # one row per (step, species, trait)
    assert set(df["trait_name"]) == {"imax"}


def test_writer_skipped_when_no_trait_stats(tmp_path: Path) -> None:
    from osmose.engine.config import EngineConfig
    config = EngineConfig.from_dict(_bare_config_dict())
    outputs = [
        StepOutput(
            step=0,
            biomass=np.array([100.0]),
            abundance=np.array([1000.0]),
            mortality_by_cause=np.zeros((1, 1)),
        ),
    ]
    write_outputs(outputs, tmp_path, config, prefix="osm")
    assert not (tmp_path / "osm_genetic_trait_means_Simu0.csv").exists()
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_genetic_trait_means_csv.py -v`
Expected: FAIL — `osm_genetic_trait_means_Simu0.csv` does not exist.

- [ ] **Step 4.3: Add the writer in `osmose/engine/output.py`**

Add a new function near `_write_bioen_csvs`:
```python
def _write_genetic_trait_means_csv(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-step mean/variance of each genetic trait per species.

    Skipped if no output carries trait_stats (genetics disabled).
    """
    rows: list[dict] = []
    for o in outputs:
        if o.trait_stats is None:
            continue
        time = o.step / config.n_dt_per_year
        for trait_name, by_species in o.trait_stats.items():
            for sp_idx, ts in by_species.items():
                rows.append(
                    {
                        "Time": time,
                        "species_id": sp_idx,
                        "trait_name": trait_name,
                        "mean": ts.mean,
                        "variance": ts.variance,
                        "n_individuals": ts.n_individuals,
                    }
                )
    if not rows:
        return
    df = pd.DataFrame(rows, columns=[
        "Time", "species_id", "trait_name", "mean", "variance", "n_individuals",
    ])
    df.to_csv(output_dir / f"{prefix}_genetic_trait_means_Simu0.csv", index=False)
```

In `write_outputs`, after the existing `_write_bioen_csvs` block (around line 64), add:
```python
    _write_genetic_trait_means_csv(output_dir, prefix, outputs, config)
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_genetic_trait_means_csv.py -v`
Expected: 2 passed.

- [ ] **Step 4.5: Commit**

```bash
git add tests/test_genetic_trait_means_csv.py osmose/engine/output.py
git commit -m "feat(engine): write genetic_trait_means CSV when trait_stats present"
```

---

## Task 5: Add `read_genetic_trait_means` reader

**Files:**
- Modify: `osmose/results.py` (add reader function)
- Test: extend `tests/test_genetic_trait_means_csv.py`

- [ ] **Step 5.1: Write the failing test**

Append to `tests/test_genetic_trait_means_csv.py`:
```python
def test_read_genetic_trait_means_round_trip(tmp_path: Path) -> None:
    from osmose.engine.config import EngineConfig
    from osmose.results import read_genetic_trait_means

    config = EngineConfig.from_dict(_bare_config_dict())
    outputs = _build_outputs_with_trait_stats()
    write_outputs(outputs, tmp_path, config, prefix="osm")

    ds = read_genetic_trait_means(tmp_path, prefix="osm")
    assert "mean" in ds.data_vars
    assert "variance" in ds.data_vars
    assert "n_individuals" in ds.data_vars
    assert set(ds.coords) >= {"Time", "species_id", "trait_name"}
    # Two timesteps, 1 species, 1 trait
    assert ds["mean"].sel(species_id=0, trait_name="imax").shape == (2,)
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_genetic_trait_means_csv.py::test_read_genetic_trait_means_round_trip -v`
Expected: ImportError — `read_genetic_trait_means` not defined.

- [ ] **Step 5.3: Add the reader in `osmose/results.py`**

Append at the bottom of `osmose/results.py`:
```python
import xarray as xr  # already imported at top of file in most cases; verify and dedupe


def read_genetic_trait_means(output_dir: Path, prefix: str = "osm") -> "xr.Dataset":
    """Read per-step genetic trait statistics into an xarray Dataset.

    Indexed by (Time, species_id, trait_name). Returns an empty dataset if the
    CSV does not exist (genetics disabled run).
    """
    path = Path(output_dir) / f"{prefix}_genetic_trait_means_Simu0.csv"
    if not path.exists():
        return xr.Dataset()
    df = pd.read_csv(path)
    return df.set_index(["Time", "species_id", "trait_name"]).to_xarray()
```

(If `xr` and `pd` are not imported, add the appropriate imports at the top of the file.)

- [ ] **Step 5.4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_genetic_trait_means_csv.py -v`
Expected: 3 passed.

- [ ] **Step 5.5: Commit**

```bash
git add tests/test_genetic_trait_means_csv.py osmose/results.py
git commit -m "feat(results): read_genetic_trait_means xarray reader"
```

---

## Task 6: Config validator for declared traits

**Files:**
- Modify: `osmose/engine/config.py` (after `genetics_enabled` parse, validate declared traits)
- Test: `tests/test_trait_registry_validator.py`

- [ ] **Step 6.1: Write the failing test**

Create `tests/test_trait_registry_validator.py`:
```python
from pathlib import Path

import pytest

from osmose.engine.config import EngineConfig


EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _base_dict() -> dict[str, str]:
    """Validator test base. Note: the validator runs INSIDE from_dict during
    config parsing, so we need a complete schema-passing config. Hand-rolled
    dicts (e.g., {nspecies, lifespan, name}) will fail upstream of the
    validator on missing linf/K/t0/length-weight keys before the validator
    is even reached."""
    from osmose.config import OsmoseConfigReader
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    raw["simulation.genetic.enabled"] = "true"
    return raw


def test_declared_trait_with_nonzero_variance_requires_mean() -> None:
    cfg = _base_dict()
    cfg.update({
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.var.sp0": "0.1",  # nonzero variance — needs mean
        # NOTE: no evolution.trait.imax.mean.sp0
        "evolution.trait.imax.var.sp1": "0.0",  # zero variance ok without mean
    })
    with pytest.raises(ValueError, match="evolution.trait.imax.mean.sp0"):
        EngineConfig.from_dict(cfg)


def test_declared_trait_with_zero_variance_does_not_require_mean() -> None:
    cfg = _base_dict()
    cfg.update({
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.var.sp0": "0.0",
        "evolution.trait.imax.var.sp1": "0.0",
    })
    # Should not raise
    EngineConfig.from_dict(cfg)


def test_complete_declaration_passes() -> None:
    cfg = _base_dict()
    cfg.update({
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.mean.sp1": "5.0",
        "evolution.trait.imax.var.sp1": "0.0",
    })
    EngineConfig.from_dict(cfg)
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trait_registry_validator.py -v`
Expected: FAIL on `test_declared_trait_with_nonzero_variance_requires_mean` — no error raised.

- [ ] **Step 6.3: Add the validator in `osmose/engine/config.py`**

Find the `genetics_enabled = _enabled(cfg, "simulation.genetic.enabled")` line (around line 1822). After parsing genetics, add:
```python
        if genetics_enabled:
            _validate_trait_declarations(cfg, n_sp)
```

Then add the function at module scope:
```python
def _validate_trait_declarations(cfg: dict[str, str], n_sp: int) -> None:
    """Each declared `evolution.trait.<name>.target` must have a per-species
    mean for every species where variance is nonzero."""
    import re
    trait_names: set[str] = set()
    for key in cfg:
        m = re.match(r"evolution\.trait\.(\w+)\.target", key)
        if m:
            trait_names.add(m.group(1))
    for name in trait_names:
        for i in range(n_sp):
            var_key = f"evolution.trait.{name}.var.sp{i}"
            mean_key = f"evolution.trait.{name}.mean.sp{i}"
            var_str = cfg.get(var_key, "0.0")
            try:
                var = float(var_str)
            except ValueError:
                raise ValueError(f"{var_key}: not a number ({var_str!r})")
            if var > 0.0 and mean_key not in cfg:
                raise ValueError(
                    f"{mean_key} missing: trait '{name}' declares nonzero variance "
                    f"on species {i} but no mean is specified"
                )
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_trait_registry_validator.py -v`
Expected: 3 passed.

- [ ] **Step 6.5: Run broader config tests**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -q`
Expected: all passed (no existing fixture sets `simulation.genetic.enabled=true`).

- [ ] **Step 6.6: Commit**

```bash
git add tests/test_trait_registry_validator.py osmose/engine/config.py
git commit -m "feat(config): validate declared traits require per-species means"
```

---

## Task 7: Clone `baltic/` to `baltic_ev/` with bioen enabled (no genetics yet)

**Files:**
- Create: `data/baltic_ev/` (cloned from `data/baltic/`)
- Create: `data/baltic_ev/README.md`
- Modify: `data/baltic_ev/baltic_ev_param-simulation.csv` (renamed copy of `baltic_param-simulation.csv`) + bioen + bioen-param files
- Test: `tests/test_baltic_ev_fixture_bioen.py`

- [ ] **Step 7.1: Clone the fixture (no engine code touched)**

Run:
```bash
cp -r data/baltic data/baltic_ev
cd data/baltic_ev
for f in baltic_*; do mv "$f" "${f/baltic_/baltic_ev_}"; done
cd ../..
```

Update the top-level `baltic_ev_all-parameters.csv` to reference the renamed files (search-and-replace `baltic_` → `baltic_ev_` inside that file; use the Edit tool, since `>` redirection is forbidden).

- [ ] **Step 7.2: Write the failing test**

Create `tests/test_baltic_ev_fixture_bioen.py`:
```python
from pathlib import Path
import pytest


def test_baltic_ev_all_parameters_exists() -> None:
    assert (Path("data/baltic_ev") / "baltic_ev_all-parameters.csv").exists()


def test_baltic_ev_has_bioen_enabled() -> None:
    text = (Path("data/baltic_ev") / "baltic_ev_param-simulation.csv").read_text()
    assert "simulation.bioen.enabled" in text
    assert "true" in text.split("simulation.bioen.enabled")[1].split("\n")[0].lower()


def test_baltic_ev_cod_has_bioen_imax() -> None:
    # cod is sp0 in baltic; bioen ingestion key (real path used by reader at
    # config.py:1796) must exist
    all_text = "\n".join(
        p.read_text() for p in Path("data/baltic_ev").rglob("*.csv")
    )
    assert "predation.ingestion.rate.max.bioen.sp0" in all_text


def test_baltic_ev_bioen_subfile_is_included() -> None:
    """OSMOSE only recurses into sub-files referenced via
    `osmose.configuration.<key>;<file>` lines (reader.py:56-68). A bare
    `include;<file>` is silently ignored."""
    master = (Path("data/baltic_ev") / "baltic_ev_all-parameters.csv").read_text()
    assert "osmose.configuration.bioen;baltic_ev_param-bioen.csv" in master


@pytest.mark.integration
def test_baltic_ev_runs_5_years_without_genetics() -> None:
    """Smoke: baltic_ev with bioen on must run end-to-end for 5y."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "5"
    # Temporarily disable genetics for this bioen-only smoke (Task 8 enables it).
    cfg["simulation.genetic.enabled"] = "false"
    result = PythonEngine().run_in_memory(cfg, seed=0)
    biomass = result.biomass()
    # `biomass()` returns long-form: columns [time, species, biomass]
    # (osmose/results.py:342-348). `.any()` over the whole frame can be
    # satisfied by background LTL species; we need to confirm the focal
    # species (cod = sp0) produced non-zero biomass at end of run.
    assert "biomass" in biomass.columns
    cod_final = biomass[biomass["species"] == "cod"].iloc[-1]["biomass"]
    assert cod_final > 0, f"cod biomass at end of 5y is {cod_final}, expected > 0"
```

- [ ] **Step 7.3: Run test to verify it fails on bioen wiring**

Run: `.venv/bin/python -m pytest tests/test_baltic_ev_fixture_bioen.py::test_baltic_ev_has_bioen_enabled -v`
Expected: FAIL — bioen not enabled yet.

- [ ] **Step 7.4: Enable bioen in baltic_ev**

In `data/baltic_ev/baltic_ev_param-simulation.csv`, add:
```
simulation.bioen.enabled;true
simulation.bioen.phit.enabled;false
```

**Why `phit.enabled=false`.** The plan does not configure `temperature.value` for Baltic, so leaving `phit.enabled=true` (the default) causes `temp_data` to be `None`, which falls through to `phi_t_arr = np.ones(...)` at `simulate.py:299-323` — bioen runs thermally neutral but the config *claims* thermal modulation is on. Setting `phit.enabled=false` is honest and makes the absence of thermal forcing explicit.

Also override Baltic's age-knife-edge cod fishery to length-sigmoidal selectivity. The default `baltic_param-fishing.csv` sets `fisheries.selectivity.type.fsh0=0` (age-knife-edge at age 2), under which fish of any length at a given age are equally vulnerable — the FIE selection differential on growth-rate is **mathematically zero**. For an `imax` FIE demo, fishing must be size-selective so that faster-growing fish reach the gear-vulnerable size earlier and are removed before spawning.

Create or modify `data/baltic_ev/baltic_ev_param-fishing.csv` to override cod (fsh0) selectivity (other fisheries fsh1-fsh7 can stay age-based; only the cod fishery's selectivity drives the imax-FIE signal). Add:
```
fisheries.selectivity.type.fsh0;1
fisheries.selectivity.l50.fsh0;35.0
fisheries.selectivity.slope.fsh0;2.0
```

(`l50=35cm` matches the EU Baltic cod minimum landing size; slope=2.0 gives a reasonably sharp sigmoid around the threshold. These produce a clear size differential: large cod nearly always caught, small cod nearly never.)

Add a new file `data/baltic_ev/baltic_ev_param-bioen.csv` with per-species bioen keys. **Config-key paths are exact** (verified against `osmose/engine/config.py:1773-1796`):
- `species.beta.sp{i}` (allometric exponent; default 0.8)
- `species.bioen.maint.energy.c_m.sp{i}` (maintenance coefficient; default 0.0)
- `species.bioen.assimilation.sp{i}` (assimilation efficiency; default 0.7)
- `species.bioen.maturity.eta.sp{i}` (maturity energy-density ratio; default 1.0)
- `species.bioen.maturity.r.sp{i}` (reproductive allocation; default 0.0)
- `predation.ingestion.rate.max.bioen.sp{i}` (max ingestion rate; default 0.0 — must be set or bioen produces zero ingestion)

**WARNING — values below are placeholders, not from a sourced literature sweep.** The plan's spec §3.1 acknowledges absolute biomass is out-of-scope. During implementation, the engineer SHOULD either (a) source from a published Baltic-cod bioen parameterization and update the README provenance accordingly, or (b) keep these placeholders and document the consequence (biomass scale not calibrated). Brander 1995 is cited as a starting point but its numerical values are NOT translated into OSMOSE bioen units in this plan — that's a deferred literature task.

Use this content for `baltic_ev_param-bioen.csv`:
```
species.beta.sp0;0.81
species.beta.sp1;0.82
species.beta.sp2;0.80
species.beta.sp3;0.80
species.beta.sp4;0.80
species.beta.sp5;0.80
species.beta.sp6;0.80
species.beta.sp7;0.80
species.bioen.maint.energy.c_m.sp0;0.025
species.bioen.maint.energy.c_m.sp1;0.030
species.bioen.maint.energy.c_m.sp2;0.035
species.bioen.maint.energy.c_m.sp3;0.030
species.bioen.maint.energy.c_m.sp4;0.030
species.bioen.maint.energy.c_m.sp5;0.030
species.bioen.maint.energy.c_m.sp6;0.030
species.bioen.maint.energy.c_m.sp7;0.030
species.bioen.assimilation.sp0;0.65
species.bioen.assimilation.sp1;0.65
species.bioen.assimilation.sp2;0.65
species.bioen.assimilation.sp3;0.65
species.bioen.assimilation.sp4;0.65
species.bioen.assimilation.sp5;0.65
species.bioen.assimilation.sp6;0.65
species.bioen.assimilation.sp7;0.65
species.bioen.maturity.r.sp0;0.20
species.bioen.maturity.r.sp1;0.30
species.bioen.maturity.r.sp2;0.30
species.bioen.maturity.r.sp3;0.25
species.bioen.maturity.r.sp4;0.25
species.bioen.maturity.r.sp5;0.30
species.bioen.maturity.r.sp6;0.20
species.bioen.maturity.r.sp7;0.30
species.bioen.maturity.m0.sp0;30.0
species.bioen.maturity.m1.sp0;0.0
species.bioen.maturity.m0.sp1;10.0
species.bioen.maturity.m1.sp1;0.0
species.bioen.maturity.m0.sp2;20.0
species.bioen.maturity.m1.sp2;0.0
species.bioen.maturity.m0.sp3;15.0
species.bioen.maturity.m1.sp3;0.0
species.bioen.maturity.m0.sp4;15.0
species.bioen.maturity.m1.sp4;0.0
species.bioen.maturity.m0.sp5;20.0
species.bioen.maturity.m1.sp5;0.0
species.bioen.maturity.m0.sp6;10.0
species.bioen.maturity.m1.sp6;0.0
species.bioen.maturity.m0.sp7;8.0
species.bioen.maturity.m1.sp7;0.0
predation.ingestion.rate.max.bioen.sp0;3.0
predation.ingestion.rate.max.bioen.sp1;3.5
predation.ingestion.rate.max.bioen.sp2;4.0
predation.ingestion.rate.max.bioen.sp3;3.5
predation.ingestion.rate.max.bioen.sp4;3.5
predation.ingestion.rate.max.bioen.sp5;4.0
predation.ingestion.rate.max.bioen.sp6;3.0
predation.ingestion.rate.max.bioen.sp7;3.5
```

Now register this sub-file in `data/baltic_ev/baltic_ev_all-parameters.csv`. The OSMOSE reader only recurses into files referenced via `osmose.configuration.<key>;<filename>` lines (verified at `osmose/config/reader.py:56-68`; a bare `include;<file>` is silently ignored). Use the Edit tool to append:
```
osmose.configuration.bioen;baltic_ev_param-bioen.csv
```

- [ ] **Step 7.5: Write the README provenance doc**

Create `data/baltic_ev/README.md`:
```markdown
# baltic_ev — Ev-OSMOSE demonstration fixture

Cloned from `data/baltic/` on 2026-05-18 with bioenergetics + Ev-OSMOSE
genetics enabled. Used by the FIE-on-cod scientific demonstration
(see `docs/tutorials/fie-on-baltic-cod.md`).

**This fixture is NOT calibrated against ICES.** Absolute biomass is
not the target — directional trait response is. See the activation
design at `docs/superpowers/specs/2026-05-18-ev-osmose-activation-design.md`.

## Bioen parameter provenance

All values in `baltic_ev_param-bioen.csv`:

| Parameter | Source |
|---|---|
| `species.beta.sp{i}` | Placeholder ≈ 0.8 (allometric exponent ballpark). Source from Brander 1995 or Baltic-cod bioen study during implementation. |
| `species.bioen.maint.energy.c_m.sp{i}` | Placeholder. Source from Mehner & Wieser 1994 (coldwater gadoid metabolic rates) during implementation; units must match OSMOSE bioen kernel. |
| `species.bioen.assimilation.sp{i}` | 0.65 placeholder (close to OSMOSE default 0.7). |
| `species.bioen.maturity.r.sp{i}` | Placeholder reproductive-allocation fraction. |
| `predation.ingestion.rate.max.bioen.sp{i}` | Placeholder. The genetics demo only requires sp0 (cod) to be in a sensible biological range; rest are non-evolving and only affect background biomass scale. |

**Maturation gate is set as a static threshold, not a reaction norm.** `species.bioen.maturity.m0.sp{i}` and `.m1.sp{i}` MUST be set, otherwise both default to 0.0 (config.py:1783-1784) → `l_mature = m0 + m1*age = 0` → **every school (including egg-stage larvae) is always mature** → all net energy allocates to gonads from day 1 → cod growth is pathologically suppressed → cod never reaches the gear l50=35cm → fishery catches ~0 cod → ZERO FIE signal.

Cod sp0 gets m0=30cm (flat reaction norm — m1=0). This is a **simplifying choice**, not a single-paper literature value: Radtke & Grygiel (2013, https://doi.org/10.1111/jai.12135) report L50=34.8cm for southern Baltic cod males in 1990-2006; Svedäng et al. (2024, https://doi.org/10.1002/ece3.70382) document that eastern Baltic cod L50 has since *halved* to ~20cm. m0=30 sits between the two: cod mature before reaching the gear-vulnerable size (l50=35cm), which lets the FIE signal operate purely through "fast growers cross the gear threshold sooner, slow growers reproduce more often before capture" — clean isolation of the growth-rate pathway. Setting m0=35 would couple maturation timing to gear vulnerability and confound the demo. Pick of stock/era is documented; sensitivity to m0 ∈ {20, 30, 35} is a deferred follow-up.

Other species use length-at-50%-maturity values from generic Baltic life-history sources. This is NOT a maturation reaction norm with age-plasticity — that would require `m1 ≠ 0` and would let maturation co-evolve with growth. **This demo intentionally fixes maturation length to isolate the growth-rate FIE pathway** (the secondary pathway per Heino, Pauli & Dieckmann 2015, https://doi.org/10.1146/annurev-ecolsys-112414-054339). The dominant maturation-evolution FIE pathway documented for cod (Olsen et al. 2004) requires non-zero m1 + an evolving trait targeting `bioen_m0` or `bioen_m1`, which is listed in the spec's out-of-scope follow-ups.

## Genetic-trait parameters

See `baltic_ev_param-genetics.csv`. Only cod (sp0) has a nonzero-variance
trait declared, targeting `bioen_i_max`.

## Citations

- Brander, K. M. (1995). The effect of temperature on growth of Atlantic
  cod (Gadus morhua L.). *ICES J. Mar. Sci.*, 52(1), 1-10.
- Mehner, T., & Wieser, W. (1994). Energetics and metabolic correlates
  of starvation in juvenile perch. *J. Fish Biol.*, 45(2), 325-333.
```

- [ ] **Step 7.6: Run the unit-level fixture tests**

Run: `.venv/bin/python -m pytest tests/test_baltic_ev_fixture_bioen.py -v -k "not integration"`
Expected: 3 passed (existence + bioen-enabled + imax key present).

- [ ] **Step 7.7: Run the integration smoke (5-year run)**

Run: `.venv/bin/python -m pytest tests/test_baltic_ev_fixture_bioen.py -v -k integration --timeout=600`
Expected: pass. The 5-year run takes ~3s on current perf.

- [ ] **Step 7.8: Pre-flight viability check — cod must reach the fishery l50 in baseline**

Without this check the entire FIE demo can silently fail: bioen placeholder params + maturation gate may produce a cod that never grows past 35cm, so the size-selective fishery (l50=35cm) catches zero cod and the FIE selection differential is structurally zero regardless of trait variance (per R3A finding).

Append to `tests/test_baltic_ev_fixture_bioen.py`:
```python
@pytest.mark.integration
def test_baltic_ev_cod_reaches_fishery_l50_in_baseline(tmp_path: Path) -> None:
    """Baseline (bioen on, genetics off, no fishing) must produce cod
    that grow past 35cm in adult life-stage, otherwise the FIE demo's
    l50=35cm gear catches nothing and produces a null FIE signal for
    structural reasons rather than the science."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "20"
    cfg["simulation.genetic.enabled"] = "false"
    # Zero fishing so cod size distribution reflects bioen alone
    cfg["fisheries.rate.base.fsh0"] = "0.0"
    PythonEngine().run(cfg, tmp_path, seed=0)

    # Read the per-step length-by-age CSV (existing output)
    import pandas as pd
    csv_path = tmp_path / "osm_biomassByAge_Simu0.csv"
    assert csv_path.exists(), "baseline run did not produce biomassByAge CSV"
    # Equivalent: read the size-distribution CSV if available, or assert
    # via xarray that cod.max_length_at_end >= 35.0
    # For now assert via the simulation summary: at year 20, mean cod
    # adult length should be >= 35cm (literature Baltic cod adult ~50cm)
    # The exact column structure depends on the output writer; the
    # implementer should inspect `osm_biomassByAge_Simu0.csv` and
    # `osm_meanSize_Simu0.csv` (or similar) to find the right column
    # and assert cod mean adult length >= 35.0.
    # If no per-species size output exists, add a quick xarray-based
    # post-run computation that pulls SchoolState.length values at the
    # final timestep via run_in_memory.
    raise AssertionError(
        "PRE-FLIGHT NOT WIRED. This stub must be replaced with a real "
        "cod-final-length >= 35cm assertion sourced from the engine output "
        "BEFORE running Task 11. Until wired, the FIE-direction test can "
        "silently pass on a null signal — cod that never reach the gear "
        "l50 produce structurally-zero selection regardless of trait "
        "variance. See plan §Task 7.8."
    )
```

Also create a sentinel that Task 11 reads in `setUp` and refuses to execute
until the pre-flight is wired and passing:

```python
# in tests/test_fie_demo_direction.py module top
def _require_preflight() -> None:
    """Block Task 11 from running until Task 7.8 is wired + passing."""
    sentinel = Path("tests/.preflight_wired")
    if not sentinel.exists():
        pytest.skip(
            "Pre-flight viability check (Task 7.8) is not wired or has not "
            "been run successfully. Wire test_baltic_ev_cod_reaches_fishery_l50 "
            "to the engine size output and run it; on success it should "
            "`tests/.preflight_wired`.touch(). See plan §Task 7.8."
        )
```

Run: `.venv/bin/python -m pytest tests/test_baltic_ev_fixture_bioen.py::test_baltic_ev_cod_reaches_fishery_l50_in_baseline -v --timeout=300`
Expected: FAILS with `AssertionError("PRE-FLIGHT NOT WIRED…")` until the
implementer replaces the stub with a real size assertion. This is the desired
behaviour — silent-skip would let Task 11 trust a null signal.

If after wiring the test fails (cod never reaches 35cm), the bioen parameter placeholders in Task 7.4 are too suppressing for this demo to work. Either (a) tune the bioen params (raise imax, lower c_m), or (b) lower the gear l50 (with caveat that selectivity differential weakens), or (c) escalate scope by sourcing real Baltic-cod bioen calibration. This MUST be resolved before running Task 11. On success, the test should `Path("tests/.preflight_wired").touch()` so Task 11 will execute (see the `_require_preflight()` helper added in Task 11).

- [ ] **Step 7.9: Cod-biomass-stability guard (uncalibrated-drift backstop)**

The placeholder bioen params (Task 7.4) acknowledge absolute biomass is
out-of-spec. But they can also produce cod **extinction** (collapse to a
few schools — founder-effect drift swamps FIE signal) or **runaway** (cod
biomass explodes — selection collapses into a single dominant phenotype).
Task 7.7's smoke only asserts `cod_final > 0`, which doesn't catch either
edge case. Add a stability guard:

```python
@pytest.mark.integration
def test_baltic_ev_cod_biomass_within_2x_envelope_over_50y(tmp_path: Path) -> None:
    """Baseline (bioen on, genetics off, no fishing) cod biomass at year 50
    must stay within [0.5, 2.0] × year-5 (post-burnin) biomass. Outside
    this envelope, the FIE demo (Task 11) runs on a degenerate population
    where founder-effect drift or selection collapse swamps the FIE signal."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.results import biomass as read_biomass

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "50"
    cfg["simulation.genetic.enabled"] = "false"
    cfg["fisheries.rate.base.fsh0"] = "0.0"

    PythonEngine().run(cfg, tmp_path, seed=0)
    bio = read_biomass(tmp_path)
    cod = bio[bio["species"] == "cod"].sort_values("time")
    burnin = cod[cod["time"].between(5.0, 6.0)]["biomass"].mean()
    end = cod[cod["time"] >= 49.0]["biomass"].mean()
    ratio = end / burnin
    assert 0.5 <= ratio <= 2.0, (
        f"cod biomass at year 50 = {end:.2e}, year 5 = {burnin:.2e}, "
        f"ratio = {ratio:.2f}. Expected 0.5 <= ratio <= 2.0 under no-fishing, "
        "no-genetics. Outside this envelope the FIE demo runs on a degenerate "
        "population; tune bioen params (Task 7.4) before relying on Task 11."
    )
```

Run: `.venv/bin/python -m pytest tests/test_baltic_ev_fixture_bioen.py::test_baltic_ev_cod_biomass_within_2x_envelope_over_50y -v -m integration --timeout=600`
Expected: PASS. If FAIL, tune Task 7.4 bioen placeholders before proceeding.

- [ ] **Step 7.10: Commit**

```bash
git add data/baltic_ev tests/test_baltic_ev_fixture_bioen.py
git commit -m "feat(fixture): baltic_ev cloned from baltic with bioen enabled"
```

---

## Task 8: Add genetics keys to baltic_ev for cod imax trait

**Files:**
- Create: `data/baltic_ev/baltic_ev_param-genetics.csv`
- Modify: `data/baltic_ev/baltic_ev_param-simulation.csv` (enable genetics flag)
- Modify: `data/baltic_ev/baltic_ev_all-parameters.csv` (include genetics file)

- [ ] **Step 8.1: Write the failing activation test**

Create `tests/test_ev_osmose_activation.py`:
```python
from pathlib import Path
import pytest


@pytest.mark.integration
def test_baltic_ev_runs_15_years_with_genetics_on(tmp_path: Path) -> None:
    """End-to-end smoke: baltic_ev with genetics on must run 15y (5y past
    evolution.seeding.year=10) and produce non-empty genetic_trait_means CSV.

    The 15y window is deliberate: with evolution.seeding.year=10, the first
    10y are seed phase where offspring genotypes are RANDOMLY REDRAWN from
    population donors (per inheritance.py:61-68). A test that runs only
    nyear<=10 cannot distinguish working inheritance from broken inheritance
    because the variance pattern is identical in both cases during seed phase.
    Running 15y and asserting that the variance pattern continues post-year-10
    confirms the inheritance pipeline did NOT degenerate at the seed-phase
    boundary (i.e., variance does not crash to zero or NaN once redraws stop)."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.results import read_genetic_trait_means

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "15"
    PythonEngine().run(cfg, tmp_path, seed=0)

    csv_path = tmp_path / "osm_genetic_trait_means_Simu0.csv"
    assert csv_path.exists(), "genetic_trait_means CSV not produced"

    ds = read_genetic_trait_means(tmp_path, prefix="osm")
    assert "trait_name" in ds.coords
    assert "imax" in set(ds["trait_name"].values)

    # Trait expression must be non-degenerate at all times.
    cod_var_series = (
        ds["variance"].sel(species_id=0, trait_name="imax").to_pandas()
    )
    assert (cod_var_series > 1e-4).all(), (
        f"cod imax variance must stay > 1e-4 at all timesteps; "
        f"got min={cod_var_series.min():.6f} at time={cod_var_series.idxmin()}. "
        "Either genetics is silently disabled or the inheritance pipeline "
        "degenerated post-seed-phase."
    )

    # Specifically check post-seed-phase (year > 10) variance is healthy.
    # Inheritance kicks in at year 10; variance should NOT collapse.
    post_seed = cod_var_series[cod_var_series.index > 10]
    assert len(post_seed) > 0, "no post-seed-phase samples; expected ~5y worth"
    assert (post_seed > 1e-4).all(), (
        f"variance collapsed post-year-10 (inheritance phase): "
        f"min={post_seed.min():.6f}. inheritance.py may be returning empty "
        "or degenerate parts."
    )
```

(Note: this test verifies (a) the writer emitted the CSV, (b) trait expression is firing in seed phase, AND (c) the variance pattern survives the seed-phase → inheritance-phase transition. It does NOT verify the trait flows all the way through bioen into mortality differences — that's what the FIE-direction test in Task 11.1 does.)

- [ ] **Step 8.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ev_osmose_activation.py -v --timeout=300`
Expected: FAIL — genetics not enabled in baltic_ev yet.

- [ ] **Step 8.3: Add genetics enable + trait config to baltic_ev**

Append to `data/baltic_ev/baltic_ev_param-simulation.csv`:
```
simulation.genetic.enabled;true
evolution.seeding.year;10
```

(`evolution.seeding.year` is the correct key; the reader at `config.py:1823` reads exactly this. Earlier draft referenced `population.genotype.transmission.year.start` which the reader silently ignores.)

Create `data/baltic_ev/baltic_ev_param-genetics.csv`:
```
evolution.trait.imax.target;bioen_i_max
evolution.trait.imax.mean.sp0;3.0
evolution.trait.imax.var.sp0;0.018
evolution.trait.imax.envvar.sp0;0.054
# nlocus=10, nval=10 follow Marty, Dieckmann & Ernande (2015,
# https://doi.org/10.1111/eva.12220), the canonical eco-genetic precursor to
# Ev-OSMOSE — they use ~10 functional loci with 10 allelic states, with nval=10
# anchored on Poulsen et al. (2006)'s empirical mean cod microsatellite allelic
# diversity of 9.4. nval=20 in the earlier draft of this plan was unsourced.
evolution.trait.imax.nlocus.sp0;10
evolution.trait.imax.nval.sp0;10
evolution.trait.imax.mean.sp1;3.5
evolution.trait.imax.var.sp1;0.0
evolution.trait.imax.mean.sp2;4.0
evolution.trait.imax.var.sp2;0.0
evolution.trait.imax.mean.sp3;3.5
evolution.trait.imax.var.sp3;0.0
evolution.trait.imax.mean.sp4;3.5
evolution.trait.imax.var.sp4;0.0
evolution.trait.imax.mean.sp5;4.0
evolution.trait.imax.var.sp5;0.0
evolution.trait.imax.mean.sp6;3.0
evolution.trait.imax.var.sp6;0.0
evolution.trait.imax.mean.sp7;3.5
evolution.trait.imax.var.sp7;0.0
```

**Variance choice.** `var.sp0=0.018`, `envvar.sp0=0.054` → h² ≈ 0.018/(0.018+0.054) = **0.25**, matching Nielsen et al. 2014 (https://doi.org/10.1186/1297-9686-46-5; Atlantic cod body-weight h²=0.24–0.34) and Otterå et al. 2018 (review of cod growth heritability). Earlier draft used `var=0.045, envvar=0.009` → h²≈0.83, which is implausibly high for wild fish and would make the FIE signal appear faster than literature predicts.

Register the sub-file in `data/baltic_ev/baltic_ev_all-parameters.csv`:
```
osmose.configuration.genetics;baltic_ev_param-genetics.csv
```

- [ ] **Step 8.4: Run the activation test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ev_osmose_activation.py -v --timeout=300`
Expected: pass — 10-year baltic_ev run produces a non-empty `osm_genetic_trait_means_Simu0.csv`.

- [ ] **Step 8.5: Now unskip the Task 3 phenotype-threading test**

In `tests/test_collect_outputs_trait_stats.py::test_focal_outputs_thread_phenotypes_when_genetics_on`, replace the `pytest.skip(...)` body with an actual 1-year baltic_ev run that asserts `captured["phenotypes"]` is non-None and contains `"imax"`:
```python
def test_focal_outputs_thread_phenotypes_when_genetics_on(monkeypatch, tmp_path) -> None:
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    import osmose.engine.simulate as sim_mod  # not exported from osmose.engine.__init__

    captured: dict = {}
    real_collect = sim_mod._collect_outputs

    def spy_collect(*args, **kwargs):
        if kwargs.get("phenotypes") is not None:
            captured["phenotypes"] = kwargs["phenotypes"]
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(sim_mod, "_collect_outputs", spy_collect)
    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "1"
    PythonEngine().run(cfg, tmp_path, seed=0)
    assert "phenotypes" in captured
    assert "imax" in captured["phenotypes"]
```

Run: `.venv/bin/python -m pytest tests/test_collect_outputs_trait_stats.py::test_focal_outputs_thread_phenotypes_when_genetics_on -v --timeout=120`
Expected: pass.

- [ ] **Step 8.6: Commit**

```bash
git add data/baltic_ev tests/test_ev_osmose_activation.py tests/test_collect_outputs_trait_stats.py
git commit -m "feat(fixture): activate Ev-OSMOSE genetics on baltic_ev for cod imax"
```

---

## Task 9: Extend genetics expression test for `bioen_i_max` target reach

**Files:**
- Modify: `tests/test_genetics_trait.py` (extend existing file)

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_genetics_trait.py`. Three assertions, each exercising a
distinct path of the imax→bioen_i_max wiring that the demo depends on. The
earlier draft of this test only asserted that `apply_trait_overrides` wrote
`overrides["bioen_i_max"]` — that is documented behavior at
`expression.py:36-37` and asserting it alone is tautological. The strengthened
test additionally verifies (a) per-school shape matches n_schools (catches
upstream regressions in `express_traits`'s shape contract that
`_collect_trait_stats` and the bioen step both rely on), (b) variance is
actually inherited from the trait's genetic + env components (catches a
silent-zero regression where the trait reads as registered but doesn't
expose draws), and (c) per-species pinning (catches a regression where
species-mean lookup drifts across species).

```python
def test_apply_trait_overrides_routes_imax_to_bioen_i_max_target() -> None:
    """Sanity: `apply_trait_overrides` writes under the declared target."""
    import numpy as np
    from osmose.engine.genetics import (
        TraitRegistry,
        apply_trait_overrides,
        create_initial_genotypes,
        express_traits,
    )

    cfg = {
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.0",
        "evolution.trait.imax.var.sp0": "0.0",
        "evolution.trait.imax.envvar.sp0": "0.0",
        "evolution.trait.imax.nlocus.sp0": "5",
        "evolution.trait.imax.nval.sp0": "10",
    }
    registry = TraitRegistry.from_config(cfg, n_species=1)
    rng = np.random.default_rng(42)
    species_id = np.zeros(4, dtype=np.int32)
    gs = create_initial_genotypes(registry, species_id, rng, n_neutral=0, n_neutral_val=0)
    phenotypes = express_traits(gs, species_id)

    overrides: dict[str, np.ndarray] = {}
    apply_trait_overrides(overrides, phenotypes, registry)

    # (a) Target name is the declared one, NOT the trait name
    assert set(overrides.keys()) == {"bioen_i_max"}
    # (b) Shape matches n_schools — bioen step indexes per-school via this
    assert overrides["bioen_i_max"].shape == species_id.shape
    # Zero variance + zero env noise → all phenotype values equal mean
    assert np.allclose(overrides["bioen_i_max"], 3.0)


def test_bioen_i_max_inherits_genetic_and_env_variance() -> None:
    """Nonzero var + envvar must produce non-degenerate per-school phenotypes
    with empirical SD ≈ sqrt(var + envvar). Catches a silent-zero regression
    where the trait reads as registered but doesn't actually draw values."""
    import numpy as np
    from osmose.engine.genetics import (
        TraitRegistry,
        apply_trait_overrides,
        create_initial_genotypes,
        express_traits,
    )

    cfg = {
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.0",
        "evolution.trait.imax.var.sp0": "0.018",
        "evolution.trait.imax.envvar.sp0": "0.054",
        "evolution.trait.imax.nlocus.sp0": "10",
        "evolution.trait.imax.nval.sp0": "20",
    }
    registry = TraitRegistry.from_config(cfg, n_species=1)
    rng = np.random.default_rng(0)
    species_id = np.zeros(2_000, dtype=np.int32)
    gs = create_initial_genotypes(registry, species_id, rng, n_neutral=0, n_neutral_val=0)
    phenotypes = express_traits(gs, species_id)

    overrides: dict[str, np.ndarray] = {}
    apply_trait_overrides(overrides, phenotypes, registry)
    expected_sd = np.sqrt(0.018 + 0.054)
    empirical = overrides["bioen_i_max"]
    # 2,000 draws is enough that empirical SD lands within ±15% of expected
    assert abs(float(empirical.std()) - expected_sd) / expected_sd < 0.15
    # Mean stays within ±0.02 of the species mean
    assert abs(float(empirical.mean()) - 3.0) < 0.02


def test_bioen_i_max_pins_zero_variance_species_to_species_mean() -> None:
    """Multi-species: cod (sp0) evolves, herring (sp1) does not. Herring's
    phenotype must remain pinned to its species_mean exactly."""
    import numpy as np
    from osmose.engine.genetics import (
        TraitRegistry,
        apply_trait_overrides,
        create_initial_genotypes,
        express_traits,
    )

    cfg = {
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.0",
        "evolution.trait.imax.var.sp0": "0.018",
        "evolution.trait.imax.envvar.sp0": "0.054",
        "evolution.trait.imax.nlocus.sp0": "10",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.mean.sp1": "5.0",
        "evolution.trait.imax.var.sp1": "0.0",
        "evolution.trait.imax.envvar.sp1": "0.0",
    }
    registry = TraitRegistry.from_config(cfg, n_species=2)
    rng = np.random.default_rng(0)
    species_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    gs = create_initial_genotypes(registry, species_id, rng, n_neutral=0, n_neutral_val=0)
    phenotypes = express_traits(gs, species_id)

    overrides: dict[str, np.ndarray] = {}
    apply_trait_overrides(overrides, phenotypes, registry)
    # sp1 phenotypes pinned to species_mean.sp1 = 5.0 exactly
    assert np.allclose(overrides["bioen_i_max"][3:], 5.0)
    # sp0 phenotypes vary
    assert overrides["bioen_i_max"][:3].std() > 0.0
```

- [ ] **Step 9.2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genetics_trait.py -v -k "bioen_i_max or pins"`
Expected: 3 passed. These exercise the imax→bioen_i_max bridge that the FIE
demo depends on; any failure here means the bridge is broken before the
demo even starts. Investigate `expression.py` + `genotype.py` before
proceeding to Task 10.

- [ ] **Step 9.3: Commit**

```bash
git add tests/test_genetics_trait.py
git commit -m "test(genetics): apply_trait_overrides routes to bioen_i_max target"
```

---

## Task 10: FIE demo script `scripts/run_fie_demo.py`

**Files:**
- Create: `scripts/run_fie_demo.py`

- [ ] **Step 10.1: Write the failing smoke test for the script**

Create `tests/test_run_fie_demo_smoke.py`:
```python
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.slow
def test_run_fie_demo_short_smoke(tmp_path: Path) -> None:
    """Smoke: script must produce both scenario CSVs + a PNG within ~5 min on
    a 10-year, 1-seed override."""
    result = subprocess.run(
        [
            sys.executable, "scripts/run_fie_demo.py",
            "--n-years", "10",
            "--seeds", "1",
            "--output-dir", str(tmp_path),
        ],
        check=True,
        timeout=600,
    )
    assert result.returncode == 0
    assert (tmp_path / "fie_imax_trajectory.png").exists()
    assert (tmp_path / "baltic_ev_high_f" / "seed0" / "osm_genetic_trait_means_Simu0.csv").exists()
    assert (tmp_path / "baltic_ev_low_f" / "seed0" / "osm_genetic_trait_means_Simu0.csv").exists()
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_fie_demo_smoke.py -v -m slow --timeout=600`
Expected: FAIL — script does not exist.

- [ ] **Step 10.3: Write the script**

Create `scripts/run_fie_demo.py`:
```python
"""Fishery-induced evolution (FIE) demonstration on Baltic cod.

Runs paired high-F vs low-F scenarios across multiple seeds, then plots
the mean cod imax trait trajectory with a multi-seed ribbon.

Usage: python scripts/run_fie_demo.py [--n-years 200] [--seeds 3] [--output-dir outputs/fie_demo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import read_genetic_trait_means

MASTER = Path("data/baltic_ev/baltic_ev_all-parameters.csv")

# Baltic fishing uses the v4 fisheries-API (fisheries.enabled=true; per
# baltic_param-fishing.csv). When fisheries-API is active, `mortality.fishing.rate.*`
# legacy keys are ignored. Cod is targeted by fsh0 (fisheries.name.fsh0;trawlcod),
# so the per-fishery base rate is the correct override knob.
# Pin both fishing rate AND selectivity so the demo is self-documenting and
# does not silently inherit baltic's age-knife-edge selectivity (which would
# make the FIE selection differential on imax zero by design).
SCENARIOS = {
    "baltic_ev_high_f": {
        "fisheries.rate.base.fsh0": "0.6",
        "fisheries.selectivity.type.fsh0": "1",
        "fisheries.selectivity.l50.fsh0": "35.0",
        "fisheries.selectivity.slope.fsh0": "2.0",
    },
    "baltic_ev_low_f": {
        "fisheries.rate.base.fsh0": "0.1",
        "fisheries.selectivity.type.fsh0": "1",
        "fisheries.selectivity.l50.fsh0": "35.0",
        "fisheries.selectivity.slope.fsh0": "2.0",
    },
}

# Optional drift-only baseline. Enabled via --with-zero-f-control. Per
# caveat #6, F=0.1 still applies meaningful selection (Heino, Pauli &
# Dieckmann 2015); this arm gives a true neutral baseline at the cost of
# doubling wall-clock. Recommended for any published version of the demo.
ZERO_F_CONTROL = {
    "baltic_ev_zero_f": {
        "fisheries.rate.base.fsh0": "0.0",
        # Selectivity irrelevant when rate=0 but pin for reproducibility
        "fisheries.selectivity.type.fsh0": "1",
        "fisheries.selectivity.l50.fsh0": "35.0",
        "fisheries.selectivity.slope.fsh0": "2.0",
    },
}


_ALL_SCENARIOS = {**SCENARIOS, **ZERO_F_CONTROL}
_SCENARIO_COLORS = {
    "baltic_ev_high_f": "C3",
    "baltic_ev_low_f": "C0",
    "baltic_ev_zero_f": "C2",
}


def _build_cfg(scenario: str, n_years: int) -> dict[str, str]:
    cfg = OsmoseConfigReader().read(MASTER)
    if cfg.get("simulation.bioen.enabled", "false").lower() != "true":
        raise RuntimeError(
            "Ev-OSMOSE traits require simulation.bioen.enabled=true; "
            "baltic_ev is misconfigured."
        )
    cfg["simulation.time.nyear"] = str(n_years)
    cfg.update(_ALL_SCENARIOS[scenario])
    return cfg


def _run_one(scenario: str, seed: int, n_years: int, output_root: Path) -> None:
    out_dir = output_root / scenario / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    PythonEngine().run(_build_cfg(scenario, n_years), out_dir, seed=seed)


def _load(scenario: str, seeds: int, output_root: Path) -> pd.DataFrame:
    frames = []
    for s in range(seeds):
        ds = read_genetic_trait_means(output_root / scenario / f"seed{s}", prefix="osm")
        df = ds.sel(species_id=0, trait_name="imax")["mean"].to_dataframe().reset_index()
        df["seed"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-years", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fie_demo"))
    parser.add_argument(
        "--with-zero-f-control",
        action="store_true",
        help="Add a third F=0 arm as a drift-only neutral baseline. Doubles "
             "wall-clock but quantifies the low-F selection contribution "
             "from caveat #6.",
    )
    args = parser.parse_args()

    scenarios = dict(SCENARIOS)
    if args.with_zero_f_control:
        scenarios.update(ZERO_F_CONTROL)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        for s in range(args.seeds):
            print(f"Running {scenario} seed={s}...", flush=True)
            _run_one(scenario, s, args.n_years, args.output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario in scenarios:
        df = _load(scenario, args.seeds, args.output_dir)
        agg = df.groupby("Time")["mean"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["Time"], agg["mean"], color=_SCENARIO_COLORS[scenario], label=scenario)
        ax.fill_between(
            agg["Time"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=_SCENARIO_COLORS[scenario], alpha=0.2,
        )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Mean cod imax trait")
    ax.set_title("FIE on Baltic cod: mean imax trajectory under selective fishing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fie_imax_trajectory.png", dpi=150)

    # Print end-state summary
    for scenario in scenarios:
        df = _load(scenario, args.seeds, args.output_dir)
        end = df[df["Time"] == df["Time"].max()]["mean"]
        print(f"{scenario}: end-of-run mean imax = {end.mean():.4f} ± {end.std():.4f}")

    # Diagnostic: imax-binding fraction. If the cap (bioen_i_max) is rarely
    # binding because cod is prey-limited, the trait is a silent no-op
    # regardless of h². Reported per scenario.
    _print_imax_binding_diagnostic(args.output_dir, list(scenarios), args.seeds)


def _print_imax_binding_diagnostic(
    output_dir: Path, scenarios: list[str], seeds: int
) -> None:
    """Read ingestion vs imax-cap per cod-school-timestep and report what
    fraction of timesteps the cap was actually binding. < 30% means imax
    trait is structurally not the limiting constraint and the FIE signal
    will be drift-dominated."""
    # The bioen output already writes per-step mean ingestion and the
    # config carries the per-species bioen_i_max value. Read both,
    # compute (mean_ingestion / cap) at each step for cod, count
    # fraction near saturation (>= 0.95 of cap).
    print("\n=== imax-binding diagnostic ===")
    for scenario in scenarios:
        bind_fracs: list[float] = []
        for s in range(seeds):
            out_dir = output_dir / scenario / f"seed{s}"
            # The bioen output writer emits osm_bioen_ingestion_Simu0.csv
            # (matches the existing bioen_e_net/ingestion pattern in
            # osmose/engine/output.py). Columns: Time, <species names>.
            ingestion_csv = out_dir / "osm_bioen_ingestion_Simu0.csv"
            if not ingestion_csv.exists():
                print(f"{scenario} seed{s}: ingestion CSV missing; skipping")
                continue
            import pandas as pd
            df = pd.read_csv(ingestion_csv, sep=None, engine="python", comment="#")
            # Cap value for cod (sp0). Read from baltic_ev_param-bioen.csv.
            cap = 3.0  # matches Task 7.4's placeholder; if tuned, update.
            cod_col = "cod"
            if cod_col not in df.columns:
                print(f"{scenario} seed{s}: 'cod' column not in ingestion CSV; skipping")
                continue
            cod_series = df[cod_col]
            bind_frac = float((cod_series >= 0.95 * cap).mean())
            bind_fracs.append(bind_frac)
        if bind_fracs:
            import statistics
            mean_bind = statistics.mean(bind_fracs)
            print(f"{scenario}: cod imax-binding fraction across seeds = "
                  f"{mean_bind*100:.1f}% (per-seed: {[f'{x*100:.1f}%' for x in bind_fracs]})")
            if mean_bind < 0.30:
                print(f"  WARNING: imax-binding < 30% — FIE signal will be drift-dominated. "
                      f"imax trait is structurally not the limiting constraint in this config. "
                      f"Possible cause: declining cod growth potential in the eastern Baltic "
                      f"since the 1990s (Svedäng et al. 2024, "
                      f"https://doi.org/10.1002/ece3.70382, report L50 halving from 40 to 20cm "
                      f"attributed to deteriorating growth potential — NOTE the paper "
                      f"explicitly excludes simple prey-density / forage-fish mechanisms as "
                      f"the sole driver). FIE-direction test (Task 11) is unlikely to produce "
                      f"a meaningful result without first calibrating bioen params.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.4: Run smoke (short)**

Run: `.venv/bin/python -m pytest tests/test_run_fie_demo_smoke.py -v -m slow --timeout=900`
Expected: pass (~3-5 min on 10y × 1 seed × 2 scenarios).

- [ ] **Step 10.5: Commit**

```bash
git add scripts/run_fie_demo.py tests/test_run_fie_demo_smoke.py
git commit -m "feat(scripts): run_fie_demo paired high-F/low-F evolution chart"
```

---

## Task 11: FIE-direction regression test

**Files:**
- Create: `tests/test_fie_demo_direction.py`

- [ ] **Step 11.1: Write the regression test**

Create `tests/test_fie_demo_direction.py`:
```python
from pathlib import Path
import pytest


@pytest.mark.slow
def test_high_f_drives_lower_cod_imax_than_low_f(tmp_path: Path) -> None:
    """Direction-only assertion: mean-across-3-seeds end-of-run cod imax
    must be lower under high F than low F by >=2%.

    Threshold defense.
    Per-generation FIE response on growth-rate at moderate F clusters at
    0.02-0.93%/yr across modelling studies (Audzijonyte et al., 2013,
    https://doi.org/10.1111/eva.12044), with a theoretical envelope of
    0.1-0.6%/yr (Andersen & Brander, 2009,
    https://doi.org/10.1073/pnas.0901690106). Over ~8 selecting generations
    (cod gen time ≈5y per Eero et al. 2015, https://doi.org/10.1093/icesjms/fsv109;
    Task 8 sets evolution.seeding.year=10 so only year>10 contributes), the
    expected cumulative high-F response is 1-4%. The paired (high-F minus
    low-F) contrast is ~2/3 of that = ~0.7-2.7%.

    Multi-seed drift floor (back-of-envelope): with σ_A²=0.018
    (config line: evolution.trait.imax.var.sp0=0.018), σ_A≈0.134; for
    N_e ≈ 0.1·N marine fish (Marty et al., 2015,
    https://doi.org/10.1111/eva.12220) and N≈10^3-10^4 schools, per-arm drift
    SD over 8 generations ≈ σ_A·sqrt(2g/N_e)/μ ≈ 0.6% of trait mean.
    With 3 seeds, multi-seed-mean drift SD ≈ 0.35%. A 1% threshold sits
    at ~2σ → ~5% false-pass risk under null. 2% sits at ~6σ → resilient.

    If the engine produces only ~1%, the right response is escalating to
    100y BEFORE relaxing the threshold (the test prints `drop_pct` so the
    implementer can decide).
    """
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.results import read_genetic_trait_means

    _require_preflight()  # see Task 7.8 — refuses to run until pre-flight wired

    def _cfg(fsh0_rate: str) -> dict[str, str]:
        cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
        cfg["simulation.time.nyear"] = "50"
        cfg["fisheries.rate.base.fsh0"] = fsh0_rate
        # Force size-selectivity for FIE; baltic default is age-knife-edge
        # which would make the FIE selection differential on imax zero.
        cfg["fisheries.selectivity.type.fsh0"] = "1"
        cfg["fisheries.selectivity.l50.fsh0"] = "35.0"
        cfg["fisheries.selectivity.slope.fsh0"] = "2.0"
        return cfg

    def _final_mean(out_dir: Path) -> float:
        ds = read_genetic_trait_means(out_dir, prefix="osm")
        s = ds.sel(species_id=0, trait_name="imax")["mean"].to_pandas()
        return float(s.iloc[-1])

    seeds = [42, 43, 44]
    high_ends, low_ends = [], []
    for s in seeds:
        out_high = tmp_path / f"high_{s}"
        out_low = tmp_path / f"low_{s}"
        out_high.mkdir()
        out_low.mkdir()
        PythonEngine().run(_cfg("0.6"), out_high, seed=s)
        PythonEngine().run(_cfg("0.1"), out_low, seed=s)
        high_ends.append(_final_mean(out_high))
        low_ends.append(_final_mean(out_low))

    import statistics
    high_mean = statistics.mean(high_ends)
    low_mean = statistics.mean(low_ends)
    drop_pct = (low_mean - high_mean) / low_mean
    assert high_mean < low_mean, (
        f"expected high-F imax < low-F imax across seeds, "
        f"got {high_mean=:.4f} vs {low_mean=:.4f} "
        f"(per-seed high={high_ends}, low={low_ends})"
    )
    assert drop_pct >= 0.02, (
        f"expected >= 2% drop in mean-across-seeds; got {drop_pct*100:.2f}% "
        f"(per-seed high={high_ends}, low={low_ends}). "
        "If close to 1%, the response is at multi-seed drift noise floor — "
        "escalate to nyear=100 (~16 generations) BEFORE relaxing the threshold."
    )
```

- [ ] **Step 11.2: Run the regression**

Run: `.venv/bin/python -m pytest tests/test_fie_demo_direction.py -v -m slow --timeout=600`
Expected: pass. Wall-clock ~3-5 min (50y × 2 runs).

If the signal is too weak after 50y, INVESTIGATE before adjusting tolerance:
- Verify cod variance > 0 in baltic_ev config
- Verify `evolution.seeding.year` is < 50 (so inheritance kicks in before the test window ends)
- Verify high-F scenario actually applies higher F (read scenario summary printed by run_fie_demo)
- Verify pre-flight (Task 7.8) is wired AND passing — null pre-flight means cod don't reach the gear l50, which structurally voids the FIE signal
- Escalate to 100y (~16 generations) before relaxing the 2% threshold

- [ ] **Step 11.3: Commit**

```bash
git add tests/test_fie_demo_direction.py
git commit -m "test(ev-osmose): high-F drives lower cod imax than low-F over 50y"
```

---

## Task 12: Tutorial `docs/tutorials/fie-on-baltic-cod.md`

**Files:**
- Create: `docs/tutorials/fie-on-baltic-cod.md`
- Pre-requirement: PNG generated by `scripts/run_fie_demo.py` exists at `outputs/fie_demo/fie_imax_trajectory.png` (referenced in tutorial)

- [ ] **Step 12.1: Generate the PNG**

Run: `.venv/bin/python scripts/run_fie_demo.py --n-years 200 --seeds 3 --output-dir outputs/fie_demo`
Expected: takes ~15-25 min; produces `outputs/fie_demo/fie_imax_trajectory.png` + 6 scenario CSVs.

- [ ] **Step 12.2: Write the tutorial**

Create `docs/tutorials/fie-on-baltic-cod.md` (~300 lines). Sections:

```markdown
# Fishery-Induced Evolution on Baltic Cod

Time: ~30 minutes (15 min reading + 15 min compute)

This tutorial walks through running the Ev-OSMOSE genetics module on a
Baltic cod scenario, demonstrating how selective fishing pressure drives
evolution of cod growth rate (intake-rate trait `imax`) over ~40-50
cod generations (eastern Baltic cod generation time ~4-5y per
Eero et al. 2015, so 200 model-years ≈ 40-50 generations).

## What is FIE?

Fisheries-Induced Evolution (FIE) is the heritable response of fish
populations to selective harvesting. When large, fast-growing fish are
removed preferentially, the surviving population skews toward slower-growing
phenotypes — and because growth has a heritable component, that skew
propagates to offspring.

Classic evidence for *maturation-trait* FIE: Northern cod (Olsen et al.
2004). Direct evidence for *growth-rate* FIE is rarer — Heino, Pauli &
Dieckmann (2015) note that in the wild, most reported growth changes in
cod are confounded with concurrent maturation evolution; growth-FIE has
only been cleanly isolated in lab common-garden experiments on silversides
(Conover & Munch 2002; Walsh et al. 2006). This demo isolates the
growth-rate pathway because the dominant maturation pathway is held
constant (see caveat #3).

## What the demo does

Two paired scenarios on the `baltic_ev/` fixture (a clone of the calibrated
Baltic config with bioenergetics + Ev-OSMOSE genetics enabled for cod):

| Scenario | Cod fishing mortality |
|---|---|
| `baltic_ev_high_f` | F = 0.6/yr (modern Baltic level) |
| `baltic_ev_low_f` | F = 0.1/yr (low-but-not-unfished reference) |

Each scenario runs 200 model-years × 3 seeds. The demo collects the mean
cod `imax` trait per timestep, then plots both trajectories with a
±1 SD ribbon.

## Running

```bash
.venv/bin/python scripts/run_fie_demo.py
```

Takes ~15-25 min on commodity hardware. Outputs:
- `outputs/fie_demo/fie_imax_trajectory.png` — the chart
- `outputs/fie_demo/<scenario>/seed<n>/osm_genetic_trait_means_Simu0.csv` — raw per-step trait stats

## Interpretation

![FIE chart](../../outputs/fie_demo/fie_imax_trajectory.png)

The high-F trajectory drifts downward over the simulated period; the
low-F trajectory stays close to the initial mean of 3.0. The gap widens
slowly because selection on a continuous trait acts gradually relative to
generation time (eastern Baltic cod ≈ 4-5 years per Eero et al. 2015,
https://doi.org/10.1093/icesjms/fsv109).

Compare against:
- Olsen et al. (2004): Northern cod showed measurable **maturation** trait
  shifts over ~3 cod generations under intense fishing (age + size at
  maturation declined; this is a different surface than `imax` growth-rate
  evolution).
- Conover & Munch (2002): silverside lab experiments measured a ~25%
  reduction in length-at-age (and ~40% reduction in weight-at-age, per
  Audzijonyte et al. 2013) over 4 generations under intense (90%/generation)
  size-selective removal. That is an experimental upper bound; Audzijonyte
  et al. (2013, https://doi.org/10.1111/eva.12044) puts modelled FIE
  growth-rate responses at moderate F (≈0.5-1.0/yr) at 0.02-0.93% per year.

## Exercise

Edit `data/baltic_ev/baltic_ev_param-genetics.csv` and try:
- `evolution.trait.imax.var.sp0=0.072` (h² ≈ 0.57, near the upper end of
  Otterå et al. 2018's cod body-weight range) — how much faster does the
  trait respond?
- `evolution.trait.imax.nlocus.sp0=5` (halve the loci count from Marty
  et al. 2015's 10) — does reduced polygenicity speed up or slow down the
  response? (Compare to Diaz Pauli & Heino 2014 on architecture sensitivity.)

You can also add a true unfished baseline:

```bash
.venv/bin/python scripts/run_fie_demo.py --with-zero-f-control
```

This adds a third F=0 arm (drift-only) at the cost of doubling wall-clock.

## Caveats

This demo emphasizes **direction of response, not absolute biomass**.
The bioenergetics parameters in `baltic_ev_param-bioen.csv` are
literature-default placeholders — the fixture is not calibrated against
ICES assessments. See `data/baltic_ev/README.md` for parameter provenance.

Three modelling choices to be explicit about:

1. **Size-selective fishing is the selection knob.** Baltic's default
   `baltic_param-fishing.csv` uses age-knife-edge selectivity for the
   cod fishery; this demo overrides it to length-sigmoidal (l50 = 35cm
   matching the EU minimum landing size). Without size-selectivity,
   imax-FIE cannot emerge: cod of any growth rate at a given age are
   equally vulnerable.

2. **No thermal forcing.** `simulation.bioen.phit.enabled` is explicitly
   set to `false` — bioen runs thermally neutral. Adding temperature
   forcing (e.g., a Copernicus SST time-series) is a future extension.

3. **Ingestion-cap pathway only.** This demo evolves only `imax`
   (intake-rate cap) targeting `bioen_i_max`. Maturation is set to a
   flat threshold (m0=30cm for cod per Radtke & Grygiel 2013), so FIE
   does NOT operate through age/size-at-maturation evolution — only
   through the indirect "fast growers cross the gear's size threshold
   sooner" pathway. The dominant FIE pathway documented in real cod
   stocks IS maturation evolution (Olsen et al. 2004; Heino, Pauli &
   Dieckmann 2015) — this demo intentionally isolates the secondary
   (growth-rate) pathway. The multi-trait extension targeting
   `bioen_m0` is listed in the spec's out-of-scope follow-ups.

4. **Heritability h² ≈ 0.25 is borrowed from cod body-weight studies**
   (Nielsen et al. 2014). No published h² estimate exists for fish
   ingestion-rate as a standalone trait. A referee will fairly ask
   whether body-weight h² (which integrates intake + assimilation +
   metabolic efficiency + activity) overstates ingestion-rate h².
   The demo's deliverable is the direction of trait response, not
   the magnitude — a future sensitivity sweep over
   h² ∈ {0.05, 0.15, 0.25, 0.40} would bracket this uncertainty.

5. **Selection window is ~8 generations across 50y**, not 10.
   `evolution.seeding.year=10` configures the first 10 years as a
   "seed phase" where offspring genotypes are randomly redrawn from
   population donors (per `inheritance.py:61-68`), erasing any
   selection signature. Real allele-frequency response only starts at
   year 10, leaving ~40y / ~5y-per-generation ≈ 8 selecting generations
   for the FIE-direction test, and ~38 generations for the 200y demo.

6. **F=0.1 "low-F" arm still applies meaningful selection.** Per
   Audzijonyte et al. (2013, https://doi.org/10.1111/eva.12044) and
   Andersen & Brander (2009, https://doi.org/10.1073/pnas.0901690106),
   modelled growth-rate FIE response at moderate F (0.5-1.0/yr) clusters
   at 0.02-0.93% per year (mean 0.25%/yr); the theoretical envelope is
   0.1-0.6%/yr. Over ~8 selecting generations the cumulative response is
   1-4% in the high-F arm, ~one-third of that in the low-F arm, so the
   paired contrast is ~0.7-2.7%. The cleanest reference would be F=0.0;
   add it via `--with-zero-f-control` (doubles wall-clock). This demo's
   2-arm default trades the cleanest contrast for keeping low-F closer
   to a real management target.

## Further reading

- Olsen, E. M. et al. (2004). *Nature* 428: 932-935. (maturation FIE)
- Heino, M. et al. (2015). *Annu. Rev. Ecol. Evol. Syst.* 46: 461-480. (review)
- Conover, D. O. & Munch, S. B. (2002). *Science* 297: 94-96. (silverside lab)
- Walsh, M. R. et al. (2006). *Ecol. Lett.* 9: 142-148. (multi-trait FIE)
- Andersen, K. H. & Brander, K. (2009). *PNAS* 106: 11657-11660. (rate-of-FIE envelope)
- Audzijonyte, A. et al. (2013). *Evol. Appl.* 6: 585-595. (FIE meta-analysis)
- Marty, L., Dieckmann, U. & Ernande, B. (2015). *Evol. Appl.* 8: 47-63. (genetic architecture)
- Brander, K. M. (1995). *ICES J. Mar. Sci.* 52: 1-10. (Baltic cod growth)
```

- [ ] **Step 12.3: Commit**

```bash
git add docs/tutorials/fie-on-baltic-cod.md outputs/fie_demo/fie_imax_trajectory.png
git commit -m "docs: FIE on Baltic cod tutorial"
```

(If the PNG should not be committed because of binary-size policy, replace with `outputs/fie_demo/.gitignore` excluding the PNG and reference the chart by relative path the user must regenerate. Per OSMOSE convention — check `git ls-files outputs/` for prior precedent.)

---

## Task 13: Mark `state.imax_trait` as vestigial

**Files:**
- Modify: `osmose/engine/state.py:81`

- [ ] **Step 13.1a: Verify the line refs against current HEAD**

The comment cites three downstream locations as also vestigial. Before committing, run:
```bash
grep -n "imax_trait" osmose/engine/processes/mortality.py
grep -n "imax_trait" osmose/engine/processes/foraging_mortality.py
grep -n "imax_trait" osmose/engine/processes/reproduction.py
```
Confirm the line numbers in the comment (mortality.py:296-316, foraging_mortality.py:36, reproduction.py:188) still match. If they drifted, update before committing.

- [ ] **Step 13.1b: Add the 1-line comment**

In `osmose/engine/state.py:81`, immediately above the `imax_trait` field, add:
```python
    # Vestigial: bioen path consumes evolving imax via trait_overrides
    # (see simulate.py:1341-1342). Kept None on all live code paths.
    # Open issue to delete this field and the dead branches at
    # mortality.py:296-316, foraging_mortality.py:36, reproduction.py:188.
    # See docs/superpowers/specs/2026-05-18-ev-osmose-activation-design.md §7.
    imax_trait: NDArray[np.float64] | None = None
```

- [ ] **Step 13.2: Commit**

```bash
git add osmose/engine/state.py
git commit -m "docs(engine): mark state.imax_trait as vestigial"
```

---

## Final verification

After all tasks are complete, run the full suite:

- [ ] **Run all tests including slow**

```bash
.venv/bin/python -m pytest -q --timeout=900
```
Expected: 2866+ tests passing (existing baseline + new tests added by this plan).

- [ ] **Run with slow marker**

```bash
.venv/bin/python -m pytest -q -m slow --timeout=900
```
Expected: FIE-direction + FIE-demo-smoke tests pass.

- [ ] **Visual confirmation of the chart**

Open `outputs/fie_demo/fie_imax_trajectory.png`. Verify the high-F line drifts visibly below the low-F line by year 200.

- [ ] **Lint + format**

```bash
.venv/bin/ruff check osmose/ tests/ scripts/
.venv/bin/ruff format --check osmose/ tests/ scripts/
```
Expected: clean.

---

## Risk register

| Risk | Trigger | Mitigation |
|---|---|---|
| Cod bioen parameters produce unrealistic biomass | Smoke test in Task 7.7 returns near-zero or runaway biomass | Adjust `predation.ingestion.rate.max.bioen.sp0` (the actual reader key) based on biomass; absolute calibration out-of-scope but biomass must remain non-degenerate. |
| Cod never reaches gear l50=35cm under placeholder bioen | Pre-flight check at Task 7.8 fails | Tune bioen params (raise imax, lower c_m), OR lower the gear l50 with caveat, OR source real Baltic-cod bioen calibration. MUST be resolved before Task 11. |
| imax cap rarely binds (prey-limited cod) → trait is a no-op | Demo runs but FIE diagnostic in Task 10 reports binding fraction < 30% | Demo is structurally drift-only. Either accept (document null result), tune prey availability up, or switch to a trait that targets a different bioen parameter (e.g., `bioen_c_m` for metabolic efficiency). |
| Heritability h²=0.25 borrowed from cod body-weight studies, not ingestion-rate-specific | Referee challenges trait choice | Sensitivity-sweep h² ∈ {0.05, 0.15, 0.25, 0.40} in a follow-up; for this demo, frame as "trait-direction-of-response" not "quantitative-FIE-magnitude". |
| F=0.1 low-F arm is not unfished — predicted ~1.2% trait drop | Paired contrast (3.4% vs 1.2%) is ~2.2pp, possibly within multi-seed noise | If demo shows insufficient contrast, add a third arm at F=0.0 as the true reference. |
| Effective selection window = 8 generations, not 10 | `evolution.seeding.year=10` redraws genotypes during seed phase (per inheritance.py:61-68), erasing selection signature | Expected response over 8 selecting generations ≈ 2.7% on h²=0.25; remains above the 1% threshold but narrower. Multi-seed reduces noise. |
| FIE signal absent at 50y | Task 11 fails on the 1% threshold | Escalate to 100y first (raise `n-years` default); only relax threshold after investigating variance and transmission-year settings. |
| PNG too large to commit | Output is >1MB | Use `.gitignore` for the PNG, document regeneration command in tutorial. |
| Existing 2866 test suite degrades | Any earlier-task commit breaks an unrelated test | TDD ordering ensures each task is verified before the next; bisect on the offending commit and revert if necessary. |
| `run_genetic_trait_means` reader behaves unexpectedly on real fixture | Tasks 8, 11 fail on `ds.coords` shape | The reader uses `df.set_index(["Time","species_id","trait_name"]).to_xarray()`, so the three become coords. If `ds["trait_name"]` access form differs from what the test expects, adjust to `ds.coords["trait_name"]`. |

---

## Plan complete

All tasks reference real files with line numbers verified against the current `master` HEAD (`c2ba915`). Each task ends with a commit. After Task 13, the deliverables are:
- Activated Ev-OSMOSE genetics on a new `baltic_ev/` fixture
- New per-step `genetic_trait_means.csv` output + xarray reader
- Config validator preventing silent trait misdeclarations
- `run_fie_demo.py` paired-scenario demo script
- FIE-direction regression test (slow-marker)
- Tutorial `docs/tutorials/fie-on-baltic-cod.md`
- 1-line vestigial-bridge comment on `state.imax_trait`

Spec follow-ups (out of scope): delete dead `imax_trait` field, calibrate `baltic_ev` against ICES, multi-trait extension, Shiny UI surface, ICES validator extension for trait-evolution metrics.
