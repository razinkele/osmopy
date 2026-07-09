# Baltic Chunk 0 — Bistability De-risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic harness that decides, with evidence, whether the deployed OSMOSE-Baltic model is a genuine *bistability* (alternative stable states, hysteresis) or a *monostable knife-edge*, and whether the system-wide over-production is driven by plankton over-accessibility — the two experiments that de-risk the whole improvement roadmap.

**Architecture:** A single self-contained script (`scripts/baltic_bistability_chunk0.py`) of *pure* helpers (classification, override builders, summary metrics) plus two experiment runners that take an **injectable model-runner callable** (defaulting to `calibrate_baltic.run_simulation`). The bistability experiment runs the model from two different initial conditions (cod-rich vs cod-poor) at each value of a driver (a global larval-mortality scale) and asks whether the two initial conditions settle into different basins. The accessibility experiment is a baseline-vs-lowered A/B. All unit tests exercise the pure helpers and the runners against a **fake runner** — no real simulations in CI (real Baltic runs are minutes-long and their emergent signals are known to be non-reproducible across CI cores; see `feedback-ci-fragile-emergent-tests`). The heavy runs are documented CLI invocations, not pytest tests.

**Tech Stack:** Python 3, numpy, pytest; existing project harness `scripts/calibrate_baltic.py` (`run_simulation`, `load_targets`, `BALTIC_CONFIG`), `osmose.config.reader.OsmoseConfigReader`, `osmose.engine.PythonEngine`. Optional matplotlib for a plot.

## Global Constraints

- **Python-engine only.** Baltic must never run on the Java engine (`nbackground>0` guard); `run_simulation` already uses `PythonEngine`. Do not add a Java path.
- **Run interpreter:** `.venv/bin/python`. Scripts are invoked from the repo root; `scripts/` is on `sys.path[0]` when a script in `scripts/` runs, so sibling imports use the bare form `from calibrate_baltic import ...` (matches `scripts/evaluate_calibration_vs_ices.py`).
- **Config-key case:** config keys are lowercase; all override keys must be lowercase (matches `run_simulation`'s `cfg.update(overrides)` and the eval script's `{k.lower(): str(v)}`).
- **Override values are strings.** Every override dict value is `str(...)` — the engine parses strings.
- **CI-safety:** no unit test may call the real engine. Inject a fake runner. Heavy runs are CLI-only.
- **Deployed config path:** `data/baltic/baltic_all-parameters.csv` (`calibrate_baltic.BALTIC_CONFIG`). Cod = sp0 … stickleback = sp7; LTL resources = sp8 (Diatoms) … sp13 (Benthos).
- **Cod collapse threshold:** cod is "collapsed" when its last-10-year mean biomass < `0.05 × cod target` = `0.05 × 120000 = 6000 t` (`biomass_targets.csv`).
- **Seeding semantics:** `population.seeding.year.max` empty ⇒ each species is re-seeded for its full lifespan (cod = 20 y), which would pin initial conditions. The bistability test sets `population.seeding.year.max = "1"` so seeding establishes the initial state in year 0 only, then the system evolves freely. Task 3 verifies this produces genuine divergence.

---

## File Structure

- **Create `scripts/baltic_bistability_chunk0.py`** — the harness: pure helpers + two experiment runners (injectable runner) + `read_base_config`/`read_base_larva_rates` + CLI `main()`. One responsibility: run and report the two Chunk-0 experiments.
- **Create `tests/test_baltic_bistability_chunk0.py`** — unit tests for every pure helper and both runners, using a fake runner. CI-safe.
- **Outputs (produced by CLI runs, not committed):** `docs/diagnostics/baltic_chunk0_bistability.json`, `docs/diagnostics/baltic_chunk0_accessibility_ab.json`, optional `docs/diagnostics/baltic_chunk0_bistability.png` (matches the existing `docs/diagnostics/` convention).

---

### Task 1: Pure classification & summary helpers

**Files:**
- Create: `scripts/baltic_bistability_chunk0.py`
- Test: `tests/test_baltic_bistability_chunk0.py`

**Interfaces:**
- Produces: `classify_state(cod_mean: float, cod_target: float, collapse_frac: float = 0.05) -> str` (`"collapsed"` | `"present"`); `bistability_gap(high_mean: float, low_mean: float) -> float`; `summarize_overproduction(stats: dict[str, float], targets: list) -> dict` where each target has `.species`, `.lower`, `.upper` attributes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baltic_bistability_chunk0.py
import sys
from collections import namedtuple
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import baltic_bistability_chunk0 as chunk0  # noqa: E402

Tgt = namedtuple("Tgt", "species lower upper")


def test_classify_state_collapsed_below_five_percent():
    # cod target 120000 -> threshold 6000
    assert chunk0.classify_state(5000.0, 120000.0) == "collapsed"
    assert chunk0.classify_state(6001.0, 120000.0) == "present"
    assert chunk0.classify_state(0.0, 120000.0) == "collapsed"


def test_bistability_gap_symmetric_and_normalized():
    assert chunk0.bistability_gap(0.0, 0.0) == 0.0
    # near-1 when one state is ~0 and the other large
    assert chunk0.bistability_gap(1_000_000.0, 0.0) > 0.99
    assert chunk0.bistability_gap(500.0, 500.0) == 0.0


def test_summarize_overproduction_counts_and_total():
    targets = [Tgt("cod", 60000, 250000), Tgt("sprat", 800000, 2500000)]
    stats = {"cod_mean": 5000.0, "sprat_mean": 3_000_000.0}
    out = chunk0.summarize_overproduction(stats, targets)
    assert out["under"] == 1          # cod below lower
    assert out["over"] == 1           # sprat above upper
    assert out["in_range"] == 0
    assert out["total_focal_biomass"] == 3_005_000.0
    assert out["per_species"]["cod"] == "under"
    assert out["per_species"]["sprat"] == "over"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'baltic_bistability_chunk0'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/baltic_bistability_chunk0.py
"""Chunk 0 de-risk harness for the Baltic OSMOSE improvement roadmap.

Two experiments:
  * bistability  — run the model from a cod-rich and a cod-poor initial state at
                   each value of a larval-mortality driver; a driver value where
                   the two initial conditions settle into different basins is a
                   bistable point (alternative stable states). No divergence at
                   any value => monostable knife-edge.
  * accessibility — baseline vs lowered plankton accessibility2fish A/B, to see
                   whether the system-wide over-production relaxes.

Pure helpers and both runners are unit-tested against a fake runner; real Baltic
runs are CLI-only (Python engine, minutes each).
"""
from __future__ import annotations


def classify_state(cod_mean: float, cod_target: float, collapse_frac: float = 0.05) -> str:
    """'collapsed' if cod mean biomass is below collapse_frac * target, else 'present'."""
    return "collapsed" if cod_mean < collapse_frac * cod_target else "present"


def bistability_gap(high_mean: float, low_mean: float) -> float:
    """Normalized separation of two equilibria: |high-low| / (max(high,low)+1)."""
    return abs(high_mean - low_mean) / (max(high_mean, low_mean) + 1.0)


def summarize_overproduction(stats: dict[str, float], targets: list) -> dict:
    """Count over/under/in-range/extinct focal species and total focal biomass.

    ``targets`` is any sequence of objects with ``.species``, ``.lower``, ``.upper``.
    """
    per_species: dict[str, str] = {}
    counts = {"over": 0, "under": 0, "in_range": 0, "extinct": 0}
    total = 0.0
    for t in targets:
        mean = float(stats.get(f"{t.species}_mean", 0.0))
        total += mean
        if mean <= 0:
            status = "extinct"
        elif mean < t.lower:
            status = "under"
        elif mean > t.upper:
            status = "over"
        else:
            status = "in_range"
        per_species[t.species] = status
        if status == "extinct":
            counts["under"] += 0  # extinct tracked separately
            counts["extinct"] += 1
        else:
            counts[status] += 1
    return {**counts, "total_focal_biomass": total, "per_species": per_species}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 pure classification & summary helpers"
```

---

### Task 2: Override builders (driver, accessibility, initial conditions)

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py`
- Test: `tests/test_baltic_bistability_chunk0.py`

**Interfaces:**
- Produces: `larva_scale_override(scale: float, base_rates: dict[int, float]) -> dict[str, str]`; `accessibility_override(value: float, resource_indices=range(8, 14)) -> dict[str, str]`; `cod_rich_seeding(seeding_stop_year: int = 1) -> dict[str, str]`; `cod_poor_seeding(seeding_stop_year: int = 1) -> dict[str, str]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_larva_scale_override_scales_each_species():
    base = {0: 360.0, 1: 192.0}
    out = chunk0.larva_scale_override(0.1, base)
    assert out["mortality.additional.larva.rate.sp0"] == "36.0"
    assert out["mortality.additional.larva.rate.sp1"] == "19.2"


def test_accessibility_override_all_resources():
    out = chunk0.accessibility_override(0.05)
    assert out["species.accessibility2fish.sp8"] == "0.05"
    assert out["species.accessibility2fish.sp13"] == "0.05"
    assert len(out) == 6  # sp8..sp13


def test_seeding_states_differ_and_stop_early():
    rich = chunk0.cod_rich_seeding()
    poor = chunk0.cod_poor_seeding()
    # cod-rich starts cod-heavy, cod-poor starts cod near-absent + clupeid-heavy
    assert float(rich["population.seeding.biomass.sp0"]) > float(poor["population.seeding.biomass.sp0"])
    assert float(poor["population.seeding.biomass.sp2"]) > float(rich["population.seeding.biomass.sp2"])
    # both stop seeding after year 1 so the states evolve freely
    assert rich["population.seeding.year.max"] == "1"
    assert poor["population.seeding.year.max"] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: module 'baltic_bistability_chunk0' has no attribute 'larva_scale_override'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def larva_scale_override(scale: float, base_rates: dict[int, float]) -> dict[str, str]:
    """Scale each focal species' additional larval-mortality rate by ``scale``."""
    return {
        f"mortality.additional.larva.rate.sp{i}": str(rate * scale)
        for i, rate in base_rates.items()
    }


def accessibility_override(value: float, resource_indices=range(8, 14)) -> dict[str, str]:
    """Set accessibility2fish for every LTL resource group (sp8..sp13) to ``value``."""
    return {f"species.accessibility2fish.sp{i}": str(value) for i in resource_indices}


def cod_rich_seeding(seeding_stop_year: int = 1) -> dict[str, str]:
    """Cod-dominated initial state; seeding stops after ``seeding_stop_year`` years."""
    return {
        "population.seeding.biomass.sp0": "300000",   # cod, 2x deployed default
        "population.seeding.biomass.sp1": "800000",    # herring (deployed default)
        "population.seeding.biomass.sp2": "600000",    # sprat (deployed default)
        "population.seeding.year.max": str(seeding_stop_year),
    }


def cod_poor_seeding(seeding_stop_year: int = 1) -> dict[str, str]:
    """Post-collapse sprat/herring-dominated initial state; cod near-absent."""
    return {
        "population.seeding.biomass.sp0": "1000",      # cod near-absent
        "population.seeding.biomass.sp1": "1500000",   # herring high
        "population.seeding.biomass.sp2": "1500000",   # sprat high
        "population.seeding.year.max": str(seeding_stop_year),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 override builders (driver, accessibility, initial conditions)"
```

---

### Task 3: Initial-condition divergence probe (instrument check)

This task de-risks the instrument itself: it confirms that two initial conditions actually evolve to **different** cod biomass under free evolution (seeding stopped after year 1). If they don't, the bistability sweep is meaningless.

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py`
- Test: `tests/test_baltic_bistability_chunk0.py`

**Interfaces:**
- Consumes: `cod_rich_seeding`, `cod_poor_seeding` (Task 2).
- Produces: `probe_ic_divergence(base_config: dict, *, runner, n_years: int, seed: int, min_gap: float = 0.25) -> dict` returning `{"rich_cod": float, "poor_cod": float, "gap": float, "diverged": bool}`. `runner(config, overrides, n_years, seed) -> dict` (same signature as `calibrate_baltic.run_simulation`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def _fake_runner_ic_sensitive(config, overrides, n_years, seed):
    """Cod equilibrium depends on the seeded initial cod biomass (bistable-like)."""
    seeded_cod = float(overrides.get("population.seeding.biomass.sp0", "0"))
    return {"cod_mean": 200000.0 if seeded_cod >= 100000 else 100.0}


def _fake_runner_ic_blind(config, overrides, n_years, seed):
    """Cod equilibrium ignores the initial condition (monostable)."""
    return {"cod_mean": 50000.0}


def test_probe_ic_divergence_detects_divergence():
    out = chunk0.probe_ic_divergence({}, runner=_fake_runner_ic_sensitive, n_years=5, seed=0)
    assert out["diverged"] is True
    assert out["rich_cod"] == 200000.0
    assert out["poor_cod"] == 100.0


def test_probe_ic_divergence_flags_no_divergence():
    out = chunk0.probe_ic_divergence({}, runner=_fake_runner_ic_blind, n_years=5, seed=0)
    assert out["diverged"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'probe_ic_divergence'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def probe_ic_divergence(
    base_config: dict,
    *,
    runner,
    n_years: int,
    seed: int,
    min_gap: float = 0.25,
) -> dict:
    """Run cod-rich and cod-poor initial states at the deployed driver; check divergence.

    Returns the two cod equilibria, their normalized gap, and whether the gap
    exceeds ``min_gap`` (i.e. the initial condition genuinely persists rather than
    being washed out). A False result means the seeding window / free-evolution
    setup is not a valid bistability instrument — reduce/raise seeding_stop_year.
    """
    rich = runner(base_config, cod_rich_seeding(), n_years, seed)
    poor = runner(base_config, cod_poor_seeding(), n_years, seed)
    rich_cod = float(rich.get("cod_mean", 0.0))
    poor_cod = float(poor.get("cod_mean", 0.0))
    gap = bistability_gap(rich_cod, poor_cod)
    return {"rich_cod": rich_cod, "poor_cod": poor_cod, "gap": gap, "diverged": gap >= min_gap}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 initial-condition divergence probe"
```

---

### Task 4: Bistability sweep runner

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py`
- Test: `tests/test_baltic_bistability_chunk0.py`

**Interfaces:**
- Consumes: `larva_scale_override`, `cod_rich_seeding`, `cod_poor_seeding`, `classify_state`, `bistability_gap` (Tasks 1–2).
- Produces: `run_bistability_point(scale, base_config, base_rates, cod_target, seeds, *, runner, n_years) -> dict`; `run_bistability_sweep(scales, base_config, base_rates, cod_target, seeds, *, runner, n_years) -> dict` returning `{"points": [...], "bistable": bool, "bistable_scales": [...], "verdict": str}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def _fake_runner_bistable_midband(config, overrides, n_years, seed):
    """Bistable only at intermediate driver: at scale 0.3 the IC decides cod fate."""
    scale = float(overrides["mortality.additional.larva.rate.sp0"]) / 360.0
    seeded_cod = float(overrides.get("population.seeding.biomass.sp0", "0"))
    if abs(scale - 0.3) < 1e-9:
        return {"cod_mean": 300000.0 if seeded_cod >= 100000 else 100.0}
    # elsewhere monostable: high larva scale -> collapse, low -> present, IC-blind
    return {"cod_mean": 100.0 if scale >= 0.9 else 300000.0}


def test_bistability_point_flags_ic_dependence():
    pt = chunk0.run_bistability_point(
        0.3, {}, {0: 360.0}, 120000.0, [0, 1],
        runner=_fake_runner_bistable_midband, n_years=5,
    )
    assert pt["rich_state"] == "present"
    assert pt["poor_state"] == "collapsed"
    assert pt["bistable"] is True


def test_bistability_sweep_verdict_bistable():
    out = chunk0.run_bistability_sweep(
        [0.1, 0.3, 1.0], {}, {0: 360.0}, 120000.0, [0],
        runner=_fake_runner_bistable_midband, n_years=5,
    )
    assert out["bistable"] is True
    assert 0.3 in out["bistable_scales"]
    assert "BISTABLE" in out["verdict"]


def test_bistability_sweep_verdict_monostable():
    def ic_blind(config, overrides, n_years, seed):
        scale = float(overrides["mortality.additional.larva.rate.sp0"]) / 360.0
        return {"cod_mean": 100.0 if scale >= 0.9 else 300000.0}
    out = chunk0.run_bistability_sweep(
        [0.1, 1.0], {}, {0: 360.0}, 120000.0, [0],
        runner=ic_blind, n_years=5,
    )
    assert out["bistable"] is False
    assert "MONOSTABLE" in out["verdict"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'run_bistability_point'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py
import statistics
from collections import Counter


def _majority(states: list[str]) -> str:
    return Counter(states).most_common(1)[0][0]


def run_bistability_point(
    scale, base_config, base_rates, cod_target, seeds, *, runner, n_years
) -> dict:
    """One driver value: run both initial conditions across seeds; classify basins."""
    driver = larva_scale_override(scale, base_rates)
    rich_means, poor_means, rich_states, poor_states = [], [], [], []
    for seed in seeds:
        rich = runner(base_config, {**driver, **cod_rich_seeding()}, n_years, seed)
        poor = runner(base_config, {**driver, **cod_poor_seeding()}, n_years, seed)
        rm = float(rich.get("cod_mean", 0.0))
        pm = float(poor.get("cod_mean", 0.0))
        rich_means.append(rm)
        poor_means.append(pm)
        rich_states.append(classify_state(rm, cod_target))
        poor_states.append(classify_state(pm, cod_target))
    high = statistics.median(rich_means)
    low = statistics.median(poor_means)
    rich_state = _majority(rich_states)
    poor_state = _majority(poor_states)
    return {
        "scale": scale,
        "rich_cod_median": high,
        "poor_cod_median": low,
        "gap": bistability_gap(high, low),
        "rich_state": rich_state,
        "poor_state": poor_state,
        "bistable": rich_state != poor_state,
    }


def run_bistability_sweep(
    scales, base_config, base_rates, cod_target, seeds, *, runner, n_years
) -> dict:
    """Sweep the larval-mortality driver; a point where the two ICs land in different
    basins is a bistable point. Any bistable point => the model already has latent
    alternative stable states; none => monostable knife-edge (feedbacks missing)."""
    points = [
        run_bistability_point(s, base_config, base_rates, cod_target, seeds,
                              runner=runner, n_years=n_years)
        for s in scales
    ]
    bistable_scales = [p["scale"] for p in points if p["bistable"]]
    bistable = bool(bistable_scales)
    if bistable:
        verdict = (
            f"BISTABLE — latent alternative stable states at larva-scale(s) "
            f"{bistable_scales}: the model already supports a cod-present and a "
            f"cod-collapsed basin at the same parameters. Focus shifts to which "
            f"basin the deployed config sits in and how to stabilize it."
        )
    else:
        verdict = (
            "MONOSTABLE / knife-edge — both initial conditions converge to the same "
            "state at every driver value. Confirms the reframe: the model lacks the "
            "feedbacks that create a self-locking bistability. Greenlight Chunks C "
            "(clupeid->cod-egg predation) and A2 (depletable plankton) to CREATE it."
        )
    return {"points": points, "bistable": bistable,
            "bistable_scales": bistable_scales, "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 bistability sweep runner"
```

---

### Task 5: Accessibility A/B runner

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py`
- Test: `tests/test_baltic_bistability_chunk0.py`

**Interfaces:**
- Consumes: `accessibility_override`, `summarize_overproduction` (Tasks 1–2).
- Produces: `run_accessibility_ab(base_config, targets, seeds, *, runner, n_years, low_value=0.05) -> dict` returning `{"baseline": {...}, "lowered": {...}, "low_value": float, "verdict": str}` where each side has median `over`, `under`, `in_range` counts and median `total_focal_biomass`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_accessibility_ab_detects_relaxed_overproduction():
    targets = [Tgt("cod", 60000, 250000), Tgt("sprat", 800000, 2500000),
               Tgt("herring", 800000, 3000000)]

    def runner(config, overrides, n_years, seed):
        lowered = "species.accessibility2fish.sp11" in overrides
        if lowered:  # everything in range when plankton is scarcer
            return {"cod_mean": 120000.0, "sprat_mean": 1_500_000.0, "herring_mean": 1_500_000.0}
        # baseline firehose: clupeids overshoot
        return {"cod_mean": 120000.0, "sprat_mean": 4_000_000.0, "herring_mean": 5_000_000.0}

    out = chunk0.run_accessibility_ab({}, targets, [0, 1], runner=runner, n_years=5)
    assert out["baseline"]["over"] == 2      # sprat + herring over
    assert out["lowered"]["over"] == 0
    assert out["lowered"]["total_focal_biomass"] < out["baseline"]["total_focal_biomass"]
    assert "relaxes" in out["verdict"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'run_accessibility_ab'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def _median_summary(summaries: list[dict]) -> dict:
    keys = ("over", "under", "in_range", "extinct", "total_focal_biomass")
    return {k: statistics.median([s[k] for s in summaries]) for k in keys}


def run_accessibility_ab(
    base_config, targets, seeds, *, runner, n_years, low_value: float = 0.05
) -> dict:
    """Baseline (config default 0.8) vs lowered plankton accessibility A/B."""
    base_summ, low_summ = [], []
    for seed in seeds:
        s_base = runner(base_config, {}, n_years, seed)
        s_low = runner(base_config, accessibility_override(low_value), n_years, seed)
        base_summ.append(summarize_overproduction(s_base, targets))
        low_summ.append(summarize_overproduction(s_low, targets))
    baseline = _median_summary(base_summ)
    lowered = _median_summary(low_summ)
    relaxed = (lowered["over"] < baseline["over"]
               or lowered["total_focal_biomass"] < baseline["total_focal_biomass"])
    verdict = (
        f"Lowering plankton accessibility 0.8 -> {low_value} relaxes over-production "
        f"(species OVER {baseline['over']:.0f} -> {lowered['over']:.0f}; total focal "
        f"biomass {baseline['total_focal_biomass']:.3e} -> "
        f"{lowered['total_focal_biomass']:.3e}). A1 is a real lever."
        if relaxed else
        f"Lowering accessibility to {low_value} does NOT relax over-production "
        f"(OVER {baseline['over']:.0f} -> {lowered['over']:.0f}); the firehose is not "
        f"the (whole) driver — reconsider before A1."
    )
    return {"baseline": baseline, "lowered": lowered, "low_value": low_value, "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 accessibility A/B runner"
```

---

### Task 6: Config loaders, CLI, and smoke gate

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py`
- Test: `tests/test_baltic_bistability_chunk0.py`

**Interfaces:**
- Consumes: all prior functions; `calibrate_baltic.run_simulation`, `calibrate_baltic.load_targets`, `calibrate_baltic.BALTIC_CONFIG`, `osmose.config.reader.OsmoseConfigReader` (lazy imports inside functions).
- Produces: `read_base_config() -> dict[str, str]`; `read_base_larva_rates(base_config, n_focal=8) -> dict[int, float]`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_read_base_larva_rates_parses_focal_species():
    cfg = {f"mortality.additional.larva.rate.sp{i}": str(10 * (i + 1)) for i in range(8)}
    rates = chunk0.read_base_larva_rates(cfg, n_focal=8)
    assert rates[0] == 10.0
    assert rates[7] == 80.0
    assert set(rates.keys()) == set(range(8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'read_base_larva_rates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py
import argparse
import json
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagnostics"
_DEFAULT_SCALES = [0.03, 0.1, 0.3, 0.5, 1.0]
_DEFAULT_SEEDS = [0, 1, 2]


def read_base_config() -> dict:
    """Load the deployed Baltic config as a lowercase-key override dict."""
    from calibrate_baltic import BALTIC_CONFIG
    from osmose.config.reader import OsmoseConfigReader

    return OsmoseConfigReader().read(str(BALTIC_CONFIG))


def read_base_larva_rates(base_config: dict, n_focal: int = 8) -> dict[int, float]:
    """Read each focal species' configured additional larval-mortality rate."""
    rates: dict[int, float] = {}
    for i in range(n_focal):
        key = f"mortality.additional.larva.rate.sp{i}"
        if key in base_config:
            rates[i] = float(base_config[key])
    return rates


def _default_runner(config, overrides, n_years, seed):
    from calibrate_baltic import run_simulation

    return run_simulation(config, overrides, n_years=n_years, seed=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic Chunk 0 de-risk experiments")
    ap.add_argument("--experiment", choices=["bistability", "accessibility", "both", "probe"],
                    default="both")
    ap.add_argument("--years", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    ap.add_argument("--scales", type=float, nargs="+", default=_DEFAULT_SCALES)
    ap.add_argument("--low-accessibility", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (3y, 1 seed, 2 scales) to prove the harness end-to-end")
    args = ap.parse_args(argv)

    seeds = [args.seeds[0]] if args.smoke else args.seeds
    scales = [1.0, 0.1] if args.smoke else args.scales
    years = 3 if args.smoke else args.years

    base_config = read_base_config()
    base_rates = read_base_larva_rates(base_config)
    from calibrate_baltic import load_targets

    targets = load_targets()
    cod_target = next(t.target for t in targets if t.species == "cod")
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment == "probe":
        out = probe_ic_divergence(base_config, runner=_default_runner, n_years=years, seed=seeds[0])
        print(f"IC probe: rich_cod={out['rich_cod']:.0f} poor_cod={out['poor_cod']:.0f} "
              f"gap={out['gap']:.3f} diverged={out['diverged']}")
        if not out["diverged"]:
            print("WARNING: initial conditions do not diverge — raise seeding_stop_year "
                  "in cod_rich_seeding/cod_poor_seeding (try 2 or 3) before trusting the sweep.")
        return 0

    if args.experiment in ("bistability", "both"):
        result = run_bistability_sweep(scales, base_config, base_rates, cod_target, seeds,
                                       runner=_default_runner, n_years=years)
        print("\n=== BISTABILITY ===")
        for p in result["points"]:
            print(f"  larva x{p['scale']:<5} rich={p['rich_state']:<9} poor={p['poor_state']:<9} "
                  f"gap={p['gap']:.3f}  ->  {'BISTABLE' if p['bistable'] else 'same basin'}")
        print(f"\nVERDICT: {result['verdict']}")
        (_DIAG_DIR / "baltic_chunk0_bistability.json").write_text(json.dumps(result, indent=2))

    if args.experiment in ("accessibility", "both"):
        result = run_accessibility_ab(base_config, targets, seeds, runner=_default_runner,
                                      n_years=years, low_value=args.low_accessibility)
        print("\n=== ACCESSIBILITY A/B ===")
        print(f"  baseline(0.8): over={result['baseline']['over']:.0f} "
              f"total={result['baseline']['total_focal_biomass']:.3e}")
        print(f"  lowered({args.low_accessibility}): over={result['lowered']['over']:.0f} "
              f"total={result['lowered']['total_focal_biomass']:.3e}")
        print(f"\nVERDICT: {result['verdict']}")
        (_DIAG_DIR / "baltic_chunk0_accessibility_ab.json").write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Smoke-run the real harness end-to-end**

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment probe --smoke`
Expected: prints an `IC probe:` line with numeric `rich_cod`, `poor_cod`, `gap`, and `diverged=True/False`, and exits 0. (If `diverged=False`, follow the printed WARNING before the full run.)

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment both --smoke`
Expected: prints a BISTABILITY block (2 scales) and an ACCESSIBILITY A/B block with numeric verdicts, writes both JSON files under `docs/diagnostics/`, exits 0. Scientific meaning is NOT expected at 3 years — this only proves the harness runs against the real engine.

- [ ] **Step 6: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 CLI, config loaders, smoke gate"
```

---

## The real experiment runs (post-implementation, not CI)

These are the actual de-risk runs; each ~minutes per simulation, so budget for the full grids.

1. **Instrument check first:**
   `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment probe --years 40 --seeds 0`
   Require `diverged=True`. If not, raise `seeding_stop_year` (2, then 3) in `cod_rich_seeding`/`cod_poor_seeding` and re-probe.
2. **Bistability sweep:**
   `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment bistability --years 40 --seeds 0 1 2 --scales 0.03 0.1 0.3 0.5 1.0`
   Read the VERDICT: BISTABLE (latent alternative stable states — the pit already exists) vs MONOSTABLE (knife-edge — feedbacks missing; greenlight Chunks C & A2).
3. **Accessibility A/B:**
   `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment accessibility --years 40 --seeds 0 1 2 --low-accessibility 0.05`
   Read whether over-production relaxes → whether roadmap Chunk A1 is a real lever.

Record the two verdicts in a short `docs/baltic_chunk0_results_YYYY-MM-DD.md` and update the roadmap: they decide whether Phase 2 emphasizes *creating* bistability (C, A2) or *choosing/stabilizing* an existing basin, and confirm/deny A1.

---

## Self-Review

**Spec coverage.** Chunk 0's two experiments (hysteresis/bistability test + accessibility A/B from the plan and investigation) are both implemented: bistability via Tasks 3–4 (two-initial-condition sweep, the tractable form given no restart files), accessibility via Task 5. The seeding-window caveat (default re-seeds for full lifespan) is handled by `population.seeding.year.max="1"` and verified by the Task-3 probe. Config/driver/target values are the real deployed ones (larva sp0=360, cod target 120000, accessibility 0.8, LTL sp8–sp13).

**Placeholder scan.** No "TBD"/"handle edge cases"/"similar to Task N". Every step has runnable code and an exact command with expected output. The one deferred decision (seeding-window length) is an explicit, tested branch with a concrete fallback, not a placeholder.

**Type consistency.** `runner(config, overrides, n_years, seed) -> dict` is used identically in Tasks 3–6 and matches `calibrate_baltic.run_simulation(config, overrides, n_years=, seed=)`. `classify_state`, `bistability_gap`, `summarize_overproduction`, `larva_scale_override`, `accessibility_override`, `cod_rich_seeding`, `cod_poor_seeding` keep the same signatures across tasks. Stats keys `"{sp}_mean"` match `run_simulation`'s output.
