# Baltic Chunk 0 — Bistability De-risk Implementation Plan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Supersedes:** v2 (`...-derisk-v2.md`). This v3 patches all 25 confirmed findings of the v2 re-review. The v1→v2 fixes that held (15-y horizon, gap-based `basins_differ`, accessibility scope sp8/10/11/12 at ≥0.1, larva read as 15, warm-start deferred, no matplotlib) are retained.

**Revised after re-review (round 3):** the re-review confirmed all 25 v2 fixes genuinely closed and refuted 5 challenges; the one real new bug (a `seed-split` aggregate leaking past `accessibility_transition`) and the worthwhile refinements are folded in here — `seed-split`/`undetermined` now withhold the accessibility verdict, the verdict gates only on weight-1.0 stocks (cod/herring/sprat) so a coarse weight-0.2 species can't spuriously veto or be spuriously relaxed, the seeding window is **4 y** (≥ cod age-at-maturity), the machine-readable `bistable` flag is gated on `establishment_fraction ≥ 0.5` (can't contradict an INSTRUMENT-LIMITED verdict), and the single-cod-axis limitation (it omits the sprat-dominated basin) is stated in the MONOSTABLE verdict and the Task 7 warm-start note.

**Goal:** Same two Chunk-0 de-risk experiments, on an instrument whose two headline guarantees actually hold: (A) an accessibility A/B where **no collapse — partial or total, real or crash — can read as "relaxation"**, and (B) a **conservative** bistability probe that can see *all three* basins (collapsed / persist / overshoot), reports seed-splits instead of voting them away, and never claims MONOSTABLE when the instrument itself couldn't establish cod.

**Architecture:** One script (`scripts/baltic_bistability_chunk0.py`) of pure helpers + two experiment runners taking an injectable model-runner. **A single `classify_state` is the only place biomass becomes a state**, shared by both experiments, so their collapse/stationarity definitions can't drift. Near-zero biomass is a *stable* "collapsed" band (checked before the stationarity gate, so `run_simulation`'s `cv=10.0` zero-mean sentinel can't hide an extinction). Every model call goes through `safe_run` → `_failed` sentinel; `_failed` seeds are dropped from **both** runners' aggregates. Verdicts are **count-based over ICES-band transitions** (robust to structural weight-0.2 overshooters). Unit tests use fake runners returning `cv`/`trend` — no real sims in CI.

## What each v2 finding maps to (all 25)

| v2 finding(s) | Fix in v3 (task) |
|---|---|
| 3 — `cv=10` zero-mean → "undetermined" → extinction basin invisible | `classify_state` returns "collapsed" for `mean < collapse_frac*target` **before** the stationarity gate (Task 1). |
| 1,2,5,6 — accessibility gate faked by partial collapse / m≤0 | Verdict vetoes on **any species dropping from {in_range,overshoot}→{low,collapsed}** (`new_undershoot`), requires in-range to **strictly increase**, shares `classify_state`'s bands (Task 2). No seeding hack needed. |
| 9,12,13 — Exp A not stationarity-gated | `species_states` runs `classify_state` (cv/trend gated) per species; any "undetermined" species **withholds** the verdict (Task 2). |
| 4,10,14 — crash folded into A/B as all-extinct | `_failed` seeds excluded from A/B aggregates; all-failed → distinct "INSTRUMENT-FAILED" verdict (Task 5). |
| 8,16,22 — median-reclassify votes seed-splits away; no ambiguous outcome; wrong comment | Aggregate from **per-seed modal band** (`aggregate_states`) with an agreement threshold → explicit `seed-split` outcome + AMBIGUOUS verdict; median used only for the gap; comment corrected (Tasks 1,4). |
| 7 — ICs differ in >1 axis | `cod_rich/poor_seeding` override **only sp0**; sp1/sp2 stay at deployed defaults → single cod axis (Task 3). |
| 17,18 — transient/regime; is 15 y stationary? | Per-scale **establishment** + `establishment_fraction`; low fraction → instrument-limited verdict, no MONOSTABLE claim; the non-default seeding regime is documented (Task 4). |
| 19 — unweighted overshoot dominated by percids | Verdict is **count-based** over band transitions; structural overshooters appear equally in both arms and cannot flip it (Task 2). |
| 20,23,25 — establishment placement/scope | Establishment computed **inside** the sweep (bistability path only) (Task 4). |
| 21,24,26 — incremental JSON schema differs | Incremental writes carry the **same top-level keys** + `complete:false` (Task 4). |
| 11,26 — `_agg` comment / budget nits | Comment corrected; budget states per-sim assumption, drops phantom "probe=2" (Tasks 1,6). |
| 15 — E702 semicolons / no `ruff format` | No compound statements; Task 6 runs `ruff check` **and** `ruff format --check` on both files. |

## Global Constraints

- **Python-engine only.** Interpreter `.venv/bin/python`. Scripts use bare `from calibrate_baltic import ...`; tests insert `scripts/` on `sys.path` (per `tests/test_fr_diagnostic.py`). **All module imports at file top** (no E402); **no `;` compound statements** (no E702).
- Config keys lowercase; override values `str(...)`. **No unit test calls the real engine.**
- Cod=sp0..stickleback=sp7; LTL sp8(Diatoms),sp9(Dino),sp10(Micro),sp11(Meso),sp12(Macro),sp13(Benthos); plankton (calibrated) = **sp8,10,11,12**.
- Larva base rates are **read at runtime** (post-4.4.1 migration → `sp0`≈15.0, not 360).
- ICES bands from `biomass_targets.csv` (cod target 120000, lower 60000, upper 250000). Collapse = `mean < 0.05×target`.
- Stationarity: `cv_max=0.30`, `trend_max=0.05`. `run_simulation` returns `{sp}_cv=10.0` when `{sp}_mean==0` (a sentinel `classify_state` must treat as collapsed, not non-stationary).
- **Runtime budget:** bistability 5 scales × 2 ICs × 3 seeds = 30 sims; accessibility 2 × 3 = 6 → **36 sims at 15 y**, serial (a single Baltic sim already multithreads via Numba). At ~1 min/sim ≈ 35–45 min; scale linearly with `--years`.

---

## File Structure

- **Create `scripts/baltic_bistability_chunk0.py`** — imports at top; pure helpers (`is_stationary`, `classify_state`, `bistability_gap`, `basins_differ`, `aggregate_states`, `species_states`, `accessibility_transition`, `accessibility_verdict`, override builders, `safe_run`) + runners (`run_bistability_point/sweep`, `run_accessibility_ab`) + loaders + CLI.
- **Create `tests/test_baltic_bistability_chunk0.py`** — CI-safe fake-runner unit tests. The per-task "Expected: PASS (N tests)" counts are INDICATIVE cumulative totals; the authoritative check is the actual `pytest` exit status — trust the run output over the printed N if they differ (e.g. after adding a test).
- **Outputs (CLI):** `docs/diagnostics/baltic_chunk0_bistability.json` (stable schema, incremental), `docs/diagnostics/baltic_chunk0_accessibility_ab.json`.
- **Create `docs/baltic_chunk0_warmstart_prerequisite.md`** (Task 7, unchanged from v2).

---

### Task 1: Core state helpers (collapse-aware, stationarity-gated, per-seed aggregation)

**Files:** Create `scripts/baltic_bistability_chunk0.py`; Test `tests/test_baltic_bistability_chunk0.py`.

**Interfaces (Produces):** `is_stationary(cv, trend, cv_max=0.30, trend_max=0.05) -> bool`; `classify_state(mean, cv, trend, target, lower, upper, collapse_frac=0.05) -> str` ∈ `{"collapsed","low","in_range","overshoot","undetermined"}`; `bistability_gap(a, b) -> float`; `basins_differ(rich, poor, gap, gap_thresh=0.5) -> bool`; `aggregate_states(states) -> str` ∈ bands ∪ `{"seed-split","undetermined"}` (consensus band if all valid seeds agree, else `seed-split`).

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

import baltic_bistability_chunk0 as c0  # noqa: E402

Tgt = namedtuple("Tgt", "species target lower upper weight", defaults=(1.0,))
COD = dict(target=120000.0, lower=60000.0, upper=250000.0)


def test_collapsed_wins_over_stationarity_sentinel():
    # run_simulation encodes mean~0 as cv=10.0; must still read 'collapsed', not 'undetermined'
    assert c0.classify_state(0.0, 10.0, 1.0, **COD) == "collapsed"
    assert c0.classify_state(3000.0, 10.0, 1.0, **COD) == "collapsed"   # < 6000


def test_classify_bands_and_stationarity():
    assert c0.classify_state(120000, 0.5, 0.01, **COD) == "undetermined"  # non-stationary, non-zero
    assert c0.classify_state(30000, 0.1, 0.01, **COD) == "low"
    assert c0.classify_state(120000, 0.1, 0.01, **COD) == "in_range"
    assert c0.classify_state(400000, 0.1, 0.01, **COD) == "overshoot"


def test_basins_differ():
    assert c0.basins_differ("in_range", "collapsed", 0.9) is True
    assert c0.basins_differ("collapsed", "collapsed", 0.9) is False
    assert c0.basins_differ("overshoot", "overshoot", 0.9) is False
    assert c0.basins_differ("in_range", "in_range", 0.8) is True   # same band, big gap
    assert c0.basins_differ("in_range", "in_range", 0.1) is False


def test_aggregate_states():
    assert c0.aggregate_states(["in_range", "in_range", "in_range"]) == "in_range"    # unanimous
    assert c0.aggregate_states(["in_range", "collapsed", "in_range"]) == "seed-split"  # any disagreement
    assert c0.aggregate_states(["in_range", "in_range", "failed"]) == "in_range"      # failed ignored, rest agree
    assert c0.aggregate_states(["failed", "undetermined"]) == "undetermined"          # none valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/baltic_bistability_chunk0.py
"""Chunk 0 de-risk harness (v3). See the plan for the design rationale.

One shared classify_state turns biomass into an ICES-band state for BOTH
experiments. Near-zero is a STABLE 'collapsed' band (checked before the
stationarity gate, so run_simulation's cv=10.0 zero-mean sentinel cannot hide an
extinction). Verdicts are count-based over band transitions; _failed seeds are
excluded; per-seed states are aggregated to a modal band or an explicit
'seed-split'. Unit-tested with fake runners; real Baltic runs are CLI-only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagnostics"
_DEFAULT_SCALES = [0.03, 0.1, 0.3, 0.5, 1.0]
_DEFAULT_SEEDS = [0, 1, 2]
_PLANKTON_GROUPS = (8, 10, 11, 12)
_SEEDING_WINDOW_Y = 4   # >= cod age-at-first-maturity (~2.6 y at L50=38); lets the seeded cohort mature and self-reproduce before seeding stops, while staying << the 15-y run and 20-y lifespan
_NONBANDS = ("failed", "undetermined")


def is_stationary(cv: float, trend: float, cv_max: float = 0.30, trend_max: float = 0.05) -> bool:
    return cv <= cv_max and trend <= trend_max


def classify_state(mean, cv, trend, target, lower, upper, collapse_frac: float = 0.05) -> str:
    """ICES-band state. Near-zero is a stable 'collapsed' (checked FIRST so the
    cv=10.0 zero-mean sentinel from run_simulation cannot mask an extinction)."""
    if mean < collapse_frac * target:
        return "collapsed"
    if not is_stationary(cv, trend):
        return "undetermined"
    if mean < lower:
        return "low"
    if mean > upper:
        return "overshoot"
    return "in_range"


def bistability_gap(a: float, b: float) -> float:
    return abs(a - b) / (max(a, b) + 1.0)


def basins_differ(rich_state, poor_state, gap, gap_thresh: float = 0.5) -> bool:
    _bands = ("collapsed", "low", "in_range", "overshoot")
    if rich_state not in _bands or poor_state not in _bands:   # 'seed-split'/'undetermined'/'failed' not comparable
        return False
    if rich_state == poor_state == "collapsed":
        return False
    if rich_state == poor_state == "overshoot":
        return False
    if rich_state != poor_state:
        return True
    return gap >= gap_thresh


def aggregate_states(states) -> str:
    """Consensus band across seeds: the band if ALL valid (non-failed, non-undetermined)
    seeds agree; 'seed-split' if valid seeds disagree (a near-tipping-point signal to
    surface, not vote away — finding 8); 'undetermined' if no seed is valid. A lone
    failed/undetermined seed among agreeing valid seeds is dropped (robust to one flaky
    seed); it only decides the result when it is the majority of seeds."""
    valid = [s for s in states if s not in _NONBANDS]
    if not valid:
        return "undetermined"
    return valid[0] if len(set(valid)) == 1 else "seed-split"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v3 collapse-aware state + per-seed aggregation"
```

---

### Task 2: Accessibility metric + verdict (collapse-proof, stationarity-gated, count-based)

**Files:** Modify both.

**Interfaces (Produces):** `species_states(stats, targets) -> dict[str,str]`; `accessibility_transition(base_states, low_states, targets, weight_threshold=1.0) -> dict` (gates only on `weight ≥ threshold` stocks; withholds on any non-band aggregate); `accessibility_verdict(transition) -> tuple[bool, str]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def _targets():
    return [Tgt("cod", 120000, 60000, 250000), Tgt("sprat", 1_500_000, 800_000, 2_500_000),
            Tgt("herring", 1_500_000, 800_000, 3_000_000)]

def _stats(**means):
    d = {}
    for sp, m in means.items():
        d[f"{sp}_mean"] = m
        d[f"{sp}_cv"] = 0.05 if m > 0 else 10.0
        d[f"{sp}_trend"] = 0.01
    return d


def test_partial_collapse_vetoes_relaxation():
    targets = _targets()
    base = c0.species_states(_stats(cod=120000, sprat=25_000_000, herring=1_500_000), targets)
    # lowering starves sprat overshoot -> below lower (collapse-through-band)
    low = c0.species_states(_stats(cod=120000, sprat=300_000, herring=1_500_000), targets)
    t = c0.accessibility_transition(base, low, targets)
    assert t["new_undershoot"] == 1
    assert c0.accessibility_verdict(t)[0] is False
    assert "collapse" in c0.accessibility_verdict(t)[1].lower()


def test_genuine_relaxation_passes():
    targets = _targets()
    base = c0.species_states(_stats(cod=120000, sprat=25_000_000, herring=20_000_000), targets)
    low = c0.species_states(_stats(cod=120000, sprat=1_500_000, herring=1_500_000), targets)
    t = c0.accessibility_transition(base, low, targets)
    assert t["new_undershoot"] == 0
    ok, msg = c0.accessibility_verdict(t)
    assert ok is True and "real lever" in msg.lower()


def test_nonstationary_withholds_verdict():
    targets = _targets()
    drifting = _stats(cod=120000, sprat=25_000_000, herring=1_500_000)
    drifting["sprat_cv"] = 0.9   # sprat not settled
    base = c0.species_states(drifting, targets)
    low = c0.species_states(_stats(cod=120000, sprat=1_500_000, herring=1_500_000), targets)
    t = c0.accessibility_transition(base, low, targets)
    assert t["undetermined"] >= 1
    assert c0.accessibility_verdict(t)[0] is False
    assert "provisional" in c0.accessibility_verdict(t)[1].lower()


def test_seed_split_species_withholds_accessibility_verdict():
    # a species non-reproducible across seeds aggregates to 'seed-split' -> must withhold,
    # not be silently ignored (v3 re-review finding #1: the accessibility_transition leak).
    targets = _targets()
    base = {"cod": "in_range", "sprat": "overshoot", "herring": "in_range"}
    low = {"cod": "in_range", "sprat": "seed-split", "herring": "in_range"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["undetermined"] >= 1
    assert c0.accessibility_verdict(t)[0] is False


def test_low_weight_species_does_not_gate():
    # perch (weight 0.2) collapsing overshoot->low must NOT veto; only weight-1.0 stocks gate.
    targets = _targets() + [Tgt("perch", 20000, 8000, 50000, 0.2)]
    base = {"cod": "in_range", "sprat": "overshoot", "herring": "overshoot", "perch": "overshoot"}
    low = {"cod": "in_range", "sprat": "in_range", "herring": "in_range", "perch": "low"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["gated_species"] == 3
    assert t["new_undershoot"] == 0
    assert c0.accessibility_verdict(t)[0] is True


def test_collapsed_stock_in_lowered_arm_blocks_real_lever():
    # cod collapsed in BOTH arms while clupeids relax overshoot->in_range must NOT read as
    # 'A1 is a real lever' — the web is still broken (round-4 findings 1/2).
    targets = _targets()
    base = {"cod": "collapsed", "sprat": "overshoot", "herring": "overshoot"}
    low = {"cod": "collapsed", "sprat": "in_range", "herring": "in_range"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["collapsed_lowered"] >= 1
    ok, msg = c0.accessibility_verdict(t)
    assert ok is False and "still broken" in msg.lower()


def test_medium_weight_collapse_blocks_real_lever():
    # flounder (weight 0.5, below the 1.0 improvement gate but above the 0.3 collapse veto)
    # starved to collapsed must block the verdict (round-4 finding 5).
    targets = _targets() + [Tgt("flounder", 50000, 20000, 100000, 0.5)]
    base = {"cod": "in_range", "sprat": "overshoot", "herring": "overshoot", "flounder": "in_range"}
    low = {"cod": "in_range", "sprat": "in_range", "herring": "in_range", "flounder": "collapsed"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["collapsed_lowered"] >= 1
    assert c0.accessibility_verdict(t)[0] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'species_states'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def species_states(stats: dict, targets) -> dict:
    """Per-species ICES-band state from one successful run (cv/trend gated).
    A species absent from the output is 'undetermined'."""
    out = {}
    for t in targets:
        mean = stats.get(f"{t.species}_mean")
        if mean is None:
            out[t.species] = "undetermined"
            continue
        cv = float(stats.get(f"{t.species}_cv", 10.0))
        trend = float(stats.get(f"{t.species}_trend", 1.0))
        out[t.species] = classify_state(float(mean), cv, trend, float(t.target), float(t.lower), float(t.upper))
    return out


def accessibility_transition(base_states: dict, low_states: dict, targets,
                             weight_threshold: float = 1.0, collapse_veto_weight: float = 0.3) -> dict:
    """Band transitions baseline->lowered. Two weight scopes:
    - IMPROVEMENT counters (in_range/overshoot up/down, new_undershoot) are over well-assessed
      stocks only (weight >= weight_threshold; default 1.0 => cod/herring/sprat), so structural
      weight-0.2 percid overshoots cannot spuriously veto or relax the verdict.
    - The ABSOLUTE collapse veto (collapsed_lowered) fires when ANY stock with weight >=
      collapse_veto_weight (default 0.3 => also flounder/smelt) ends 'collapsed' in the LOWERED
      arm — catching a pre-existing collapse, a low->collapsed drop, and a lever-caused one alike,
      so 'no collapse can read as relaxation' holds regardless of the baseline band.
    Any non-band aggregate ('seed-split'/'undetermined'/absent) on a gated stock withholds the verdict."""
    bands = ("collapsed", "low", "in_range", "overshoot")
    high = {"in_range", "overshoot"}
    below = {"low", "collapsed"}
    c = {"in_range_base": 0, "in_range_low": 0, "overshoot_base": 0, "overshoot_low": 0,
         "new_undershoot": 0, "undetermined": 0, "gated_species": 0, "collapsed_lowered": 0}
    for t in targets:
        w = float(getattr(t, "weight", 1.0))
        low = low_states.get(t.species, "undetermined")
        if w >= collapse_veto_weight and low == "collapsed":
            c["collapsed_lowered"] += 1        # absolute end-state veto (any baseline band)
        if w < weight_threshold:
            continue
        c["gated_species"] += 1
        b = base_states.get(t.species, "undetermined")
        if b not in bands or low not in bands:   # seed-split / undetermined / missing => withhold
            c["undetermined"] += 1
            continue
        c["in_range_base"] += b == "in_range"
        c["in_range_low"] += low == "in_range"
        c["overshoot_base"] += b == "overshoot"
        c["overshoot_low"] += low == "overshoot"
        if b in high and low in below:
            c["new_undershoot"] += 1
    return c


def accessibility_verdict(t: dict) -> tuple[bool, str]:
    """Relaxation ONLY if: no undetermined gated species, no weight-relevant stock collapsed in
    the lowered arm (absolute veto), no gated stock pushed below lower, in-range strictly up, and
    overshoot count not up. Count-based, so structural weight-0.2 overshooters (present equally in
    both arms) cannot flip it, and a pre-existing/medium-weight collapse cannot read as relaxation."""
    if t["undetermined"] > 0:
        return False, (f"PROVISIONAL — {t['undetermined']} species non-stationary/absent in one arm; "
                       f"verdict withheld (raise --years or --seeds).")
    if t.get("collapsed_lowered", 0) > 0:
        return False, (f"WEB STILL BROKEN — {t['collapsed_lowered']} weight-relevant stock(s) are collapsed "
                       f"in the lowered arm. Accessibility may relieve overshoot but does not fix the "
                       f"collapse; NOT a green light for A1 on its own.")
    if t["new_undershoot"] > 0:
        return False, (f"COLLAPSES the web, not relaxes it — {t['new_undershoot']} well-assessed species "
                       f"dropped from in-range/overshoot to below-lower. NOT evidence for A1.")
    relaxed = t["in_range_low"] > t["in_range_base"] and t["overshoot_low"] <= t["overshoot_base"]
    if relaxed:
        return True, (f"Relaxes over-production toward ICES bands (in-range {t['in_range_base']} -> "
                      f"{t['in_range_low']}; overshooters {t['overshoot_base']} -> {t['overshoot_low']}; "
                      f"no weight-relevant stock collapsed). A1 is a real lever.")
    return False, (f"No clean relaxation (in-range {t['in_range_base']} -> {t['in_range_low']}; "
                   f"overshooters {t['overshoot_base']} -> {t['overshoot_low']}). Reconsider A1.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v3 collapse-proof, stationarity-gated accessibility verdict"
```

---

### Task 3: Override builders (single cod axis) + safe_run

**Files:** Modify both.

**Interfaces (Produces):** `larva_scale_override(scale, base_rates) -> dict`; `accessibility_override(value, resource_indices=_PLANKTON_GROUPS) -> dict`; `cod_rich_seeding(window=_SEEDING_WINDOW_Y) -> dict`; `cod_poor_seeding(window=_SEEDING_WINDOW_Y) -> dict`; `safe_run(runner, config, overrides, n_years, seed) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_ics_vary_only_cod():
    rich, poor = c0.cod_rich_seeding(), c0.cod_poor_seeding()
    # ONLY sp0 differs; sp1/sp2 are NOT overridden (stay at deployed defaults) -> single axis
    assert set(rich) == {"population.seeding.biomass.sp0", "population.seeding.year.max"}
    assert set(poor) == set(rich)
    assert float(rich["population.seeding.biomass.sp0"]) > float(poor["population.seeding.biomass.sp0"])
    assert rich["population.seeding.year.max"] == "4"


def test_accessibility_scope_and_safe_run():
    assert set(c0.accessibility_override(0.1)) == {f"species.accessibility2fish.sp{i}" for i in (8, 10, 11, 12)}
    assert c0.safe_run(lambda *a: {"cod_mean": 5.0}, {}, {}, 5, 0) == {"cod_mean": 5.0}
    assert c0.safe_run(lambda *a: {}, {}, {}, 5, 0)["_failed"] is True
    assert c0.safe_run(lambda *a: {"herring_mean": 1.0}, {}, {}, 5, 0)["_failed"] is True  # no cod_mean

    def boom(*a):
        raise RuntimeError("x")

    assert c0.safe_run(boom, {}, {}, 5, 0)["_failed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'cod_rich_seeding'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def larva_scale_override(scale: float, base_rates: dict) -> dict:
    return {f"mortality.additional.larva.rate.sp{i}": str(r * scale) for i, r in base_rates.items()}


def accessibility_override(value: float, resource_indices=_PLANKTON_GROUPS) -> dict:
    return {f"species.accessibility2fish.sp{i}": str(value) for i in resource_indices}


def cod_rich_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    # Vary ONLY the cod SEEDING BIOMASS; herring/sprat biomass stay at deployed defaults.
    # NOTE: population.seeding.year.max is a GLOBAL key -> it truncates the seeding window
    # for ALL species to `window` years in both arms (deployed default is per-lifespan);
    # a low establishment_fraction can therefore reflect whole-web truncation, not just cod.
    return {"population.seeding.biomass.sp0": "300000", "population.seeding.year.max": str(window)}


def cod_poor_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    return {"population.seeding.biomass.sp0": "1000", "population.seeding.year.max": str(window)}


def safe_run(runner, config, overrides, n_years, seed) -> dict:
    """Model call; `_failed` sentinel (distinct from a real cod_mean==0) on crash or
    empty/partial output. Callers drop `_failed` runs from every aggregate."""
    try:
        stats = runner(config, overrides, n_years, seed)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not abort the whole grid
        return {"_failed": True, "_error": repr(exc)}
    if not stats or "cod_mean" not in stats:
        return {"_failed": True, "_error": "empty or partial stats (no cod_mean)"}
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v3 single-axis ICs + scoped accessibility + safe_run"
```

---

### Task 4: Bistability point + sweep (per-seed outcomes, establishment, stable-schema persistence)

**Files:** Modify both.

**Interfaces (Produces):** `run_bistability_point(scale, base_config, base_rates, cod_bands, seeds, *, runner, n_years) -> dict` (`outcome` ∈ `{"bistable","same-basin","seed-split","undetermined"}`, plus `established`); `run_bistability_sweep(scales, base_config, base_rates, cod_bands, seeds, *, runner, n_years, on_point=None) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def _bands():
    return {"target": 120000.0, "lower": 60000.0, "upper": 250000.0}

def _runner_bistable(config, overrides, n_years, seed):
    scale = float(overrides["mortality.additional.larva.rate.sp0"]) / 15.0
    seeded = float(overrides.get("population.seeding.biomass.sp0", "0"))
    if abs(scale - 0.3) < 1e-9:            # IC-dependent basin at 0.3
        cod = 120000.0 if seeded >= 100000 else 0.0
    else:
        cod = 120000.0 if scale < 0.9 else 0.0
    cv = 0.05 if cod > 0 else 10.0
    return {"cod_mean": cod, "cod_cv": cv, "cod_trend": 0.01}


def test_point_detects_bistable_including_collapsed_basin():
    pt = c0.run_bistability_point(0.3, {}, {0: 15.0}, _bands(), [0, 1, 2],
                                  runner=_runner_bistable, n_years=15)
    assert pt["rich_state"] == "in_range"
    assert pt["poor_state"] == "collapsed"     # cv=10 sentinel no longer hides this
    assert pt["outcome"] == "bistable"
    assert pt["established"] is True


def test_seed_split_outcome():
    # cod-rich lands in-range for 2 seeds, collapsed for 1 -> a per-seed disagreement that is
    # SURFACED as 'seed-split' (near a tipping point), not voted away to the majority (finding 8).
    def flaky(config, overrides, n_years, seed):
        seeded = float(overrides.get("population.seeding.biomass.sp0", "0"))
        cod = 120000.0 if (seeded >= 100000 and seed != 1) else 0.0
        return {"cod_mean": cod, "cod_cv": 0.05 if cod > 0 else 10.0, "cod_trend": 0.01}
    pt = c0.run_bistability_point(0.3, {}, {0: 15.0}, _bands(), [0, 1, 2],
                                  runner=flaky, n_years=15)
    assert pt["rich_state"] == "seed-split"
    assert pt["outcome"] == "seed-split"


def test_sweep_verdict_and_stable_persistence():
    seen = []
    out = c0.run_bistability_sweep([0.1, 0.3, 1.0], {}, {0: 15.0}, _bands(), [0, 1, 2],
                                   runner=_runner_bistable, n_years=15, on_point=seen.append)
    assert out["bistable"] is True and 0.3 in out["bistable_scales"]
    assert "conservative" in out["verdict"].lower()
    assert 0.0 <= out["establishment_fraction"] <= 1.0
    # incremental payload carries the SAME top-level keys as the final result
    assert set(seen[-1]) >= {"points", "bistable", "verdict", "complete"}
    assert seen[0]["complete"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'run_bistability_point'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def _cod_state(stats: dict, bands: dict) -> tuple[str, float]:
    if stats.get("_failed"):
        return "failed", 0.0
    mean = float(stats.get("cod_mean", 0.0))
    st = classify_state(mean, float(stats.get("cod_cv", 10.0)), float(stats.get("cod_trend", 1.0)),
                        bands["target"], bands["lower"], bands["upper"])
    return st, mean


def _median_valid(states, means) -> float:
    vals = [m for s, m in zip(states, means) if s not in _NONBANDS]
    return statistics.median(vals) if vals else 0.0


def _partial(points: list) -> dict:
    return {"points": points, "bistable": None, "bistable_scales": [], "seed_split_scales": [],
            "undetermined_scales": [], "establishment_fraction": None, "trustworthy": None,
            "verdict": "incomplete", "complete": False}


def run_bistability_point(scale, base_config, base_rates, cod_bands, seeds, *, runner, n_years) -> dict:
    driver = larva_scale_override(scale, base_rates)
    rich_states, poor_states, rich_means, poor_means = [], [], [], []
    for seed in seeds:
        r = safe_run(runner, base_config, {**driver, **cod_rich_seeding()}, n_years, seed)
        p = safe_run(runner, base_config, {**driver, **cod_poor_seeding()}, n_years, seed)
        rs, rm = _cod_state(r, cod_bands)
        ps, pm = _cod_state(p, cod_bands)
        rich_states.append(rs)
        poor_states.append(ps)
        rich_means.append(rm)
        poor_means.append(pm)
    rich_agg = aggregate_states(rich_states)      # consensus band (all valid seeds agree) or 'seed-split'
    poor_agg = aggregate_states(poor_states)
    gap = bistability_gap(_median_valid(rich_states, rich_means), _median_valid(poor_states, poor_means))
    established = rich_agg in ("low", "in_range", "overshoot")
    if rich_agg == "seed-split" or poor_agg == "seed-split":
        outcome = "seed-split"
    elif rich_agg == "undetermined" or poor_agg == "undetermined":
        outcome = "undetermined"
    elif basins_differ(rich_agg, poor_agg, gap):
        outcome = "bistable"
    else:
        outcome = "same-basin"
    return {"scale": scale, "rich_state": rich_agg, "poor_state": poor_agg,
            "per_seed_rich": rich_states, "per_seed_poor": poor_states,
            "rich_cod_median": _median_valid(rich_states, rich_means),
            "poor_cod_median": _median_valid(poor_states, poor_means),
            "gap": gap, "established": established, "outcome": outcome,
            "bistable": outcome == "bistable"}


def run_bistability_sweep(scales, base_config, base_rates, cod_bands, seeds, *, runner, n_years, on_point=None) -> dict:
    points = []
    for s in scales:
        pt = run_bistability_point(s, base_config, base_rates, cod_bands, seeds,
                                   runner=runner, n_years=n_years)
        points.append(pt)
        if on_point is not None:
            on_point(_partial(points))   # same top-level schema, complete=False
    bistable = [p["scale"] for p in points if p["outcome"] == "bistable"]
    seed_split = [p["scale"] for p in points if p["outcome"] == "seed-split"]
    undet = [p["scale"] for p in points if p["outcome"] == "undetermined"]
    est_frac = sum(p["established"] for p in points) / len(points) if points else 0.0
    trustworthy = est_frac >= 0.5
    if not trustworthy:
        split_note = (f" (a basin split WAS seen at scale(s) {bistable} — treat as tentative)"
                      if bistable else "")
        verdict = (f"INSTRUMENT-LIMITED — cod-rich established a non-collapsed stock at only "
                   f"{est_frac:.0%} of scales, so egg-seeding (not the biology) may set the outcome"
                   f"{split_note}. No MONOSTABLE conclusion; a definitive test needs the warm-start "
                   f"primitive (Task 7).")
    elif bistable:
        verdict = (f"BISTABLE (conservative) — different cod basins from the two ICs at larva-scale(s) "
                   f"{bistable}. Egg-only ICs + Beverton-Holt bias this test toward MONOSTABLE, so a "
                   f"positive result is strong. Confirm with a warm-start standing IC (Task 7).")
    elif seed_split:
        verdict = (f"AMBIGUOUS — seed-split (per-seed basin disagreement) at scale(s) {seed_split}; "
                   f"near a tipping point. Re-run with more --seeds before concluding.")
    else:
        verdict = (f"MONOSTABLE by this CONSERVATIVE method — no basin split at any established scale "
                   f"(undetermined: {undet}). Cannot rule out bistability (egg-only ICs, and the "
                   f"single-cod-axis ICs omit the sprat-dominated start); add the warm-start primitive "
                   f"(Task 7) for a definitive test, or proceed to Chunks C & A2 to CREATE a self-locking "
                   f"bistability. Read the rich/poor cod response curve.")
    return {"points": points, "bistable": bool(bistable) and trustworthy, "bistable_scales": bistable,
            "seed_split_scales": seed_split, "undetermined_scales": undet,
            "establishment_fraction": est_frac, "trustworthy": trustworthy,
            "verdict": verdict, "complete": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v3 bistability sweep (per-seed outcomes + establishment + stable persistence)"
```

---

### Task 5: Accessibility A/B runner (_failed excluded, per-seed aggregate)

**Files:** Modify both.

**Interfaces (Produces):** `run_accessibility_ab(base_config, targets, seeds, *, runner, n_years, low_value=0.1) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_ab_excludes_failed_and_flags_all_failed():
    targets = _targets()
    # low arm always crashes -> must NOT read as 'collapse'; all seeds fail -> INSTRUMENT-FAILED
    def low_crashes(config, overrides, n_years, seed):
        if "species.accessibility2fish.sp11" in overrides:
            raise RuntimeError("blowup")
        return _stats(cod=120000, sprat=1_500_000, herring=1_500_000)
    out = c0.run_accessibility_ab({}, targets, [0, 1, 2], runner=low_crashes, n_years=15)
    assert out["relaxed"] is False
    assert "instrument-failed" in out["verdict"].lower()
    assert "collapse" not in out["verdict"].lower()


def test_ab_real_relaxation():
    targets = _targets()
    def runner(config, overrides, n_years, seed):
        low = "species.accessibility2fish.sp11" in overrides
        if low:
            return _stats(cod=120000, sprat=1_500_000, herring=1_500_000)
        return _stats(cod=120000, sprat=25_000_000, herring=20_000_000)
    out = c0.run_accessibility_ab({}, targets, [0, 1], runner=runner, n_years=15)
    assert out["relaxed"] is True and out["n_failed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'run_accessibility_ab'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def run_accessibility_ab(base_config, targets, seeds, *, runner, n_years, low_value: float = 0.1) -> dict:
    per_seed = []   # (base_states, low_states) for seeds where BOTH arms succeeded
    n_failed = 0
    for seed in seeds:
        b = safe_run(runner, base_config, {}, n_years, seed)
        low = safe_run(runner, base_config, accessibility_override(low_value), n_years, seed)
        if b.get("_failed") or low.get("_failed"):
            n_failed += 1
            continue
        per_seed.append((species_states(b, targets), species_states(low, targets)))
    if not per_seed:
        return {"relaxed": False, "n_failed": n_failed, "low_value": low_value,
                "verdict": (f"INSTRUMENT-FAILED — all {len(seeds)} seeds crashed or returned empty "
                            f"output; no accessibility verdict (instrument failure, not an ecological signal).")}
    base_agg = {t.species: aggregate_states([ps[0][t.species] for ps in per_seed]) for t in targets}
    low_agg = {t.species: aggregate_states([ps[1][t.species] for ps in per_seed]) for t in targets}
    transition = accessibility_transition(base_agg, low_agg, targets)
    relaxed, verdict = accessibility_verdict(transition)
    return {"baseline_states": base_agg, "lowered_states": low_agg, "transition": transition,
            "low_value": low_value, "n_failed": n_failed, "relaxed": relaxed, "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v3 accessibility A/B runner (failed-excluded, per-seed aggregate)"
```

---

### Task 6: Loaders, CLI, lint gate, smoke

**Files:** Modify both.

**Interfaces (Produces):** `read_base_config() -> dict`; `read_base_larva_rates(base_config, n_focal=8) -> dict`; `read_cod_bands(targets) -> dict`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_loaders():
    cfg = {f"mortality.additional.larva.rate.sp{i}": str(i + 1) for i in range(8)}
    rates = c0.read_base_larva_rates(cfg)
    assert rates[0] == 1.0 and rates[7] == 8.0
    assert c0.read_cod_bands([Tgt("cod", 120000, 60000, 250000)]) == {
        "target": 120000.0, "lower": 60000.0, "upper": 250000.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'read_base_larva_rates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def read_base_config() -> dict:
    from calibrate_baltic import BALTIC_CONFIG
    from osmose.config.reader import OsmoseConfigReader
    return OsmoseConfigReader().read(str(BALTIC_CONFIG))


def read_base_larva_rates(base_config: dict, n_focal: int = 8) -> dict:
    rates = {}
    for i in range(n_focal):
        key = f"mortality.additional.larva.rate.sp{i}"
        if key in base_config:
            rates[i] = float(base_config[key])   # post-4.4.1 migration => per-dt (~15 for cod)
    return rates


def read_cod_bands(targets) -> dict:
    t = next(x for x in targets if x.species == "cod")
    return {"target": float(t.target), "lower": float(t.lower), "upper": float(t.upper)}


def _default_runner(config, overrides, n_years, seed):
    from calibrate_baltic import run_simulation
    return run_simulation(config, overrides, n_years=n_years, seed=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic Chunk 0 de-risk experiments (v3)")
    ap.add_argument("--experiment", choices=["bistability", "accessibility", "both"], default="both")
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    ap.add_argument("--scales", type=float, nargs="+", default=_DEFAULT_SCALES)
    ap.add_argument("--low-accessibility", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    from calibrate_baltic import load_targets
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    scales = [1.0, 0.1] if args.smoke else args.scales
    years = 3 if args.smoke else args.years

    base_config = read_base_config()
    base_rates = read_base_larva_rates(base_config)
    targets = load_targets()
    cod_bands = read_cod_bands(targets)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"base larva rates (post-migration, per-dt): {base_rates}")

    if args.experiment in ("bistability", "both"):
        out_path = _DIAG_DIR / "baltic_chunk0_bistability.json"
        result = run_bistability_sweep(
            scales, base_config, base_rates, cod_bands, seeds, runner=_default_runner,
            n_years=years, on_point=lambda payload: out_path.write_text(json.dumps(payload, indent=2)))
        print("\n=== BISTABILITY (conservative) ===")
        print(f"establishment fraction (cod-rich reaches a non-collapsed stock): {result['establishment_fraction']:.0%}")
        for p in result["points"]:
            print(f"  larva x{p['scale']:<5} rich={p['rich_state']:<11} poor={p['poor_state']:<11} "
                  f"gap={p['gap']:.3f} rich_seeds={p['per_seed_rich']} -> {p['outcome']}")
        print(f"\nVERDICT: {result['verdict']}")
        out_path.write_text(json.dumps(result, indent=2))

    if args.experiment in ("accessibility", "both"):
        result = run_accessibility_ab(base_config, targets, seeds, runner=_default_runner,
                                      n_years=years, low_value=args.low_accessibility)
        print("\n=== ACCESSIBILITY A/B ===")
        print(f"  n_failed_seeds={result.get('n_failed')}")
        print(f"  baseline_states={result.get('baseline_states')}")
        print(f"  lowered_states={result.get('lowered_states')}")
        print(f"\nVERDICT: {result['verdict']}")
        (_DIAG_DIR / "baltic_chunk0_accessibility_ab.json").write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, lint, and format-check**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q` → PASS (15 tests).
Run: `.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py` → no E402/E702 (clean, or only the intended `# noqa: BLE001`).
Run: `.venv/bin/ruff format --check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py` → clean (run `.venv/bin/ruff format ...` and re-commit if it reports changes).

- [ ] **Step 5: Smoke-run end-to-end**

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment both --smoke`
Expected: prints the read base larva rates, a BISTABILITY block (2 scales, with per-scale outcomes and an establishment fraction) and an ACCESSIBILITY block (with `n_failed_seeds` and per-species states), writes both JSON files, exits 0. Scientific meaning is NOT expected at 3 y; this only proves the harness runs and that `safe_run` handled every point. Note if `establishment_fraction` is 0% at 3 y (expected — cod cannot establish that fast), which is why the real runs use 15 y.

- [ ] **Step 6: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v3 CLI, loaders, lint/format gate, smoke"
```

---

### Task 7: Record the warm-start prerequisite (docs only)

Create `docs/baltic_chunk0_warmstart_prerequisite.md` with this content:

- The Python engine has **no standing-stock init**: `osmose/engine/simulate.py:1188` `initialize()` returns `SchoolState.create(n_schools=0)`; every school is created by the egg-seeding mechanism (SSB==0 → `seeding_biomass` injected as virtual SSB → eggs). `restart.file` / `population.initialization.*` are Java-side allowlist keys with no Python read path.
- **Consequence:** Chunk-0's bistability test can only seed EGGS, filtered through the swept larval mortality and Beverton-Holt compensation — a CONSERVATIVE test that can confirm bistability but cannot rule it out. Moreover, the single-cod-axis ICs vary only cod seeding, so they cannot initialize the real Baltic **cod↔sprat regime-shift alternative state** (a clupeid-dominated start). **A MONOSTABLE result therefore does NOT rule out the sprat-dominated basin.**
- **Minimal primitive for a DEFINITIVE test** (future engine sub-chunk): an age-structured standing-stock initializer (given a per-species initial biomass, distribute it across the size/age structure at t=0 — OSMOSE "initialization by relative biomass") OR a `SchoolState` restart reader. It must be able to seed a standing ADULT clupeid-dominated state (high herring+sprat, near-zero cod) *and* a cod-dominated one, then let both evolve under fixed parameters. Estimated effort: medium (engine + `initialize()` + snapshot format + parity tests). Do this before treating any MONOSTABLE result as definitive.

- [ ] **Step 1: Write the note** (content above).
- [ ] **Step 2: Commit** (`docs(baltic): record warm-start prerequisite for a definitive bistability test`).

---

### Task 8: Run the grid and record results (committed)

- [ ] **Step 1: Run the 15-y grid.**
Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment both --years 15 --seeds 0 1 2`
Expected: ~35–45 min; a BISTABILITY block (per-scale outcomes + establishment fraction + verdict) and an ACCESSIBILITY block (per-species states + verdict); both JSON files written.

- [ ] **Step 2: If bistability is INSTRUMENT-LIMITED or heavily `undetermined`, re-run at `--years 30`** and compare; a basin split (or accessibility relaxation) that survives both horizons is stronger.

- [ ] **Step 3: Record results.** Write `docs/baltic_chunk0_results_YYYY-MM-DD.md` with both verdicts, the establishment fraction, and the roadmap implication (create bistability via Chunks C/A2, add the warm-start primitive per Task 7, or build A1). Commit (`docs(baltic): chunk0 de-risk results`).

---

## The real experiment runs (post-implementation, not CI)

~36 sims at 15 y (~35–45 min, serial). Read the printed verdicts + the two JSON files.

1. `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment both --years 15 --seeds 0 1 2`
   - **Bistability:** trust the verdict only if `establishment_fraction ≥ 0.5`; INSTRUMENT-LIMITED or many `undetermined`/`seed-split` scales ⇒ inconclusive (needs the Task-7 warm-start primitive or more seeds), never a MONOSTABLE claim.
   - **Accessibility:** a `relaxes … A1 is a real lever` verdict now requires more in-range AND no species pushed below its lower band AND no crashes; PROVISIONAL/COLLAPSES/INSTRUMENT-FAILED are all distinct, non-false-positive outcomes.
2. Optional horizon check: re-run bistability at `--years 30`; a basin split that survives both horizons is stronger.

Record verdicts in `docs/baltic_chunk0_results_YYYY-MM-DD.md`.

---

## Self-Review

**Spec coverage.** The Changes table maps all 25 v2 findings to a task; each is implemented and unit-tested: collapse-aware `classify_state` (Task 1, finding 3), collapse-proof/stationarity-gated/count-based accessibility verdict (Task 2, findings 1/2/5/6/9/12/13/19), single-axis ICs + `safe_run` (Task 3, finding 7), per-seed outcomes + `seed-split` + establishment + stable persistence (Task 4, findings 8/16/22/17/18/20/23/25/21/24/11/26), `_failed`-excluded A/B (Task 5, findings 4/10/14), lint/format gate (Task 6, finding 15), warm-start note (Task 7).

**Placeholder scan.** No "TBD"/"handle edge cases"/unshown code; every step has runnable code + an exact command with expected output. All imports at file top; no `;` compound statements.

**Type consistency.** `runner(config, overrides, n_years, seed) -> dict` wrapped by `safe_run` everywhere; one `classify_state(mean, cv, trend, target, lower, upper, ...)` shared by `species_states` and `_cod_state`; `aggregate_states` used identically by both runners; incremental and final bistability payloads share the top-level keys `points/bistable/…/verdict/complete`. Stats keys `{sp}_mean|_cv|_trend` match `run_simulation`; the `_failed` sentinel is the only non-`{sp}_*` key and is checked before any classification.
