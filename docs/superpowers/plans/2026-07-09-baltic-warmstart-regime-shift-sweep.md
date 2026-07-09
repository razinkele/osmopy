# Baltic warm-start regime-shift sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the Chunk-0 bistability harness so it can run the warm-start standing-stock reciprocal-invasion test — a cod-axis contrast and a cod↔clupeid regime-shift contrast — while keeping the egg-only cod-axis path byte-identical.

**Architecture:** All changes are additive to `scripts/baltic_bistability_chunk0.py`. The reviewed v3 machinery (`classify_state`, `basins_differ`, `aggregate_states`, `_median_valid`, the incremental `_partial` writer) is reused unchanged. `run_bistability_point`/`run_bistability_sweep` gain keyword-only params (`ic_a`, `ic_b`, `warmstart`, `contrast`, `clupeid_targets`) whose defaults reproduce the current egg-only cod-axis behavior exactly. A new directional regime-shift verdict is added alongside the existing cod-axis verdict; the sweep dispatches on `contrast`. Real Baltic runs stay CLI-only; all new tests use fake runners.

**Tech Stack:** Python 3.11+, stdlib only (`argparse`, `json`, `statistics`, `math`, `pathlib`). Tests via pytest with the existing fake-runner pattern. Ruff for lint/format.

## Global Constraints

- **No new dependencies.** Stdlib only. `ruff check` and `ruff format --check` must pass on `scripts/ tests/`.
- **Egg-only cod-axis parity is sacred.** With `--warmstart` absent and `contrast="cod-axis"` (the defaults), the harness must produce the exact same behavior and byte-identical JSON as v3. All 19 existing tests in `tests/test_baltic_bistability_chunk0.py` must still pass unchanged.
- **Species→index map (verbatim):** sp0 cod, sp1 herring, sp2 sprat, sp3 flounder, sp4 perch, sp5 pikeperch, sp6 smelt, sp7 stickleback.
- **Warm-start flag (verbatim):** the canonical key `module.population.initialisation.enabled` set to the string `"true"` (read by `osmose/engine/initialization.py`). Init biomass reuses `population.seeding.biomass.sp{i}` — no new config key.
- **Regime-shift IC values (verbatim, from spec §Components):** cod-dominated → cod sp0 `250000`, herring sp1 `800000`, sprat sp2 `600000`; clupeid-dominated → cod sp0 `1000`, herring sp1 `1500000`, sprat sp2 `2500000`.
- **CI discipline:** real emergent Baltic runs are non-reproducible across runner cores (`feedback-ci-fragile-emergent-tests`) — every new automated test uses a fake runner; the real sweep (Task 6) is manual/CLI-only.
- **Test command:** `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`

---

### Task 1: Warm-start injection helper + regime-shift IC builders

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py` (add three functions after `cod_poor_seeding`, ~line 175)
- Test: `tests/test_baltic_bistability_chunk0.py` (append after the Task 6 block, ~line 251)

**Interfaces:**
- Consumes: module constant `_SEEDING_WINDOW_Y` (existing).
- Produces:
  - `warmstart_override(enabled: bool) -> dict` — `{"module.population.initialisation.enabled": "true"}` when `enabled`, else `{}`.
  - `cod_dominated_seeding(window: int = _SEEDING_WINDOW_Y) -> dict` — cod-rich / clupeid-suppressed standing-stock IC override.
  - `clupeid_dominated_seeding(window: int = _SEEDING_WINDOW_Y) -> dict` — cod-remnant / clupeid-rich standing-stock IC override.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baltic_bistability_chunk0.py`:

```python
# ---------------------------------------------------------------- Task 1 (warm-start)
def test_warmstart_override():
    assert c0.warmstart_override(False) == {}
    assert c0.warmstart_override(True) == {"module.population.initialisation.enabled": "true"}


def test_regime_shift_ic_builders():
    cd = c0.cod_dominated_seeding()
    cl = c0.clupeid_dominated_seeding()
    # cod axis: cod high in the cod-dominated IC, a remnant in the clupeid-dominated IC
    assert float(cd["population.seeding.biomass.sp0"]) > float(cl["population.seeding.biomass.sp0"])
    # clupeid axis: herring (sp1) + sprat (sp2) high in clupeid-dominated, suppressed in cod-dominated
    assert float(cl["population.seeding.biomass.sp1"]) > float(cd["population.seeding.biomass.sp1"])
    assert float(cl["population.seeding.biomass.sp2"]) > float(cd["population.seeding.biomass.sp2"])
    # exact spec values
    assert cd["population.seeding.biomass.sp0"] == "250000"
    assert cl["population.seeding.biomass.sp2"] == "2500000"
    # both carry the (now-inert-under-warmstart) global seeding window key
    assert "population.seeding.year.max" in cd and "population.seeding.year.max" in cl
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k "warmstart_override or regime_shift_ic_builders" -v`
Expected: FAIL — `AttributeError: module 'baltic_bistability_chunk0' has no attribute 'warmstart_override'`

- [ ] **Step 3: Implement the three functions**

In `scripts/baltic_bistability_chunk0.py`, immediately after `cod_poor_seeding` (which ends at ~line 175), insert:

```python
def warmstart_override(enabled: bool) -> dict:
    """Merge-in override that turns the warm-start standing-stock init ON (canonical flag).
    Empty when off, so an egg-only run's overrides stay byte-identical."""
    return {_ENABLE_KEY: "true"} if enabled else {}


def cod_dominated_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    """Cod-dominated standing-stock IC: cod at the ICES upper band, clupeids suppressed."""
    return {
        "population.seeding.biomass.sp0": "250000",  # cod, ICES upper
        "population.seeding.biomass.sp1": "800000",  # herring, lower
        "population.seeding.biomass.sp2": "600000",  # sprat, suppressed
        "population.seeding.year.max": str(window),
    }


def clupeid_dominated_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    """Clupeid-dominated (sprat-dominated) standing-stock IC: cod a remnant/invader,
    herring + sprat at target/upper — the real post-1990 Baltic regime."""
    return {
        "population.seeding.biomass.sp0": "1000",  # cod, remnant/invader
        "population.seeding.biomass.sp1": "1500000",  # herring, target
        "population.seeding.biomass.sp2": "2500000",  # sprat, upper
        "population.seeding.year.max": str(window),
    }
```

Add the module-level constant near the other constants (after `_NONBANDS`, ~line 23) so `warmstart_override` and later tasks share one spelling:

```python
_ENABLE_KEY = "module.population.initialisation.enabled"  # canonical warm-start flag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k "warmstart_override or regime_shift_ic_builders" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full test file to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (21 passed)

- [ ] **Step 6: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
.venv/bin/ruff format scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): warm-start injection helper + regime-shift IC builders"
```

---

### Task 2: Clupeid-axis signal helper

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py` (add `_VALID_BANDS` constant + `clupeid_axis` after `_median_valid`, ~line 207)
- Test: `tests/test_baltic_bistability_chunk0.py` (append)

**Interfaces:**
- Consumes: `classify_state`, `aggregate_states` (existing); each `t` in `clupeid_targets` is a target with `.species`, `.target`, `.lower`, `.upper` (the `Tgt` namedtuple / `BiomassTarget`).
- Produces:
  - `_VALID_BANDS = ("collapsed", "low", "in_range", "overshoot")` — the determinate (non-`seed-split`, non-`undetermined`) bands.
  - `clupeid_axis(runs, clupeid_targets) -> tuple[float, bool]` — `runs` is the list of per-seed `safe_run` dicts for ONE arm. Returns `(median_summed_biomass, valid)`: the median over non-failed seeds of `herring_mean + sprat_mean`, and `valid=True` iff **both** stocks aggregate to a determinate band across seeds (stationary + seed-consensus). `(0.0, False)` if every seed failed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baltic_bistability_chunk0.py` (note: `_stats` and `Tgt` already exist near the top of the file):

```python
# ---------------------------------------------------------------- Task 2 (clupeid axis)
def _clup_targets():
    return [
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]


def test_clupeid_axis_valid_and_sum():
    runs = [
        _stats(herring=1_500_000, sprat=2_500_000),
        _stats(herring=1_500_000, sprat=2_500_000),
    ]
    biomass, valid = c0.clupeid_axis(runs, _clup_targets())
    assert valid is True
    assert biomass == 4_000_000


def test_clupeid_axis_nonstationary_is_invalid():
    drifting = _stats(herring=1_500_000, sprat=2_500_000)
    drifting["herring_cv"] = 0.9  # non-stationary -> herring 'undetermined'
    biomass, valid = c0.clupeid_axis([drifting, drifting], _clup_targets())
    assert valid is False


def test_clupeid_axis_seed_split_is_invalid():
    runs = [
        _stats(herring=1_500_000, sprat=2_500_000),  # in_range
        _stats(herring=100_000, sprat=2_500_000),  # herring 'collapsed' -> disagreement
    ]
    _, valid = c0.clupeid_axis(runs, _clup_targets())
    assert valid is False


def test_clupeid_axis_all_failed():
    biomass, valid = c0.clupeid_axis([{"_failed": True}], _clup_targets())
    assert biomass == 0.0 and valid is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k clupeid_axis -v`
Expected: FAIL — `AttributeError: ... has no attribute 'clupeid_axis'`

- [ ] **Step 3: Implement `_VALID_BANDS` + `clupeid_axis`**

In `scripts/baltic_bistability_chunk0.py`, add the constant near the other constants (after `_NONBANDS`):

```python
_VALID_BANDS = ("collapsed", "low", "in_range", "overshoot")  # determinate bands
```

Then insert `clupeid_axis` after `_median_valid` (~line 207):

```python
def clupeid_axis(runs, clupeid_targets) -> tuple[float, bool]:
    """Clupeid regime signal for ONE arm: median summed herring+sprat biomass over non-failed
    seeds, plus a validity flag. Valid iff BOTH stocks aggregate to a determinate band across
    seeds (stationary + seed-consensus). Summing sidesteps banding two stocks with different
    ICES ranges; validity gating mirrors the cod-axis stationarity discipline."""
    bands = {t.species: [] for t in clupeid_targets}
    sums = []
    for st in runs:
        if st.get("_failed"):
            continue
        total = 0.0
        for t in clupeid_targets:
            mean = float(st.get(f"{t.species}_mean", 0.0))
            total += mean
            bands[t.species].append(
                classify_state(
                    mean,
                    float(st.get(f"{t.species}_cv", 10.0)),
                    float(st.get(f"{t.species}_trend", 1.0)),
                    float(t.target),
                    float(t.lower),
                    float(t.upper),
                )
            )
        sums.append(total)
    if not sums:
        return 0.0, False
    valid = all(aggregate_states(bands[t.species]) in _VALID_BANDS for t in clupeid_targets)
    return statistics.median(sums), valid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k clupeid_axis -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full test file**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (25 passed)

- [ ] **Step 6: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
.venv/bin/ruff format scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): clupeid-axis signal helper (summed biomass + validity)"
```

---

### Task 3: Outcome helpers — cod-axis refactor + regime-shift verdict

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py` (add `_ESTABLISHED` + two functions before `run_bistability_point`, ~line 222; refactor the inline outcome branch inside `run_bistability_point`, lines 244-252)
- Test: `tests/test_baltic_bistability_chunk0.py` (append)

**Interfaces:**
- Consumes: `basins_differ`, `bistability_gap` (existing).
- Produces:
  - `_ESTABLISHED = ("low", "in_range", "overshoot")` — cod-present bands.
  - `cod_axis_outcome(rich_agg, poor_agg, gap) -> str` — returns one of `"seed-split"`, `"undetermined"`, `"bistable"`, `"same-basin"`. Exactly the existing inline logic, extracted.
  - `regime_shift_outcome(cod_a, cod_b, clup_a, clup_b, clup_a_valid, clup_b_valid, gap_thresh: float = 0.5) -> str` — returns `"provisional"`, `"regime-shift"`, `"partial"`, or `"same-basin"`. `cod_a`/`cod_b` are the consensus cod bands for the cod-dominated (A) and clupeid-dominated (B) arms; `clup_a`/`clup_b` the summed clupeid biomasses; `clup_*_valid` the clupeid validity flags.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baltic_bistability_chunk0.py`:

```python
# ---------------------------------------------------------------- Task 3 (outcome helpers)
def test_cod_axis_outcome_extracted_logic():
    assert c0.cod_axis_outcome("in_range", "collapsed", 0.9) == "bistable"
    assert c0.cod_axis_outcome("seed-split", "in_range", 0.0) == "seed-split"
    assert c0.cod_axis_outcome("undetermined", "in_range", 0.0) == "undetermined"
    assert c0.cod_axis_outcome("in_range", "in_range", 0.1) == "same-basin"
    assert c0.cod_axis_outcome("in_range", "in_range", 0.8) == "bistable"  # gap-driven split


def test_regime_shift_outcome_both_axes_diverge():
    # cod persists in cod-dominated arm (a), collapses in clupeid-dominated arm (b);
    # clupeids boom in b (4.0M) vs suppressed in a (0.5M)
    assert (
        c0.regime_shift_outcome("in_range", "collapsed", 500_000.0, 4_000_000.0, True, True)
        == "regime-shift"
    )


def test_regime_shift_outcome_cod_only_is_partial():
    # cod diverges but clupeid gap is tiny (3.9M vs 4.0M)
    assert (
        c0.regime_shift_outcome("in_range", "collapsed", 3_900_000.0, 4_000_000.0, True, True)
        == "partial"
    )


def test_regime_shift_outcome_clupeid_only_is_partial():
    # clupeids diverge but cod persists in BOTH arms (no collapse in b)
    assert (
        c0.regime_shift_outcome("in_range", "in_range", 500_000.0, 4_000_000.0, True, True)
        == "partial"
    )


def test_regime_shift_outcome_neither_is_monostable():
    assert (
        c0.regime_shift_outcome("in_range", "in_range", 3_900_000.0, 4_000_000.0, True, True)
        == "same-basin"
    )


def test_regime_shift_outcome_withheld_when_undetermined_or_invalid():
    # cod arm undetermined -> provisional
    assert (
        c0.regime_shift_outcome("seed-split", "collapsed", 500_000.0, 4_000_000.0, True, True)
        == "provisional"
    )
    # clupeid arm invalid -> provisional
    assert (
        c0.regime_shift_outcome("in_range", "collapsed", 500_000.0, 4_000_000.0, False, True)
        == "provisional"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k "cod_axis_outcome or regime_shift_outcome" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'cod_axis_outcome'`

- [ ] **Step 3: Implement the two outcome helpers**

In `scripts/baltic_bistability_chunk0.py`, insert directly before `run_bistability_point` (~line 222):

```python
_ESTABLISHED = ("low", "in_range", "overshoot")  # cod-present (non-collapsed) bands


def cod_axis_outcome(rich_agg, poor_agg, gap) -> str:
    """Cod-axis point outcome (extracted verbatim from the v3 inline branch)."""
    if rich_agg == "seed-split" or poor_agg == "seed-split":
        return "seed-split"
    if rich_agg == "undetermined" or poor_agg == "undetermined":
        return "undetermined"
    if basins_differ(rich_agg, poor_agg, gap):
        return "bistable"
    return "same-basin"


def regime_shift_outcome(
    cod_a, cod_b, clup_a, clup_b, clup_a_valid, clup_b_valid, gap_thresh: float = 0.5
) -> str:
    """Directional regime-shift point outcome. A regime shift is the SPECIFIC pattern of cod
    down where clupeids are up, so BOTH axes must diverge in that direction:
      - cod-collapse axis: cod persists in the cod-dominated arm (a) AND is collapsed in the
        clupeid-dominated arm (b);
      - clupeid-boom axis: summed clupeid biomass is higher in b than a by a relative gap.
    Any non-stationary / seed-split / invalid gated arm withholds the call ('provisional')."""
    if cod_a in ("seed-split", "undetermined") or cod_b in ("seed-split", "undetermined"):
        return "provisional"
    if not (clup_a_valid and clup_b_valid):
        return "provisional"
    cod_diverge = cod_a in _ESTABLISHED and cod_b == "collapsed"
    clup_diverge = clup_b > clup_a and bistability_gap(clup_a, clup_b) >= gap_thresh
    if cod_diverge and clup_diverge:
        return "regime-shift"
    if cod_diverge or clup_diverge:
        return "partial"
    return "same-basin"
```

- [ ] **Step 4: Refactor `run_bistability_point` to call `cod_axis_outcome`**

In `run_bistability_point`, replace the inline branch (currently lines 245-252):

```python
    if rich_agg == "seed-split" or poor_agg == "seed-split":
        outcome = "seed-split"
    elif rich_agg == "undetermined" or poor_agg == "undetermined":
        outcome = "undetermined"
    elif basins_differ(rich_agg, poor_agg, gap):
        outcome = "bistable"
    else:
        outcome = "same-basin"
```

with:

```python
    outcome = cod_axis_outcome(rich_agg, poor_agg, gap)
```

This is behavior-preserving — the returned dict and JSON are unchanged.

- [ ] **Step 5: Run the new tests AND the full file (parity check)**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (31 passed) — including the existing `test_point_detects_bistable_including_collapsed_basin`, `test_seed_split_outcome`, `test_sweep_verdict_and_stable_persistence`, which exercise the refactored path.

- [ ] **Step 6: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
.venv/bin/ruff format scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): extract cod-axis outcome + add directional regime-shift verdict"
```

---

### Task 4: Generalize `run_bistability_point` and `run_bistability_sweep`

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py` (`run_bistability_point` ~lines 223-265; `run_bistability_sweep` ~lines 268-325; add `_regime_shift_verdict` before the sweep)
- Test: `tests/test_baltic_bistability_chunk0.py` (append)

**Interfaces:**
- Consumes: `warmstart_override` (Task 1), `clupeid_axis` (Task 2), `cod_axis_outcome`/`regime_shift_outcome` (Task 3), IC builders (Task 1), and the existing `larva_scale_override`, `safe_run`, `_cod_state`, `aggregate_states`, `_median_valid`, `bistability_gap`, `_partial`.
- Produces:
  - `run_bistability_point(scale, base_config, base_rates, cod_bands, seeds, *, runner, n_years, ic_a=cod_rich_seeding, ic_b=cod_poor_seeding, warmstart=False, contrast="cod-axis", clupeid_targets=None) -> dict` — cod-axis dict unchanged when defaults hold; when `contrast="regime-shift"` also carries `a_clupeid_biomass`, `b_clupeid_biomass`, `a_clupeid_valid`, `b_clupeid_valid`, `clupeid_gap`, and a regime-shift `outcome`/`regime_shift`.
  - `run_bistability_sweep(scales, base_config, base_rates, cod_bands, seeds, *, runner, n_years, on_point=None, ic_a=cod_rich_seeding, ic_b=cod_poor_seeding, warmstart=False, contrast="cod-axis", clupeid_targets=None) -> dict` — dispatches to the existing cod-axis verdict or `_regime_shift_verdict`.
  - `_regime_shift_verdict(points) -> dict` — sweep-level regime-shift verdict with keys `points, contrast, regime_shift, bistable, regime_shift_scales, partial_scales, provisional_scales, determinate_fraction, trustworthy, verdict, complete`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baltic_bistability_chunk0.py`:

```python
# ---------------------------------------------------------------- Task 4 (generalized sweep)
def _runner_regime(config, overrides, n_years, seed):
    """Cod-dominated arm (cod seed >= 100k) -> cod in_range + clupeids 'low';
    clupeid-dominated arm -> cod collapsed + clupeids booming."""
    cod_seed = float(overrides.get("population.seeding.biomass.sp0", "0"))
    if cod_seed >= 100_000:
        return _stats(cod=120_000, herring=400_000, sprat=300_000)
    return _stats(cod=0, herring=1_500_000, sprat=2_500_000)


def test_point_regime_shift_records_clupeid_and_outcome():
    pt = c0.run_bistability_point(
        1.0, {}, {0: 15.0}, _bands(), [0, 1, 2],
        runner=_runner_regime, n_years=15,
        ic_a=c0.cod_dominated_seeding, ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift", clupeid_targets=_clup_targets(),
    )
    assert pt["rich_state"] == "in_range"  # cod persists in cod-dominated arm
    assert pt["poor_state"] == "collapsed"  # cod collapses in clupeid-dominated arm
    assert pt["b_clupeid_biomass"] > pt["a_clupeid_biomass"]
    assert pt["a_clupeid_valid"] is True and pt["b_clupeid_valid"] is True
    assert pt["outcome"] == "regime-shift"
    assert pt["regime_shift"] is True


def test_regime_shift_sweep_verdict_and_incremental():
    seen = []
    out = c0.run_bistability_sweep(
        [1.0, 0.3], {}, {0: 15.0}, _bands(), [0, 1, 2],
        runner=_runner_regime, n_years=15, on_point=seen.append,
        ic_a=c0.cod_dominated_seeding, ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift", clupeid_targets=_clup_targets(),
    )
    assert out["regime_shift"] is True
    assert 1.0 in out["regime_shift_scales"]
    assert "regime shift" in out["verdict"].lower()
    assert out["complete"] is True
    assert seen[0]["complete"] is False


def test_regime_shift_sweep_monostable_when_convergent():
    def convergent(config, overrides, n_years, seed):
        # both arms -> cod in_range + clupeids in_range: no divergence on either axis
        return _stats(cod=120_000, herring=1_500_000, sprat=1_500_000)

    out = c0.run_bistability_sweep(
        [1.0, 0.3], {}, {0: 15.0}, _bands(), [0, 1],
        runner=convergent, n_years=15,
        ic_a=c0.cod_dominated_seeding, ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift", clupeid_targets=_clup_targets(),
    )
    assert out["regime_shift"] is False
    assert "monostable" in out["verdict"].lower()


def test_warmstart_flag_injected_into_overrides():
    captured = []

    def spy(config, overrides, n_years, seed):
        captured.append(dict(overrides))
        return _stats(cod=120_000, herring=400_000, sprat=300_000)

    c0.run_bistability_point(
        1.0, {}, {0: 15.0}, _bands(), [0],
        runner=spy, n_years=5, warmstart=True,
        ic_a=c0.cod_dominated_seeding, ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift", clupeid_targets=_clup_targets(),
    )
    assert captured  # both arms ran
    assert all(o.get("module.population.initialisation.enabled") == "true" for o in captured)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k "regime_shift_records or regime_shift_sweep or warmstart_flag_injected" -v`
Expected: FAIL — `TypeError: run_bistability_point() got an unexpected keyword argument 'ic_a'`

- [ ] **Step 3: Rewrite `run_bistability_point`**

Replace the whole `run_bistability_point` function (lines 223-265) with:

```python
def run_bistability_point(
    scale,
    base_config,
    base_rates,
    cod_bands,
    seeds,
    *,
    runner,
    n_years,
    ic_a=cod_rich_seeding,
    ic_b=cod_poor_seeding,
    warmstart=False,
    contrast="cod-axis",
    clupeid_targets=None,
) -> dict:
    driver = larva_scale_override(scale, base_rates)
    ws = warmstart_override(warmstart)
    rich_states, poor_states, rich_means, poor_means = [], [], [], []
    a_runs, b_runs = [], []
    for seed in seeds:
        r = safe_run(runner, base_config, {**driver, **ic_a(), **ws}, n_years, seed)
        p = safe_run(runner, base_config, {**driver, **ic_b(), **ws}, n_years, seed)
        a_runs.append(r)
        b_runs.append(p)
        rs, rm = _cod_state(r, cod_bands)
        ps, pm = _cod_state(p, cod_bands)
        rich_states.append(rs)
        poor_states.append(ps)
        rich_means.append(rm)
        poor_means.append(pm)
    rich_agg = aggregate_states(rich_states)  # consensus band or 'seed-split'
    poor_agg = aggregate_states(poor_states)
    rich_med = _median_valid(rich_states, rich_means)
    poor_med = _median_valid(poor_states, poor_means)
    gap = bistability_gap(rich_med, poor_med)
    established = rich_agg in ("low", "in_range", "overshoot")
    out = {
        "scale": scale,
        "rich_state": rich_agg,
        "poor_state": poor_agg,
        "per_seed_rich": rich_states,
        "per_seed_poor": poor_states,
        "rich_cod_median": rich_med,
        "poor_cod_median": poor_med,
        "gap": gap,
        "established": established,
    }
    if contrast == "regime-shift":
        ct = clupeid_targets or []
        clup_a, clup_a_valid = clupeid_axis(a_runs, ct)
        clup_b, clup_b_valid = clupeid_axis(b_runs, ct)
        outcome = regime_shift_outcome(
            rich_agg, poor_agg, clup_a, clup_b, clup_a_valid, clup_b_valid
        )
        out.update(
            {
                "a_clupeid_biomass": clup_a,
                "b_clupeid_biomass": clup_b,
                "a_clupeid_valid": clup_a_valid,
                "b_clupeid_valid": clup_b_valid,
                "clupeid_gap": bistability_gap(clup_a, clup_b),
                "outcome": outcome,
                "regime_shift": outcome == "regime-shift",
            }
        )
    else:
        outcome = cod_axis_outcome(rich_agg, poor_agg, gap)
        out.update({"outcome": outcome, "bistable": outcome == "bistable"})
    return out
```

Parity note: for `contrast="cod-axis"` with default ICs and `warmstart=False`, `ws={}` and `{**driver, **ic_a()}` equals the original `{**driver, **cod_rich_seeding()}`; the `out` dict is built in the original key order, so the cod-axis JSON is byte-identical.

- [ ] **Step 4: Add `_regime_shift_verdict` and generalize `run_bistability_sweep`**

Insert `_regime_shift_verdict` directly before `run_bistability_sweep`:

```python
def _regime_shift_verdict(points) -> dict:
    shift = [p["scale"] for p in points if p["outcome"] == "regime-shift"]
    partial = [p["scale"] for p in points if p["outcome"] == "partial"]
    provisional = [p["scale"] for p in points if p["outcome"] == "provisional"]
    det = [p for p in points if p["outcome"] != "provisional"]
    det_frac = len(det) / len(points) if points else 0.0
    trustworthy = det_frac >= 0.5
    if not trustworthy:
        verdict = (
            f"INSTRUMENT-LIMITED — only {det_frac:.0%} of scales gave a determinate outcome "
            f"(provisional at {provisional}); withhold. Raise --seeds/--years."
        )
    elif shift:
        verdict = (
            f"REGIME SHIFT / BISTABLE — both axes diverge in the regime-shift direction at "
            f"scale(s) {shift}: cod persists in the cod-dominated IC and collapses in the "
            f"clupeid-dominated IC, while clupeids boom. SCRUTINIZE before trusting — re-run "
            f"with more seeds and rule out a seeding/parameter artifact (Chunks C & A2 are the "
            f"expected source of a real second attractor)."
        )
    elif partial:
        verdict = (
            f"PARTIAL — NOT a regime shift. Only one axis moved at scale(s) {partial} "
            f"(cod-only or clupeid-only); the other axis is monostable. A regime shift "
            f"requires BOTH axes to diverge."
        )
    else:
        verdict = (
            f"MONOSTABLE (warm-start) — cod-dominated and clupeid-dominated standing-stock ICs "
            f"converge at every determinate scale (provisional: {provisional}). No alternative "
            f"regime-shift attractor under the deployed parameters; bistability must be CREATED "
            f"(Chunk C clupeid->cod-egg predation; Chunk A2 depletable plankton)."
        )
    return {
        "points": points,
        "contrast": "regime-shift",
        "regime_shift": bool(shift) and trustworthy,
        "bistable": bool(shift) and trustworthy,
        "regime_shift_scales": shift,
        "partial_scales": partial,
        "provisional_scales": provisional,
        "determinate_fraction": det_frac,
        "trustworthy": trustworthy,
        "verdict": verdict,
        "complete": True,
    }
```

Then update `run_bistability_sweep`'s signature and point-loop, and dispatch on `contrast`. Change the signature line and the `run_bistability_point(...)` call, and add the dispatch immediately after the loop (before the existing cod-axis verdict code). The new signature:

```python
def run_bistability_sweep(
    scales,
    base_config,
    base_rates,
    cod_bands,
    seeds,
    *,
    runner,
    n_years,
    on_point=None,
    ic_a=cod_rich_seeding,
    ic_b=cod_poor_seeding,
    warmstart=False,
    contrast="cod-axis",
    clupeid_targets=None,
) -> dict:
    points = []
    for s in scales:
        pt = run_bistability_point(
            s,
            base_config,
            base_rates,
            cod_bands,
            seeds,
            runner=runner,
            n_years=n_years,
            ic_a=ic_a,
            ic_b=ic_b,
            warmstart=warmstart,
            contrast=contrast,
            clupeid_targets=clupeid_targets,
        )
        points.append(pt)
        if on_point is not None:
            on_point(_partial(points))
    if contrast == "regime-shift":
        return _regime_shift_verdict(points)
    # ---- cod-axis verdict (unchanged from v3) ----
    bistable = [p["scale"] for p in points if p["outcome"] == "bistable"]
```

Leave the rest of the cod-axis verdict body (from `seed_split = [...]` through the final `return {...}`) exactly as it is.

- [ ] **Step 5: Run the new tests AND the full file (parity check)**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (35 passed) — the four existing Task-4/5 cod-axis tests still pass because their calls omit the new kwargs (defaults reproduce v3).

- [ ] **Step 6: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
.venv/bin/ruff format scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): generalize bistability sweep with warm-start + regime-shift contrast"
```

---

### Task 5: CLI wiring, contrast selection, pre-flight check

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py` (add `import math`; add `_load_targets`, `_clupeid_targets_from`, `contrast_specs`, `preflight_check`; add CLI flags and the warm-start branch in `main`; change `main` to call `_load_targets()`)
- Test: `tests/test_baltic_bistability_chunk0.py` (append)

**Interfaces:**
- Consumes: IC builders (Task 1), `run_bistability_sweep` (Task 4), `safe_run`, `larva_scale_override`, `warmstart_override`, `read_base_config`, `read_base_larva_rates`, `read_cod_bands`, `_default_runner`, `_DIAG_DIR`.
- Produces:
  - `_load_targets() -> list` — module-level wrapper around `calibrate_baltic.load_targets` (so `main` is monkeypatchable).
  - `_clupeid_targets_from(targets) -> list` — the herring + sprat targets.
  - `contrast_specs(contrast: str, targets) -> list[dict]` — sweep specs; each dict has `label`, `ic_a`, `ic_b`, `clupeid_targets`, `out_name`.
  - `preflight_check(stats: dict, species=("cod", "herring", "sprat")) -> tuple[bool, str]` — `(ok, message)` de-risk gate on a single standing-stock run.
  - `main` accepts `--warmstart`, `--contrast {cod-axis,regime-shift,both}`, `--preflight`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baltic_bistability_chunk0.py`:

```python
# ---------------------------------------------------------------- Task 5 (CLI + preflight)
def test_contrast_specs():
    tgts = [
        Tgt("cod", 120_000, 60_000, 250_000),
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]
    both = c0.contrast_specs("both", tgts)
    assert [s["label"] for s in both] == ["cod-axis", "regime-shift"]
    assert both[0]["clupeid_targets"] is None
    assert both[1]["ic_a"] is c0.cod_dominated_seeding
    assert both[1]["ic_b"] is c0.clupeid_dominated_seeding
    assert {t.species for t in both[1]["clupeid_targets"]} == {"herring", "sprat"}
    assert both[1]["out_name"] == "baltic_chunk0_warmstart_bistability_regime-shift.json"
    assert len(c0.contrast_specs("cod-axis", tgts)) == 1
    assert len(c0.contrast_specs("regime-shift", tgts)) == 1


def test_preflight_check():
    ok, msg = c0.preflight_check(_stats(cod=120_000, herring=800_000, sprat=600_000))
    assert ok is True and "ok" in msg.lower()
    assert c0.preflight_check({"_failed": True, "_error": "boom"})[0] is False
    nan_stats = {"cod_mean": float("nan"), "herring_mean": 1.0, "sprat_mean": 1.0}
    assert c0.preflight_check(nan_stats)[0] is False
    assert c0.preflight_check(_stats(cod=0, herring=0, sprat=0))[0] is False


def test_cli_warmstart_writes_both_contrasts(tmp_path, monkeypatch):
    tgts = [
        Tgt("cod", 120_000, 60_000, 250_000),
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]
    monkeypatch.setattr(c0, "read_base_config", lambda: {})
    monkeypatch.setattr(c0, "read_base_larva_rates", lambda cfg, n_focal=8: {0: 15.0})
    monkeypatch.setattr(c0, "_load_targets", lambda: tgts)
    monkeypatch.setattr(c0, "_default_runner", _runner_regime)
    monkeypatch.setattr(c0, "_DIAG_DIR", tmp_path)
    rc = c0.main(["--warmstart", "--contrast", "both", "--smoke"])
    assert rc == 0
    assert (tmp_path / "baltic_chunk0_warmstart_bistability_cod-axis.json").exists()
    assert (tmp_path / "baltic_chunk0_warmstart_bistability_regime-shift.json").exists()


def test_cli_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(c0, "read_base_config", lambda: {})
    monkeypatch.setattr(c0, "read_base_larva_rates", lambda cfg, n_focal=8: {0: 15.0})
    monkeypatch.setattr(c0, "_load_targets", lambda: [Tgt("cod", 120_000, 60_000, 250_000)])
    monkeypatch.setattr(c0, "_default_runner", _runner_regime)
    monkeypatch.setattr(c0, "_DIAG_DIR", tmp_path)
    rc = c0.main(["--preflight"])
    assert rc == 0  # _runner_regime cod-dominated arm returns a persisting stock
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k "contrast_specs or preflight or cli_warmstart or cli_preflight" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'contrast_specs'`

- [ ] **Step 3: Add `import math` and the four helper functions**

At the top of `scripts/baltic_bistability_chunk0.py`, add `import math` to the imports block (after `import json`).

Add `_load_targets` next to the other loaders (after `read_cod_bands`, ~line 384):

```python
def _load_targets():
    from calibrate_baltic import load_targets

    return load_targets()
```

Add the contrast + preflight helpers after `_default_runner` (~line 389):

```python
def _clupeid_targets_from(targets):
    return [t for t in targets if t.species in ("herring", "sprat")]


def contrast_specs(contrast: str, targets) -> list[dict]:
    """Sweep specs to run for the requested contrast (label, IC pair, clupeid targets, out file)."""
    cod_axis = {
        "label": "cod-axis",
        "ic_a": cod_rich_seeding,
        "ic_b": cod_poor_seeding,
        "clupeid_targets": None,
        "out_name": "baltic_chunk0_warmstart_bistability_cod-axis.json",
    }
    regime = {
        "label": "regime-shift",
        "ic_a": cod_dominated_seeding,
        "ic_b": clupeid_dominated_seeding,
        "clupeid_targets": _clupeid_targets_from(targets),
        "out_name": "baltic_chunk0_warmstart_bistability_regime-shift.json",
    }
    if contrast == "cod-axis":
        return [cod_axis]
    if contrast == "regime-shift":
        return [regime]
    return [cod_axis, regime]


def preflight_check(stats: dict, species=("cod", "herring", "sprat")) -> tuple[bool, str]:
    """De-risk gate: one standing-stock run must complete finite and non-vanishing.
    A pathological t=0 decay is itself a finding — stop, do not run the full sweep."""
    if stats.get("_failed"):
        return False, f"FAILED — run crashed/empty: {stats.get('_error')}"
    total = 0.0
    for sp in species:
        mean = stats.get(f"{sp}_mean")
        if mean is None or not math.isfinite(float(mean)):
            return False, f"NON-FINITE — {sp}_mean = {mean!r}"
        total += float(mean)
    if total <= 0.0:
        return False, (
            "VANISHED — cod+herring+sprat summed to zero; the standing-stock IC is not "
            "self-consistent with the deployed parameters. Stop and reassess."
        )
    return True, f"OK — standing stock persists (cod+herring+sprat mean = {total:.0f} t)."
```

- [ ] **Step 4: Wire the CLI (`main`)**

In `main`, add the three arguments after the existing `--smoke` line:

```python
    ap.add_argument("--warmstart", action="store_true")
    ap.add_argument("--contrast", choices=["cod-axis", "regime-shift", "both"], default="cod-axis")
    ap.add_argument("--preflight", action="store_true")
```

Change the local targets import — replace `from calibrate_baltic import load_targets` and the later `targets = load_targets()` with a single call to the module-level wrapper. The `targets = load_targets()` line becomes:

```python
    targets = _load_targets()
```

(and delete the now-unused `from calibrate_baltic import load_targets` line).

Then, immediately after `print(f"base larva rates ...")` (~line 415) and BEFORE the existing `if args.experiment in ("bistability", "both"):` block, insert the pre-flight and warm-start branches:

```python
    if args.preflight:
        ic = cod_dominated_seeding()
        driver = larva_scale_override(1.0, base_rates)
        stats = safe_run(
            _default_runner,
            base_config,
            {**driver, **ic, **warmstart_override(True)},
            years,
            seeds[0],
        )
        ok, msg = preflight_check(stats)
        print(f"\n=== PRE-FLIGHT (cod-dominated standing stock, warm-start ON) ===\n{msg}")
        return 0 if ok else 1

    if args.warmstart:
        for spec in contrast_specs(args.contrast, targets):
            out_path = _DIAG_DIR / spec["out_name"]
            result = run_bistability_sweep(
                scales,
                base_config,
                base_rates,
                cod_bands,
                seeds,
                runner=_default_runner,
                n_years=years,
                ic_a=spec["ic_a"],
                ic_b=spec["ic_b"],
                warmstart=True,
                contrast=spec["label"],
                clupeid_targets=spec["clupeid_targets"],
                on_point=lambda payload, p=out_path: p.write_text(json.dumps(payload, indent=2)),
            )
            print(f"\n=== WARM-START {spec['label'].upper()} ===")
            for pt in result["points"]:
                print(f"  larva x{pt['scale']:<5} outcome={pt['outcome']}")
            print(f"VERDICT: {result['verdict']}")
            out_path.write_text(json.dumps(result, indent=2))
        return 0
```

The existing `--experiment` egg-only blocks remain unchanged below this and only run when `--warmstart` is absent (byte-identical to v3).

- [ ] **Step 5: Run the new tests AND the full file**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (39 passed)

- [ ] **Step 6: Confirm egg-only parity end-to-end (smoke, fake runner path)**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q -k "sweep or point or ab or loaders"`
Expected: PASS — the cod-axis sweep/point tests confirm the default path is unchanged.

- [ ] **Step 7: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
.venv/bin/ruff format scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): CLI --warmstart/--contrast/--preflight + contrast selection"
```

---

### Task 6: Pre-flight + full sweep (real engine, manual) and results write-up

**This task is real-engine, CLI-only, NOT CI** — it produces the scientific result. Run it only after Tasks 1–5 are merged (or on the feature branch after all unit tests pass). Real Baltic emergent runs are non-reproducible across cores, so there is no automated test here; verification is that the runs complete, the pre-flight passes, the JSON outputs are written, and the results doc is filled from them.

**Files:**
- Create: `docs/baltic_chunk0_warmstart_results_2026-07-09.md` (write-up)
- Produces (real outputs): `docs/diagnostics/baltic_chunk0_warmstart_bistability_cod-axis.json`, `docs/diagnostics/baltic_chunk0_warmstart_bistability_regime-shift.json`

- [ ] **Step 1: Pre-flight de-risk (must pass before the full sweep)**

Run one cod-dominated standing stock forward ~5 y:

```bash
cd /home/razinka/osmopy
.venv/bin/python scripts/baltic_bistability_chunk0.py --preflight --years 5
```
Expected: `=== PRE-FLIGHT ... === OK — standing stock persists (...)`, exit code 0.
If it reports `FAILED`, `NON-FINITE`, or `VANISHED`: **STOP.** A standing stock that decays pathologically at t=0 means the IC is not self-consistent with the deployed parameters — that is itself a finding. Record it in the results doc and do not run the full sweep.

- [ ] **Step 2: Fast smoke of both contrasts (1 seed, 2 scales, 3 y)**

```bash
.venv/bin/python scripts/baltic_bistability_chunk0.py --warmstart --contrast both --smoke
```
Expected: both `=== WARM-START COD-AXIS ===` and `=== WARM-START REGIME-SHIFT ===` sections print a VERDICT; both JSON files appear under `docs/diagnostics/`. This confirms the real engine accepts the warm-start flag and the IC overrides without a crash/NaN/1e22 blow-up.

- [ ] **Step 3: Full sweep (both contrasts, default 5 scales × 3 seeds × 15 y)**

```bash
.venv/bin/python scripts/baltic_bistability_chunk0.py --warmstart --contrast both 2>&1 | tee /tmp/warmstart_sweep.log
```
Expected: ~60 real Baltic runs, order 1–3 h wall clock. Incremental JSON is written after each point (safe to inspect mid-run). Two final JSONs written.

- [ ] **Step 4: Write the results doc**

Create `docs/baltic_chunk0_warmstart_results_2026-07-09.md` mirroring `docs/baltic_chunk0_results_2026-07-08.md`. Include, filled from the two JSONs:
- the pre-flight outcome;
- a cod-axis table (per larva scale: rich_state, poor_state, gap, outcome) and its verdict;
- a regime-shift table (per larva scale: rich/poor cod band, a/b clupeid biomass, clupeid_gap, outcome) and its verdict;
- the honest-scope interpretation from the spec (a t=0 standing stock does not manufacture a second attractor; a MONOSTABLE result confirms monostability more rigorously; a REGIME-SHIFT result must be scrutinized, not celebrated);
- the follow-on: if monostable, the roadmap is unchanged — bistability must be created (Chunk C clupeid→cod-egg predation; Chunk A2 depletable plankton).

- [ ] **Step 5: Commit the results and outputs**

```bash
cd /home/razinka/osmopy
git add docs/baltic_chunk0_warmstart_results_2026-07-09.md \
        docs/diagnostics/baltic_chunk0_warmstart_bistability_cod-axis.json \
        docs/diagnostics/baltic_chunk0_warmstart_bistability_regime-shift.json
git commit -m "docs(baltic): warm-start regime-shift sweep results (2026-07-09)"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-09-baltic-warmstart-regime-shift-sweep-design.md`):
- warm-start injection helper → Task 1 (`warmstart_override`). ✓
- IC builders (existing unchanged + `cod_dominated_seeding` / `clupeid_dominated_seeding`) → Task 1. ✓
- clupeid-dominance signal (summed herring+sprat, recorded per arm) → Task 2 (`clupeid_axis`) + recorded in the point dict in Task 4. ✓
- generalized `run_bistability_point` / `run_bistability_sweep` (labelled IC pair + warmstart + contrast) → Task 4. ✓
- CLI `--warmstart` + `--contrast {cod-axis,regime-shift,both}` → Task 5. ✓
- verdict logic: cod-axis unchanged (Task 3 extraction is behavior-preserving; Task 4 reuses it), regime-shift directional conjunction (both axes diverge) → Task 3 `regime_shift_outcome` + Task 4 `_regime_shift_verdict`. ✓
- pre-flight de-risk before the full sweep → Task 5 `preflight_check` + `--preflight`; Task 6 Step 1 runs it as the gate. ✓
- testing (CI-safe fake runner unit tests for both-axes/cod-only/clupeid-only/neither/non-stationary; existing cod-axis tests still pass) → Tasks 3–5. ✓
- parity (`--warmstart` absent ⇒ same egg-only sweep) → enforced by defaults + the unchanged existing tests, checked in Tasks 3–5 Step 5/6. ✓
- outputs: two JSONs + `baltic_chunk0_warmstart_results_2026-07-09.md` → Task 5 (paths via `contrast_specs`) + Task 6. ✓
- runtime / `--smoke` availability → `--smoke` reused from v3; Task 6 Step 2/3. ✓

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step shows complete code; every command shows expected output.

**Type consistency:** `warmstart_override`, `cod_dominated_seeding`, `clupeid_dominated_seeding`, `clupeid_axis`, `cod_axis_outcome`, `regime_shift_outcome`, `_regime_shift_verdict`, `contrast_specs`, `preflight_check`, `_load_targets`, `_clupeid_targets_from` are named identically at definition (Tasks 1–5) and at every call site. Point-dict keys (`a_clupeid_biomass`, `b_clupeid_biomass`, `a_clupeid_valid`, `b_clupeid_valid`, `clupeid_gap`, `outcome`, `regime_shift`) are written in Task 4 and asserted in the Task 4 tests. `contrast` values (`"cod-axis"`, `"regime-shift"`) match between `contrast_specs` labels, the sweep dispatch, and `spec["label"]` passed as `contrast=`. Flag key `module.population.initialisation.enabled` matches `_ENABLE_KEY` and `osmose/engine/initialization.py`.
