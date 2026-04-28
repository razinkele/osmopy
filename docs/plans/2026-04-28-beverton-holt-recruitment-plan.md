# Beverton-Holt Stock-Recruitment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add density-dependent stock-recruitment (Beverton-Holt + Ricker) to OSMOSE-Python's reproduction process so DE calibration can no longer defeat mortality bounds by tuning the linear SSB→eggs path.

**Architecture:** Multiplicative correction applied to the existing linear `n_eggs` formula in `osmose/engine/processes/reproduction.py`. Two new per-species config keys (`stock.recruitment.{type,ssbhalf}`) with `type=none` default that preserves byte-for-byte parity with current Java behavior. SSB-driven, applied per-step against per-step SSB; the `season_factor` distribution is unchanged.

**Per-step semantic (deliberate divergence from ICES annual SR convention):** B-H/Ricker is applied independently to each spawning event in the year. For multi-season spawners (e.g., herring spring + autumn), each event sees the current SSB and gets its own density-dependent reduction. Total annual recruitment is the sum across events, not a single annual B-H draw. This is biologically defensible (each cohort experiences the density at its own spawning time) but means literature α/β values calibrated against annual ICES SR pairs need to be reinterpreted as per-event. The DE calibration step (out-of-scope §1) should fit `ssb_half` directly against simulated trajectories rather than ICES point estimates.

**Tech Stack:** Python 3.12, NumPy, pytest. Schema-driven (`osmose/schema/species.py`). Pure-function helper exposed for direct unit testing. Opt-in via config; existing Baltic / EEC / Bay of Biscay configs run unchanged.

**Mathematical formulation (locked):**

```
linear_eggs_step = sex_ratio * relative_fecundity * SSB * season_factor * 1e6     # current code

# Beverton-Holt (asymptotic)
eggs_step = linear_eggs_step / (1 + SSB / ssb_half)

# Ricker (over-compensating)
eggs_step = linear_eggs_step * exp(-SSB / ssb_half)

# none (default) — current behaviour preserved
eggs_step = linear_eggs_step
```

`SSB` is in tonnes (per-step, focal species only). `ssb_half` is in tonnes and represents the SSB at which Beverton-Holt halves recruitment relative to the linear regime; for Ricker it is the SSB at which recruitment peaks. Linear regime at low SSB is identical across all three modes — preserves Java parity at low density.

**Open questions (resolved here, not for the executor):**
1. **SR applied at egg-production**, not at egg-to-larva transition. Egg-production is where the SSB→eggs transformation already lives; splitting state machinery into a separate larval bucket is out of scope.
2. **Per-step**, not per-year. SSB is computed per-step; multiplicative correction acts on per-step SSB. `season_factor` already handles annual→per-step distribution.
3. **Multi-season spawners (herring spring vs autumn)**: handled implicitly. Single per-species SSB and single `ssb_half`; `season_factor` carries the bimodal pattern. Per-cohort SR is a v2 follow-on.
4. **Calibration**: ship with literature defaults from ICES SR pairs for cod and flounder; expose `ssb_half` as DE-tunable for the four priority species (cod, perch, pikeperch, flounder) with biologically-informed bounds.

**Java parity implication:** Java OSMOSE has no B-H SR. This is a deliberate divergence. `type=none` is the default for every species; existing Java-compat tests must remain byte-equivalent.

---

## File Structure

**New files:**
- `tests/test_engine_stock_recruitment.py` — unit tests for the pure SR helper (mathematical correctness, edge cases)
- `tests/test_engine_reproduction_sr_regression.py` — `type=none` byte-for-byte regression vs the un-modified linear path

**Modified files:**
- `osmose/schema/species.py` — two new `OsmoseField` entries in the reproduction category
- `osmose/engine/config.py` — parsing in `_parse_reproduction_params` (line 462), dataclass fields on `EngineConfig` (line 1094), focal/background merge in `_merge_focal_background` (line 662), unpack/wiring inside `from_dict` (line 1348)
- `osmose/engine/config_validation.py` — schema-driven (auto-picked by AST walker); verify in Task 2
- `osmose/engine/processes/reproduction.py` — new pure helper `apply_stock_recruitment()` and one-line wiring
- `tests/test_engine_reproduction.py` — extend with 4 config-parse tests (`TestStockRecruitmentConfig`) + 2 wiring tests (`TestReproduction`)
- `docs/parity-roadmap.md` — add a "Post-parity divergences" subsection
- `CHANGELOG.md` — entry under Unreleased

**Out of scope (separate follow-ons):**
- DE search-bound updates in `osmose/calibration/` — operational task, no code change to the engine. Tracked as a memory note, not a task here.
- Phase 12 re-calibration run — operational task scheduled by cron `926f3ab5`.
- Per-cohort (spring/autumn) SR — v2.

---

## Pre-flight (no code change)

### Task 0: Read the touchpoints

**Files (read-only):**
- `osmose/engine/processes/reproduction.py:30-75` — current SSB and `n_eggs` block
- `osmose/engine/config.py:462-501` — `_parse_reproduction_params`
- `osmose/engine/config.py:680-755` — focal/background merge for reproduction fields
- `osmose/engine/config.py:1057-1110` — `EngineConfig` dataclass reproduction block
- `osmose/schema/species.py:170-244` — reproduction-category schema fields
- `osmose/engine/config_validation.py:40-260` — `_SUPPLEMENTARY_ALLOWLIST` and AST walker
- `tests/test_engine_reproduction.py` — existing reproduction test patterns

- [ ] **Step 0.1: Read each file above to confirm line numbers haven't drifted.** No changes; just verify the architecture description matches HEAD before starting Task 1.

---

## Task 1: Add schema fields for stock-recruitment

**Files:**
- Modify: `osmose/schema/species.py` (add two `OsmoseField` entries inside the reproduction block, after the existing `reproduction.season.file.sp{idx}` field around line 244)

- [ ] **Step 1.1: Add the two OsmoseField entries**

Insert immediately after the existing `reproduction.season.file.sp{idx}` field (currently around line 244, before the `# ── Life History ──` divider):

```python
    OsmoseField(
        key_pattern="stock.recruitment.type.sp{idx}",
        param_type=ParamType.ENUM,
        default="none",
        choices=["none", "beverton_holt", "ricker"],
        description=(
            "Stock-recruitment relationship applied to per-step egg production. "
            "'none' preserves the linear SSB→eggs formula (Java parity). "
            "'beverton_holt' caps recruitment asymptotically at high SSB. "
            "'ricker' over-compensates (recruitment peaks then declines)."
        ),
        category="reproduction",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="stock.recruitment.ssbhalf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=0.0,
        min_val=0.0,
        max_val=1e9,
        description=(
            "SSB (tonnes) at which Beverton-Holt halves recruitment, or at which "
            "Ricker recruitment peaks. Ignored when stock.recruitment.type=none. "
            "0.0 with type!=none is a config error."
        ),
        category="reproduction",
        unit="tonnes",
        indexed=True,
        required=False,
    ),
```

Note: only **two** OsmoseField entries — `type` and `ssbhalf`. The third `gamma` from earlier brainstorming is dropped; B-H and Ricker each need exactly one shape parameter beyond the linear α.

- [ ] **Step 1.2: Run schema-registry tests**

Run: `.venv/bin/python -m pytest tests/test_schema.py tests/test_schema_species.py tests/test_schema_all.py -v`
Expected: PASS. The only field-count assertion is `tests/test_schema_species.py:25` — `assert len(SPECIES_FIELDS) >= 25` — which still holds with two added fields. No expected counts to update.

- [ ] **Step 1.3: Commit**

```bash
git add osmose/schema/species.py
git commit -m "schema: add stock.recruitment.{type,ssbhalf} fields per species"
```

---

## Task 2: Parse stock-recruitment params in EngineConfig

**Files:**
- Modify: `osmose/engine/config.py` — `_parse_reproduction_params` (line 462-501), `_merge_focal_background` merge dicts (with-bkg branch line 700; focal-only branch line 748), `EngineConfig` dataclass reproduction section (line 1094-1101), `from_dict` unpack (line 1409, 1505) and `cls(...)` call (line 1747)
- Test: extend `tests/test_engine_reproduction.py` (Step 2.1 below)

- [ ] **Step 2.1: Write a failing test for the new EngineConfig fields**

Add to `tests/test_engine_reproduction.py` at the end:

```python
class TestStockRecruitmentConfig:
    def test_default_type_is_none(self):
        """Without stock.recruitment.* keys, all species default to 'none'."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        assert cfg.recruitment_type[0] == "none"
        assert cfg.recruitment_ssb_half[0] == 0.0

    def test_beverton_holt_type_parsed(self):
        """Setting type=beverton_holt and ssbhalf is round-tripped."""
        d = _make_reprod_config()
        d["stock.recruitment.type.sp0"] = "beverton_holt"
        d["stock.recruitment.ssbhalf.sp0"] = "12500.0"
        cfg = EngineConfig.from_dict(d)
        assert cfg.recruitment_type[0] == "beverton_holt"
        assert cfg.recruitment_ssb_half[0] == 12500.0

    def test_unknown_type_rejected(self):
        """Misspelled SR types fail loudly at config parse time."""
        import pytest
        d = _make_reprod_config()
        d["stock.recruitment.type.sp0"] = "berverton_holdt"  # typo
        with pytest.raises(ValueError, match="stock.recruitment.type"):
            EngineConfig.from_dict(d)

    def test_ssbhalf_zero_with_active_sr_rejected(self):
        """type!=none with ssbhalf=0 is a configuration error."""
        import pytest
        d = _make_reprod_config()
        d["stock.recruitment.type.sp0"] = "beverton_holt"
        d["stock.recruitment.ssbhalf.sp0"] = "0.0"
        with pytest.raises(ValueError, match="stock.recruitment.ssbhalf"):
            EngineConfig.from_dict(d)
```

- [ ] **Step 2.2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py::TestStockRecruitmentConfig -v`
Expected: FAIL — all four tests fail. Two with `AttributeError: 'EngineConfig' object has no attribute 'recruitment_type'` (the parsed-field tests); two with `Failed: DID NOT RAISE <class 'ValueError'>` (the validation tests, since the parse-time validator added in Step 2.4 is not yet in place). All four go green after Step 2.11.

- [ ] **Step 2.3: Add a string-array helper next to `_species_float_optional`**

Insert into `osmose/engine/config.py` immediately after `_species_int_optional` (around line 75):

```python
def _species_str_optional(
    cfg: dict[str, str], pattern: str, n: int, default: str, allowed: set[str] | None = None
) -> list[str]:
    """Extract a per-species string array, using default if key is missing.

    If `allowed` is given, raise ValueError on any value not in the set.
    """
    out: list[str] = []
    for i in range(n):
        key = pattern.format(i=i)
        val = cfg.get(key, default).strip().lower()
        if allowed is not None and val not in allowed:
            raise ValueError(
                f"{key}={val!r} is not one of {sorted(allowed)}"
            )
        out.append(val)
    return out
```

- [ ] **Step 2.4: Extend `_parse_reproduction_params` to parse the two new keys**

Inside `_parse_reproduction_params` (around line 462-501), add before the `return {...}`:

```python
    recruitment_type = _species_str_optional(
        cfg,
        "stock.recruitment.type.sp{i}",
        n_sp,
        default="none",
        allowed={"none", "beverton_holt", "ricker"},
    )
    recruitment_ssb_half = _species_float_optional(
        cfg, "stock.recruitment.ssbhalf.sp{i}", n_sp, default=0.0
    )
    for i in range(n_sp):
        if recruitment_type[i] != "none" and recruitment_ssb_half[i] <= 0.0:
            raise ValueError(
                f"stock.recruitment.ssbhalf.sp{i} must be > 0 when "
                f"stock.recruitment.type.sp{i}={recruitment_type[i]!r}"
            )
```

Add to the returned dict (replacing the existing `return {...}`):

```python
    return {
        "focal_sex_ratio": sex_ratio,
        "focal_relative_fecundity": relative_fecundity,
        "focal_maturity_size": maturity_size,
        "focal_seeding_biomass": seeding_biomass,
        "focal_seeding_max_step": seeding_max_step,
        "focal_larva_mortality_rate": larva_mortality_rate,
        "focal_maturity_age_dt": maturity_age_dt,
        "focal_recruitment_type": recruitment_type,
        "focal_recruitment_ssb_half": recruitment_ssb_half,
    }
```

- [ ] **Step 2.5: Add the dataclass fields to `EngineConfig`**

Append to the `# Reproduction` block (currently line 1094-1101) — keep existing fields untouched and add two new lines after `larva_mortality_by_dt`:

```python
    # Stock-recruitment (post-parity divergence; Java has no equivalent)
    recruitment_type: list[str]  # one of {"none","beverton_holt","ricker"} per species
    recruitment_ssb_half: NDArray[np.float64]  # tonnes; ignored when type=="none"
```

- [ ] **Step 2.6: Unpack `_repro` into local focal vars**

In `from_dict` at line 1409-1415 (immediately after `_repro = _parse_reproduction_params(...)`), add two more lines after `focal_maturity_age_dt`:

```python
        focal_recruitment_type = _repro["focal_recruitment_type"]
        focal_recruitment_ssb_half = _repro["focal_recruitment_ssb_half"]
```

- [ ] **Step 2.7: Stuff focal vars into the `_focal` dict**

In `from_dict` at line 1454-1474, append to the `_focal` dict literal (after `"focal_maturity_age_dt"`):

```python
            "focal_recruitment_type": focal_recruitment_type,
            "focal_recruitment_ssb_half": focal_recruitment_ssb_half,
```

- [ ] **Step 2.8: Add to the with-background merge dict**

In `_merge_focal_background` (function defined at `config.py:662`) at line 700-709 (after `"maturity_age_dt": np.concatenate([focal["focal_maturity_age_dt"], bkg_zeros_i]),`), add:

```python
            "recruitment_type": (
                focal["focal_recruitment_type"] + ["none"] * len(background_list)
            ),
            "recruitment_ssb_half": np.concatenate(
                [focal["focal_recruitment_ssb_half"], bkg_zeros_f]
            ),
```

Note: `recruitment_type` is a `list[str]`, not an `np.ndarray`, so `+` concat is correct (not `np.concatenate`). Background species don't reproduce in OSMOSE — `"none"` is the right default.

- [ ] **Step 2.9: Add to the focal-only merge dict**

In `_merge_focal_background` (still inside the function defined at `config.py:662`, focal-only branch) at line 748-754 (after `"maturity_age_dt": focal["focal_maturity_age_dt"],`), add:

```python
            "recruitment_type": focal["focal_recruitment_type"],
            "recruitment_ssb_half": focal["focal_recruitment_ssb_half"],
```

- [ ] **Step 2.10: Unpack `_merged` into post-merge local vars**

In `from_dict` at line 1505-1510 (after `larva_mortality_rate = _merged["larva_mortality_rate"]`), add:

```python
        recruitment_type = _merged["recruitment_type"]
        recruitment_ssb_half = _merged["recruitment_ssb_half"]
```

- [ ] **Step 2.11: Pass to the `cls(...)` constructor**

In `from_dict` at line 1747+ (the `return cls(` block), find the existing reproduction fields (`sex_ratio=sex_ratio,` at line 1775, through `larva_mortality_by_dt=...` at line 1784). Add immediately after `larva_mortality_by_dt=...`:

```python
            recruitment_type=recruitment_type,
            recruitment_ssb_half=recruitment_ssb_half,
```

- [ ] **Step 2.12: Run the Task 2 config-parse tests**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py::TestStockRecruitmentConfig -v`
Expected: PASS — all four tests green.

- [ ] **Step 2.13: Run the existing reproduction + config tests for regression**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py tests/test_config_reader.py tests/test_config_validation.py -v`
Expected: PASS — no existing test breaks.

- [ ] **Step 2.14: Verify config_validation allowlist auto-picks the new keys**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -v -k "warn_mode_clean"`
Expected: PASS — the AST walker should pick up the new patterns automatically because they appear as string literals in `_parse_reproduction_params`. If it fails with "unknown key stock.recruitment.*", add the two patterns to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py:43`:

```python
_SUPPLEMENTARY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # ... existing entries ...
        "stock.recruitment.type.sp{i}",
        "stock.recruitment.ssbhalf.sp{i}",
    }
)
```

- [ ] **Step 2.15: Commit**

```bash
git add osmose/engine/config.py osmose/engine/config_validation.py tests/test_engine_reproduction.py
git commit -m "engine(config): parse stock.recruitment.{type,ssbhalf} per species"
```

---

## Task 3: Pure stock-recruitment helper with unit tests (TDD)

**Files:**
- Create: `tests/test_engine_stock_recruitment.py`
- Modify: `osmose/engine/processes/reproduction.py` (add `apply_stock_recruitment` near top)

- [ ] **Step 3.1: Write the failing unit tests**

Create `tests/test_engine_stock_recruitment.py`:

```python
"""Unit tests for the pure stock-recruitment helper."""

import numpy as np
import pytest

from osmose.engine.processes.reproduction import apply_stock_recruitment


class TestApplyStockRecruitment:
    def test_none_returns_input_unchanged(self):
        linear = np.array([1000.0, 2000.0])
        ssb = np.array([10.0, 20.0])
        ssb_half = np.array([0.0, 0.0])
        types = ["none", "none"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_array_equal(out, linear)

    def test_beverton_holt_low_ssb_approaches_linear(self):
        """At SSB << ssb_half, B-H ≈ linear (within 1%)."""
        linear = np.array([1000.0])
        ssb = np.array([1.0])
        ssb_half = np.array([1000.0])
        types = ["beverton_holt"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        # 1000 / (1 + 1/1000) = 999.0...
        assert abs(out[0] - linear[0]) / linear[0] < 0.01

    def test_beverton_holt_at_half_saturation(self):
        """At SSB == ssb_half, B-H = linear / 2."""
        linear = np.array([1000.0])
        ssb = np.array([500.0])
        ssb_half = np.array([500.0])
        types = ["beverton_holt"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_allclose(out, [500.0])

    def test_beverton_holt_asymptote(self):
        """At SSB >> ssb_half, B-H plateaus at linear * (ssb_half/ssb)."""
        linear = np.array([1_000_000.0])
        ssb = np.array([100_000.0])
        ssb_half = np.array([100.0])
        types = ["beverton_holt"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        # 1e6 / (1 + 1e5/100) = 1e6 / 1001 ≈ 999
        assert out[0] < linear[0] * 0.01

    def test_ricker_at_peak(self):
        """Ricker peaks at SSB = ssb_half (where d eggs / d SSB = 0).

        At SSB = ssb_half, the multiplier is exp(-1) ≈ 0.368.
        """
        linear = np.array([1000.0])
        ssb = np.array([500.0])
        ssb_half = np.array([500.0])
        types = ["ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_allclose(out, [1000.0 * np.exp(-1.0)], rtol=1e-6)

    def test_ricker_high_ssb_collapses(self):
        """Ricker recruitment goes to ~0 at very high SSB / ssb_half ratios."""
        linear = np.array([1000.0])
        ssb = np.array([10000.0])
        ssb_half = np.array([100.0])
        types = ["ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        assert out[0] < 1e-30

    def test_mixed_types_per_species(self):
        """Different SR types can coexist across species in one call."""
        linear = np.array([1000.0, 1000.0, 1000.0])
        ssb = np.array([500.0, 500.0, 500.0])
        ssb_half = np.array([0.0, 500.0, 500.0])
        types = ["none", "beverton_holt", "ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_allclose(out[0], 1000.0)
        np.testing.assert_allclose(out[1], 500.0)
        np.testing.assert_allclose(out[2], 1000.0 * np.exp(-1.0), rtol=1e-6)

    def test_zero_ssb_returns_zero(self):
        """SSB=0 with any SR type returns 0 (linear is already 0)."""
        linear = np.array([0.0, 0.0])
        ssb = np.array([0.0, 0.0])
        ssb_half = np.array([100.0, 100.0])
        types = ["beverton_holt", "ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_array_equal(out, [0.0, 0.0])

    def test_unknown_type_raises(self):
        linear = np.array([1000.0])
        ssb = np.array([100.0])
        ssb_half = np.array([100.0])
        with pytest.raises(ValueError, match="unknown stock-recruitment type"):
            apply_stock_recruitment(linear, ssb, ssb_half, ["sigmoid"])
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_stock_recruitment.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_stock_recruitment' from 'osmose.engine.processes.reproduction'`

- [ ] **Step 3.3: Implement `apply_stock_recruitment`**

In `osmose/engine/processes/reproduction.py`, add after the imports (before the existing `def reproduction(...)`):

```python
def apply_stock_recruitment(
    linear_eggs: np.ndarray,
    ssb: np.ndarray,
    ssb_half: np.ndarray,
    recruitment_type: list[str],
) -> np.ndarray:
    """Apply per-species density-dependent stock-recruitment.

    Multiplicative correction over the linear SSB→eggs formula. At low SSB,
    every variant approaches `linear_eggs` (preserves Java-linear regime).

    Parameters
    ----------
    linear_eggs : (n_sp,) per-step linear egg production = sex_ratio * relative_fecundity
        * SSB * season_factor * 1e6 (tonnes→grams). All non-negative.
    ssb : (n_sp,) spawning stock biomass in tonnes (per-step).
    ssb_half : (n_sp,) half-saturation SSB in tonnes; ignored where type=="none".
    recruitment_type : per-species, one of {"none","beverton_holt","ricker"}.

    Returns
    -------
    (n_sp,) corrected egg counts.
    """
    n_sp = linear_eggs.shape[0]
    if not (ssb.shape[0] == ssb_half.shape[0] == len(recruitment_type) == n_sp):
        raise ValueError("apply_stock_recruitment: input length mismatch")

    out = linear_eggs.copy()
    for sp in range(n_sp):
        t = recruitment_type[sp]
        if t == "none":
            continue
        if ssb[sp] <= 0.0:
            continue  # nothing to scale; linear_eggs is already 0
        if t == "beverton_holt":
            out[sp] = linear_eggs[sp] / (1.0 + ssb[sp] / ssb_half[sp])
        elif t == "ricker":
            out[sp] = linear_eggs[sp] * np.exp(-ssb[sp] / ssb_half[sp])
        else:
            raise ValueError(f"unknown stock-recruitment type: {t!r}")
    return out
```

- [ ] **Step 3.4: Run unit tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_stock_recruitment.py -v`
Expected: PASS — all nine tests green.

- [ ] **Step 3.5: Commit**

```bash
git add osmose/engine/processes/reproduction.py tests/test_engine_stock_recruitment.py
git commit -m "engine(reproduction): add apply_stock_recruitment helper (B-H + Ricker)"
```

---

## Task 4: Wire the helper into `reproduction()`

**Files:**
- Modify: `osmose/engine/processes/reproduction.py` (the `reproduction()` function around line 68-75)

- [ ] **Step 4.1: Write a failing integration test for B-H wiring**

Append to `tests/test_engine_reproduction.py` inside `class TestReproduction`:

```python
    def test_beverton_holt_caps_eggs_at_high_ssb(self):
        """Setting type=beverton_holt with low ssbhalf reduces eggs vs linear."""
        d = _make_reprod_config()
        # Linear baseline egg count
        cfg_linear = EngineConfig.from_dict(d)
        # B-H with very tight cap
        d_bh = dict(d)
        d_bh["stock.recruitment.type.sp0"] = "beverton_holt"
        d_bh["stock.recruitment.ssbhalf.sp0"] = "1.0"  # very low cap
        cfg_bh = EngineConfig.from_dict(d_bh)

        def _make_state():
            s = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
            return s.replace(
                abundance=np.array([100_000.0]),
                length=np.array([15.0]),
                weight=np.array([20.25]),
                biomass=np.array([2_025_000.0]),  # 2025 t
                age_dt=np.array([24], dtype=np.int32),
            )

        rng = np.random.default_rng(42)
        st_lin = reproduction(_make_state(), cfg_linear, step=0, rng=rng)
        st_bh = reproduction(_make_state(), cfg_bh, step=0, rng=rng)

        eggs_lin = st_lin.abundance[st_lin.is_egg].sum()
        eggs_bh = st_bh.abundance[st_bh.is_egg].sum()
        assert eggs_bh < eggs_lin * 0.01, (
            f"B-H should heavily cap eggs: linear={eggs_lin}, bh={eggs_bh}"
        )

    def test_recruitment_type_none_matches_pre_sr_byte_for_byte(self):
        """type=none must preserve the exact linear formula (Java parity)."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        assert cfg.recruitment_type[0] == "none"
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            length=np.array([15.0]),
            weight=np.array([20.25]),
            biomass=np.array([20250.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)

        # Manual linear computation
        ssb = 20250.0  # 1 mature school
        sex_ratio = 0.5
        rel_fec = 800.0
        season = 1.0 / 12  # n_dt_per_year=12, no spawning_season CSV
        expected = sex_ratio * rel_fec * ssb * season * 1_000_000.0
        actual = new_state.abundance[new_state.is_egg].sum()
        np.testing.assert_allclose(actual, expected, rtol=1e-9)
```

- [ ] **Step 4.2: Run to verify the first test fails (helper not yet wired)**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py::TestReproduction::test_beverton_holt_caps_eggs_at_high_ssb -v`
Expected: FAIL — eggs_bh equals eggs_lin because the helper isn't invoked yet.

The second test (`test_recruitment_type_none_matches_pre_sr_byte_for_byte`) should already PASS because the helper is not yet called.

- [ ] **Step 4.3: Wire `apply_stock_recruitment` into `reproduction()`**

In `osmose/engine/processes/reproduction.py`, modify the `n_eggs = ...` block (currently lines 68-75). Replace:

```python
    TONNES_TO_GRAMS = 1_000_000.0
    n_eggs = (
        config.sex_ratio[:n_sp]
        * config.relative_fecundity[:n_sp]
        * ssb
        * season_factor
        * TONNES_TO_GRAMS
    )
```

with:

```python
    TONNES_TO_GRAMS = 1_000_000.0
    n_eggs_linear = (
        config.sex_ratio[:n_sp]
        * config.relative_fecundity[:n_sp]
        * ssb
        * season_factor
        * TONNES_TO_GRAMS
    )
    n_eggs = apply_stock_recruitment(
        n_eggs_linear,
        ssb,
        config.recruitment_ssb_half[:n_sp],
        config.recruitment_type[:n_sp],
    )
```

- [ ] **Step 4.4: Run both new tests + the existing reproduction suite**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py -v`
Expected: PASS — all reproduction tests including the two new SR ones.

- [ ] **Step 4.5: Run the full Java parity suite to confirm no regression with `type=none`**

Run: `.venv/bin/python -m pytest tests/test_baltic_java_compat.py -v`
Expected: PASS — Java compat is unchanged because every existing config defaults to `type=none`.

- [ ] **Step 4.6: Commit**

```bash
git add osmose/engine/processes/reproduction.py tests/test_engine_reproduction.py
git commit -m "engine(reproduction): apply stock-recruitment to per-step eggs"
```

---

## Task 5: Long-trajectory regression test

**Files:**
- Create: `tests/test_engine_reproduction_sr_regression.py`

This task ensures that a multi-year run with `type=none` is byte-for-byte identical to a control run produced **before** the SR wiring landed. Because we cannot run pre-Task-4 code from Task 5, the test instead asserts that `type=none` matches a manually computed linear trajectory for one species across ten timesteps.

- [ ] **Step 5.1: Create the regression test**

Create `tests/test_engine_reproduction_sr_regression.py`:

```python
"""Regression: stock.recruitment.type=none reproduces the linear formula exactly."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState


def _make_cfg() -> EngineConfig:
    return EngineConfig.from_dict({
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "800",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "50000",
    })


def _make_state(biomass_t: float, n_schools: int = 1) -> SchoolState:
    state = SchoolState.create(
        n_schools=n_schools, species_id=np.zeros(n_schools, dtype=np.int32)
    )
    return state.replace(
        abundance=np.full(n_schools, 1000.0, dtype=np.float64),
        length=np.full(n_schools, 15.0, dtype=np.float64),
        weight=np.full(n_schools, biomass_t / 1000.0, dtype=np.float64),
        biomass=np.full(n_schools, biomass_t, dtype=np.float64),
        age_dt=np.full(n_schools, 24, dtype=np.int32),
    )


def test_type_none_matches_linear_formula_across_ssb_sweep():
    """For five SSB levels at step=0, type=none returns sex_ratio*fec*SSB*season*1e6 exactly."""
    cfg = _make_cfg()
    rng = np.random.default_rng(42)
    sex_ratio = 0.5
    rel_fec = 800.0
    season = 1.0 / 12  # uniform when no spawning_season CSV is configured

    biomasses_t = [10000.0, 50000.0, 100000.0, 250000.0, 1_000_000.0]
    for bm in biomasses_t:
        state = _make_state(bm)
        out = reproduction(state, cfg, step=0, rng=rng)
        eggs = out.abundance[out.is_egg].sum()
        # Reproduction uses weight*abundance for SSB (not the biomass field).
        # Recompute the same way for the manual expected value.
        ssb_t = (state.abundance * state.weight).sum()
        expected = sex_ratio * rel_fec * ssb_t * season * 1_000_000.0
        np.testing.assert_allclose(eggs, expected, rtol=1e-9)


def test_type_none_matches_linear_formula_across_step_phase():
    """Sample three steps within a year; the season factor must drop out of the byte-equality."""
    cfg = _make_cfg()
    rng = np.random.default_rng(42)
    sex_ratio = 0.5
    rel_fec = 800.0
    season = 1.0 / 12  # uniform; not phase-dependent without a spawning CSV
    state = _make_state(100_000.0)
    ssb_t = (state.abundance * state.weight).sum()
    expected = sex_ratio * rel_fec * ssb_t * season * 1_000_000.0

    for step in (0, 6, 11):
        out = reproduction(state, cfg, step=step, rng=rng)
        eggs = out.abundance[out.is_egg].sum()
        np.testing.assert_allclose(eggs, expected, rtol=1e-9)
```

- [ ] **Step 5.2: Run the regression test**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction_sr_regression.py -v`
Expected: PASS for all biomass levels.

- [ ] **Step 5.3: Commit**

```bash
git add tests/test_engine_reproduction_sr_regression.py
git commit -m "test(reproduction): pin linear-formula regression for type=none across SSB sweep"
```

---

## Task 6: Multi-step integration smoke test (B-H caps biomass)

**Files:**
- Create: `tests/test_engine_reproduction_sr_integration.py`

A multi-step in-process loop driving `reproduction()` directly with a synthetic single-species config (no Baltic fixture, no `simulate()` chain). Asserts: (1) with B-H + low `ssb_half`, cumulative eggs over 12 steps are strictly less than the linear baseline; (2) Ricker with `ssb_half << SSB` collapses recruitment to ≈0.

We deliberately **avoid calling `simulate()`**: it requires ~50 config keys (predation stages, accessibility, movement, fishing, etc.) that the synthetic dict does not supply, and Java-compat Step 4.5 already exercises the full simulate loop with `type=none`. End-to-end SR validation against Baltic is the phase-12 re-cal job (out-of-scope §1).

- [ ] **Step 6.1: Create the synthetic integration test**

Create `tests/test_engine_reproduction_sr_integration.py`:

```python
"""Integration smoke: B-H caps cumulative egg production over a multi-step run."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState


def _base_cfg() -> dict[str, str]:
    """Single-species synthetic config sized to produce non-trivial SSB quickly."""
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "2",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "800",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "50000",
    }


def _seed_state(n_schools: int = 5) -> SchoolState:
    """Seed mature schools that will spawn at every step."""
    s = SchoolState.create(
        n_schools=n_schools, species_id=np.zeros(n_schools, dtype=np.int32)
    )
    return s.replace(
        abundance=np.full(n_schools, 10_000.0, dtype=np.float64),
        length=np.full(n_schools, 20.0, dtype=np.float64),
        weight=np.full(n_schools, 0.006 * 20.0**3, dtype=np.float64),  # 48.0
        biomass=np.full(n_schools, 480_000.0, dtype=np.float64),
        age_dt=np.full(n_schools, 24, dtype=np.int32),
    )


def _run_steps(cfg: EngineConfig, n_steps: int) -> float:
    """Run reproduction n_steps times, return total eggs produced across the run."""
    state = _seed_state()
    rng = np.random.default_rng(42)
    total_eggs = 0.0
    for step in range(n_steps):
        n_before = len(state)
        state = reproduction(state, cfg, step=step, rng=rng)
        new_egg_mask = np.zeros(len(state), dtype=np.bool_)
        new_egg_mask[n_before:] = state.is_egg[n_before:]
        total_eggs += float(state.abundance[new_egg_mask].sum())
    return total_eggs


def test_bh_produces_strictly_fewer_eggs_than_linear():
    """Over 12 reproduction steps, B-H caps cumulative eggs vs linear baseline."""
    cfg_lin = EngineConfig.from_dict(_base_cfg())
    eggs_lin = _run_steps(cfg_lin, n_steps=12)

    cfg_bh = EngineConfig.from_dict({
        **_base_cfg(),
        "stock.recruitment.type.sp0": "beverton_holt",
        "stock.recruitment.ssbhalf.sp0": "1000.0",  # ssb_half << per-step SSB
    })
    eggs_bh = _run_steps(cfg_bh, n_steps=12)

    assert eggs_bh < eggs_lin, (
        f"B-H must cap eggs vs linear: linear={eggs_lin:.3e}, bh={eggs_bh:.3e}"
    )
    # ssb_half << SSB → B-H multiplier ≈ ssb_half / SSB → near-zero ratio.
    assert eggs_bh / eggs_lin < 0.1, (
        f"B-H cap should be aggressive at low ssb_half: ratio={eggs_bh/eggs_lin:.4f}"
    )


def test_ricker_collapses_eggs_at_high_ssb():
    """Ricker with ssb_half << SSB drives eggs to near-zero (over-compensation)."""
    cfg_ricker = EngineConfig.from_dict({
        **_base_cfg(),
        "stock.recruitment.type.sp0": "ricker",
        "stock.recruitment.ssbhalf.sp0": "100.0",  # SSB / ssb_half ≈ 24000 → exp(-24000) ≈ 0
    })
    eggs_ricker = _run_steps(cfg_ricker, n_steps=6)
    assert eggs_ricker < 1.0, (
        f"Ricker should collapse to ~0 at SSB/h=24000: got {eggs_ricker}"
    )
```

- [ ] **Step 6.2: Run the integration test**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction_sr_integration.py -v`
Expected: PASS for both tests. Do **not** weaken the ratio assertions if the test fails — investigate whether the helper or wiring produces different math than expected.

- [ ] **Step 6.3: Commit**

```bash
git add tests/test_engine_reproduction_sr_integration.py
git commit -m "test(reproduction): multi-step integration smoke for B-H + Ricker"
```

---

## Task 7: Documentation + version bump

**Files:**
- Modify: `docs/parity-roadmap.md` (add a "Post-parity divergences" section if not present)
- Modify: `CHANGELOG.md` (Unreleased section)
- Modify: `osmose/__version__.py`

- [ ] **Step 7.1: Update parity-roadmap.md**

Append a new section before the "What's next (post-parity)" section (or to the end of the document if that section is missing):

```markdown
## Post-parity divergences

OSMOSE-Python preserves Java parity by default but adds the following opt-in
features that have no Java counterpart. These are documented here so any future
parity audit knows to expect a divergence when the corresponding config keys
are set.

### Beverton-Holt / Ricker stock-recruitment (2026-04-28)

- Config: `stock.recruitment.type.sp{i}` ∈ `{none, beverton_holt, ricker}`,
  `stock.recruitment.ssbhalf.sp{i}` (tonnes).
- Default: `type=none` for every species → byte-for-byte equivalent to the
  Java linear formula `n_eggs = sex_ratio · relative_fecundity · SSB · season · 1e6`.
- Rationale: Java OSMOSE has no SR; DE calibration of the linear regime can
  trade off adult vs larval mortality to defeat single-axis biomass bounds
  (verified 2026-04-27 with cod-floor experiment: forcing adult mortality up
  14× moved cod biomass only +8% because larval mortality dropped 24× to
  compensate). Density-dependent recruitment is the structural fix.
- Code: `osmose/engine/processes/reproduction.py:apply_stock_recruitment`
```

- [ ] **Step 7.2: Add CHANGELOG entry**

Insert under the most recent `## [Unreleased]` header (or create one if missing):

```markdown
### Added
- Stock-recruitment subsystem: per-species `stock.recruitment.type` ∈
  `{none, beverton_holt, ricker}` plus `stock.recruitment.ssbhalf` for the
  shape parameter. Default is `none` (linear, Java parity preserved). Opt-in
  fix for the cod/perch/pikeperch/flounder structural overshoots observed in
  the 2026-04-27 phase-12 cod-floor calibration. See
  `docs/parity-roadmap.md` § Post-parity divergences.
```

- [ ] **Step 7.3: Bump version**

Current version (verified 2026-04-28): `osmose/__version__.py:3` reads `__version__ = "0.9.3"`. Latest tag: `v0.9.3`. Bump the minor → `0.10.0` (feature add, opt-in). Update `osmose/__version__.py`:

```python
__version__ = "0.10.0"
```

- [ ] **Step 7.4: Run lint + the full test suite**

Per `CLAUDE.md` lint convention, run ruff check + format check before the release commit:

```bash
.venv/bin/ruff check osmose/ ui/ tests/
.venv/bin/ruff format --check osmose/ ui/ tests/
```

Expected: zero diagnostics. Fix any violations introduced by the SR work and re-run; do not commit a release with unfixed lint.

Then run the full pytest suite:

```bash
.venv/bin/python -m pytest -x
```

Expected: all green. If any test fails outside the SR scope, do **not** suppress it — investigate and fix root cause.

- [ ] **Step 7.5: Commit**

```bash
git add docs/parity-roadmap.md CHANGELOG.md osmose/__version__.py
git commit -m "docs+version: Beverton-Holt SR documented as post-parity divergence"
```

- [ ] **Step 7.6: Tag (only after explicit user approval)**

**Do not tag without asking.** Stop here and wait for the user to confirm the release. Once approved:

```bash
git tag v0.10.0
git push origin master --tags
```

---

## Out-of-scope follow-ons (do NOT do as part of this plan)

These are operational tasks that depend on the engine change but are tracked separately:

1. **DE calibration bound updates.** Add `ssb_half` for cod (sp0), perch (sp4), pikeperch (sp5), flounder (sp3) to the calibration parameter file with biologically-informed bounds (cod ≈ 50–250 kt based on ICES SR pairs for cod.27.24-32; flounder ≈ 5–50 kt; perch and pikeperch ≈ 1–20 kt with very wide priors since literature is thin for the Curonian Lagoon). Set `stock.recruitment.type.spN=beverton_holt` for these four in the Baltic fixture. Save as a separate commit on master after this plan ships.

2. **Phase 12 re-calibration.** Re-run joint 24+4-param DE with the four new ssbhalf parameters. Cron `926f3ab5` is already scheduled. Compare: f-objective, in-range species count, cod overshoot ratio.

3. **Per-cohort SR for herring.** Spring vs autumn cohorts may need separate ssb_half values if the v1 single-cohort treatment under-fits. Defer until phase-12 re-cal results show whether single-cohort is sufficient.

4. **Java parity audit (recurring).** Java OSMOSE has no SR. Existing parity tests (`tests/test_baltic_java_compat.py`, EEC and BoB suites) compare against Java baselines using configs without the new SR keys, so `type=none` default keeps them green. No action needed unless a future audit adds SR-on configs to the parity suite.

---

## Self-review checklist (already applied; documented for the executor)

- [x] **Spec coverage:** every requirement from `project_beverton_holt_recruitment_planned.md` maps to a task: formulation (Task 3), code change (Tasks 3-4), config keys (Tasks 1-2), backward-compat (Task 5), tests (Tasks 3-6), priority species (out-of-scope §1), Java parity (Task 7 docs).
- [x] **No placeholders:** every code step shows complete code; every test step shows assertion text; every command shows expected output.
- [x] **Type consistency:** the helper signature `apply_stock_recruitment(linear_eggs, ssb, ssb_half, recruitment_type)` matches between Task 3 (definition) and Task 4 (call site). Field names `recruitment_type` (list[str]) and `recruitment_ssb_half` (NDArray[float64]) are consistent across config parsing (Task 2), dataclass (Task 2), wiring (Task 4), and tests (Tasks 3-6).
- [x] **No spec drift:** the brainstormed `gamma` parameter was reduced to a single `ssb_half` per species — both B-H and Ricker need exactly one shape parameter beyond α, and α is already implicit in `sex_ratio · relative_fecundity · 1e6`. Documented in the Architecture preamble.
