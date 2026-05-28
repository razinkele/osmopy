# Density-Dependent Recruitment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hockey_stick` and `shepherd` stock-recruitment forms to the OSMOSE Python engine, wire Shepherd into the Baltic calibrator, and run a calibration experiment evaluating whether the new forms put more species in ICES range than the B-H baseline.

**Architecture:** Extend the existing multiplicative-correction function `apply_stock_recruitment` in place (no new module). All forms keep the multiplicative-over-linear framing so low SSB → Java-linear. Shepherd adds one per-species parameter β; at β=1 it is identically Beverton-Holt. Hockey-stick reuses the existing `ssb_half` slot as a breakpoint.

**Tech Stack:** Python 3.12, NumPy, pytest, scipy `differential_evolution`. Use `.venv/bin/python` for everything.

**Spec:** `docs/plans/2026-05-28-density-dependent-recruitment-design.md`

**Key facts verified against current master (HEAD 5198ed9):**
- `apply_stock_recruitment` is at `osmose/engine/processes/reproduction.py:15`; its only caller is `reproduction()` at `reproduction.py:124`.
- `recruitment_ssb_half` / `recruitment_type` are threaded through `config.py` at lines: parse 528-553, merge 763-766 (+bkg) and 812-813 (no-bkg), dataclass field 1204-1205, `from_dict` 1520-1521 / 1581-1582 / 1620-1621 / 1915-1916.
- The schema fields are at `osmose/schema/species.py:245-274`.
- The calibrator objective applies `10 ** x[i]` to **every** param (`scripts/calibrate_baltic.py:248`), so β goes in log10 space like all other params.
- The only direct `EngineConfig(**cfg)` constructions are 6 in `tests/test_engine_config_validation.py`, all fed by `_minimal_config()` (the `shepherd_beta` field must be added there).

---

## Task 1: Engine — Shepherd branch + β parameter

**Files:**
- Modify: `osmose/engine/processes/reproduction.py:15-59` (`apply_stock_recruitment`)
- Test: `tests/test_engine_stock_recruitment.py`

- [ ] **Step 1: Write the failing tests**

Add to the `TestApplyStockRecruitment` class in `tests/test_engine_stock_recruitment.py`:

```python
    def test_shepherd_beta_one_equals_beverton_holt(self):
        """Shepherd at beta=1 is identically Beverton-Holt (correctness anchor)."""
        linear = np.array([1000.0, 2000.0])
        ssb = np.array([500.0, 1500.0])
        ssb_half = np.array([500.0, 1000.0])
        bh = apply_stock_recruitment(
            linear, ssb, ssb_half, ["beverton_holt", "beverton_holt"]
        )
        shep = apply_stock_recruitment(
            linear, ssb, ssb_half, ["shepherd", "shepherd"], np.array([1.0, 1.0])
        )
        np.testing.assert_array_equal(shep, bh)

    def test_shepherd_low_ssb_approaches_linear(self):
        """At SSB << ssb_half, Shepherd ≈ linear for any beta."""
        linear = np.array([1000.0])
        ssb = np.array([1.0])
        ssb_half = np.array([1000.0])
        out = apply_stock_recruitment(
            linear, ssb, ssb_half, ["shepherd"], np.array([2.0])
        )
        assert abs(out[0] - linear[0]) / linear[0] < 0.01

    def test_shepherd_high_beta_overcompensates(self):
        """beta>1: with linear ∝ ssb, recruitment turns down at very high SSB."""
        alpha = 1.0
        ssb_half = np.array([500.0])
        beta = np.array([3.0])
        r_peak = apply_stock_recruitment(
            np.array([alpha * 500.0]), np.array([500.0]), ssb_half, ["shepherd"], beta
        )
        r_high = apply_stock_recruitment(
            np.array([alpha * 5000.0]), np.array([5000.0]), ssb_half, ["shepherd"], beta
        )
        assert r_high[0] < r_peak[0]

    def test_shepherd_low_beta_gentler_than_bh(self):
        """beta<1 caps less aggressively than B-H at the same high SSB."""
        linear = np.array([1000.0])
        ssb = np.array([2000.0])
        ssb_half = np.array([500.0])
        bh = apply_stock_recruitment(linear, ssb, ssb_half, ["beverton_holt"])
        shep = apply_stock_recruitment(
            linear, ssb, ssb_half, ["shepherd"], np.array([0.5])
        )
        assert shep[0] > bh[0]

    def test_shepherd_defaults_beta_one_when_array_omitted(self):
        """If shepherd_beta is not passed, beta defaults to 1.0 (≡ B-H)."""
        linear = np.array([1000.0])
        ssb = np.array([500.0])
        ssb_half = np.array([500.0])
        out = apply_stock_recruitment(linear, ssb, ssb_half, ["shepherd"])
        np.testing.assert_allclose(out, [500.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_stock_recruitment.py -k shepherd -v`
Expected: FAIL — `apply_stock_recruitment()` takes 4 positional args; `"shepherd"` hits the `else: raise ValueError(f"unknown stock-recruitment type")`.

- [ ] **Step 3: Implement the Shepherd branch + β param**

In `osmose/engine/processes/reproduction.py`, change the signature and add the branch. Replace lines 15-59 so the function reads:

```python
def apply_stock_recruitment(
    linear_eggs: NDArray[np.float64],
    ssb: NDArray[np.float64],
    ssb_half: NDArray[np.float64],
    recruitment_type: list[str],
    shepherd_beta: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply per-species density-dependent stock-recruitment.

    Multiplicative correction over the linear SSB→eggs formula. At low SSB,
    every variant approaches `linear_eggs` (preserves Java-linear regime).

    Parameters
    ----------
    linear_eggs : (n_sp,) per-step linear egg production = sex_ratio * relative_fecundity
        * SSB * season_factor * 1e6 (tonnes→grams). All non-negative.
    ssb : (n_sp,) spawning stock biomass in tonnes (per-step).
    ssb_half : (n_sp,) characteristic SSB in tonnes; for beverton_holt it is the
        half-saturation SSB, for ricker the peak, for hockey_stick the breakpoint,
        for shepherd the inflection scale. Ignored where type=="none".
    recruitment_type : per-species, one of
        {"none","beverton_holt","ricker","hockey_stick","shepherd"}.
    shepherd_beta : (n_sp,) Shepherd exponent; only read where type=="shepherd".
        None means beta=1.0 everywhere (≡ beverton_holt).

    Returns
    -------
    (n_sp,) corrected egg counts.
    """
    n_sp = linear_eggs.shape[0]
    if not (ssb.shape[0] == ssb_half.shape[0] == len(recruitment_type) == n_sp):
        raise ValueError(
            f"apply_stock_recruitment: shape mismatch — "
            f"linear_eggs={n_sp}, ssb={ssb.shape[0]}, "
            f"ssb_half={ssb_half.shape[0]}, recruitment_type={len(recruitment_type)}"
        )
    if shepherd_beta is not None and shepherd_beta.shape[0] != n_sp:
        raise ValueError(
            f"apply_stock_recruitment: shepherd_beta length {shepherd_beta.shape[0]} "
            f"!= n_sp {n_sp}"
        )

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
        elif t == "hockey_stick":
            if ssb[sp] > ssb_half[sp]:
                out[sp] = linear_eggs[sp] * (ssb_half[sp] / ssb[sp])
            # else: below/at breakpoint, no correction (out stays linear_eggs[sp])
        elif t == "shepherd":
            beta = 1.0 if shepherd_beta is None else float(shepherd_beta[sp])
            out[sp] = linear_eggs[sp] / (1.0 + (ssb[sp] / ssb_half[sp]) ** beta)
        else:
            raise ValueError(f"unknown stock-recruitment type: {t!r}")
    return out
```

(Note: the `hockey_stick` branch is exercised by Task 2's tests; it is included here to write the function once.)

- [ ] **Step 4: Run the Shepherd tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_stock_recruitment.py -k shepherd -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/reproduction.py tests/test_engine_stock_recruitment.py
git commit -m "feat(engine): add Shepherd stock-recruitment (beta=1 identical to B-H)"
```

---

## Task 2: Engine — hockey-stick branch tests

The hockey-stick code was written in Task 1; this task adds its tests. The key subtlety: hockey-stick produces a *constant* cap only because `linear_eggs ∝ ssb`, so the tests must pass `linear = alpha * ssb`.

**Files:**
- Test: `tests/test_engine_stock_recruitment.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestApplyStockRecruitment`:

```python
    def test_hockey_stick_below_breakpoint_is_linear(self):
        """At or below the breakpoint, hockey-stick applies no correction."""
        linear = np.array([800.0])
        ssb = np.array([400.0])
        ssb_half = np.array([500.0])  # breakpoint
        out = apply_stock_recruitment(linear, ssb, ssb_half, ["hockey_stick"])
        np.testing.assert_array_equal(out, linear)

    def test_hockey_stick_continuous_at_breakpoint(self):
        """With linear ∝ ssb (alpha=2), output is continuous across the breakpoint."""
        alpha = 2.0
        ssb_half = np.array([500.0])
        at = apply_stock_recruitment(
            np.array([alpha * 500.0]), np.array([500.0]), ssb_half, ["hockey_stick"]
        )
        just_above = apply_stock_recruitment(
            np.array([alpha * 501.0]), np.array([501.0]), ssb_half, ["hockey_stick"]
        )
        np.testing.assert_allclose(at, [alpha * 500.0])
        np.testing.assert_allclose(just_above, [alpha * 500.0])

    def test_hockey_stick_flat_cap_above_breakpoint(self):
        """With linear ∝ ssb, recruitment is constant (alpha*ssb_half) above breakpoint."""
        alpha = 2.0
        ssb_half = np.array([500.0])
        out_vals = []
        for s in (600.0, 1000.0, 5000.0):
            out = apply_stock_recruitment(
                np.array([alpha * s]), np.array([s]), ssb_half, ["hockey_stick"]
            )
            out_vals.append(out[0])
        np.testing.assert_allclose(out_vals, [alpha * 500.0] * 3)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_stock_recruitment.py -k hockey -v`
Expected: PASS (3 tests) — the branch already exists from Task 1.

- [ ] **Step 3: Run the full SR unit file to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/test_engine_stock_recruitment.py -v`
Expected: PASS (original 9 + 5 Shepherd + 3 hockey-stick = 17).

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_stock_recruitment.py
git commit -m "test(engine): hockey-stick continuity + flat-cap coverage"
```

---

## Task 3: Config & schema — Shepherd shape parameter

**Files:**
- Modify: `osmose/schema/species.py:245-274` (add field, extend choices)
- Modify: `osmose/engine/config.py` (allow-set 533, parse 535, validation 538, focal dict 553, merge 763-766 + 812-813, dataclass field 1205, from_dict 1521/1582/1621/1916)
- Modify: `tests/test_engine_config_validation.py:88` (`_minimal_config` adds `shepherd_beta`)
- Test: `tests/test_engine_config_validation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_engine_config_validation.py` (the `_load_example_config` helper already exists at line 187):

```python
def test_shepherd_shape_defaults_to_one(tmp_path):
    """A config with no shape key parses shepherd_beta defaulting to 1.0."""
    from osmose.engine.config import EngineConfig as _EC

    cfg = _load_example_config("baltic")
    cfg["stock.recruitment.type.sp0"] = "shepherd"
    cfg["stock.recruitment.ssbhalf.sp0"] = "120000"
    ec = _EC.from_dict(cfg)
    assert ec.shepherd_beta.shape[0] == ec.n_species + ec.n_background
    assert ec.shepherd_beta[0] == 1.0


def test_shepherd_negative_shape_raises():
    """type=shepherd with beta<=0 must raise."""
    from osmose.engine.config import EngineConfig as _EC

    cfg = _load_example_config("baltic")
    cfg["stock.recruitment.type.sp0"] = "shepherd"
    cfg["stock.recruitment.ssbhalf.sp0"] = "120000"
    cfg["stock.recruitment.shape.sp0"] = "-1.0"
    with pytest.raises(ValueError, match="stock.recruitment.shape.sp0"):
        _EC.from_dict(cfg)


def test_hockey_stick_type_accepted():
    """hockey_stick is a valid recruitment type."""
    from osmose.engine.config import EngineConfig as _EC

    cfg = _load_example_config("baltic")
    cfg["stock.recruitment.type.sp0"] = "hockey_stick"
    cfg["stock.recruitment.ssbhalf.sp0"] = "120000"
    ec = _EC.from_dict(cfg)
    assert ec.recruitment_type[0] == "hockey_stick"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -k "shepherd or hockey" -v`
Expected: FAIL — `shepherd`/`hockey_stick` rejected by the allow-set; `ec.shepherd_beta` attribute does not exist.

- [ ] **Step 3a: Extend the schema** (`osmose/schema/species.py`)

Line 249, extend choices:
```python
        choices=["none", "beverton_holt", "ricker", "hockey_stick", "shepherd"],
```
Line 250-255, extend the description's final sentence:
```python
        description=(
            "Stock-recruitment relationship applied to per-step egg production. "
            "'none' preserves the linear SSB→eggs formula (Java parity). "
            "'beverton_holt' caps recruitment asymptotically at high SSB. "
            "'ricker' over-compensates (recruitment peaks then declines). "
            "'hockey_stick' is linear up to a breakpoint then a flat cap. "
            "'shepherd' generalizes B-H via a shape exponent (see "
            "stock.recruitment.shape)."
        ),
```
After the `stock.recruitment.ssbhalf.sp{idx}` field (insert after line 274, before the closing of the field list), add a new field:
```python
    OsmoseField(
        key_pattern="stock.recruitment.shape.sp{idx}",
        param_type=ParamType.FLOAT,
        default=1.0,
        min_val=0.0,
        max_val=10.0,
        description=(
            "Shepherd stock-recruitment exponent beta. beta<1 under-compensates, "
            "beta=1 is identical to Beverton-Holt, beta>1 over-compensates "
            "(recruitment peaks then declines). Ignored unless "
            "stock.recruitment.type=shepherd."
        ),
        category="reproduction",
        indexed=True,
        required=False,
    ),
```

- [ ] **Step 3b: Extend config parsing + validation** (`osmose/engine/config.py`)

Line 533, extend the allow-set:
```python
        allowed={"none", "beverton_holt", "ricker", "hockey_stick", "shepherd"},
```
After line 537 (the `recruitment_ssb_half = ...` block), add:
```python
    recruitment_shepherd_beta = _species_float_optional(
        cfg, "stock.recruitment.shape.sp{i}", n_sp, default=1.0
    )
```
Extend the validation loop at lines 538-543:
```python
    for i in range(n_sp):
        if recruitment_type[i] != "none" and recruitment_ssb_half[i] <= 0.0:
            raise ValueError(
                f"stock.recruitment.ssbhalf.sp{i} must be > 0 when "
                f"stock.recruitment.type.sp{i}={recruitment_type[i]!r}"
            )
        if recruitment_type[i] == "shepherd" and recruitment_shepherd_beta[i] <= 0.0:
            raise ValueError(
                f"stock.recruitment.shape.sp{i} must be > 0 when "
                f"stock.recruitment.type.sp{i}='shepherd'"
            )
```
Add to the focal return dict (after line 553):
```python
        "focal_recruitment_shepherd_beta": recruitment_shepherd_beta,
```

- [ ] **Step 3c: Thread through the merge** (`osmose/engine/config.py`)

In the with-background branch, after line 766 (`recruitment_ssb_half` concat):
```python
            "recruitment_shepherd_beta": np.concatenate(
                [focal["focal_recruitment_shepherd_beta"], np.ones(n_bkg, dtype=np.float64)]
            ),
```
In the no-background branch, after line 813:
```python
            "recruitment_shepherd_beta": focal["focal_recruitment_shepherd_beta"],
```

- [ ] **Step 3d: Add the dataclass field** (`osmose/engine/config.py`)

First, update the stale comment on line 1204 — it currently lists only the
three pre-existing forms. Replace:
```python
    recruitment_type: list[str]  # one of {"none","beverton_holt","ricker"} per species
```
with:
```python
    recruitment_type: list[str]  # one of {"none","beverton_holt","ricker","hockey_stick","shepherd"} per species
```

Then, after line 1205 (`recruitment_ssb_half: NDArray[np.float64] ...`), add the new field:
```python
    shepherd_beta: NDArray[np.float64]  # per-species Shepherd exponent; 1.0 ≡ B-H
```
(If `EngineConfig.__post_init__` validates per-species array lengths via an explicit field list, add `shepherd_beta` to that list. The field is built at length `n_species + n_background` by `from_dict`, so it satisfies a generic length check.)

- [ ] **Step 3e: Thread through `from_dict`** (`osmose/engine/config.py`)

After line 1521:
```python
        focal_recruitment_shepherd_beta = _repro["focal_recruitment_shepherd_beta"]
```
After line 1582 (in the `_focal` dict):
```python
            "focal_recruitment_shepherd_beta": focal_recruitment_shepherd_beta,
```
After line 1621:
```python
        recruitment_shepherd_beta = _merged["recruitment_shepherd_beta"]
```
After line 1916 (in the constructor call):
```python
            shepherd_beta=recruitment_shepherd_beta,
```

- [ ] **Step 3f: Update the test constructor helper** (`tests/test_engine_config_validation.py`)

After line 89 (`recruitment_ssb_half=np.zeros(n_total),`):
```python
        shepherd_beta=np.ones(n_total),
```

- [ ] **Step 4: Run the new config tests**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -k "shepherd or hockey" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full config-validation file**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -v`
Expected: PASS (all, including the 6 `_minimal_config`-based tests).

- [ ] **Step 6: Commit**

```bash
git add osmose/schema/species.py osmose/engine/config.py tests/test_engine_config_validation.py
git commit -m "feat(config): stock.recruitment.shape (Shepherd beta) + new SR types in allow-set"
```

---

## Task 4: Wire `reproduction()` + verify parity & allowlist

**Files:**
- Modify: `osmose/engine/processes/reproduction.py:124-129` (pass `shepherd_beta`)
- Verify: `osmose/engine/config_validation.py` allowlist; parity tests

- [ ] **Step 1: Pass `shepherd_beta` from `reproduction()`**

In `osmose/engine/processes/reproduction.py`, change the call at lines 124-129 to:
```python
    n_eggs = apply_stock_recruitment(
        n_eggs_linear,
        ssb,
        config.recruitment_ssb_half[:n_sp],
        config.recruitment_type[:n_sp],
        config.shepherd_beta[:n_sp],
    )
```

- [ ] **Step 2: Verify the config-validation allowlist accepts the new key**

The CLAUDE.md note says the AST walker usually auto-captures keys the engine reads. Confirm:

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS with no warnings. If it warns that `stock.recruitment.shape.sp*` is unknown, add `"stock.recruitment.shape.sp{i}"` to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py` and re-run.

- [ ] **Step 3: Verify Java parity stays bit-exact**

Default fixtures don't use the new forms and β default only matters under `type=="shepherd"`, so parity must be unchanged.

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
Expected: PASS / SKIP exactly as on master HEAD (no new failures; the value-match tests are CI-gated and run locally).

- [ ] **Step 4: Run the reproduction + SR integration/regression suites**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py tests/test_engine_reproduction_sr_integration.py tests/test_engine_reproduction_sr_regression.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/reproduction.py osmose/engine/config_validation.py
git commit -m "feat(engine): thread shepherd_beta from reproduction() into stock-recruitment"
```

---

## Task 5: Calibrator — `get_phase13_shepherd_params()` + phase-13 dispatch

**Files:**
- Modify: `scripts/calibrate_baltic.py` (add param function near line 561; add dispatch + fixed type-key injection near lines 885-942)
- Test: `tests/test_calibrate_baltic_parallelism.py` (a unit test on the param function)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibrate_baltic_parallelism.py`:

```python
def test_phase13_shepherd_params_shape():
    """All 8 species get a shape key; cod sp0 ssb_half is NOT tunable (fixed)."""
    from scripts.calibrate_baltic import get_phase13_shepherd_params

    keys, bounds, x0 = get_phase13_shepherd_params()
    assert len(keys) == len(bounds) == len(x0)
    shape_keys = [k for k in keys if k.startswith("stock.recruitment.shape.sp")]
    assert len(shape_keys) == 8
    ssbhalf_keys = [k for k in keys if k.startswith("stock.recruitment.ssbhalf.sp")]
    assert "stock.recruitment.ssbhalf.sp0" not in ssbhalf_keys  # cod fixed at Bpa
    assert len(ssbhalf_keys) == 7  # sp1..sp7
    # beta x0 is log10(1.0) = 0.0 for every shape key
    for k, x in zip(keys, x0):
        if k.startswith("stock.recruitment.shape.sp"):
            assert abs(x - 0.0) < 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_parallelism.py::test_phase13_shepherd_params_shape -v`
Expected: FAIL — `get_phase13_shepherd_params` does not exist.

- [ ] **Step 3: Implement the param function**

In `scripts/calibrate_baltic.py`, add after `get_phase12_params()` (after line 574):

```python
def get_phase13_shepherd_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 13: all 8 species on Shepherd SR; joint mortality + fishing +
    per-species ssb_half (sp1-7) + shape beta (sp0-7).

    cod sp0 ssb_half stays FIXED at 120 kt (Bpa) via base_config; only its beta
    is tunable. Bounds in log10 space (the objective applies 10**x); beta in
    (0.3, 5.0) -> (log10(0.3), log10(5.0)), x0 = log10(1.0) = 0.0 (≡ B-H start).
    Bounds widened from initial (0.2, 3.0) to give DE more room to find strong
    over-compensation (beta > 2) for the perch/pikeperch x100+ overshoots
    motivating phase 13; under-compensation (beta < 1) stays accessible.

    ssb_half log10 bounds are first-pass, scaled to each species' biomass target;
    verify against data/baltic/reference/biomass_targets.csv before a long run.
    """
    keys1, bounds1, x01 = get_phase1_params()
    keys2, bounds2, x02 = get_phase2_params()

    # ssb_half for sp1..sp7 (cod sp0 fixed). log10(tonnes).
    ssbhalf_log_bounds = {
        1: (4.7, 6.3),   # herring: 50k-2M t
        2: (4.7, 6.3),   # sprat:   50k-2M t
        3: (3.7, 5.3),   # flounder: 5k-200k t
        4: (2.7, 4.7),   # perch:    0.5k-50k t
        5: (2.7, 4.7),   # pikeperch:0.5k-50k t
        6: (3.0, 5.0),   # smelt:    1k-100k t
        7: (3.0, 5.7),   # stickleback: 1k-500k t (recent HELCOM bloom estimates 150-300k t)
    }
    ssbhalf_x0_tonnes = {1: 3e5, 2: 3e5, 3: 5e4, 4: 1e4, 5: 1e4, 6: 1e4, 7: 1e4}
    ssbhalf_keys, ssbhalf_bounds, ssbhalf_x0 = [], [], []
    for i in range(1, 8):
        ssbhalf_keys.append(f"stock.recruitment.ssbhalf.sp{i}")
        ssbhalf_bounds.append(ssbhalf_log_bounds[i])
        ssbhalf_x0.append(np.log10(ssbhalf_x0_tonnes[i]))

    # shape beta for all 8 species.
    shape_keys, shape_bounds, shape_x0 = [], [], []
    for i in range(8):
        shape_keys.append(f"stock.recruitment.shape.sp{i}")
        shape_bounds.append((np.log10(0.3), np.log10(5.0)))
        shape_x0.append(np.log10(1.0))

    keys = keys1 + keys2 + ssbhalf_keys + shape_keys
    bounds = bounds1 + bounds2 + ssbhalf_bounds + shape_bounds
    x0 = x01 + x02 + ssbhalf_x0 + shape_x0
    return keys, bounds, x0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_parallelism.py::test_phase13_shepherd_params_shape -v`
Expected: PASS.

- [ ] **Step 5: Add the phase-13 dispatch + fixed type-key injection**

`--phase` is `type=str` (argparse line 1305) and the dispatch compares a local `phase` string variable. In `main()`, insert a new branch into the dispatch chain immediately **before** the trailing `else: raise ValueError(f"Unknown phase: {phase}")` (the `elif phase == "12":` block is just above it):
```python
    elif phase == "13":
        param_keys, bounds, x0 = get_phase13_shepherd_params()
```
After the existing fixed-config block (the accessibility/dynamic-predation settings around lines 894-918) and alongside the `if phase == "12":` banner block (line ~939), add a phase-13 setup block that pins every species to Shepherd and fixes cod sp0 ssb_half:
```python
    if phase == "13":
        for sp_idx in range(8):
            base_config[f"stock.recruitment.type.sp{sp_idx}"] = "shepherd"
        base_config["stock.recruitment.ssbhalf.sp0"] = "120000"  # cod Bpa, fixed
        print("Phase 13: all 8 species on Shepherd SR; tuning mortality + fishing "
              "+ ssb_half (sp1-7) + shape beta (sp0-7). cod sp0 ssb_half fixed at 120 kt.")
```
(Strings throughout — matches the existing `phase == "12"` style. The smoke run in Step 6 passes `--phase 13`, which argparse keeps as the string `"13"`.)

- [ ] **Step 6: Smoke-run the phase-13 config wiring (1 short eval, no full DE)**

Run a 1-iteration, single-seed sanity check that the config assembles and the engine runs:
```bash
.venv/bin/python scripts/calibrate_baltic.py --phase 13 --maxiter 1 --seeds 1
```
Expected: completes without a config-validation error; prints the phase-13 banner; produces a result JSON. (This is a wiring smoke test, not the real calibration.)

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_parallelism.py
git commit -m "feat(calibration): phase 13 — all-8-species Shepherd SR param set + dispatch"
```

---

## Task 6: Phase-3 experiment — baseline, run, evaluate (runbook)

This task is an experiment, not unit-tested software. It has a concrete pass/fail check. **Outcome is not guaranteed** even with a correct implementation (see spec risks).

**Files:**
- Uses: `scripts/calibrate_baltic.py`, `scripts/validate_outputs_vs_ices.py`
- Produces: result JSONs under `data/baltic/calibration_results/`, a short markdown summary under `docs/`

- [ ] **Step 1: Sanity-check the ssb_half bounds against targets**

Run: `.venv/bin/python -c "import csv,sys; [print(r) for r in csv.DictReader(open('data/baltic/reference/biomass_targets.csv'))]"`
For each of sp1-7, confirm the `ssbhalf_log_bounds` from Task 5 straddle a plausible fraction of the species' target SSB. If a bound is clearly off (e.g. upper bound below the target lower bound), adjust the dict in `get_phase13_shepherd_params()` and re-commit before the long run.

- [ ] **Step 2: Establish the B-H baseline in-ICES-range count**

Locate the best existing B-H phase-12 result (the most recent `data/baltic/calibration_results/phase12*_results.json`), run that parameter set through validation:
```bash
.venv/bin/python scripts/validate_outputs_vs_ices.py --help
```
Run the validator on the B-H baseline outputs and record: number of species strictly in their ICES envelope, and per-species magnitude factors. Capture this as the **baseline bar**.

- [ ] **Step 3: Run the phase-13 Shepherd calibration under the runtime guards**

```bash
OSMOSE_DE_WORKERS=16 .venv/bin/python scripts/calibrate_baltic.py \
  --phase 13 --optimizer de --seeds 3 \
  --warm-start data/baltic/calibration_results/phase12_results.json \
  --skip-warm-start-keys mortality.additional.rate.sp0 \
  --patience 20 --wall-clock-cap-h 12 --checkpoint-every 5
```
This is the multi-hour run. It checkpoints every 5 generations and is interrupt-safe.

`--warm-start` realises the design's "warm-start x0 from phase-12-best" intent
via the existing CLI in `scripts/calibrate_baltic.py:808` (`apply_warm_start`):
phase-12-overlapping keys (phase-1/2 mortality + fishing + sp3/4/5 ssb_half)
are loaded from the JSON; new phase-13 keys (Shepherd shape, sp1/2/6/7
ssb_half) keep their computed `x0` from Task 5. `--skip-warm-start-keys`
excludes cod sp0 adult mortality for the same reason
`launch_phase12_bh_fast.sh:33-36` does — its phase-12 optimum sat against the
cod-floor ceiling, an artefact closed first by B-H and now generalised by
Shepherd; warm-starting from that value would bias DE toward the artefact.

- [ ] **Step 4: Validate the Shepherd result against ICES**

Run the best phase-13 parameter set through `scripts/validate_outputs_vs_ices.py` (same invocation as Step 2). Record the in-ICES-range count and per-species magnitude factors.

- [ ] **Step 5: Evaluate against the success criterion**

- **Primary (pass/fail):** does the Shepherd result place **strictly more of the 8 species inside their ICES envelope** than the baseline from Step 2?
- **Secondary diagnostic:** per-species magnitude-factor change, especially perch (sp4) and pikeperch (sp5).

- [ ] **Step 6: Write a short results summary**

Create `docs/baltic_shepherd_calibration_<YYYY-MM-DD>.md` recording: baseline vs Shepherd in-range counts, per-species magnitude factors, the best β per species (interpret β>1 as over-compensation engaged), and the verdict. Commit it.

```bash
git add docs/baltic_shepherd_calibration_*.md data/baltic/calibration_results/
git commit -m "docs(calibration): phase-13 Shepherd experiment results vs B-H baseline"
```

---

## Self-review notes

- **Spec coverage:** engine Shepherd (Task 1) + hockey-stick (Tasks 1-2); config/schema/validation (Task 3); reproduction wiring + parity + allowlist (Task 4); calibrator Shepherd param set + all-8 + cod-fixed (Task 5); baseline + run + "more species in ICES range" evaluation (Task 6). All spec sections map to a task.
- **Type consistency:** `apply_stock_recruitment(..., shepherd_beta=None)` (Task 1) ↔ called with `config.shepherd_beta[:n_sp]` (Task 4); `EngineConfig.shepherd_beta` field (Task 3d) ↔ constructor kwarg `shepherd_beta=recruitment_shepherd_beta` (Task 3e) ↔ `_minimal_config` key `shepherd_beta` (Task 3f); config key `stock.recruitment.shape.sp{i}` used identically in schema, parse, validation, and calibrator.
- **Open implementation judgment (flagged, not a blocker):** the sp1-7 ssb_half bounds in Task 5 are first-pass; Task 6 Step 1 verifies them against the targets CSV before the long run.
