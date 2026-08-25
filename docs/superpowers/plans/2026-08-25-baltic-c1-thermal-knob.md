# Baltic C1 Thermal Scenario Knob — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Voss & Quaas exponential temperature-recruitment response to the thermal gate,
derive its parameters (herring quoted, cod_west fitted with pre-registered criteria), and
validate the knob with constant-temperature A/B arms whose identity arm is bit-identical.

**Architecture:** One new response shape in the existing thermal gate (load-time factor
computation, unchanged runtime); an offline series builder and an offline fit script; a 4-arm
A/B harness modeled on `scripts/baltic_f_hindcast.py`'s `run_in_memory` pattern.

**Tech Stack:** Python 3.12, NumPy, pandas, scipy, pytest. Always `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-25-baltic-c1-temperature-recruitment-scenario-knob-design.md`
— read it first; decisions 1–9 are binding and pre-registered (fit criteria, arm design,
bit-identity). Thresholds are NOT tunable.

## Global Constraints

- `.venv/bin/python` always; ruff (line 100) clean on touched files.
- EXISTING files in `data/baltic/` byte-identical; new files may be ADDED
  (`data/baltic/forcing/baltic_thermal_sr_series.csv` + `.README.md`,
  `data/baltic/calibration_results/c1_thermal_knob_arm.json`).
- The tree carries the USER'S unrelated uncommitted changes (.mcp.json, mcp_servers/,
  osmose/cli.py, **osmose/runner.py**, osmose/engine/movement_maps.py, tests/test_runner.py,
  tests/test_engine_map_movement.py, tests/test_hpc_container_touchups.py). Stage ONLY each
  task's explicit file list — never `git add tests/`, `-A`, or `commit -a`. **The spec's Java
  block-reason edit targets runner.py, which is user-dirty: it CANNOT be staged cleanly — defer
  it (Task 6 records the deferral in the results doc and the ledger).**
- Shell rules: no `$()`, no heredocs containing `#` lines, no `>` redirection, no `cd&&git`
  (use `git -C`). Multi-line python checks go to a `/tmp/*.py` file via the Write tool.
- Engine runs: check `uptime` first; never two engine jobs concurrently. Only Task 6 runs the
  engine (4 arms × 5 seeds × 50 yr ≈ 70 min).
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Guard tests that must stay green in any task touching schema/config:
  `tests/test_schema_engine_key_parity.py`, `tests/test_issue_123_known_but_unread_keys.py`,
  `tests/test_engine_config_validation.py` (warn-mode cleanliness).

---

### Task 1: Engine — exponential response, mode matrix, explicit tref, offset guards, schema

**Files:**
- Modify: `osmose/engine/processes/thermal_gate.py` (add `exponential_response`; extend
  `normalize_factor` with `raw`)
- Modify: `osmose/engine/config.py` (`_load_thermal_gate` ~1241-1320; `_load_rv_gate` offset
  guard ~1147-1150)
- Modify: `osmose/schema/species.py` (two new fields), `CLAUDE.md` (registry count)
- Test: the module where thermal-gate tests live (find it:
  `grep -rln "thermal.gate\|thermal_gate" tests/` — expected `tests/test_engine_thermal_gate.py`
  or similar; append there) and the RV-gate test module for the RV offset guard.

**Interfaces:**
- Produces: config keys `reproduction.thermal.gate.response` (`logistic`|`exponential`, default
  `logistic`), `reproduction.thermal.gate.beta.sp{N}` (required per enabled species under
  exponential), explicit-`tref.sp{N}` requirement under exponential; mode `raw`; ValueError on
  negative `start.year` offsets in BOTH year-indexed gates. Tasks 5–6 rely on: factor ≡ 1.0
  exactly when the series column equals tref everywhere.

- [ ] **Step 1: Write the failing tests** (append to the thermal-gate test module; adapt the
  existing tests' config-fixture idiom — they already construct loader configs with a series
  CSV; follow that pattern for `_cfg()` below):

```python
class TestExponentialResponse:
    """C1 spec decisions 2, 8, 9 — Voss & Quaas exponential response."""

    def _series_csv(self, tmp_path, temps_sp0, first_year=1974):
        rows = ["year,temp_sp0"]
        for i, t in enumerate(temps_sp0):
            rows.append(f"{first_year + i},{t}")
        p = tmp_path / "thermal.csv"
        p.write_text("\n".join(rows) + "\n")
        return p

    def _cfg(self, tmp_path, temps, **over):
        cfg = {
            "simulation.nspecies": "1",
            "simulation.time.ndtperyear": "4",
            "simulation.time.nyear": str(len(temps)),
            "_osmose.config.dir": str(tmp_path),
            "reproduction.thermal.gate.enabled": "true",
            "reproduction.thermal.gate.series.file": str(self._series_csv(tmp_path, temps)),
            "reproduction.thermal.gate.species.enabled.sp0": "true",
            "reproduction.thermal.gate.response": "exponential",
            "reproduction.thermal.gate.beta.sp0": "-0.51",
            "reproduction.thermal.gate.tref.sp0": "7.0",
        }
        cfg.update(over)
        return cfg

    def test_factor_is_exactly_one_at_tref(self, tmp_path):
        from osmose.engine.config import _load_thermal_gate

        factor, enabled, offset = _load_thermal_gate(self._cfg(tmp_path, [7.0] * 5), 1, 4, 5)
        assert (factor[:, 0] == 1.0).all()  # exp(0) == 1.0 exactly — bit-identity rests on this

    def test_exponential_scaling(self, tmp_path):
        import numpy as np
        from osmose.engine.config import _load_thermal_gate

        factor, _, _ = _load_thermal_gate(self._cfg(tmp_path, [9.0] * 3), 1, 4, 3)
        assert np.allclose(factor[:, 0], np.exp(-0.51 * 2.0))

    def test_missing_beta_raises(self, tmp_path):
        import pytest
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [7.0] * 3)
        del cfg["reproduction.thermal.gate.beta.sp0"]
        with pytest.raises(ValueError, match="beta.sp0"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_missing_tref_raises_not_defaults(self, tmp_path):
        """The key has a silent 20.0 thermal_cap default the exponential path must refuse."""
        import pytest
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [7.0] * 3)
        del cfg["reproduction.thermal.gate.tref.sp0"]
        with pytest.raises(ValueError, match="tref.sp0"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_mode_matrix(self, tmp_path):
        import pytest
        from osmose.engine.config import _load_thermal_gate

        for bad in ("thermal_cap", "mean_preserving"):
            cfg = self._cfg(tmp_path, [7.0] * 3)
            cfg["reproduction.thermal.gate.mode"] = bad
            with pytest.raises(ValueError, match="raw"):
                _load_thermal_gate(cfg, 1, 4, 3)
        cfg = self._cfg(tmp_path, [7.0] * 3)
        cfg["reproduction.thermal.gate.mode"] = "raw"
        _load_thermal_gate(cfg, 1, 4, 3)  # explicit raw OK
        cfg = self._cfg(tmp_path, [7.0] * 3)
        cfg["reproduction.thermal.gate.response"] = "logistic"
        cfg["reproduction.thermal.gate.mode"] = "raw"
        cfg["reproduction.thermal.gate.t50.sp0"] = "18.5"
        with pytest.raises(ValueError, match="raw"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_negative_offset_raises(self, tmp_path):
        import pytest
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [7.0] * 3)
        cfg["reproduction.thermal.gate.start.year"] = "1960"  # < series first year 1974
        with pytest.raises(ValueError, match="negative|predates"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_floor_applies_under_raw(self, tmp_path):
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [27.0] * 3)  # exp(-0.51*20) ~ 4e-5
        cfg["reproduction.thermal.gate.floor"] = "0.05"
        factor, _, _ = _load_thermal_gate(cfg, 1, 4, 3)
        assert (factor[:, 0] == 0.05).all()
```

  And in the RV-gate test module, the sibling guard test:

```python
def test_rv_gate_negative_offset_raises(tmp_path):
    """B1-audit latent bug: negative offset feeds Python negative indexing (reads the
    series END). C1 spec decision 9 adds load-time rejection."""
    import pytest

    # Reuse the module's existing minimal RV-gate config fixture; set
    # reproduction.rv.gate.start.year to a year BEFORE the series' first year
    # and assert ValueError at load (match="negative|predates").
```

  (Complete the RV test by copying the module's existing RV-gate fixture setup verbatim — the
  fixture builds a small `year,spawning_rv` CSV; only the `start.year` override and the
  `pytest.raises` differ.)

- [ ] **Step 2: Run to verify failures**

Run: `.venv/bin/python -m pytest <thermal test module> -k Exponential -v`
Expected: FAIL — `response`/`beta` keys unknown to the loader (it computes the logistic and
raises on the missing `t50` default path or produces wrong factors).

- [ ] **Step 3: Implement.** In `osmose/engine/processes/thermal_gate.py` add after
  `logistic_response`:

```python
def exponential_response(
    temp: NDArray[np.float64], beta: float, tref: float
) -> NDArray[np.float64]:
    """Voss & Quaas (2026, doi:10.1093/icesjms/fsag033) productivity factor
    exp(beta * (T - tref)). beta < 0 encodes warming-reduces-recruitment; the
    factor is exactly 1.0 at T == tref and is deliberately uncapped above 1
    (the paper's Beverton-Holt numerator has no cap). Scenario knob — see spec
    2026-08-25; NOT a validated mechanism.
    """
    return np.exp(beta * (temp - tref))
```

  In `normalize_factor`, add the `raw` branch first and extend the docstring:

```python
    if mode == "raw":
        # exponential response only: the factor IS the response; tref anchoring
        # replaces normalisation (C1 spec decision 8).
        factor = r.copy()
    elif mode == "thermal_cap":
        ...
```

  In `_load_thermal_gate`: parse `response` right after the mode block, restructure per the
  spec's decision 8 rules:

```python
    response = cfg.get("reproduction.thermal.gate.response", "logistic")
    if response not in ("logistic", "exponential"):
        raise ValueError(f"unknown reproduction.thermal.gate.response: {response!r}")
    mode_cfg = cfg.get("reproduction.thermal.gate.mode")
    if response == "exponential":
        mode = mode_cfg if mode_cfg is not None else "raw"
        if mode != "raw":
            raise ValueError(
                "response=exponential requires mode 'raw' (or unset): thermal_cap/"
                "mean_preserving would renormalise away the scenario offsets "
                "(C1 spec decision 8)."
            )
    else:
        mode = mode_cfg if mode_cfg is not None else "thermal_cap"
        if mode == "raw":
            raise ValueError("mode 'raw' is only valid with response=exponential.")
        if mode not in ("thermal_cap", "mean_preserving"):
            raise ValueError(f"unknown reproduction.thermal.gate.mode: {mode!r}")
```

  (this REPLACES the existing `mode = cfg.get(...)`/validation lines), the offset guard right
  after `offset = start_year - first_year`:

```python
    if offset < 0:
        raise ValueError(
            f"reproduction.thermal.gate.start.year={start_year} predates the series "
            f"first year {first_year}: a negative offset silently wraps the year "
            "index (modulo) to the wrong year. Use a start.year >= the series start."
        )
```

  and the per-species branch (inside the loop, replacing the single `r = logistic_response(...)`
  line and the existing tref/r_ref block):

```python
        if response == "exponential":
            beta_key = f"reproduction.thermal.gate.beta.sp{sp}"
            tref_key = f"reproduction.thermal.gate.tref.sp{sp}"
            if beta_key not in cfg:
                raise ValueError(f"{beta_key} is required with response=exponential.")
            if tref_key not in cfg:
                raise ValueError(
                    f"{tref_key} is required with response=exponential (the key's 20.0 C "
                    "thermal_cap default would be a silently wrong anchor)."
                )
            r = exponential_response(temp, float(cfg[beta_key]), float(cfg[tref_key]))
            r_ref = 0.0
        else:
            t50 = float(cfg.get(f"reproduction.thermal.gate.t50.sp{sp}", "18.5"))
            <existing logistic branch, unchanged, incl. its thermal_cap tref/r_ref logic>
```

  Import `exponential_response` beside `logistic_response` at the top of the function. In
  `_load_rv_gate` (`config.py` ~1147-1150), after its `offset = start_year - first_year`
  equivalent, add the sibling guard (message: RV mechanism is Python **negative indexing** —
  reads from the series END — not modulo; word it accordingly).

  Schema (`osmose/schema/species.py`, beside the existing `reproduction.thermal.gate.*`
  fields — find them, follow their exact kwargs idiom):
  `reproduction.thermal.gate.response` (STRING/choices if the idiom supports choices, default
  `"logistic"`, not indexed) and `reproduction.thermal.gate.beta.sp{idx}` (FLOAT, indexed,
  `required=False`, no min/max clamp — beta is signed). Check whether `tref.sp{idx}` already
  has a schema field (`grep -n "tref" osmose/schema/species.py`); add it only if absent. Update
  CLAUDE.md's registry count to the computed value:
  `.venv/bin/python -c "from osmose.schema import build_registry; print(len(build_registry().all_fields()))"`.

- [ ] **Step 4: Run the tests + guards**

Run: `.venv/bin/python -m pytest <thermal test module> <rv test module> tests/test_schema.py tests/test_schema_engine_key_parity.py tests/test_issue_123_known_but_unread_keys.py tests/test_engine_config_validation.py -v`
Expected: all PASS (the new keys are AST-captured literals/f-strings — no allowlist edits; the
warn-mode example-config test must stay warning-free).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check osmose/ tests/
git add osmose/engine/processes/thermal_gate.py osmose/engine/config.py osmose/schema/species.py CLAUDE.md <thermal test module> <rv test module> tests/test_schema.py
git commit -m "feat(engine): exponential thermal-gate response + negative-offset guards (C1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

  (Stage `tests/test_schema.py` only if you touched it for a registry-count or field test;
  otherwise drop it from the list. NEVER stage runner.py — it carries user changes; the spec's
  Java block-reason edit is DEFERRED, see Global Constraints.)

---

### Task 2: Series builder `scripts/build_baltic_thermal_sr_series.py` + unit tests

**Files:**
- Create: `scripts/build_baltic_thermal_sr_series.py`
- Test: `tests/test_build_baltic_thermal_sr_series.py`

**Interfaces:**
- Produces: pure functions `quarter_mean(monthly_temps: dict[int, float], quarter: int) -> float`
  (monthly dict keyed 1–12; Q3 = {7,8,9}, Q4 = {10,11,12}),
  `assemble_series(hist: dict[int, float], tref: float, first_hist_year: int = 1993, spinup: int = 19) -> list[tuple[int, float]]`
  (rows (year, T): synthetic years `first_hist_year-spinup .. first_hist_year-1` at `tref`, then
  the historical years ascending — contiguity guaranteed or ValueError),
  `write_series_csv(path, rows_by_species: dict[int, list[tuple[int, float]]])` (columns
  `year,temp_sp0,temp_sp1`, NO comment lines — the loader rejects them; provenance goes to
  `<path>.README.md` via `write_readme(...)`). CLI `main()` performs the CMEMS
  extraction/downloads. Task 4 runs `main()`; Task 5's harness reads only tref values from the
  README/overlay, not this file.

- [ ] **Step 1: Failing tests** (synthetic only — no network, no NetCDF):

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_baltic_thermal_sr_series",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_thermal_sr_series.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_quarter_means():
    monthly = {mo: float(mo) for mo in range(1, 13)}
    assert m.quarter_mean(monthly, 3) == 8.0    # (7+8+9)/3
    assert m.quarter_mean(monthly, 4) == 11.0   # (10+11+12)/3


def test_assemble_series_layout():
    hist = {1993 + i: 10.0 + i * 0.1 for i in range(31)}
    rows = m.assemble_series(hist, tref=11.5)
    assert len(rows) == 50
    assert rows[0] == (1974, 11.5) and rows[18] == (1992, 11.5)
    assert rows[19] == (1993, 10.0) and rows[-1] == (2023, 13.0)
    years = [y for y, _ in rows]
    assert years == list(range(1974, 2024))  # contiguous — the loader requires it


def test_assemble_series_rejects_gap():
    import pytest

    hist = {1993: 10.0, 1995: 10.2}
    with pytest.raises(ValueError):
        m.assemble_series(hist, tref=10.0)


def test_write_series_csv_no_comments(tmp_path):
    p = tmp_path / "s.csv"
    rows = [(1974 + i, 7.0) for i in range(3)]
    m.write_series_csv(p, {0: rows, 1: [(y, 8.0) for y, _ in rows]})
    text = p.read_text()
    assert "#" not in text                       # comments crash the loader
    assert text.splitlines()[0] == "year,temp_sp0,temp_sp1"
    assert text.splitlines()[1] == "1974,7.0,8.0"
```

- [ ] **Step 2: Verify FAIL at import.** Run:
  `.venv/bin/python -m pytest tests/test_build_baltic_thermal_sr_series.py -v`

- [ ] **Step 3: Implement.** Pure functions exactly per the Interfaces block (straightforward —
  `quarter_mean` averages the three months, raising KeyError→ValueError on missing months;
  `assemble_series` validates historical-year contiguity then prepends the spin-up rows;
  `write_series_csv` emits the header + comma rows with `repr`-free plain `str(float)` values).
  For `main()`: read the two precedent files FIRST and reuse their machinery —
  `scripts/build_percid_thermal_series.py` (bbox extraction, monthly means from CMEMS NetCDF,
  cache layout) and `scripts/download_baltic_rv_forcing.py` (credentialed download pattern,
  `.env` via the copernicus conventions). Behavior: cod_west ← surface `thetao` Q3 mean over
  SD22–24 (bbox 9.5–15.0E, 53.5–56.5N — same box the review used); herring ← `bottomT` Q4 mean,
  same box; window 1993 → the product's available end; download only what
  `data/cmems_cache/cmems_downloads/` lacks; on download failure print an explicit
  `DEGRADED:` line naming what is missing and continue with what exists (the spec's decision-5
  fallback), writing the README with the degradation recorded. tref per species = mean over the
  obtained window, printed and written to the README.

- [ ] **Step 4: Run tests + ruff**, expected 4 PASS, clean.

- [ ] **Step 5: Commit** (script + tests only — NO data yet):

```bash
git add scripts/build_baltic_thermal_sr_series.py tests/test_build_baltic_thermal_sr_series.py
git commit -m "feat(scripts): C1 thermal series builder (SST Q3 / bottom-T Q4, SD22-24)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Fit script `scripts/fit_codwest_thermal_sr.py` + unit tests

**Files:**
- Create: `scripts/fit_codwest_thermal_sr.py`
- Test: `tests/test_fit_codwest_thermal_sr.py`

**Interfaces:**
- Produces: `paired_data(recs: list[dict], temps: dict[int, float], hatch_years: range) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  returning (R, SSB, T) where **R comes from assessment row hatch_year+1** (recruitment_age=1)
  and SSB/T from hatch_year — rows with missing values dropped;
  `fit_bh_exp(r, ssb, temp) -> dict` with keys `beta1`, `se`, `p` (two-sided, asymptotic from
  the least-squares jacobian), `b0`, `b3`, `n`;
  `detrended(temp: np.ndarray, years: np.ndarray) -> np.ndarray` (residuals of T ~ year OLS);
  `verdict(fit, fit_detrended) -> dict` implementing spec decision 4:
  `enabled = fit['beta1'] < 0 and fit['p'] < 0.1 and fit_detrended['beta1'] < 0`.
  `main()` loads the cod.27.22-24 snapshot + the Task-4 series, runs primary, detrended, and
  leave-one-out-terminal fits, writes `docs/baltic_c1_codwest_fit_YYYY-MM-DD.md`.

- [ ] **Step 1: Failing tests** (synthetic, deterministic — seed the noise):

```python
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "fit_codwest_thermal_sr",
    Path(__file__).resolve().parent.parent / "scripts" / "fit_codwest_thermal_sr.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _synthetic(beta1, n=29, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n)
    temp = 16.0 + rng.normal(0, 1.0, n)
    ssb = np.exp(rng.normal(9.0, 0.3, n))
    b0, b3 = -1.0, 1e-4
    ln_r = -b0 + beta1 * temp + np.log(ssb) - np.log1p(b3 * ssb) + rng.normal(0, 0.1, n)
    return np.exp(ln_r), ssb, temp, years


def test_fit_recovers_known_beta():
    r, ssb, temp, _ = _synthetic(beta1=-0.4)
    fit = m.fit_bh_exp(r, ssb, temp)
    assert abs(fit["beta1"] - (-0.4)) < 0.05
    assert fit["p"] < 0.01


def test_paired_data_applies_age1_lag():
    recs = [
        {"year": "2000", "ssb": "100.0", "recruitment": "555.0"},
        {"year": "2001", "ssb": "110.0", "recruitment": "666.0"},
    ]
    r, ssb, t = m.paired_data(recs, {2000: 15.0}, range(2000, 2001))
    assert r[0] == 666.0 and ssb[0] == 100.0 and t[0] == 15.0   # R_{y+1} <- SSB_y, T_y


def test_detrend_kills_trend_only_signal():
    rng = np.random.default_rng(1)
    n = 29
    years = np.arange(1993, 1993 + n)
    temp = 14.0 + 0.05 * (years - 1993) + rng.normal(0, 0.15, n)  # strong trend
    ssb = np.exp(rng.normal(9.0, 0.3, n))
    ln_r = np.log(ssb) - 0.02 * (years - 1993) + rng.normal(0, 0.1, n)  # trend, NOT T-driven
    fit = m.fit_bh_exp(np.exp(ln_r), ssb, temp)
    fit_d = m.fit_bh_exp(np.exp(ln_r), ssb, m.detrended(temp, years))
    v = m.verdict(fit, fit_d)
    assert isinstance(v["enabled"], bool)  # runs end-to-end; and on this fixture:
    assert abs(fit_d["beta1"]) < abs(fit["beta1"])  # detrending shrinks the spurious signal
```

- [ ] **Step 2: Verify FAIL at import**, run the test file.

- [ ] **Step 3: Implement.** `fit_bh_exp`: `scipy.optimize.least_squares` on residuals
  `ln(r) - (-b0 + b1*T + ln(ssb) - log1p(b3*ssb))`, `b3` bounded ≥ 0, x0 = (0, 0, 1e-4);
  asymptotic covariance `s² (JᵀJ)⁻¹` from the solution jacobian, two-sided p from Student-t
  with n−3 dof (`scipy.stats.t.sf`). `paired_data` per the Interfaces contract (snapshot values
  are strings; skip rows with empty `ssb`/`recruitment`). `main()`: hatch years 1993–2021,
  R rows 1994–2022 (the snapshot's last row IS 2022 — the review corrected the spec's "~2023");
  temperature from the Task-4 series CSV's `temp_sp0` column (historical rows only); primary +
  detrended + leave-one-out-terminal fits; verdict per decision 4; write the dated fit doc with
  all three fits' numbers and the verdict, plus the no-supplement note (self-fit is the sole
  source; no cross-check exists).

- [ ] **Step 4: Run tests + ruff**, expected 3 PASS.

- [ ] **Step 5: Commit** (script + tests only):

```bash
git add scripts/fit_codwest_thermal_sr.py tests/test_fit_codwest_thermal_sr.py
git commit -m "feat(scripts): C1 cod_west BH-exp(T) fit, pre-registered criteria (lag, detrend)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Real data + fit run — series CSV, README, fit verdict doc

**Files:**
- Create (generated): `data/baltic/forcing/baltic_thermal_sr_series.csv`,
  `data/baltic/forcing/baltic_thermal_sr_series.README.md`,
  `docs/baltic_c1_codwest_fit_2026-MM-DD.md` (actual date)

- [ ] **Step 1:** `uptime` low (downloads + light compute only, but be polite). CMEMS
  credentials come from `.env` (never echo them). Run
  `.venv/bin/python scripts/build_baltic_thermal_sr_series.py` (background if downloads are
  slow). Record: window obtained per variable, any `DEGRADED:` lines, tref values.
- [ ] **Step 2:** Sanity: the CSV has 50 rows + header, years 1974–2023, no `#`; trefs plausible
  (SST Q3 ~15–18 °C, bottom-T Q4 ~6–11 °C). Load-through-engine check: write `/tmp/c1_smoke.py`
  (Write tool) that builds a minimal 2-species config pointing
  `reproduction.thermal.gate.series.file` at the real CSV with `response=exponential`,
  `beta.sp0=-0.1`, `tref.sp0=<the README's sp0 tref>`, enabled sp0, and asserts
  `_load_thermal_gate` returns 50-row factors with the 19 spin-up rows == 1.0 exactly.
  If `bottomT` was DEGRADED: herring tref falls back per spec decision 5 — set it from the
  literature constant recorded in the README and say so in every downstream doc.
- [ ] **Step 3:** Run `.venv/bin/python scripts/fit_codwest_thermal_sr.py`. Read the verdict.
  Either outcome proceeds (pre-registered); the verdict decides whether Task 6's overlay
  enables sp0.
- [ ] **Step 4: Commit** the three files:

```bash
git add data/baltic/forcing/baltic_thermal_sr_series.csv data/baltic/forcing/baltic_thermal_sr_series.README.md docs/baltic_c1_codwest_fit_2026-*.md
git commit -m "data(baltic): C1 thermal series + cod_west fit verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: A/B harness `scripts/baltic_c1_knob_ab.py` + helper tests

**Files:**
- Create: `scripts/baltic_c1_knob_ab.py`
- Test: `tests/test_baltic_c1_knob_helpers.py`

**Interfaces:**
- Consumes: tref values + enabled-species set from Task 4's README/fit verdict (passed as
  constants at the top of the script, edited to match Task 4's outputs);
  `PythonEngine().run_in_memory(raw, seed)` with `.biomass()` (per-species-column DataFrame —
  the F1 harness precedent, `scripts/baltic_f_hindcast.py`).
- Produces: `write_arm_series(path, trefs: dict[int, float], dT: float)` (50 rows, years
  1974–2023; spin-up rows 1974–1992 at tref, 1993–2023 at tref+dT — spin-up shared across
  arms); `expected_factors(beta: float, dT: float, n_year: int = 50, spinup: int = 19) -> np.ndarray`
  ( [1.0]*spinup + [exp(beta*dT)]*(n_year-spinup) ); `arm_overrides(mode, series_path, trefs, betas, enabled)`;
  `run_ab(seeds=(42, 123, 7, 999, 2024)) -> dict` writing `/tmp/c1_knob_report.json`.

- [ ] **Step 1: Failing tests** (CI-safe, no engine):

```python
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "baltic_c1_knob_ab",
    Path(__file__).resolve().parent.parent / "scripts" / "baltic_c1_knob_ab.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_arm_series_layout(tmp_path):
    p = tmp_path / "arm.csv"
    m.write_arm_series(p, trefs={0: 16.0, 1: 8.0}, dT=2.0)
    lines = p.read_text().splitlines()
    assert lines[0] == "year,temp_sp0,temp_sp1"
    assert lines[1] == "1974,16.0,8.0"          # spin-up at tref in EVERY arm
    assert lines[19] == "1992,16.0,8.0"
    assert lines[20] == "1993,18.0,10.0"        # +dT on the historical block only
    assert len(lines) == 51 and "#" not in p.read_text()


def test_expected_factors():
    f = m.expected_factors(beta=-0.51, dT=2.0)
    assert (f[:19] == 1.0).all()
    assert np.allclose(f[19:], np.exp(-1.02))
    assert (m.expected_factors(beta=-0.51, dT=0.0) == 1.0).all()   # identity arm


def test_arm_overrides_identity_vs_scenario(tmp_path):
    p = tmp_path / "arm.csv"
    m.write_arm_series(p, {1: 8.0}, dT=0.0)
    base = m.arm_overrides("knob", str(p), trefs={1: 8.0}, betas={1: -0.51}, enabled=(1,))
    assert base["reproduction.thermal.gate.enabled"] == "true"
    assert base["reproduction.thermal.gate.response"] == "exponential"
    assert base["reproduction.thermal.gate.beta.sp1"] == "-0.51"
    assert base["reproduction.thermal.gate.tref.sp1"] == "8.0"
    off = m.arm_overrides("off", str(p), {1: 8.0}, {1: -0.51}, (1,))
    assert "reproduction.thermal.gate.enabled" not in off
```

- [ ] **Step 2: FAIL at import**, run the file.
- [ ] **Step 3: Implement.** Model `run_ab` on `scripts/baltic_f_hindcast.py`'s loop: base
  config from `osmose_demo("baltic", tmp)` + reader; arms `off`, `knob0` (dT=0), `knob2`,
  `knob4`; both `simulation.time.nyear=50` everywhere. Per seed: run all four arms, store
  per-species `.biomass()` arrays. **Instrument (spec §4c):** before any run, call
  `_load_thermal_gate` on each knob arm's assembled config and assert the returned factor
  column for each enabled species equals `expected_factors(beta, dT)` exactly (loader-level
  determinism check). **Identity (spec §4a):** for every seed and every species,
  `np.array_equal(bio_off, bio_knob0)` — collect violations, do not raise (report them; the
  verdict logic marks FAIL). **Monotonicity (§4b):** final-decade means per enabled species
  strictly decreasing over knob0→knob2→knob4 (5-seed means). Elasticity (§4d): ratio of
  (knob_dT final-decade mean / knob0 final-decade mean) to exp(beta*dT), reported. Write
  everything to `/tmp/c1_knob_report.json` (json.dump — no stdout redirection).
- [ ] **Step 4: Run tests + ruff**, expected 3 PASS.
- [ ] **Step 5: Commit** (script + tests):

```bash
git add scripts/baltic_c1_knob_ab.py tests/test_baltic_c1_knob_helpers.py
git commit -m "feat(scripts): C1 knob A/B harness — constant-T arms, bit-identity check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: A/B run + results doc + overlay JSON

**Files:**
- Create: `docs/baltic_c1_knob_ab_2026-MM-DD.md` (actual date),
  `data/baltic/calibration_results/c1_thermal_knob_arm.json`

- [ ] **Step 1:** `uptime` low; run `.venv/bin/python scripts/baltic_c1_knob_ab.py` via Bash
  `run_in_background` (4 arms × 5 seeds × 50 yr ≈ 70 min; never concurrent with other engine
  jobs).
- [ ] **Step 2: Gates in order.** (1) Instrument assertions passed (in-report flags); (2)
  identity: ZERO bit-identity violations — any violation is a wiring bug: STOP, debug, do not
  interpret; (3) monotonicity per enabled species; (4) elasticities reported (expected damped
  vs exp(betaΔT) — the Non-goals double-counting caveat, restated).
- [ ] **Step 3: Results doc** with: verdict up top (PASS = identity + monotonicity for all
  enabled species); the cod_west enable/disable status from Task 4's fit; every labelled
  approximation (herring ~9% catch-share transplant, CMEMS-for-BSIO, any bottomT degradation);
  the deferred Java block-reason item (runner.py user-dirty — follow-up when that file is
  committed); the B2 interface sentence (swap the series CSV = the whole future hookup); run
  provenance (arms, seeds, commit range, NOT a CI gate).
- [ ] **Step 4: Overlay JSON** — the knob keys only (enabled, response, series.file pointing at
  the committed CSV path relative to the config dir, per-species enabled/beta/tref for the
  species the fit verdict enables; NO nyear key — horizon is the harness's business).
- [ ] **Step 5: Commit**:

```bash
git add docs/baltic_c1_knob_ab_2026-*.md data/baltic/calibration_results/c1_thermal_knob_arm.json
git commit -m "docs(baltic): C1 thermal knob A/B — results, verdict, scenario overlay

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution notes

- Tasks 1–3 and 5 are CI-safe; Task 4 needs CMEMS credentials (`.env`) and network; Task 6 is
  the only engine run.
- Task 5's constants (trefs, betas, enabled set) come from Task 4's outputs — the Task 5
  implementer receives them in its dispatch, not by reading Task 4's artifacts blind.
- If `run_in_memory(...).biomass()` surprises, read `scripts/baltic_f_hindcast.py` and
  `osmose/results.py:455-500` first — the accessor contract is established there (wide
  DataFrame, per-species-name columns, 50 annual rows at the Baltic output cadence).
- The spec is the authority on every pre-registered value; contradictions STOP the task and
  surface to the controller, never get patched silently.
