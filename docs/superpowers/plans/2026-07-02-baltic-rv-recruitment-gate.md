# Baltic cod reproductive-volume recruitment gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-gated, cod-only recruitment gate that multiplies cod egg production by a per-model-year spawning-season reproductive-volume factor, so cod recruitment responds to the real inflow/stagnation cycle instead of overshooting unconstrained.

**Architecture:** A precomputed per-year RV series (built from the diagnostic) is loaded at config build into three `EngineConfig` fields (a 1-D factor-by-series-index array, a per-species enable mask, a year offset). A pure helper `rv_gate_factor(config, step)` returns a per-species egg multiplier (1.0 where disabled). `reproduction()` applies it to `n_eggs` after stock-recruitment, skipping steps where cod SSB was seeded. Inert by default: when the master switch is off, no field is built and no multiply happens.

**Tech Stack:** Python 3.12+, NumPy, pandas, pytest, ruff, pyright. OSMOSE schema-driven config (`osmose/schema/`), engine (`osmose/engine/`).

**Spec:** `docs/superpowers/specs/2026-07-02-baltic-rv-recruitment-gate-design.md` (read it first).

## Global Constraints

- Python `.venv/bin/python`; run tests with `.venv/bin/python -m pytest`.
- Lint: `.venv/bin/ruff check osmose/ scripts/ tests/` AND `.venv/bin/ruff format --check osmose/ scripts/ tests/`. Line length 100.
- Types: `.venv/bin/pyright` must be clean on changed files.
- Config keys are lowercase dot-separated. New per-species keys use `sp{idx}`.
- **Inert by default:** with `reproduction.rv.gate.enabled=false` (the default, and the only setting in every bundled config), Baltic/EEC/BoB engine output must be **bit-identical** to pre-change. The feature is exercised only by an explicit opt-in overlay, never by a bundled config.
- Mode formulas (verbatim from spec §3.2): `mean_preserving`: `m(y) = rv[idx(y)] / D`, `D` = multiset mean of `rv[idx(y')]` over model years `y'=0..nyear-1`. `raw_cap`: `m(y) = clip(rv[idx(y)] / ref, 0, 1)`. Then `m = max(m, floor)`. `idx(y) = (start_year - first_year + y) mod n_years`.
- Cod is species index `sp0` in the Baltic config.
- Commit after each task with a `feat:`/`test:`/`docs:` message ending with the Co-Authored-By trailer used in this repo.

---

## File Structure

- `osmose/schema/species.py` — **modify**: append 7 `OsmoseField`s to `SPECIES_FIELDS` (the gate config keys). Auto-registered via `osmose/schema/__init__.py`.
- `scripts/baltic_rv_overshoot_diagnostic.py` — **modify**: add `build_rv_gate_series(...)` + `--emit-gate-series` CLI flag; add a `window` arg to `characterise_instability`.
- `data/baltic/forcing/baltic_rv_gate_series.csv` — **create** (generated artifact): per-year spawning-season RV, 1993–2021.
- `osmose/engine/config.py` — **modify**: add 3 `EngineConfig` dataclass fields; add `_load_rv_gate(...)` loader with fail-fast validation; call it in `from_dict`.
- `osmose/engine/processes/recruitment_gate.py` — **create**: pure `rv_gate_factor(config, step)`.
- `osmose/engine/processes/reproduction.py` — **modify**: track `seeded_this_step`; apply the gate to `n_eggs`.
- `tests/test_rv_recruitment_gate.py` — **create**: unit + integration + parity tests.

---

## Task 1: Schema fields for the gate

**Files:**
- Modify: `osmose/schema/species.py` (append to `SPECIES_FIELDS`, before the closing `]` at line ~440)
- Test: `tests/test_rv_recruitment_gate.py`

**Interfaces:**
- Produces: config keys `reproduction.rv.gate.enabled` (bool), `reproduction.rv.gate.mode` (enum), `reproduction.rv.gate.series.file` (file path), `reproduction.rv.gate.ref` (float), `reproduction.rv.gate.floor` (float), `reproduction.rv.gate.start.year` (int), `reproduction.rv.gate.species.enabled.sp{idx}` (bool) — all discoverable via `build_registry()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rv_recruitment_gate.py`:

```python
from osmose.schema import build_registry


def test_rv_gate_keys_registered():
    reg = build_registry()
    keys = {f.key_pattern for f in reg.all_fields()}
    assert "reproduction.rv.gate.enabled" in keys
    assert "reproduction.rv.gate.mode" in keys
    assert "reproduction.rv.gate.series.file" in keys
    assert "reproduction.rv.gate.ref" in keys
    assert "reproduction.rv.gate.floor" in keys
    assert "reproduction.rv.gate.start.year" in keys
    assert "reproduction.rv.gate.species.enabled.sp{idx}" in keys
```

(`build_registry()` returns a `ParameterRegistry`; `.all_fields()` yields the `OsmoseField`s — confirmed API at `osmose/schema/registry.py:36`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py::test_rv_gate_keys_registered -v`
Expected: FAIL (keys absent).

- [ ] **Step 3: Add the fields**

In `osmose/schema/species.py`, immediately before the closing `]` of `SPECIES_FIELDS` (line ~440), insert:

```python
    OsmoseField(
        key_pattern="reproduction.rv.gate.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description=(
            "Master switch for the reproductive-volume recruitment gate "
            "(Baltic cod). When false the gate is inert and output is unchanged."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.gate.mode",
        param_type=ParamType.ENUM,
        default="mean_preserving",
        choices=["mean_preserving", "raw_cap"],
        description=(
            "RV gate mode. 'mean_preserving' normalises the per-year factor to "
            "mean 1 over the run window (variability test). 'raw_cap' applies "
            "clip(rv/ref, 0, 1) (literal environmental cap; needs recalibration)."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.gate.series.file",
        param_type=ParamType.FILE_PATH,
        default="",
        description="CSV of per-year spawning-season reproductive volume (year,spawning_rv).",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.gate.ref",
        param_type=ParamType.FLOAT,
        default=0.20,
        min_val=1e-9,
        max_val=1.0,
        description="Reference RV at which raw_cap saturates to 1.0 (~95th pctile).",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.gate.floor",
        param_type=ParamType.FLOAT,
        default=0.0,
        min_val=0.0,
        max_val=1.0,
        description="Optional lower bound on the gate factor (sensitivity knob).",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.gate.start.year",
        param_type=ParamType.INT,
        default=1993,
        min_val=0,
        max_val=3000,
        description="Real calendar year that model year 0 maps to for the RV series.",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.gate.species.enabled.sp{idx}",
        param_type=ParamType.BOOL,
        default=False,
        description="Per-species enable for the RV gate (cod only for Baltic).",
        category="reproduction",
        indexed=True,
        required=False,
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py::test_rv_gate_keys_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/schema/species.py tests/test_rv_recruitment_gate.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: schema fields for the RV recruitment gate"
```

---

## Task 2: Gate-series builder + generated data file

**Files:**
- Modify: `scripts/baltic_rv_overshoot_diagnostic.py` (add `build_rv_gate_series`, `--emit-gate-series`)
- Create: `data/baltic/forcing/baltic_rv_gate_series.csv` (generated)
- Test: `tests/test_rv_recruitment_gate.py`

**Interfaces:**
- Consumes: the `rv` dict returned by `reproductive_volume(...)` (keys `available`, `both_criteria`, `times`, `fraction`) and `annual_rv(times, fraction, months=SPAWNING_MONTHS)`.
- Produces: `build_rv_gate_series(rv: dict, out_path: Path) -> Path` writing `year,spawning_rv` rows; a `--emit-gate-series PATH` CLI flag.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rv_recruitment_gate.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import baltic_rv_overshoot_diagnostic as diag  # noqa: E402


def _rv_dict(years, vals, both=True):
    # 12 monthly steps/year; put the annual value in the Mar-Aug months.
    times, frac = [], []
    for y, v in zip(years, vals):
        for m in range(1, 13):
            times.append(np.datetime64(f"{y}-{m:02d}-01"))
            frac.append(v if m in diag.SPAWNING_MONTHS else 0.0)
    return {
        "available": True,
        "both_criteria": both,
        "times": np.array(times),
        "fraction": np.array(frac),
    }


def test_build_rv_gate_series_writes_rows(tmp_path):
    rv = _rv_dict([1993, 1994, 1995], [0.00, 0.07, 0.12])
    out = diag.build_rv_gate_series(rv, tmp_path / "series.csv")
    text = out.read_text().strip().splitlines()
    assert text[0] == "year,spawning_rv"
    assert text[1].startswith("1993,")
    assert len(text) == 4  # header + 3 years
    # spawning value round-trips (Mar-Aug mean == the injected value)
    assert abs(float(text[2].split(",")[1]) - 0.07) < 1e-6


def test_build_rv_gate_series_requires_both_criteria(tmp_path):
    rv = _rv_dict([1993, 1994], [0.0, 0.07], both=False)
    with pytest.raises(ValueError, match="both"):
        diag.build_rv_gate_series(rv, tmp_path / "series.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k build_rv_gate_series -v`
Expected: FAIL (`build_rv_gate_series` not defined).

- [ ] **Step 3: Implement the builder + CLI**

In `scripts/baltic_rv_overshoot_diagnostic.py`, add near the other helpers:

```python
def build_rv_gate_series(rv: dict, out_path: Path) -> Path:
    """Write per-year spawning-season RV (year,spawning_rv) for the engine gate.

    Requires the full salinity+oxygen RV (not the oxygen-only proxy) and a
    calendar time axis spanning >= 2 years. Raises rather than emitting a
    degenerate/optimistic file.
    """
    if not rv.get("available") or not rv.get("both_criteria"):
        raise ValueError("RV gate series requires both criteria (salinity + oxygen).")
    yrs, spawn = annual_rv(rv.get("times"), rv["fraction"], months=SPAWNING_MONTHS)
    if yrs is None:
        raise ValueError("RV series needs a calendar time axis spanning >= 2 years.")
    if np.any(~np.isfinite(spawn)):
        raise ValueError("RV series has NaN spawning-season year(s); cannot emit gate series.")
    lines = ["year,spawning_rv"] + ["%d,%.6f" % (int(y), v) for y, v in zip(yrs, spawn)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
```

Add the CLI flag in `main()` (in the argparse block):

```python
    ap.add_argument(
        "--emit-gate-series",
        type=Path,
        default=None,
        help="write the per-year RV gate series CSV for the engine and exit",
    )
```

And, in `main()` right after `rv = reproductive_volume(...)` is computed:

```python
    if args.emit_gate_series is not None:
        path = build_rv_gate_series(rv, args.emit_gate_series)
        log.info("wrote gate series %s", path)
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k build_rv_gate_series -v`
Expected: PASS.

- [ ] **Step 5: Generate the real data file**

Run (the 26 GB CMEMS reanalysis is already on disk):

```bash
cd /home/razinka/osmose/osmose-python && PYTHONPATH=. .venv/bin/python scripts/baltic_rv_overshoot_diagnostic.py --emit-gate-series data/baltic/forcing/baltic_rv_gate_series.csv
```

Expected: writes `data/baltic/forcing/baltic_rv_gate_series.csv` with a `year,spawning_rv` header and 29 rows (1993–2021). Verify the 2004/2016 rows are higher than the 2002/2012 rows (inflow pulses vs troughs).

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add scripts/baltic_rv_overshoot_diagnostic.py data/baltic/forcing/baltic_rv_gate_series.csv tests/test_rv_recruitment_gate.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: RV gate series builder and generated 1993-2021 data file"
```

---

## Task 3: EngineConfig fields + loader with fail-fast validation

**Files:**
- Modify: `osmose/engine/config.py` (add 3 dataclass fields after line 1333; add `_load_rv_gate`; call it in `from_dict`)
- Test: `tests/test_rv_recruitment_gate.py`

**Interfaces:**
- Consumes: the CSV format from Task 2; `EngineConfig.n_dt_per_year`, `n_year`, `n_species`.
- Produces: `EngineConfig.rv_gate_factor_by_index: NDArray[np.float64] | None` (shape `(n_years,)`, mode applied), `EngineConfig.rv_gate_enabled: NDArray[np.bool_] | None` (shape `(n_species,)`), `EngineConfig.rv_gate_offset: int`. And `_load_rv_gate(cfg, n_species, n_dt_per_year, n_year) -> tuple[NDArray|None, NDArray|None, int]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rv_recruitment_gate.py`:

```python
from osmose.engine.config import _load_rv_gate


def _write_series(tmp_path, years, vals, name="s.csv"):
    # `name` MUST be unique per file so tests that need two different series in
    # the same tmp_path do not overwrite each other (a good series written by
    # _cfg would otherwise clobber a bad series written for a validation test).
    p = tmp_path / name
    rows = ["year,spawning_rv"] + ["%d,%.6f" % (y, v) for y, v in zip(years, vals)]
    p.write_text("\n".join(rows) + "\n")
    return p


def _cfg(tmp_path, **over):
    series = _write_series(
        tmp_path, range(1993, 1998), [0.0, 0.10, 0.20, 0.05, 0.15], name="good.csv"
    )
    base = {
        "reproduction.rv.gate.enabled": "true",
        "reproduction.rv.gate.mode": "mean_preserving",
        "reproduction.rv.gate.series.file": str(series),
        "reproduction.rv.gate.start.year": "1993",
        "reproduction.rv.gate.species.enabled.sp0": "true",
        "_osmose.config.dir": str(tmp_path),
    }
    base.update(over)
    return base


def test_load_rv_gate_disabled_returns_none():
    fac, mask, off = _load_rv_gate({"reproduction.rv.gate.enabled": "false"}, 3, 24, 5)
    assert fac is None and mask is None and off == 0


def test_load_rv_gate_mean_preserving_full_window(tmp_path):
    # 5-year run over the full 5-year series -> window == all rows -> mean(fac) == 1.
    fac, mask, off = _load_rv_gate(_cfg(tmp_path), n_species=1, n_dt_per_year=24, n_year=5)
    assert off == 0
    assert mask.tolist() == [True]
    assert abs(float(np.mean(fac)) - 1.0) < 1e-9


def test_load_rv_gate_mean_preserving_window_subset(tmp_path):
    # 3-year run over a 5-row series with asymmetric values: the denominator MUST
    # be the mean over the SAMPLED window (rows 0..2), not the whole array. This
    # is the test that actually proves the windowing (the full-window test above
    # would pass even for a whole-array denominator).
    series = _write_series(
        tmp_path, range(1993, 1998), [0.10, 0.20, 0.30, 0.40, 0.50], name="sub.csv"
    )
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.gate.series.file"] = str(series)
    fac, _, _ = _load_rv_gate(cfg, 1, 24, 3)  # window = rows 0,1,2 -> D = mean(.1,.2,.3) = 0.20
    assert fac.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])
    assert abs(float(np.mean(fac[[0, 1, 2]])) - 1.0) < 1e-9  # window mean == 1
    assert float(np.mean(fac)) == pytest.approx(1.5)  # whole-array mean != 1 -> scoping proven


def test_load_rv_gate_offset_indexes_window(tmp_path):
    # start_year 1995 -> offset 2; 3-year window = rows 2,3,4 -> D = mean(.3,.4,.5) = 0.40.
    series = _write_series(
        tmp_path, range(1993, 1998), [0.10, 0.20, 0.30, 0.40, 0.50], name="off.csv"
    )
    cfg = _cfg(tmp_path, **{"reproduction.rv.gate.start.year": "1995"})
    cfg["reproduction.rv.gate.series.file"] = str(series)
    fac, _, off = _load_rv_gate(cfg, 1, 24, 3)
    assert off == 2
    assert fac[2] == pytest.approx(0.75)  # 0.30 / 0.40
    assert fac[4] == pytest.approx(1.25)  # 0.50 / 0.40


def test_load_rv_gate_raw_cap_clips(tmp_path):
    cfg = _cfg(tmp_path, **{"reproduction.rv.gate.mode": "raw_cap", "reproduction.rv.gate.ref": "0.10"})
    fac, _, _ = _load_rv_gate(cfg, 1, 24, 5)
    assert fac.min() >= 0.0 and fac.max() <= 1.0
    assert fac[0] == 0.0  # rv=0.0 -> 0
    assert fac[1] == pytest.approx(1.0)  # rv=0.10 == ref -> 1


@pytest.mark.parametrize("bad,exc", [
    ({"reproduction.rv.gate.mode": "nope"}, "mode"),
    ({"reproduction.rv.gate.species.enabled.sp0": "false"}, "no species"),
    ({"reproduction.rv.gate.mode": "raw_cap", "reproduction.rv.gate.ref": "0"}, "ref"),
    ({"reproduction.rv.gate.floor": "2.0"}, "floor"),
])
def test_load_rv_gate_fail_fast_config(tmp_path, bad, exc):
    with pytest.raises(ValueError, match=exc):
        _load_rv_gate(_cfg(tmp_path, **bad), 1, 24, 5)


def test_load_rv_gate_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        _load_rv_gate(_cfg(tmp_path, **{"reproduction.rv.gate.series.file": ""}), 1, 24, 5)


def test_load_rv_gate_nan_rv_raises(tmp_path):
    p = tmp_path / "nan.csv"
    p.write_text("year,spawning_rv\n1993,0.1\n1994,nan\n1995,0.2\n")
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.gate.series.file"] = str(p)
    with pytest.raises(ValueError, match="NaN|negative"):
        _load_rv_gate(cfg, 1, 24, 3)


def test_load_rv_gate_zero_denominator_raises(tmp_path):
    # all-zero window under mean_preserving -> D == 0.
    series = _write_series(tmp_path, range(1993, 1996), [0.0, 0.0, 0.0], name="zero.csv")
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.gate.series.file"] = str(series)
    with pytest.raises(ValueError, match="denominator"):
        _load_rv_gate(cfg, 1, 24, 3)


def test_load_rv_gate_nonascending_years_raises(tmp_path):
    cfg = _cfg(tmp_path)  # writes good.csv and points series.file at it
    bad = _write_series(tmp_path, [1993, 1995, 1994], [0.1, 0.1, 0.1], name="bad.csv")
    cfg["reproduction.rv.gate.series.file"] = str(bad)  # now point at the bad file
    with pytest.raises(ValueError, match="contiguous"):
        _load_rv_gate(cfg, 1, 24, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k load_rv_gate -v`
Expected: FAIL (`_load_rv_gate` not defined).

- [ ] **Step 3: Add the dataclass fields**

In `osmose/engine/config.py`, in the `EngineConfig` dataclass right after line 1333 (`spawning_season: ...`), add:

```python
    # Reproductive-volume recruitment gate (all None when disabled)
    rv_gate_factor_by_index: NDArray[np.float64] | None  # (n_years,), mode already applied
    rv_gate_enabled: NDArray[np.bool_] | None  # (n_species,) per-species enable mask
    rv_gate_offset: int  # start_year - first_year (see _load_rv_gate)
```

- [ ] **Step 4: Implement the loader**

Add near `_load_spawning_seasons` in `osmose/engine/config.py`:

```python
def _load_rv_gate(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_year: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None, int]:
    """Load the reproductive-volume recruitment gate (spec §3.2/§4/§8).

    Returns (factor_by_index, enabled_mask, offset). factor_by_index has length
    n_years (number of series rows), is indexed by series index, and has the
    mode formula already applied. All three are (None, None, 0) when the master
    switch is off. Raises a clear error on any invalid configuration (fail-fast):
    ValueError for bad content/values, FileNotFoundError for a missing file.
    """
    if cfg.get("reproduction.rv.gate.enabled", "false").lower() != "true":
        return None, None, 0

    file_key = cfg.get("reproduction.rv.gate.series.file", "")
    if not file_key:
        raise ValueError("RV gate enabled but reproduction.rv.gate.series.file is empty.")
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.rv.gate.series.file")
    df = pd.read_csv(path)
    if df.shape[0] == 0 or "year" not in df.columns or "spawning_rv" not in df.columns:
        raise ValueError(f"RV gate series {path} has no data rows or wrong columns.")
    years = df["year"].to_numpy()
    rv = df["spawning_rv"].to_numpy(dtype=np.float64)
    first_year = int(years[0])
    if not np.array_equal(years, np.arange(first_year, first_year + len(years))):
        raise ValueError(f"RV gate series {path} years must be contiguous and ascending.")
    if np.any(~np.isfinite(rv)) or np.any(rv < 0):
        raise ValueError(f"RV gate series {path} has NaN or negative spawning_rv.")

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"reproduction.rv.gate.species.enabled.sp{sp}", "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError("RV gate enabled but no species enabled (…species.enabled.sp{idx}).")

    mode = cfg.get("reproduction.rv.gate.mode", "mean_preserving")
    floor = float(cfg.get("reproduction.rv.gate.floor", "0.0"))
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"reproduction.rv.gate.floor must be in [0,1], got {floor}.")
    start_year = int(cfg.get("reproduction.rv.gate.start.year", str(first_year)))
    n_years = len(rv)
    offset = start_year - first_year

    if mode == "mean_preserving":
        # Multiset mean over the sampled model years y=0..n_year-1 (with repeats).
        window_idx = [(offset + y) % n_years for y in range(n_year)]
        denom = float(np.mean(rv[window_idx]))
        if denom == 0.0:
            raise ValueError("RV gate mean_preserving denominator is 0 over the run window.")
        factor = rv / denom
    elif mode == "raw_cap":
        ref = float(cfg.get("reproduction.rv.gate.ref", "0.20"))
        if ref <= 0.0:
            raise ValueError(f"reproduction.rv.gate.ref must be > 0, got {ref}.")
        factor = np.clip(rv / ref, 0.0, 1.0)
    else:
        raise ValueError(f"unknown reproduction.rv.gate.mode: {mode!r}")

    factor = np.maximum(factor, floor)
    return factor.astype(np.float64), enabled, offset
```

- [ ] **Step 5: Call the loader in `from_dict`**

In `EngineConfig.from_dict`, before the `return EngineConfig(` construction (n_sp, n_dt, n_yr are already defined at lines ~1629-1630), add:

```python
        rv_gate_factor_by_index, rv_gate_enabled, rv_gate_offset = _load_rv_gate(
            cfg, n_sp, n_dt, n_yr
        )
```

Then, in the `EngineConfig(...)` keyword arguments (adjacent to `spawning_season=_load_spawning_seasons(cfg, n_sp, n_dt),` at line 2116), add:

```python
            rv_gate_factor_by_index=rv_gate_factor_by_index,
            rv_gate_enabled=rv_gate_enabled,
            rv_gate_offset=rv_gate_offset,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k load_rv_gate -v`
Expected: PASS (all load_rv_gate tests).

- [ ] **Step 7: Guard the config-validation test**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS with no new warnings (default configs don't set the keys; the loader's `cfg.get(...)` calls are AST-captured because they live in `config.py`). If it warns about unknown keys, the schema fields from Task 1 cover them — confirm both landed.

- [ ] **Step 8: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/engine/config.py tests/test_rv_recruitment_gate.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: EngineConfig RV gate loader with fail-fast validation"
```

---

## Task 4: Pure factor helper

**Files:**
- Create: `osmose/engine/processes/recruitment_gate.py`
- Test: `tests/test_rv_recruitment_gate.py`

**Interfaces:**
- Consumes: `EngineConfig.rv_gate_factor_by_index`, `rv_gate_enabled`, `rv_gate_offset`, `n_dt_per_year`, `n_species`.
- Produces: `rv_gate_factor(config, step) -> NDArray[np.float64]` of shape `(n_species,)`, value `1.0` for disabled species, the mode factor for the current model year otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rv_recruitment_gate.py`:

```python
from types import SimpleNamespace

from osmose.engine.processes.recruitment_gate import rv_gate_factor


def _fake_cfg(factor, enabled, offset=0, n_dt=24):
    return SimpleNamespace(
        rv_gate_factor_by_index=factor,
        rv_gate_enabled=enabled,
        rv_gate_offset=offset,
        n_dt_per_year=n_dt,
        n_species=len(enabled) if enabled is not None else 1,
    )


def test_rv_gate_factor_disabled_all_ones():
    cfg = _fake_cfg(None, None)
    cfg.n_species = 3
    assert rv_gate_factor(cfg, 100).tolist() == [1.0, 1.0, 1.0]


def test_rv_gate_factor_selects_year_and_species():
    factor = np.array([0.5, 2.0, 1.0])  # 3-year series
    enabled = np.array([True, False])
    cfg = _fake_cfg(factor, enabled, offset=0, n_dt=24)
    # model year 0 -> idx 0 -> 0.5 for cod, 1.0 for the disabled species
    assert rv_gate_factor(cfg, 0).tolist() == [0.5, 1.0]
    # model year 1 (step 24..47) -> idx 1 -> 2.0
    assert rv_gate_factor(cfg, 30).tolist() == [2.0, 1.0]


def test_rv_gate_factor_wraps_and_offsets():
    factor = np.array([0.5, 2.0, 1.0])
    enabled = np.array([True])
    cfg = _fake_cfg(factor, enabled, offset=2, n_dt=24)
    # model year 0 -> idx (2+0)%3 = 2 -> 1.0
    assert rv_gate_factor(cfg, 0).tolist() == [1.0]
    # model year 4 -> idx (2+4)%3 = 0 -> 0.5 (wrap)
    assert rv_gate_factor(cfg, 4 * 24).tolist() == [0.5]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k rv_gate_factor -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the helper**

Create `osmose/engine/processes/recruitment_gate.py`:

```python
"""Reproductive-volume recruitment gate — pure per-step factor helper.

Engine-state-free: reads only precomputed EngineConfig fields (see
osmose/engine/config.py:_load_rv_gate). Returns a per-species egg multiplier,
constant within a model year, 1.0 for species with the gate disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig


def rv_gate_factor(config: "EngineConfig", step: int) -> NDArray[np.float64]:
    """Per-species egg-production multiplier for this timestep.

    1.0 for every species when the gate is off or the species is disabled;
    otherwise the mode factor for the current model year's series index.
    """
    out = np.ones(config.n_species, dtype=np.float64)
    factor = config.rv_gate_factor_by_index
    if factor is None:
        return out
    n_years = factor.shape[0]
    year = step // config.n_dt_per_year
    idx = (config.rv_gate_offset + year) % n_years
    out[config.rv_gate_enabled] = factor[idx]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k rv_gate_factor -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/engine/processes/recruitment_gate.py tests/test_rv_recruitment_gate.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: pure rv_gate_factor helper"
```

---

## Task 5: Apply the gate in reproduction + parity regression

**Files:**
- Modify: `osmose/engine/processes/reproduction.py` (seeding loop lines 122-126; after `n_eggs` at line 152)
- Test: `tests/test_rv_recruitment_gate.py`

**Interfaces:**
- Consumes: `rv_gate_factor(config, step)` (Task 4); the `seeded_this_step` bool computed in the seeding loop.
- Produces: gated `n_eggs` for enabled, non-seeded species.

- [ ] **Step 1: Write the failing integration + parity tests**

Append to `tests/test_rv_recruitment_gate.py`:

```python
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

BALTIC = Path("/home/razinka/osmose/osmose-python/data/baltic/baltic_all-parameters.csv")
SERIES = Path("/home/razinka/osmose/osmose-python/data/baltic/forcing/baltic_rv_gate_series.csv")


def _baltic_cfg(**over):
    cfg = dict(OsmoseConfigReader().read(BALTIC))
    cfg["simulation.time.nyear"] = "6"
    cfg.update(over)
    return cfg


def test_gate_off_bit_identical():
    base = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()
    gated_off = PythonEngine().run_in_memory(
        _baltic_cfg(**{"reproduction.rv.gate.enabled": "false"}), seed=0
    ).biomass()
    np.testing.assert_array_equal(base["cod"].to_numpy(), gated_off["cod"].to_numpy())


def test_gate_on_changes_cod_and_cod_dominates():
    off = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()
    on = PythonEngine().run_in_memory(_baltic_cfg(**{
        "reproduction.rv.gate.enabled": "true",
        "reproduction.rv.gate.mode": "raw_cap",
        "reproduction.rv.gate.ref": "0.20",
        "reproduction.rv.gate.series.file": str(SERIES),
        "reproduction.rv.gate.start.year": "1993",
        "reproduction.rv.gate.species.enabled.sp0": "true",
    }), seed=0).biomass()

    def rel_change(sp):
        a, b = off[sp].to_numpy(), on[sp].to_numpy()
        denom = float(np.abs(a).sum())
        return float(np.abs(b - a).sum()) / denom if denom else 0.0

    # Cod (the only gated species) changes; and its relative change dominates a
    # coupled species (sprat), whose change is only a secondary predation/RNG
    # effect. We do NOT assert sprat is bit-identical — cod preys on sprat and
    # cod's changed survival desyncs the shared RNG stream, so sprat legitimately
    # shifts. The per-species enable-mask restriction to sp0 is proven directly
    # by the helper unit tests (Task 4).
    assert rel_change("cod") > 0.05  # gate meaningfully changes cod
    assert rel_change("cod") > rel_change("sprat")  # cod is the primary effect
```

Note: keep `test_gate_off_bit_identical` strict — that is the parity guarantee. The "gate applies only to enabled species" property is proven at the helper unit level (Task 4: `out[config.rv_gate_enabled] = factor`, `1.0` elsewhere); the full-run test asserts the weaker, robust cod-dominant behavior.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k "gate_off or gate_on" -v`
Expected: `test_gate_on_changes_cod_and_cod_dominates` FAILS (gate not applied → cod unchanged). `test_gate_off_bit_identical` should already PASS.

- [ ] **Step 3: Track seeded species in the seeding loop**

In `osmose/engine/processes/reproduction.py`, replace the seeding loop (lines 122-126):

```python
    # Seeding: if SSB is zero and within seeding period, use seeding biomass
    seeded_this_step = np.zeros(n_sp, dtype=np.bool_)
    for sp in range(n_sp):
        if ssb[sp] == 0.0:
            if step < config.seeding_max_step[sp]:
                ssb[sp] = config.seeding_biomass[sp]
                seeded_this_step[sp] = True
```

- [ ] **Step 4: Apply the gate after stock-recruitment**

In `osmose/engine/processes/reproduction.py`, immediately after the `n_eggs = apply_stock_recruitment(...)` block (ends line 152) and before the "Create new schools from eggs" loop (line 154), add:

```python
    # Reproductive-volume recruitment gate (Baltic cod). Inert unless enabled;
    # skipped on steps where SSB was seeded (bootstrap must not be gated).
    if config.rv_gate_factor_by_index is not None:
        from osmose.engine.processes.recruitment_gate import rv_gate_factor

        gate = rv_gate_factor(config, step)
        for sp in range(n_sp):
            if config.rv_gate_enabled[sp] and not seeded_this_step[sp]:
                n_eggs[sp] *= gate[sp]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k "gate_off or gate_on" -v`
Expected: PASS (cod changes with gate on; gate-off bit-identical).

- [ ] **Step 6: Run the parity/regression suite**

Run: `.venv/bin/python -m pytest -k "parity or cross_engine" -q`
Also run the migration-check skill if available. Expected: PASS — the gate is inert by default, so EEC/BoB/Baltic parity is unchanged. If any parity test regresses, the default-off path is leaking; re-check Step 4's `is not None` guard.

- [ ] **Step 7: Lint, format, types**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ scripts/ tests/ && .venv/bin/ruff format --check osmose/ scripts/ tests/ && .venv/bin/pyright osmose/engine/processes/recruitment_gate.py osmose/engine/processes/reproduction.py osmose/engine/config.py
```
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/engine/processes/reproduction.py tests/test_rv_recruitment_gate.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: apply RV recruitment gate to cod egg production"
```

---

## Task 6: Effect measurement (gate on vs off)

**Files:**
- Modify: `scripts/baltic_rv_overshoot_diagnostic.py` (add a `window` arg to `characterise_instability`)
- Create: (optional) a short comparison entry the diagnostic prints

**Interfaces:**
- Consumes: `OsmoseResults.biomass("cod")`, `characterise_instability(t, b, window=...)`.
- Produces: a printed/measured boom-bust comparison over model years 3–14.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rv_recruitment_gate.py`:

```python
def test_characterise_instability_window():
    t = np.arange(15.0)
    b = np.array([1, 2, 3, 100, 50, 40, 30, 25, 22, 20, 18, 16, 15, 14, 13], dtype=float)
    full = diag.characterise_instability(t, b)
    win = diag.characterise_instability(t, b, window=(3, 14))
    # windowed max/min excludes the tiny spin-up values (1,2,3)
    assert win["boom_bust_ratio"] < full["boom_bust_ratio"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k characterise_instability_window -v`
Expected: FAIL (`window` kwarg not accepted).

- [ ] **Step 3: Add the window parameter**

In `scripts/baltic_rv_overshoot_diagnostic.py`, replace `characterise_instability` (lines 293-313) with the version below. Only the signature's `window` param, the `t = np.asarray(...)` line, and the `if window is not None:` slice are new; the stats body is byte-for-byte the existing one.

```python
def characterise_instability(t, b, window: tuple[int, int] | None = None) -> dict:
    """Summary stats describing how unstable the trajectory is.

    window=(lo, hi) restricts to model years [lo, hi] (inclusive) before the
    stats, so the spin-up transient can be excluded.
    """
    t = np.asarray(t, dtype=float)
    b = np.asarray(b, dtype=float)
    if window is not None:
        sel = (t >= window[0]) & (t <= window[1])
        b = b[sel]
    finite = b[np.isfinite(b) & (b > 0)]
    if finite.size == 0:
        return {"empty": True}
    mean = float(finite.mean())
    cv = float(finite.std() / mean) if mean else float("nan")
    boom_bust = float(finite.max() / finite.min()) if finite.min() > 0 else float("inf")
    # trend over the last third of the run (is it still moving?)
    tail = b[max(0, len(b) - max(3, len(b) // 3)) :]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size >= 2 else float("nan")
    return {
        "empty": False,
        "mean": mean,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "cv": cv,
        "boom_bust_ratio": boom_bust,
        "tail_slope_per_step": slope,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -k characterise_instability_window -v`
Expected: PASS.

- [ ] **Step 5: Measure the real effect**

Run a gate-on vs gate-off Baltic comparison and record the numbers (this is a measurement, not an assertion):

```bash
cd /home/razinka/osmose/osmose-python && PYTHONPATH=. .venv/bin/python - <<'PY'
import numpy as np
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

base = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
base["simulation.time.nyear"] = "15"
gate = dict(base, **{
    "reproduction.rv.gate.enabled": "true",
    "reproduction.rv.gate.mode": "mean_preserving",
    "reproduction.rv.gate.series.file": "data/baltic/forcing/baltic_rv_gate_series.csv",
    "reproduction.rv.gate.start.year": "1993",
    "reproduction.rv.gate.species.enabled.sp0": "true",
})
def bb(cfg):
    b = PythonEngine().run_in_memory(cfg, seed=0).biomass()["cod"].to_numpy()
    w = b[3:15]; w = w[np.isfinite(w) & (w > 0)]
    return w.max()/w.min(), w.mean()
off_bb, off_mean = bb(base); on_bb, on_mean = bb(gate)
print(f"boom/bust off={off_bb:.1f} on={on_bb:.1f} reduction={100*(1-on_bb/off_bb):.0f}%")
print(f"mean cod biomass off={off_mean:.0f} on={on_mean:.0f} delta={100*(on_mean/off_mean-1):+.0f}%")
PY
```

Expected (success criteria §10.2): boom/bust reduced by ≥25%; mean cod biomass within ±10%. Record the actual numbers in `docs/diagnostics/`. If the mean drifts >10%, that is a finding to report (the closed-loop mean shift of spec §3.2), not an automatic failure — note it and, if desired, follow up by fitting the normalisation constant `k`.

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add scripts/baltic_rv_overshoot_diagnostic.py tests/test_rv_recruitment_gate.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: windowed instability metric + RV gate effect measurement"
```

---

## Final verification

- [ ] Full test file green: `.venv/bin/python -m pytest tests/test_rv_recruitment_gate.py -v`
- [ ] Lint/format/types clean (Task 5 Step 7 command over all changed files).
- [ ] Parity suite green (gate off ⇒ bit-identical).
- [ ] The effect measurement recorded; boom/bust reduction and mean drift documented.
