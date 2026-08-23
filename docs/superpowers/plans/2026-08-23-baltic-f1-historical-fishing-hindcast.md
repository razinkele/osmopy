# Baltic F1 Historical-Fishing Hindcast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Force the certified 9-species Baltic config with observed ICES fishing-mortality
history (1993–2023) and measure hindcast skill against a constant-F baseline, per the
pre-registered criteria.

**Architecture:** Three layers, no engine features: (1) small load-path hardening in
`osmose/engine/config.py` (case fix + fail-fast) plus one schema field; (2) an offline
derivation script that turns cached ICES snapshots into per-species by-year F CSVs; (3) a
2-arm hindcast harness modeled line-for-line on `scripts/baltic_rv_hindcast.py`
(`osmose_demo` → reader dict → arm overrides → `PythonEngine().run_in_memory(raw, seed)`).

**Tech Stack:** Python 3.12, NumPy, pandas, pytest. Always `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-23-baltic-f1-historical-fishing-hindcast-design.md`
(read it first — decisions 1–7 are binding; §4's margins are pre-registered and NOT tunable).

## Global Constraints

- Run everything with `.venv/bin/python` (system python may not exist).
- Production config: every file currently in `data/baltic/` stays byte-identical. New files may
  be ADDED under `data/baltic/reference/` only.
- OSMOSE config keys are lowercase dot-separated; `OsmoseConfigReader` lowercases every key at
  read time (`osmose/config/reader.py:173`) — tests for config lookups must go through the
  reader or use lowercase dicts.
- The hindcast run itself is NOT a CI gate (emergent, seed/machine-sensitive). CI-safe unit
  tests only in `tests/`.
- Engine runs: check `uptime` first; NEVER run two engine jobs concurrently (user may have a
  large ESTAS_II job; load must be low before Tasks 7–8).
- Ruff line length 100: `.venv/bin/ruff check osmose/ ui/ tests/` must stay clean.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Case fix — by-year/by-dt fishing keys resolvable from real configs

The reader lowercases all keys, but `config.py` looks up camelCase in three places, so
`byDt.byAge`, `byDt.bySize`, and `catches.byYear` can never match a reader-produced config.
Fix the lookups, the `_FISHING_SCENARIOS` table, its four dispatch tests, and the
validation allowlist.

**Files:**
- Modify: `osmose/engine/config.py:1569-1578` (`_FISHING_SCENARIOS`), `:2296` (byDt variants),
  `:2319` (catches.byYear)
- Modify: `osmose/engine/config_validation.py` (`_ALLOWLIST_PY_HONORED`, lines ~47-111)
- Modify: `tests/test_engine_fishing_variants.py:344-392` (dispatch tests)
- Test: `tests/test_engine_fishing_variants.py` (new reader-path regression test)

**Interfaces:**
- Produces: `detect_fishing_scenario(config, idx)` recognizing lowercase keys;
  `EngineConfig.from_dict` loading `fishing_catches_by_year` / `fishing_rate_by_dt_by_class`
  from lowercase keys. Task 2 builds directly on the (already-lowercase)
  `mortality.fishing.rate.byyear.file.sp{i}` path.

- [ ] **Step 1: Write the failing reader-path regression test** (append to the detection test
  class in `tests/test_engine_fishing_variants.py`):

```python
def test_camelcase_keys_survive_the_reader(self, tmp_path) -> None:
    """Keys written camelCase in a config FILE are lowercased by the reader and must
    still be detected — regression for the dead camelCase lookups (B1 audit)."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine.config import detect_fishing_scenario

    cfg_file = tmp_path / "cfg.csv"
    cfg_file.write_text(
        "mortality.fishing.rate.byYear.file.sp0;f0.csv\n"
        "mortality.fishing.catches.byYear.file.sp1;c1.csv\n"
        "mortality.fishing.rate.byDt.byAge.file.sp2;a2.csv\n"
    )
    cfg = dict(OsmoseConfigReader().read(str(cfg_file)))
    assert detect_fishing_scenario(cfg, 0) == "rate_by_year"
    assert detect_fishing_scenario(cfg, 1) == "catches_by_year"
    assert detect_fishing_scenario(cfg, 2) == "rate_by_dt_by_class"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py -k camelcase -v`
Expected: FAIL — all three asserts return `None` (camelCase prefixes never match lowercased keys).

- [ ] **Step 3: Lowercase the `_FISHING_SCENARIOS` table** in `osmose/engine/config.py:1569-1578`:

```python
# NOTE: OsmoseConfigReader lowercases every key at read time (osmose/config/reader.py),
# so these prefixes are lowercase. Scenario names still mirror Java's
# FishingMortality.Scenario enum.
_FISHING_SCENARIOS = [
    ("rate_annual", "mortality.fishing.rate.sp"),
    ("rate_by_year", "mortality.fishing.rate.byyear.file.sp"),
    ("rate_by_dt_by_class", "mortality.fishing.rate.bydt.byage.file.sp"),
    ("rate_by_dt_by_class", "mortality.fishing.rate.bydt.bysize.file.sp"),
    ("catches_annual", "mortality.fishing.catches.sp"),
    ("catches_by_year", "mortality.fishing.catches.byyear.file.sp"),
    ("catches_by_dt_by_class", "mortality.fishing.catches.bydt.byage.file.sp"),
    ("catches_by_dt_by_class", "mortality.fishing.catches.bydt.bysize.file.sp"),
]
```

- [ ] **Step 4: Lowercase the two loader lookups.** At `config.py:2296` change
  `for variant in ["byDt.byAge", "byDt.bySize"]:` to
  `for variant in ["bydt.byage", "bydt.bysize"]:` (the key is built as
  `f"mortality.fishing.rate.{variant}.file.sp{i}"`). At `config.py:2319` change
  `year_key = f"mortality.fishing.catches.byYear.file.sp{i}"` to
  `year_key = f"mortality.fishing.catches.byyear.file.sp{i}"`.

- [ ] **Step 5: Update the four camelCase dispatch tests** in
  `tests/test_engine_fishing_variants.py:351-392` — change the dict keys only, assertions stay:
  `mortality.fishing.rate.byYear.file.sp0` → `mortality.fishing.rate.byyear.file.sp0`;
  `mortality.fishing.rate.byDt.byAge.file.sp0` → `mortality.fishing.rate.bydt.byage.file.sp0`;
  `mortality.fishing.catches.byYear.file.sp0` → `mortality.fishing.catches.byyear.file.sp0`;
  `mortality.fishing.catches.byDt.byAge.file.sp0` → `mortality.fishing.catches.bydt.byage.file.sp0`.
  Then grep for remaining producers of camelCase FISHING keys only
  (`grep -rn "mortality.fishing.*byYear\|mortality.fishing.*byDt" tests/`) — any test that
  hand-builds these keys and feeds `EngineConfig.from_dict` must be lowercased too (they bypass
  the reader, which is the blind spot this task closes). Two files the broader grep would
  surface must stay camelCase — do NOT touch them: `tests/test_engine_timeseries.py` feeds
  `load_timeseries` (its own camelCase lookups at `osmose/engine/timeseries.py:421-434` are out
  of scope here), and `tests/test_engine_additional_mortality.py` exercises the larva
  byDt detection (`config.py:1526-1542`), also intentionally untouched.

- [ ] **Step 6: Add the AST-invisible keys to the validation allowlist AND its frozen-snapshot
  guard.** Two guard tests police the allowlist and both must be updated in this same commit:
  `tests/test_schema_engine_key_parity.py` (accept-set = AST-walked keys ∪
  `_SUPPLEMENTARY_ALLOWLIST` — NOT the schema registry) and
  `tests/test_issue_123_known_but_unread_keys.py` (asserts the allowlists equal a deliberately
  independent `FROZEN_ALLOWLIST_SNAPSHOT` literal, ~149 keys).

  In `osmose/engine/config_validation.py`, `_ALLOWLIST_PY_HONORED` (alphabetical, lines
  ~47-111), add these FIVE keys:

```python
        "mortality.fishing.catches.bydt.byage.file.sp{idx}",
        "mortality.fishing.catches.bydt.bysize.file.sp{idx}",
        "mortality.fishing.rate.bydt.byage.file.sp{idx}",
        "mortality.fishing.rate.bydt.bysize.file.sp{idx}",
        "mortality.fishing.rate.byyear.file.sp{idx}",
```

  (sorted into place). Rationale per key: the bydt keys are `{variant}`-f-string-built
  (`config.py:2296`) and the walker cannot capture them; `rate.byyear` is read via a
  caller-arg `key_pattern` inside `_load_per_species_timeseries` (`config.py:451`) — also
  walker-invisible, and Task 3's schema field for it would otherwise fail the parity gate.
  Do NOT add `mortality.fishing.catches.byyear.file.sp{idx}` — after Step 4 lowercases
  `config.py:2319` its literal f-string IS captured by the walker automatically.

  Then `grep -n "byYear\|byDt" osmose/engine/config_validation.py` — delete any camelCase
  variants found (they can never match a canonicalized key).

  Finally, mirror EVERY addition and deletion into `FROZEN_ALLOWLIST_SNAPSHOT` in
  `tests/test_issue_123_known_but_unread_keys.py` (same five keys added, same camelCase
  removals) and update its key-count comment to match the new length.

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py tests/test_engine_config_validation.py tests/test_schema_engine_key_parity.py tests/test_issue_123_known_but_unread_keys.py -v`
Expected: PASS, including `test_camelcase_keys_survive_the_reader`, both allowlist guard files
(parity + frozen snapshot), and the warn-mode cleanliness test
`test_from_dict_warn_mode_clean_on_example_configs` (must stay warning-free).

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff check osmose/ tests/
git add osmose/engine/config.py osmose/engine/config_validation.py tests/test_engine_fishing_variants.py tests/test_issue_123_known_but_unread_keys.py
git commit -m "fix(engine): by-year/by-dt fishing keys now resolve from reader-produced configs

The reader lowercases every key; three lookups in config.py were camelCase and
could never match a real config file (B1 audit finding). byDt remains
reference-path-only and warned unsupported — this removes dead lookups, it does
not ship byDt support.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Fail-fast on short by-year F series

Today a series shorter than the run silently falls back to base F mid-run
(`fishing.py:44`, `mortality.py:750`). Make `EngineConfig.from_dict` raise instead.

**Files:**
- Modify: `osmose/engine/config.py` (immediately after `:2285`,
  `fishing_rate_by_year = _load_fishing_rate_by_year(cfg, n_sp)`)
- Test: `tests/test_engine_fishing_v2.py` (the by-year fixture lives at `:104-114`)

**Interfaces:**
- Consumes: `_load_fishing_rate_by_year` (Task-1-touched file, unchanged function).
- Produces: `ValueError` at config load when `len(series) < nyear`. The Task 6 harness relies
  on this guard (50-row files vs `nyear=50`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_engine_fishing_v2.py`, next to
  the existing by-year tests):

```python
class TestByYearShortSeriesGuard:
    """A by-year F series shorter than the run must fail at load, not silently
    fall back to base F mid-run (B1 audit / F1 spec §2b)."""

    def _cfg_with_series(self, tmp_path, n_values: int, nyear: int) -> dict:
        f_csv = tmp_path / "f_byyear_sp0.csv"
        f_csv.write_text("\n".join(["0.2"] * n_values) + "\n")
        cfg = _base_config(n_sp=1, n_dt=4)  # the module's existing minimal-config helper
        cfg["_osmose.config.dir"] = str(tmp_path)
        cfg["simulation.time.nyear"] = str(nyear)
        cfg["mortality.fishing.rate.byyear.file.sp0"] = str(f_csv)
        return cfg

    def test_short_series_raises_at_load(self, tmp_path) -> None:
        import pytest
        from osmose.engine.config import EngineConfig

        cfg = self._cfg_with_series(tmp_path, n_values=3, nyear=5)
        with pytest.raises(ValueError, match=r"byyear.*sp0.*3.*5"):
            EngineConfig.from_dict(cfg)

    def test_exact_and_longer_series_pass(self, tmp_path) -> None:
        from osmose.engine.config import EngineConfig

        for n in (5, 8):
            cfg = self._cfg_with_series(tmp_path, n_values=n, nyear=5)
            ec = EngineConfig.from_dict(cfg)
            assert ec.fishing_rate_by_year[0] is not None
            assert len(ec.fishing_rate_by_year[0]) == n
```

  (`_base_config(n_sp=1, n_dt=4)` is the module's verified existing helper; keep its import
  path as the other tests in that file use it.)

- [ ] **Step 2: Run to verify the guard test fails**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_v2.py -k ShortSeries -v`
Expected: `test_short_series_raises_at_load` FAILS (no exception today);
`test_exact_and_longer_series_pass` may already pass.

- [ ] **Step 3: Implement the guard** in `osmose/engine/config.py`, directly after the
  `fishing_rate_by_year = _load_fishing_rate_by_year(cfg, n_sp)` line (`:2285`):

```python
        # F1 spec §2b: a by-year series shorter than the run would silently fall
        # back to the base rate mid-run (fishing.py / mortality.py guard on
        # `year < len(arr)`). Fail at load instead. Longer-than-run is fine
        # (extra years ignored). Intentionally stricter than Java's fallback.
        if fishing_rate_by_year is not None:
            _byyear_nyear = int(cfg.get("simulation.time.nyear", "1"))
            for _sp_i, _byyear_arr in enumerate(fishing_rate_by_year):
                if _byyear_arr is not None and len(_byyear_arr) < _byyear_nyear:
                    raise ValueError(
                        f"mortality.fishing.rate.byyear.file.sp{_sp_i}: series has "
                        f"{len(_byyear_arr)} rows but simulation.time.nyear="
                        f"{_byyear_nyear}; past the series end the engine would "
                        "silently revert to the base rate. Provide >= nyear rows."
                    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_v2.py -v`
Expected: PASS (all, including pre-existing by-year tests — none ships a short series; audited).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check osmose/ tests/
git add osmose/engine/config.py tests/test_engine_fishing_v2.py
git commit -m "feat(engine): fail fast when a by-year F series is shorter than the run

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Schema field for the by-year F key

**Files:**
- Modify: `osmose/schema/species.py` (the Fishing block, after
  `mortality.fishing.rate.sp{idx}` at `:396`)
- Test: `tests/test_schema.py` (or the module the registry tests live in — find with
  `grep -rln "build_registry" tests/`)

**Interfaces:**
- Produces: registry-known key `mortality.fishing.rate.byyear.file.sp{idx}` → no unknown-key
  warning for Task 6's overlay configs; field renders in the UI like other FILE_PATH params.

- [ ] **Step 1: Write the failing test** (append to the registry test module):

```python
def test_byyear_fishing_file_is_registry_known() -> None:
    """F1 spec §2c: the by-year F file key must be schema-known so overlay configs
    validate clean."""
    from osmose.engine.config_validation import _check, build_known_keys

    known = build_known_keys()
    assert _check("mortality.fishing.rate.byyear.file.sp0", known) is None
```

  (`KnownKeys` is a bare dataclass with no `.matches`; `_check(key, known) -> None` when the
  key is known is the same path `validate()` uses — verified.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schema.py -k byyear -v`
Expected: FAIL (key unknown; the pre-fix suggester used to propose the camelCase ghost).

- [ ] **Step 3: Add the field** in `osmose/schema/species.py`, directly after the
  `mortality.fishing.rate.sp{idx}` field (`:395-405`). Copy the kwarg shape of an existing
  FILE_PATH field (`grep -n "FILE_PATH" osmose/schema/*.py` — e.g. `ltl.netcdf.file` in
  `osmose/schema/ltl.py`) and adapt:

```python
    OsmoseField(
        key_pattern="mortality.fishing.rate.byyear.file.sp{idx}",
        param_type=ParamType.FILE_PATH,
        default="",
        description=(
            "Per-year fishing mortality CSV (one F per line, sim-year 0 first; "
            "overrides the annual rate; must cover >= simulation.time.nyear rows)"
        ),
        category="fishing",
        indexed=True,
    ),
```

- [ ] **Step 4: Run schema + both allowlist-guard test files; update CLAUDE.md's stale count**

Run: `.venv/bin/python -m pytest tests/test_schema.py tests/test_engine_config_validation.py tests/test_schema_engine_key_parity.py tests/test_issue_123_known_but_unread_keys.py -v`
Expected: PASS — the parity gate passes because Task 1 Step 6 already allowlisted
`mortality.fishing.rate.byyear.file.sp{idx}` (schema fields must be engine-accepted, and this
key is AST-invisible). No test asserts a registry count (verified), but CLAUDE.md's "223
params" is stale twice over: set it to the actual post-change value —
`.venv/bin/python -c "from osmose.schema import build_registry; print(len(build_registry().all_fields()))"`
(263 before this task → 264 after).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check osmose/ tests/
git add osmose/schema/species.py tests/test_schema.py CLAUDE.md
git commit -m "feat(schema): register mortality.fishing.rate.byyear.file.sp{idx}

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Derivation script `scripts/build_baltic_f_byyear.py` + unit tests

Pure functions + a `__main__` that writes the four CSVs. Spec decisions 2, 3, 5 are binding:
factors per stock anchored on mean F over **available** years in 2018–2022; herring =
catch-weighted mean of per-stock **factors**; flounder gets **no file**.

**Files:**
- Create: `scripts/build_baltic_f_byyear.py`
- Test: `tests/test_build_baltic_f_byyear.py`

**Interfaces:**
- Produces: functions `load_stock(snap_dir, stock_key) -> tuple[dict[int, float], dict[int, float]]`
  (F-by-year, catches-by-year; empty-string values skipped),
  `hold_last(series: dict[int, float], years: list[int]) -> list[float]`,
  `anchor_mean(f: dict[int, float]) -> float`,
  `factor_series(f: dict[int, float]) -> list[float]` (over `YEARS`),
  `herring_factor_series(stocks: list[tuple[dict, dict]]) -> list[float]`,
  `read_base_f_strings(fishing_csv: Path) -> dict[int, str]` (raw strings by sp index),
  `build_rows(base_str: str, factors: list[float]) -> list[str]` (19 verbatim + 31 scaled),
  `write_csv(path, rows, header_lines)`. Module constants
  `YEARS = list(range(1993, 2024))`, `SPINUP = 19`, `ANCHOR = (2018, 2022)`,
  `STOCKS = {0: ("cod_west", ["cod.27.22-24"]), 1: ("herring", ["her.27.25-2932",
  "her.27.28", "her.27.3031", "her.27.20-24"]), 2: ("sprat", ["spr.27.22-32"]),
  8: ("cod_east", ["cod.27.24-32"])}`. Task 5 runs the `__main__`; Task 6 reads the CSVs.

- [ ] **Step 1: Write the failing unit tests**:

```python
"""Unit tests for the F1 by-year derivation (spec decisions 2, 3, 5).
Fixtures are tiny synthetic snapshot dicts — no I/O beyond tmp_path."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_baltic_f_byyear",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_f_byyear.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_anchor_mean_uses_available_years_only():
    f = {2018: 1.0, 2019: 1.2, 2020: 0.8, 2021: 1.0}  # 2022 missing (cod_west case)
    assert m.anchor_mean(f) == 1.0


def test_hold_last_fills_trailing_gap():
    f = {1993: 0.5, 1994: 0.7}
    assert m.hold_last(f, [1993, 1994, 1995, 1996]) == [0.5, 0.7, 0.7, 0.7]


def test_factor_series_is_anchored():
    f = {y: 2.0 for y in range(1993, 2024)}
    for y in range(2018, 2023):
        f[y] = 4.0
    fac = m.factor_series(f)
    assert fac[0] == 0.5   # 1993: 2.0 / anchor 4.0
    assert fac[27] == 1.0  # 2020 is inside the anchor window: 4.0 / 4.0


def test_factor_series_final_year():
    f = {y: 1.0 for y in range(1993, 2024)}
    assert m.factor_series(f) == [1.0] * 31


def test_herring_factors_are_scale_free():
    """An index-scaled stock (F around 1) and an absolute stock (F around 0.2)
    with identical PATTERNS and equal catches must give the same aggregate as
    either stock alone — the scale must cancel (spec decision 3)."""
    years = m.YEARS
    pattern = {y: 1.0 + 0.5 * ((y % 5) - 2) / 2 for y in years}
    f_index = dict(pattern)
    f_abs = {y: 0.2 * v for y, v in pattern.items()}
    catches = {y: 100.0 for y in years}
    agg = m.herring_factor_series([(f_index, catches), (f_abs, catches)])
    solo = m.factor_series(f_index)
    assert all(abs(a - s) < 1e-12 for a, s in zip(agg, solo))


def test_build_rows_layout_and_verbatim_spinup():
    rows = m.build_rows("0.3799687566571175", [1.0] * 31)
    assert len(rows) == 50
    assert rows[:19] == ["0.3799687566571175"] * 19   # verbatim string
    assert float(rows[19]) == 0.3799687566571175      # repr round-trips exactly


def test_herring_factors_weighting_unequal():
    """3:1 catch weights, hand-computed value (transposition/indexing canary —
    herring is a pass/fail stock)."""
    years = m.YEARS
    f_flat = {y: 1.0 for y in years}                 # factor 1.0 everywhere
    f_step = {y: 1.0 for y in years}                 # factor 0.5 outside the anchor
    for y in years:
        if not (2018 <= y <= 2022):
            f_step[y] = 0.5
    big = {y: 300.0 for y in years}
    small = {y: 100.0 for y in years}
    agg = m.herring_factor_series([(f_flat, big), (f_step, small)])
    assert agg[0] == 0.875                            # (3*1.0 + 1*0.5) / 4, exact in FP
    assert agg[years.index(2020)] == 1.0              # inside the anchor window


def test_no_flounder_in_stocks():
    assert 3 not in m.STOCKS  # spec decision 5
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_baltic_f_byyear.py -v`
Expected: FAIL at import (script does not exist).

- [ ] **Step 3: Implement `scripts/build_baltic_f_byyear.py`**:

```python
#!/usr/bin/env python
"""Derive by-year fishing-mortality CSVs for the F1 hindcast (spec 2026-08-23).

Offline: reads data/baltic/reference/ices_snapshots/*.assessment.json (cached; no
network) and data/baltic/baltic_param-fishing.csv (base F, verbatim strings).
Writes data/baltic/reference/f_byyear_sp{0,1,2,8}.csv — 50 rows each: 19 spin-up
rows carrying the base-F string verbatim (arms must share the pre-period
bit-exactly), then 31 rows base_F * factor(1993..2023).

Scaling (spec decisions 2-3): factor_s(y) = F_s(y) / mean(F_s over available
years in 2018-2022); herring aggregates the four stocks' FACTORS (scale-free)
with per-year catch weights in tonnes. Flounder (sp3) gets NO file (decision 5:
its calibrated base F is 6.4x its ICES anchor — incommensurable). cod_west F
ends 2021 -> hold-last.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data/baltic/reference/ices_snapshots"
FISHING_CSV = ROOT / "data/baltic/baltic_param-fishing.csv"
OUT_DIR = ROOT / "data/baltic/reference"

YEARS = list(range(1993, 2024))  # 31 hindcast years
SPINUP = 19                      # sim-years 0-18; sim-year 19 = 1993
ANCHOR = (2018, 2022)

STOCKS: dict[int, tuple[str, list[str]]] = {
    0: ("cod_west", ["cod.27.22-24"]),
    1: ("herring", ["her.27.25-2932", "her.27.28", "her.27.3031", "her.27.20-24"]),
    2: ("sprat", ["spr.27.22-32"]),
    8: ("cod_east", ["cod.27.24-32"]),
}


def load_stock(snap_dir: Path, stock_key: str) -> tuple[dict[int, float], dict[int, float]]:
    """(F-by-year, catches-by-year); snapshot values are strings, '' = missing."""
    recs = json.loads((snap_dir / f"{stock_key}.assessment.json").read_text())
    f = {int(r["year"]): float(r["f"]) for r in recs if r.get("f") not in ("", None)}
    c = {int(r["year"]): float(r["catches"]) for r in recs if r.get("catches") not in ("", None)}
    return f, c


def hold_last(series: dict[int, float], years: list[int]) -> list[float]:
    out: list[float] = []
    last: float | None = None
    for y in years:
        if y in series:
            last = series[y]
        if last is None:
            raise ValueError(f"no value at or before {y}")
        out.append(last)
    return out


def anchor_mean(f: dict[int, float]) -> float:
    vals = [f[y] for y in range(ANCHOR[0], ANCHOR[1] + 1) if y in f]
    if not vals:
        raise ValueError(f"no F values in anchor window {ANCHOR}")
    return sum(vals) / len(vals)


def factor_series(f: dict[int, float]) -> list[float]:
    a = anchor_mean(f)
    return [v / a for v in hold_last(f, YEARS)]


def herring_factor_series(stocks: list[tuple[dict[int, float], dict[int, float]]]) -> list[float]:
    per_stock = [factor_series(f) for f, _ in stocks]
    weights = [hold_last(c, YEARS) for _, c in stocks]
    out: list[float] = []
    for i in range(len(YEARS)):
        w = [wt[i] for wt in weights]
        out.append(sum(wi * fs[i] for wi, fs in zip(w, per_stock)) / sum(w))
    return out


def read_base_f_strings(fishing_csv: Path) -> dict[int, str]:
    """Raw base-F strings by species index. Relies on the identity sp<->fsh mapping
    (data/baltic/fishery-catchability.csv)."""
    out: dict[int, str] = {}
    for line in fishing_csv.read_text().splitlines():
        if line.startswith("fisheries.rate.base.fsh"):
            key, val = line.split(";", 1)
            out[int(key.rsplit("fsh", 1)[1])] = val.strip()
    return out


def build_rows(base_str: str, factors: list[float]) -> list[str]:
    base = float(base_str)
    # repr() is the shortest round-trip representation: float(repr(x)) == x exactly,
    # so the scaled rows lose no precision through np.loadtxt.
    return [base_str] * SPINUP + [repr(base * f) for f in factors]


def write_csv(path: Path, rows: list[str], header_lines: list[str]) -> None:
    text = "".join(f"# {h}\n" for h in header_lines) + "\n".join(rows) + "\n"
    path.write_text(text)


def main() -> None:
    base_strings = read_base_f_strings(FISHING_CSV)
    for sp_idx, (name, stock_keys) in STOCKS.items():
        loaded = [load_stock(SNAP, k) for k in stock_keys]
        factors = (
            herring_factor_series(loaded) if len(loaded) > 1 else factor_series(loaded[0][0])
        )
        rows = build_rows(base_strings[sp_idx], factors)
        header = [
            f"F1 hindcast by-year F for {name} (sp{sp_idx}) — generated {date.today()}",
            f"stocks: {', '.join(stock_keys)}; anchor: mean F over available years "
            f"{ANCHOR[0]}-{ANCHOR[1]}; base F (verbatim): {base_strings[sp_idx]}",
            f"layout: {SPINUP} spin-up rows at base F, then {len(YEARS)} rows "
            f"base*factor for {YEARS[0]}-{YEARS[-1]} (sim-year {SPINUP} = {YEARS[0]})",
            f"factor range: {min(factors):.3g}-{max(factors):.3g}",
            "spec: docs/superpowers/specs/2026-08-23-baltic-f1-historical-fishing-hindcast-design.md",
        ]
        out = OUT_DIR / f"f_byyear_sp{sp_idx}.csv"
        write_csv(out, rows, header)
        print(f"wrote {out} (factor range {min(factors):.3g}-{max(factors):.3g})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_build_baltic_f_byyear.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint and commit** (script + tests only — no data yet)

```bash
.venv/bin/ruff check scripts/build_baltic_f_byyear.py tests/test_build_baltic_f_byyear.py
git add scripts/build_baltic_f_byyear.py tests/test_build_baltic_f_byyear.py
git commit -m "feat(scripts): F1 by-year F derivation from cached ICES snapshots

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Generate the real CSVs and verify against the spec's claims

**Files:**
- Create (generated): `data/baltic/reference/f_byyear_sp0.csv`, `f_byyear_sp1.csv`,
  `f_byyear_sp2.csv`, `f_byyear_sp8.csv`

**Interfaces:**
- Consumes: Task 4's `main()`.
- Produces: the four committed CSVs Task 6 points config keys at.

- [ ] **Step 1: Generate**

Run: `.venv/bin/python scripts/build_baltic_f_byyear.py`
Expected: four `wrote ...` lines with factor ranges.

- [ ] **Step 2: Verify the ranges against the spec/review numbers.** Reported factor ranges
  must be consistent with the audited claims: cod_east ≈ 0.17–14.6; herring and sprat scaled F
  (base × max factor) must stay below ~0.6/yr (bases ≈ 0.380 and 0.164, base/anchor ratios
  ≈ 0.39/0.44); cod_west factors within roughly 0.7–1.4 (its ICES F spans 0.896–1.214).
  Check with:

```bash
head -6 data/baltic/reference/f_byyear_sp8.csv
grep -c "" data/baltic/reference/f_byyear_sp1.csv
```

  Each file: 5 header lines + 50 rows = 55 lines. If any range is far off, STOP — do not
  commit; re-check the derivation against the snapshots (the review's numbers were
  independently computed twice, so a large mismatch means a bug in the script).

- [ ] **Step 3: Load-through-engine smoke check** (the spin-up verbatim guarantee end-to-end).
  Shell rule: NO heredocs with `#` lines and no output redirection — write the check to
  `/tmp/f1_smoke.py` with the Write tool, then run it. Content of `/tmp/f1_smoke.py`:

```python
import tempfile
from pathlib import Path

from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.config import EngineConfig

tmp = Path(tempfile.mkdtemp())
cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
cfg["simulation.time.nyear"] = "50"
root = Path("/home/razinka/osmopy")
for i in (0, 1, 2, 8):
    cfg[f"mortality.fishing.rate.byyear.file.sp{i}"] = str(
        root / f"data/baltic/reference/f_byyear_sp{i}.csv"
    )
ec = EngineConfig.from_dict(cfg)
for i in (0, 1, 2, 8):
    arr = ec.fishing_rate_by_year[i]
    assert arr is not None and len(arr) == 50, (i, None if arr is None else len(arr))
    assert (arr[:19] == ec.fishing_rate[i]).all(), f"sp{i} spin-up rows != base F"
assert ec.fishing_rate_by_year[3] is None, "flounder must be unforced (decision 5)"
print("OK: 4 series loaded, 50 rows, verbatim spin-up, flounder unforced")
```

Run: `.venv/bin/python /tmp/f1_smoke.py`
Expected: `OK: ...`. (This also exercises Task 2's guard on the true 50-row/50-year shape.)

- [ ] **Step 4: Commit the data**

```bash
git add data/baltic/reference/f_byyear_sp0.csv data/baltic/reference/f_byyear_sp1.csv data/baltic/reference/f_byyear_sp2.csv data/baltic/reference/f_byyear_sp8.csv
git commit -m "data(baltic): by-year F series 1993-2023 for the F1 hindcast (4 stocks)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Hindcast harness `scripts/baltic_f_hindcast.py` + CI-safe helper tests

Modeled on `scripts/baltic_rv_hindcast.py` (read it first). Metrics implement spec §4 and
decisions 6–7 exactly.

**Files:**
- Create: `scripts/baltic_f_hindcast.py`
- Test: `tests/test_baltic_f_hindcast_helpers.py`

**Interfaces:**
- Consumes: Task 5's CSVs; `PythonEngine().run_in_memory(raw, seed)` returning results with
  `.ssb()` and `.yield_biomass()` (`osmose/results.py:474,492`) — DataFrames with one column
  per species name.
- Produces: `run_hindcast(seeds=(42, 123, 7, 999, 2024)) -> dict` and pure helpers
  `annualize(x, n_year) -> np.ndarray`, `zscore(x) -> np.ndarray`,
  `decadal_trend_signs(values, years) -> list[int]`,
  `pearson(a, b) -> float`,
  `skill_verdict(dr_per_seed) -> dict` (fields: `mean_dr`, `sd_dr`, `passes`),
  `observed_stock_z(snap_dir, stock_key, years) -> np.ndarray` (nan-padded),
  `observed_herring_z(snap_dir, years) -> np.ndarray`,
  `instrument_check(factors, yields, biomass) -> float` (Spearman rank corr of factor vs
  yield/biomass). Task 8 runs `run_hindcast` and formats the results doc from its dict.

- [ ] **Step 1: Write the failing helper tests**:

```python
"""CI-safe tests for the F1 hindcast harness helpers (spec §4, decisions 6-7).
The hindcast RUN is not a CI gate; these cover only the pure functions."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "baltic_f_hindcast",
    Path(__file__).resolve().parent.parent / "scripts" / "baltic_f_hindcast.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

YEARS = list(range(1993, 2024))


def test_annualize_per_step_and_per_year():
    per_year = np.arange(50.0)
    assert (m.annualize(per_year, 50) == per_year).all()
    per_step = np.repeat(np.arange(50.0), 24)
    assert (m.annualize(per_step, 50) == np.arange(50.0)).all()


def test_decadal_trend_signs():
    rising_then_falling = [float(y) for y in YEARS[:10]] + [0.0] * 10 + [
        float(2024 - y) for y in YEARS[20:]
    ]
    signs = m.decadal_trend_signs(rising_then_falling, YEARS)
    assert signs[0] == 1 and signs[2] == -1


def test_skill_verdict_margins():
    # spec decision 7: mean dr >= 0.10 AND mean dr > 2*sd
    assert m.skill_verdict([0.12, 0.11, 0.13, 0.12, 0.12])["passes"] is True
    assert m.skill_verdict([0.009, 0.010, 0.008, 0.011, 0.009])["passes"] is False  # July spike
    assert m.skill_verdict([0.30, -0.10, 0.25, -0.05, 0.15])["passes"] is False  # noisy


def test_zscore_unit_variance():
    z = m.zscore([1.0, 2.0, 3.0, 4.0])
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1.0) < 1e-12


def test_observed_herring_z_is_catch_share_weighted(tmp_path):
    """Two synthetic stocks, opposite SSB trends, 3:1 mean catch share ->
    composite must tilt to the big stock's trend (spec decision 6)."""
    import json

    def snap(key, ssb_by_year, catches):
        recs = [
            {"year": str(y), "ssb": str(ssb_by_year(y)), "f": "0.2", "catches": str(catches)}
            for y in YEARS
        ]
        (tmp_path / f"{key}.assessment.json").write_text(json.dumps(recs))

    snap("her.27.25-2932", lambda y: y - 1990, 300.0)   # rising, weight 3
    snap("her.27.28", lambda y: 2030 - y, 100.0)        # falling, weight 1
    snap("her.27.3031", lambda y: 1.0, 0.0)
    snap("her.27.20-24", lambda y: 1.0, 0.0)
    z = m.observed_herring_z(tmp_path, YEARS)
    assert len(z) == len(YEARS)
    assert z[-1] > z[0]  # composite follows the dominant stock upward
```

  Note the two zero-catch stocks: `observed_herring_z` must tolerate zero weights (their
  z-series contribute nothing) and constant SSB (zscore of a constant series must return
  zeros, not divide by zero — implement `zscore` with a `std == 0 -> zeros` branch and cover
  it in `test_zscore_unit_variance` with an extra assert:
  `assert (m.zscore([2.0, 2.0, 2.0]) == 0).all()`).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_f_hindcast_helpers.py -v`
Expected: FAIL at import.

- [ ] **Step 3: Implement `scripts/baltic_f_hindcast.py`**:

```python
#!/usr/bin/env python
"""F1 historical-fishing hindcast (spec 2026-08-23, Stage 1 of B1). Two arms x 5
seeds x 50 yr on the certified Baltic config: A = constant F, B = by-year ICES F
(4 stocks). Sim-year 19 = 1993. Scores herring+sprat (pass/fail, decision 7
margins); cod_west/cod_east/flounder reported-only. NOT a CI gate (emergent).

Instrument check (blocking for herring/sprat/cod_east): arm B realized
yield-per-biomass must rank-correlate with the imposed factor pattern.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data/baltic/reference/ices_snapshots"
YEARS = list(range(1993, 2024))
SPINUP = 19
N_YEAR = 50
SEEDS = (42, 123, 7, 999, 2024)
FORCED = (0, 1, 2, 8)
SPECIES = {0: "cod_west", 1: "herring", 2: "sprat", 3: "flounder", 8: "cod_east"}
SCORED = ("herring", "sprat")
BLOCKING_INSTRUMENT = ("herring", "sprat", "cod_east")
DECADES = ((1993, 2002), (2003, 2012), (2013, 2023))
HERRING_STOCKS = ["her.27.25-2932", "her.27.28", "her.27.3031", "her.27.20-24"]
OBS_STOCK = {"cod_west": "cod.27.22-24", "sprat": "spr.27.22-32",
             "flounder": "fle.27.2223", "cod_east": "cod.27.24-32"}


def annualize(x, n_year: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == n_year:
        return x
    if len(x) % n_year == 0:
        return x.reshape(n_year, -1).mean(axis=1)
    raise ValueError(f"series of {len(x)} not divisible into {n_year} years")


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() <= 2 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def decadal_trend_signs(values, years) -> list[int]:
    v, y = np.asarray(values, float), np.asarray(years)
    out = []
    for lo, hi in DECADES:
        mask = (y >= lo) & (y <= hi) & ~np.isnan(v)
        slope = np.polyfit(y[mask], v[mask], 1)[0]
        out.append(1 if slope > 0 else -1)
    return out


def skill_verdict(dr_per_seed) -> dict:
    dr = np.asarray(dr_per_seed, float)
    mean, sd = float(np.nanmean(dr)), float(np.nanstd(dr, ddof=1))
    return {"mean_dr": mean, "sd_dr": sd, "passes": bool(mean >= 0.10 and mean > 2 * sd)}


def _ssb_series(snap_dir: Path, stock_key: str, years) -> np.ndarray:
    recs = json.loads((snap_dir / f"{stock_key}.assessment.json").read_text())
    by_year = {int(r["year"]): float(r["ssb"]) for r in recs if r.get("ssb") not in ("", None)}
    return np.array([by_year.get(y, np.nan) for y in years], dtype=float)


def observed_stock_z(snap_dir: Path, stock_key: str, years) -> np.ndarray:
    return zscore(_ssb_series(snap_dir, stock_key, years))


def observed_herring_z(snap_dir: Path, years) -> np.ndarray:
    """Decision 6: fixed-weight mean of per-stock z-scores; weights = mean catch
    share over the window."""
    zs, weights = [], []
    for key in HERRING_STOCKS:
        recs = json.loads((snap_dir / f"{key}.assessment.json").read_text())
        catches = {int(r["year"]): float(r["catches"])
                   for r in recs if r.get("catches") not in ("", None)}
        w = np.nanmean([catches.get(y, np.nan) for y in years])
        zs.append(observed_stock_z(snap_dir, key, years))
        weights.append(0.0 if np.isnan(w) else w)
    z, w = np.stack(zs), np.asarray(weights, float)
    wsum = np.where(np.isnan(z), 0.0, w[:, None]).sum(axis=0)
    return np.nansum(z * w[:, None], axis=0) / np.where(wsum == 0, np.nan, wsum)


def _spearman(a, b) -> float:
    from scipy.stats import rankdata  # scipy is in the venv (SALib dependency)

    return pearson(rankdata(a), rankdata(b))


def instrument_check(factors, yields, biomass) -> float:
    """Rank corr between the imposed factor pattern and realized yield-per-biomass
    over 1993-2023. Wrong-mapping / silent-no-op canary."""
    ypb = np.asarray(yields, float) / np.maximum(np.asarray(biomass, float), 1e-9)
    return _spearman(np.asarray(factors, float), ypb)


def load_factors(sp_idx: int) -> np.ndarray:
    arr = np.loadtxt(ROOT / f"data/baltic/reference/f_byyear_sp{sp_idx}.csv")
    return arr[SPINUP:] / arr[0]  # scaled rows / base F = factor series


def arm_overrides(mode: str) -> dict:
    base = {"simulation.time.nyear": str(N_YEAR), "output.ssb.enabled": "true"}
    if mode == "fhist":
        for i in FORCED:
            base[f"mortality.fishing.rate.byyear.file.sp{i}"] = str(
                ROOT / f"data/baltic/reference/f_byyear_sp{i}.csv"
            )
    return base


def run_hindcast(seeds=SEEDS) -> dict:
    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    tmp = Path(tempfile.mkdtemp())
    base_cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))

    # Spec §3 startup assertion: spin-up rows must equal the live base F bit-exactly,
    # so the two arms share the 1974-1992 pre-period.
    for i in FORCED:
        arr = np.loadtxt(ROOT / f"data/baltic/reference/f_byyear_sp{i}.csv")
        base_f = float(base_cfg[f"fisheries.rate.base.fsh{i}"])
        assert (arr[:SPINUP] == base_f).all(), (
            f"sp{i}: spin-up rows != base F {base_f}; regenerate the CSVs "
            "(scripts/build_baltic_f_byyear.py) after any recalibration"
        )

    ssb: dict[str, dict[str, list[np.ndarray]]] = {a: {} for a in ("A", "B")}
    ypb_inputs: dict[str, list] = {}
    for seed in seeds:
        for arm, mode in (("A", "base"), ("B", "fhist")):
            raw = {**base_cfg, **arm_overrides(mode)}
            res = PythonEngine().run_in_memory(raw, seed=seed)
            ssb_df, yld_df, bio_df = res.ssb(), res.yield_biomass(), res.biomass()
            for name in SPECIES.values():
                series = annualize(ssb_df[name].to_numpy(dtype=float), N_YEAR)[SPINUP:]
                ssb[arm].setdefault(name, []).append(series)
                if arm == "B":
                    # spec §3: yield-per-BIOMASS (not SSB) — selectivity admits
                    # pre-mature fish, so the denominators differ.
                    yld = annualize(yld_df[name].to_numpy(dtype=float), N_YEAR)[SPINUP:]
                    bio = annualize(bio_df[name].to_numpy(dtype=float), N_YEAR)[SPINUP:]
                    ypb_inputs.setdefault(name, []).append((yld, bio))

    obs = {"herring": observed_herring_z(SNAP, YEARS)}
    for name, key in OBS_STOCK.items():
        obs[name] = observed_stock_z(SNAP, key, YEARS)

    report: dict = {"stocks": {}, "instrument": {}}
    for sp_idx in FORCED:
        name = SPECIES[sp_idx]
        factors = load_factors(sp_idx)
        rhos = [instrument_check(factors, y, b) for y, b in ypb_inputs[name]]
        report["instrument"][name] = {
            "rho_per_seed": rhos,
            "blocking": name in BLOCKING_INSTRUMENT,
        }
    for name in SPECIES.values():
        a_runs, b_runs = np.stack(ssb["A"][name]), np.stack(ssb["B"][name])
        dr = [pearson(zscore(b), obs[name]) - pearson(zscore(a), obs[name])
              for a, b in zip(a_runs, b_runs)]
        report["stocks"][name] = {
            "scored": name in SCORED,
            "trend_model_B": decadal_trend_signs(b_runs.mean(axis=0), YEARS),
            "trend_observed": decadal_trend_signs(obs[name], YEARS),
            "skill": skill_verdict(dr),
            "r_A_mean": pearson(zscore(a_runs.mean(axis=0)), obs[name]),
            "r_B_mean": pearson(zscore(b_runs.mean(axis=0)), obs[name]),
            "ssb_A_mean": a_runs.mean(axis=0).tolist(),
            "ssb_B_mean": b_runs.mean(axis=0).tolist(),
            "obs_z": np.asarray(obs[name], float).tolist(),
        }
    return report


REPORT_PATH = Path("/tmp/f1_hindcast_report.json")

if __name__ == "__main__":
    out = run_hindcast()
    REPORT_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"report written to {REPORT_PATH}")
```

- [ ] **Step 4: Run the helper tests**

Run: `.venv/bin/python -m pytest tests/test_baltic_f_hindcast_helpers.py -v`
Expected: PASS (5 tests). Also confirm scipy imports:
`.venv/bin/python -c "from scipy.stats import rankdata; print('ok')"` — if scipy is absent,
replace `_spearman` with a rank via `np.argsort(np.argsort(x))` (no new dependency).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check scripts/baltic_f_hindcast.py tests/test_baltic_f_hindcast_helpers.py
git add scripts/baltic_f_hindcast.py tests/test_baltic_f_hindcast_helpers.py
git commit -m "feat(scripts): F1 hindcast harness — 2 arms, pre-registered metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Certification guard (local engine run)

**Files:** none (verification run).

- [ ] **Step 1: Confirm the machine is free**

Run: `uptime` — load must be low (< ~2) and no ESTAS_II job hogging cores. If busy, wait.

- [ ] **Step 2: Run the certification with an explicit --out**

Run (Bash `run_in_background`, ~35 min):
`.venv/bin/python scripts/baltic_stability_certify.py --out /tmp/f1_cert_guard.md`

All other flags default to the standard protocol (50 yr, seeds [42, 123, 7, 999, 2024],
`--params current`, config-default seeding — verified against the certifier's argparse).
`--out` is MANDATORY: the certifier's default output path is
`docs/baltic_stability_certification_2026-07-01.md`, a committed historical doc it would
silently overwrite.

- [ ] **Step 3: Compare** `/tmp/f1_cert_guard.md` against the last committed certification
  (`docs/baltic_certification_2026-08-14.md` — same protocol, same seeds). Expected: identical
  verdicts for the identity set {cod_west, cod_east, herring, sprat, flounder, perch,
  stickleback} and per-species figures matching (cod_east ≈ 65,209 t etc.). Any regression →
  STOP, bisect Tasks 1–3 (they must be behaviorally inert on configs without by-year keys).

- [ ] **Step 4: Record.** Append one line to the results doc draft (Task 8 creates it): guard
  date, invocation, verdict "unchanged".

---

### Task 8: The hindcast run + results doc (local engine run)

**Files:**
- Create: `docs/baltic_f_hindcast_2026-MM-DD.md` (actual run date)
- Create: `docs/diagnostics/baltic_f_hindcast.png`

- [ ] **Step 1: Machine check** — `uptime` low, no concurrent engine jobs (10 engine runs
  ≈ 2× the certification cost; run in the background).

- [ ] **Step 2: Run** `.venv/bin/python scripts/baltic_f_hindcast.py` via Bash
  `run_in_background` — the harness writes its own report to `/tmp/f1_hindcast_report.json`
  (no shell redirection needed or allowed).

- [ ] **Step 3: Instrument gate FIRST.** From `report["instrument"]`: for herring, sprat,
  cod_east (blocking), the per-seed Spearman rho of factor vs yield-per-biomass must be clearly
  positive (median rho ≥ 0.5 is the expectation for a working override; if any blocking stock
  is near 0 or negative, STOP — debug the key wiring before reading any SSB result; check
  cod_west's rho for context but do not block on it).

- [ ] **Step 4: Apply the pre-registered criteria** (no post-hoc adjustment): per scored stock
  (herring, sprat) — trend test: `trend_model_B` matches `trend_observed` in ≥2 of 3 decades;
  skill test: `skill["passes"]`. Verdict: PASS = both stocks pass both tests; PARTIAL = one;
  NULL = zero. Cod stocks and flounder: report trajectories and (non-binding) the same numbers.

- [ ] **Step 5: Figure.** One panel per stock (5): observed z (line), arm A and arm B 5-seed
  mean z (lines), x = 1993–2023. Save `docs/diagnostics/baltic_f_hindcast.png`. Load the
  `dataviz` skill before writing the plotting code.

- [ ] **Step 6: Results doc** `docs/baltic_f_hindcast_2026-MM-DD.md`: verdict up top
  (PASS/PARTIAL/NULL + the gating consequence from spec §4 — a NULL demotes B1 Stages 2–3 to
  capability-motivated); instrument-check table; per-stock table (trend signs, r_A, r_B, mean
  Δr ± sd, pass margins); the honest washout framing (herring/sprat face the July spike's
  mechanism on the certified config); limitations (relative scaling, herring composite
  construction, cod_west data gaps); certification-guard line from Task 7.

- [ ] **Step 7: Commit**

```bash
git add docs/baltic_f_hindcast_2026-*.md docs/diagnostics/baltic_f_hindcast.png
git commit -m "docs(baltic): F1 historical-fishing hindcast — results and verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution notes

- Tasks 1–6 are CI-safe and independent of machine load; Tasks 7–8 are heavy local engine runs
  (never concurrent — memory `uq-parallel-threading` cost hours to learn).
- If `run_in_memory(...).ssb()` or `.yield_biomass()` raises or lacks a species column, read
  `scripts/baltic_rv_hindcast.py` and `osmose/results.py:474-530` before changing the harness —
  the accessor contract is established there.
- The spec is the authority on every threshold; if implementation reveals a contradiction,
  STOP and surface it rather than adjusting a margin.
