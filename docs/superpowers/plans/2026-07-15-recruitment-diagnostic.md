# Recruitment Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Recruitment (model vs ICES)" section to `scripts/evaluate_calibration_vs_ices.py` comparing the model's age-matched recruitment to the ICES R geomean for sprat + herring, with an order-of-magnitude verdict.

**Architecture:** Pure ICES/verdict helpers live in `evaluate_calibration_vs_ices.py` (reusing `validate_baltic_vs_ices_sag.py`'s snapshot loaders); the model-side recruitment stat is extracted inside `calibrate_baltic.run_simulation` via a new **gated** `recruitment_ages` param (inert for the DE loop); `evaluate()` resolves the ages, enables `abundance_by_age` for its own run, threads both, and renders a report section.

**Tech Stack:** Python 3.12, numpy, pandas, pytest. No new dependency.

## Global Constraints

- **No optimizer/loss/`biomass_targets.csv` change. No change to DE-loop behavior.** `run_simulation` gains a `recruitment_ages: dict[str, str] | None = None` param; the recruitment block runs ONLY when it is non-None AND `output.abundance.byage.enabled` is set — which only `evaluate()` does, never `_ObjectiveWrapper`. Every DE eval stays byte-identical.
- **Scope: sprat + herring only.** cod (eastern index + recruitment_age mismatch 0 vs 1) and flounder (no recruitment) print as "no clean ICES R" — never silently dropped.
- **`recruitment_age` is a STRING** (`"0"`/`"1"`); `abundance_by_age()` is **long-format** `[time, species, bin, value]` with `bin` a string. Filter `bin == str(recruitment_age)`; compare as strings.
- **Guard the disabled/empty abundance-by-age path:** `abundance_by_age()` returns a bare `pd.DataFrame()` (no columns) when the output is off → guard `if not df.empty and "bin" in df.columns`, else leave the stat unset.
- **Verdict:** `ratio = model_R / ices_geomean`; OK if `1/3 ≤ ratio ≤ 3` (inclusive), else FLAG.
- **Window: 2018–2022** (`validate_baltic_vs_ices_sag.WINDOW_YEARS = range(2018, 2023)`), keeping only years all mapped stocks report R. ICES R reference = **geometric mean** (+ report per-year min/max).
- **Age-0 science caveat:** the report text must note a low **herring** ratio is partly the annual-mean-vs-cohort-census artifact (weaker evidence of SR miscalibration than sprat).
- **The report/formatter test must NOT run the engine** ([[feedback-ci-fragile-emergent-tests]]).
- Spec: `docs/superpowers/specs/2026-07-15-recruitment-diagnostic-design.md`.

---

## File Structure

- **Modify** `scripts/evaluate_calibration_vs_ices.py` — add `_species_recruitment_age`, `_ices_recruitment_geomean`, `_recruitment_verdict`, `_format_recruitment_section` + reuse-imports from `validate_baltic_vs_ices_sag`; thread the flag + `recruitment_ages` in `evaluate()`; render the section in `_print_report`.
- **Modify** `scripts/calibrate_baltic.py` — `run_simulation` gains `recruitment_ages` + a gated recruitment-stat block.
- **Test** `tests/test_recruitment_diagnostic.py` (Tasks 1, 3), `tests/test_calibrate_baltic_recruitment_stat.py` (Task 2).

---

## Task 1: ICES recruitment helpers + verdict

**Files:**
- Modify: `scripts/evaluate_calibration_vs_ices.py` (add helpers + reuse imports)
- Test: `tests/test_recruitment_diagnostic.py`

**Interfaces:**
- Consumes: `validate_baltic_vs_ices_sag.{WINDOW_YEARS, _load_manifest, _load_assessment, _load_reference_points, _series_by_year}`.
- Produces:
  - `_species_recruitment_age(species: str) -> str | None`
  - `_ices_recruitment_geomean(species: str) -> tuple[float, float, float] | None` (geomean, min, max)
  - `_recruitment_verdict(model_R: float, ices_geomean: float) -> tuple[float, str]`
  - `RECRUITMENT_ASSESSED = ("cod", "herring", "sprat", "flounder")`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recruitment_diagnostic.py`:

```python
"""Recruitment diagnostic helpers (Spec 2)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ modules use BARE sibling imports (e.g. `from calibrate_baltic import ...`), so scripts/
# must be on sys.path and we import UNQUALIFIED — mirrors tests/test_fr_diagnostic.py. A dotted
# `from scripts.evaluate_calibration_vs_ices import ...` fails at collection (ModuleNotFoundError:
# calibrate_baltic), because a package-qualified import never puts scripts/ itself on sys.path.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluate_calibration_vs_ices import (  # noqa: E402
    _ices_recruitment_geomean,
    _recruitment_verdict,
    _species_recruitment_age,
)


def test_recruitment_age_sprat_herring_clean_cod_flounder_none():
    assert _species_recruitment_age("sprat") == "1"
    assert _species_recruitment_age("herring") == "0"
    assert _species_recruitment_age("cod") is None  # stocks disagree (age 0 vs 1)
    assert _species_recruitment_age("flounder") is None  # no recruitment_age


def test_sprat_geomean_matches_independent_computation():
    from validate_baltic_vs_ices_sag import WINDOW_YEARS, _load_assessment, _series_by_year

    rec = _series_by_year(_load_assessment("spr.27.22-32"), "recruitment")
    vals = [rec[y] for y in WINDOW_YEARS if y in rec]
    expected_geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
    geo, lo, hi = _ices_recruitment_geomean("sprat")
    assert geo == pytest.approx(expected_geo, rel=1e-9)
    assert lo == pytest.approx(min(vals)) and hi == pytest.approx(max(vals))


def test_herring_geomean_sums_four_stocks():
    from validate_baltic_vs_ices_sag import (
        WINDOW_YEARS,
        _load_assessment,
        _load_manifest,
        _series_by_year,
    )

    stocks = _load_manifest()["model_species_to_ices_stocks"]["herring"]
    series = [_series_by_year(_load_assessment(s), "recruitment") for s in stocks]
    per_year = [sum(s[y] for s in series) for y in WINDOW_YEARS if all(y in s for s in series)]
    expected_geo = math.exp(sum(math.log(v) for v in per_year) / len(per_year))
    geo, _, _ = _ices_recruitment_geomean("herring")
    assert geo == pytest.approx(expected_geo, rel=1e-9)
    assert geo > 100_000  # central stock included -> not the western-only undercount


def test_no_clean_r_species_return_none():
    assert _ices_recruitment_geomean("cod") is None
    assert _ices_recruitment_geomean("flounder") is None


def test_verdict_thresholds_inclusive():
    assert _recruitment_verdict(1.0, 1.0) == (1.0, "OK")
    assert _recruitment_verdict(1.0, 3.0)[1] == "OK"  # ratio 1/3 -> OK (inclusive)
    assert _recruitment_verdict(3.0, 1.0)[1] == "OK"  # ratio 3 -> OK (inclusive)
    assert _recruitment_verdict(0.33, 1.0)[1] == "FLAG"  # just below 1/3
    assert _recruitment_verdict(5.0, 1.0)[1] == "FLAG"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recruitment_diagnostic.py -q`
Expected: FAIL — `ImportError: cannot import name '_species_recruitment_age' from 'evaluate_calibration_vs_ices'` (the module imports fine via the sys.path shim; the name doesn't exist yet).

- [ ] **Step 3: Implement the helpers**

In `scripts/evaluate_calibration_vs_ices.py`, add near the top (after the existing imports):

```python
from validate_baltic_vs_ices_sag import (  # reuse snapshot loaders (dependency-free leaf)
    WINDOW_YEARS,
    _load_assessment,
    _load_manifest,
    _load_reference_points,
    _series_by_year,
)

RECRUITMENT_ASSESSED = ("cod", "herring", "sprat", "flounder")


def _species_recruitment_age(species: str) -> str | None:
    """Common ICES recruitment_age (as a string) across a species' mapped stocks, or None if
    the species has no mapped stocks, a stock lacks the age, or the stocks disagree."""
    stocks = _load_manifest()["model_species_to_ices_stocks"].get(species, [])
    if not stocks:
        return None
    ages = set()
    for st in stocks:
        a = _load_reference_points(st).get("recruitment_age")
        if a in (None, ""):
            return None
        ages.add(str(a))
    return ages.pop() if len(ages) == 1 else None


def _ices_recruitment_geomean(species: str) -> tuple[float, float, float] | None:
    """(geomean, min, max) of the per-year SUMMED ICES recruitment across a species' mapped
    stocks over WINDOW_YEARS, keeping only years all stocks report R. None if no clean numeric R.

    Summability is an inferred assumption: the snapshot records SSB units but not recruitment
    units; the mapped stocks' recruitments are all absolute counts on a self-consistent scale.
    """
    if _species_recruitment_age(species) is None:
        return None
    stocks = _load_manifest()["model_species_to_ices_stocks"][species]
    series = [_series_by_year(_load_assessment(st), "recruitment") for st in stocks]
    per_year = [
        sum(s[y] for s in series) for y in WINDOW_YEARS if all(y in s for s in series)
    ]
    if not per_year:
        return None
    arr = np.asarray(per_year, dtype=float)
    geomean = float(np.exp(np.mean(np.log(arr))))
    return geomean, float(arr.min()), float(arr.max())


def _recruitment_verdict(model_R: float, ices_geomean: float) -> tuple[float, str]:
    """(ratio, verdict). OK if 1/3 <= ratio <= 3 (order-of-magnitude), else FLAG."""
    ratio = model_R / ices_geomean if ices_geomean > 0 else float("inf")
    verdict = "OK" if (1.0 / 3.0) <= ratio <= 3.0 else "FLAG"
    return ratio, verdict
```

Ensure `import numpy as np` is present at module scope (add it if not).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recruitment_diagnostic.py -q`
Expected: all 5 tests PASS.

- [ ] **Step 5: Lint**

Run: `.venv/bin/python -m ruff check scripts/evaluate_calibration_vs_ices.py tests/test_recruitment_diagnostic.py && .venv/bin/python -m ruff format --check scripts/evaluate_calibration_vs_ices.py tests/test_recruitment_diagnostic.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_calibration_vs_ices.py tests/test_recruitment_diagnostic.py
git commit -m "feat(diagnostic): ICES recruitment geomean + verdict helpers"
```

---

## Task 2: `run_simulation` recruitment stat (gated)

**Files:**
- Modify: `scripts/calibrate_baltic.py` (`run_simulation`)
- Test: `tests/test_calibrate_baltic_recruitment_stat.py`

**Interfaces:**
- Consumes: `OsmoseResults.abundance_by_age()` (long frame `[time, species, bin, value]`, string bins).
- Produces: `run_simulation(config, overrides, n_years=40, seed=42, timeout_s=None, recruitment_ages: dict[str, str] | None = None)` now emits `{sp}_recruitment_mean` for each `(sp, age)` in `recruitment_ages` (when abundance-by-age is present).

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibrate_baltic_recruitment_stat.py`:

```python
"""run_simulation emits {sp}_recruitment_mean when recruitment_ages is passed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.calibrate_baltic as cb


def _fake_results(bio_df, yld_df, abd_df):
    class _R:
        def __init__(self, *a, **k): ...
        def biomass(self):
            return bio_df
        def yield_biomass(self):
            return yld_df
        def abundance_by_age(self):
            return abd_df
        def close(self): ...

    class _Engine:
        def run(self, *a, **k):
            class _Ret:
                returncode = 0
            return _Ret()

    return _R, _Engine


def _wire(monkeypatch, R, Engine):
    monkeypatch.setattr("osmose.engine.PythonEngine", Engine, raising=False)
    monkeypatch.setattr("osmose.results.OsmoseResults", R)


def test_recruitment_mean_from_age_bin(monkeypatch):
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0), "herring": np.full(12, 2000.0)})
    yld = pd.DataFrame({"sprat": np.full(12, 100.0), "herring": np.full(12, 200.0)})
    # Long abundance-by-age. Vary EARLY (t<2) vs TRAILING (last 10) values so a wrong-window slice
    # (full-mean / head-slice / off-by-one) is actually caught; a decoy wrong-age bin proves age
    # selection. mean of the LAST 10 = 5e7/4e7; a full-12 mean would be inflated by the early decoy.
    rows = []
    for t in range(12):
        early = t < 2
        rows += [
            {"time": t, "species": "sprat", "bin": "1", "value": 9e9 if early else 5e7},
            {"time": t, "species": "sprat", "bin": "0", "value": 1e11},  # decoy wrong age
            {"time": t, "species": "herring", "bin": "0", "value": 9e9 if early else 4e7},
        ]
    abd = pd.DataFrame(rows)
    R, Engine = _fake_results(bio, yld, abd)
    _wire(monkeypatch, R, Engine)

    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0,
                              recruitment_ages={"sprat": "1", "herring": "0"})
    assert stats["sprat_recruitment_mean"] == pytest.approx(5e7)   # last-10 window, not full mean
    assert stats["herring_recruitment_mean"] == pytest.approx(4e7)


def test_missing_bin_for_one_species(monkeypatch):
    # Non-empty, correctly-columned frame but with NO herring rows: exercises the per-species
    # no-matching-(species,bin) path (distinct from the bare-empty-frame guard) — herring stat
    # stays unset while sprat is still emitted.
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0), "herring": np.full(12, 2000.0)})
    yld = pd.DataFrame({"sprat": np.full(12, 100.0), "herring": np.full(12, 200.0)})
    abd = pd.DataFrame([{"time": t, "species": "sprat", "bin": "1", "value": 5e7} for t in range(12)])
    R, Engine = _fake_results(bio, yld, abd)
    _wire(monkeypatch, R, Engine)
    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0,
                              recruitment_ages={"sprat": "1", "herring": "0"})
    assert stats["sprat_recruitment_mean"] == pytest.approx(5e7)
    assert "herring_recruitment_mean" not in stats  # no matching rows -> unset, no crash


def test_no_recruitment_ages_emits_nothing(monkeypatch):
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0)})
    R, Engine = _fake_results(bio, pd.DataFrame({"sprat": np.full(12, 100.0)}), pd.DataFrame())
    _wire(monkeypatch, R, Engine)
    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0)  # recruitment_ages default None
    assert not any(k.endswith("_recruitment_mean") for k in stats)


def test_empty_abundance_frame_guarded(monkeypatch):
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0)})
    # abundance-by-age off -> bare empty frame (no columns); must not KeyError.
    R, Engine = _fake_results(bio, pd.DataFrame({"sprat": np.full(12, 100.0)}), pd.DataFrame())
    _wire(monkeypatch, R, Engine)
    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0, recruitment_ages={"sprat": "1"})
    assert "sprat_recruitment_mean" not in stats  # gracefully unset, no crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_recruitment_stat.py -q`
Expected: FAIL — `TypeError: run_simulation() got an unexpected keyword argument 'recruitment_ages'`.

- [ ] **Step 3: Add the param + gated extraction**

In `scripts/calibrate_baltic.py::run_simulation`, add the parameter to the signature (after `timeout_s`):

```python
    timeout_s: float | None = None,
    recruitment_ages: dict[str, str] | None = None,
```

In the results-reading block, read abundance-by-age BEFORE `results.close()` (only when needed). Change:

```python
        results = OsmoseResults(output_dir, strict=False)
        bio = results.biomass()
        try:
            yld = results.yield_biomass()
        except Exception:  # noqa: BLE001 — yield CSV absent/empty: leave yield stats unset
            yld = None
        results.close()
```

to:

```python
        results = OsmoseResults(output_dir, strict=False)
        bio = results.biomass()
        try:
            yld = results.yield_biomass()
        except Exception:  # noqa: BLE001 — yield CSV absent/empty: leave yield stats unset
            yld = None
        abd = None
        if recruitment_ages:
            try:
                abd = results.abundance_by_age()
            except Exception:  # noqa: BLE001 — abundance-by-age absent: leave recruitment unset
                abd = None
        results.close()
```

Then, immediately before `return species_stats`, add the gated recruitment extraction:

```python
    # Recruitment stat (diagnostic only; DE loop passes recruitment_ages=None -> skipped).
    # abundance_by_age() is LONG [time, species, bin, value] with string bins; mean the
    # ICES recruitment-age bin over the same trailing window as {sp}_mean.
    if recruitment_ages and abd is not None and not abd.empty and "bin" in abd.columns:
        for sp, age in recruitment_ages.items():
            sub = abd[(abd["species"] == sp) & (abd["bin"].astype(str) == str(age))]
            if not sub.empty:
                rvals = sub.sort_values("time")["value"].to_numpy(dtype=float)
                reval = rvals[-n_eval_years:] if len(rvals) > n_eval_years else rvals
                if reval.size > 0:
                    species_stats[f"{sp}_recruitment_mean"] = float(np.mean(reval))

    return species_stats
```

(`n_eval_years` is the existing `= 10` local above the species loop.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_recruitment_stat.py -q`
Expected: 3 passed.

- [ ] **Step 5: Confirm the DE path is unaffected**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_yield_stat.py tests/ -q -k "calibrat or objective or yield" 2>&1 | tail -8`
Expected: green — `run_simulation`'s existing callers (DE loop, yield stat) pass `recruitment_ages=None` by default and are unchanged.

- [ ] **Step 6: Lint**

Run: `.venv/bin/python -m ruff check scripts/calibrate_baltic.py tests/test_calibrate_baltic_recruitment_stat.py && .venv/bin/python -m ruff format --check scripts/calibrate_baltic.py tests/test_calibrate_baltic_recruitment_stat.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_recruitment_stat.py
git commit -m "feat(calibration): gated {sp}_recruitment_mean stat in run_simulation"
```

---

## Task 3: Wire `evaluate()` + render the report section

**Files:**
- Modify: `scripts/evaluate_calibration_vs_ices.py` (`evaluate`, `_print_report`, + `_format_recruitment_section`)
- Test: `tests/test_recruitment_diagnostic.py` (append)

**Interfaces:**
- Consumes: `_species_recruitment_age`, `_ices_recruitment_geomean`, `_recruitment_verdict` (Task 1); `run_simulation(..., recruitment_ages=)` + `{sp}_recruitment_mean` (Task 2).
- Produces: `_format_recruitment_section(rows: list[dict]) -> str`; `evaluate()` result dict gains a `"recruitment"` key (list of row dicts).

- [ ] **Step 1: Write the failing tests (append)**

Append to `tests/test_recruitment_diagnostic.py`:

```python
def test_format_recruitment_section_is_pure():
    from evaluate_calibration_vs_ices import _format_recruitment_section  # scripts/ on path (top of file)

    rows = [
        {"species": "sprat", "age": "1", "model_R": 6.0e7, "ices_geomean": 7.0e7,
         "ices_min": 2.4e7, "ices_max": 1.1e8, "ratio": 0.86, "verdict": "OK", "reason": None},
        {"species": "herring", "age": "0", "model_R": 2.0e7, "ices_geomean": 4.5e7,
         "ices_min": 2.7e7, "ices_max": 7.3e7, "ratio": 0.44, "verdict": "OK", "reason": None},
        {"species": "cod", "age": None, "model_R": None, "ices_geomean": None,
         "ices_min": None, "ices_max": None, "ratio": None, "verdict": None,
         "reason": "no clean ICES R (eastern index + age mismatch 0 vs 1)"},
        {"species": "flounder", "age": None, "model_R": None, "ices_geomean": None,
         "ices_min": None, "ices_max": None, "ratio": None, "verdict": None,
         "reason": "no clean ICES R (none reported)"},
    ]
    out = _format_recruitment_section(rows)
    assert "Recruitment" in out
    assert "sprat" in out and "0.86" in out
    assert "no clean ICES R" in out
    assert "age-0" in out.lower()  # the herring caveat text


def test_evaluate_adds_recruitment_rows(monkeypatch):
    import evaluate_calibration_vs_ices as ev  # scripts/ on path (top of file)

    # Stub the sim so no engine runs; return biomass + recruitment stats.
    def _fake_run(base_config, overrides, n_years, seed, recruitment_ages=None):
        assert base_config.get("output.abundance.byage.enabled") == "true"
        assert recruitment_ages == {"sprat": "1", "herring": "0"}
        stats = {f"{sp}_mean": 1000.0 for sp in ev.SPECIES_NAMES}
        stats["sprat_recruitment_mean"] = 6.0e7
        stats["herring_recruitment_mean"] = 2.0e7
        return stats

    monkeypatch.setattr(ev, "run_simulation", _fake_run)
    # Minimal params file
    import json
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkstemp(suffix=".json")[1])
    p.write_text(json.dumps({"parameters": {}}))
    result = ev.evaluate(p, mode="bh", n_years=1, seed=0)
    rec = {r["species"]: r for r in result["recruitment"]}
    assert set(rec) == {"cod", "herring", "sprat", "flounder"}
    assert rec["sprat"]["verdict"] in ("OK", "FLAG") and rec["sprat"]["ices_geomean"] is not None
    assert rec["cod"]["ices_geomean"] is None and "no clean ICES R" in rec["cod"]["reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recruitment_diagnostic.py -k "format or evaluate_adds" -q`
Expected: FAIL — `ImportError`/`AttributeError` (`_format_recruitment_section` missing; `result` has no `"recruitment"` key).

- [ ] **Step 3: Add `_format_recruitment_section`**

In `scripts/evaluate_calibration_vs_ices.py`, add:

```python
def _format_recruitment_section(rows: list[dict]) -> str:
    """Pure formatter for the recruitment table (never runs the engine)."""
    lines = ["\nRecruitment (model vs ICES R geomean, 2018-2022)"]
    lines.append(
        f"  {'species':10s} {'age':>3s} {'model_R':>14s} "
        f"{'ICES_geomean [min-max]':>34s} {'ratio':>7s}  verdict"
    )
    for r in rows:
        if r.get("ices_geomean") is None:
            lines.append(
                f"  {r['species']:10s} {'—':>3s} {'—':>14s} {'—':>34s} {'—':>7s}  {r['reason']}"
            )
        else:
            ref = f"{r['ices_geomean']:,.0f} [{r['ices_min']:,.0f}-{r['ices_max']:,.0f}]"
            model = f"{r['model_R']:,.0f}" if r["model_R"] is not None else "—"
            ratio = f"{r['ratio']:.2f}x" if r["ratio"] is not None else "—"
            note = "  (age-0: model reads ~0.4-0.6x low; see note)" if r["age"] == "0" else ""
            lines.append(
                f"  {r['species']:10s} {r['age']:>3s} {model:>14s} {ref:>34s} "
                f"{ratio:>7s}  {r['verdict']}{note}"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: Wire `evaluate()`**

In `evaluate()`, BEFORE the `run_simulation(...)` call, enable abundance-by-age and resolve the ages:

```python
    base_config["output.abundance.byage.enabled"] = "true"
    recruitment_ages = {
        sp: age
        for sp in RECRUITMENT_ASSESSED
        if (age := _species_recruitment_age(sp)) is not None
    }
    stats = run_simulation(
        base_config, overrides, n_years=n_years, seed=seed, recruitment_ages=recruitment_ages
    )
```

(Replace the existing `stats = run_simulation(base_config, overrides, n_years=n_years, seed=seed)` line.)

After the biomass `rows` loop, build the recruitment rows and add them to the returned dict:

```python
    recruitment = []
    for sp in RECRUITMENT_ASSESSED:
        age = _species_recruitment_age(sp)
        geo = _ices_recruitment_geomean(sp) if age is not None else None
        if age is None or geo is None:
            reason = (
                "no clean ICES R (eastern index + age mismatch 0 vs 1)"
                if sp == "cod"
                else "no clean ICES R (none reported)"
            )
            recruitment.append({
                "species": sp, "age": None, "model_R": None, "ices_geomean": None,
                "ices_min": None, "ices_max": None, "ratio": None, "verdict": None,
                "reason": reason,
            })
            continue
        geomean, gmin, gmax = geo
        model_R = stats.get(f"{sp}_recruitment_mean")
        if model_R is None:
            ratio, verdict = None, None
        else:
            ratio, verdict = _recruitment_verdict(model_R, geomean)
        recruitment.append({
            "species": sp, "age": age, "model_R": model_R, "ices_geomean": geomean,
            "ices_min": gmin, "ices_max": gmax, "ratio": ratio, "verdict": verdict,
            "reason": None,
        })
```

Then add `"recruitment": recruitment,` to the returned dict (alongside `"species": rows`).

Finally, in `_print_report(result)`, after the biomass table's final print, add:

```python
    print(_format_recruitment_section(result.get("recruitment", [])))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recruitment_diagnostic.py -q`
Expected: all tests PASS (7 total).

- [ ] **Step 6: Lint + import-sanity**

Run: `.venv/bin/python -m ruff check scripts/evaluate_calibration_vs_ices.py tests/test_recruitment_diagnostic.py && .venv/bin/python -m ruff format --check scripts/evaluate_calibration_vs_ices.py && PYTHONPATH=scripts .venv/bin/python -c "import evaluate_calibration_vs_ices"`
Expected: clean; the import exits 0 (scripts/ on path via PYTHONPATH; no circular import from reusing `validate_baltic_vs_ices_sag`).

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_calibration_vs_ices.py tests/test_recruitment_diagnostic.py
git commit -m "feat(diagnostic): render recruitment section in evaluate_calibration_vs_ices"
```

---

## Verification (whole-branch, after all tasks)

- [ ] Tests: `.venv/bin/python -m pytest tests/test_recruitment_diagnostic.py tests/test_calibrate_baltic_recruitment_stat.py tests/ -q -k "recruit or calibrat or objective or yield or ices" 2>&1 | tail -15` — green.
- [ ] Lint/format: `.venv/bin/python -m ruff check osmose/ scripts/ tests/ && .venv/bin/python -m ruff format --check osmose/ tests/` — clean (CI scope is `osmose/ ui/ tests/`; pre-existing `scripts/` drift is out of scope).
- [ ] DE-loop unaffected: the calibration objective tests stay green (no `recruitment_ages` passed by `_ObjectiveWrapper`).
- [ ] Manual smoke (optional, NOT CI — real Baltic run): `.venv/bin/python scripts/evaluate_calibration_vs_ices.py --mode bh --params <a calibration result JSON> --years 15` prints the biomass table AND a Recruitment section with sprat/herring ratios + the two "no clean ICES R" rows; sprat/herring ICES geomeans are tens of millions.

---

## Self-Review

**1. Spec coverage:**
- §1 model recruitment in `run_simulation` (gated `recruitment_ages`, long-frame `bin==str(age)`, empty guard) → Task 2. ✅
- §2 ICES geomean + `_species_recruitment_age` + reuse of `validate_baltic_vs_ices_sag` helpers → Task 1. ✅
- §3 report (age column, geomean+min/max, age-0 caveat, cod/flounder reasons, verdict [1/3,3], JSON key) → Task 3. ✅
- §4 boundaries (`_species_recruitment_age`/`_ices_recruitment_geomean`/`_recruitment_verdict`/`_format_recruitment_section`; ages passed into `run_simulation`) → Tasks 1–3. ✅
- Non-goals (no loss/optimizer/CSV change; DE byte-identical) → Task 2 gating + Verification. ✅
- Testing (geomean, run_simulation stat + empty-frame + None, verdict boundaries, pure formatter no-engine, evaluate wiring) → Tasks 1–3. ✅

**2. Placeholder scan:** No TBD/TODO; every code step is complete; the manual smoke names a real command. The `<...>` in the smoke step is a user-supplied params path, not a code placeholder. ✅

**3. Type consistency:** `_species_recruitment_age(str)->str|None`, `_ices_recruitment_geomean(str)->tuple[float,float,float]|None`, `_recruitment_verdict(float,float)->tuple[float,str]`, `_format_recruitment_section(list[dict])->str`, `run_simulation(..., recruitment_ages: dict[str,str]|None=None)`, stat key `{sp}_recruitment_mean` — consistent across Tasks 1–3 and match the spec + the current `run_simulation`/`evaluate` signatures. ✅
