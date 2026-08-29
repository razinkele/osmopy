# Baltic B2 Literature-Delta Scenarios — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the four RCP×load scenario arms (Meier 2022 Table 7/10 deltas) through the C1
knob and the O₂→benthos-K coupling, with the spec's four blocking wiring checks and the 2×2
results table.

**Architecture:** No engine changes. One cited delta-spec JSON; a builder that emits per-arm
forcing (knob series + offset O₂ NetCDF) with deterministic self-checks; a 6-arm harness
modeled on `scripts/baltic_c1_knob_ab.py`; one engine run + results doc.

**Tech Stack:** Python 3.12, NumPy, xarray, pytest. Always `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-29-baltic-b2-literature-delta-scenarios-design.md`
— decisions 1–7 and §4's blocking/reported split are pre-registered and NOT tunable.

## Global Constraints

- `.venv/bin/python` always; ruff (line 100) clean on touched files.
- EXISTING `data/baltic/` files byte-identical; this stage ADDS `data/baltic/scenarios/`
  (one JSON) and docs artifacts. Generated forcing lives in run dirs, never committed.
- The tree carries the USER'S unrelated uncommitted changes (osmose/runner.py, osmose/cli.py,
  movement_maps, three test files, .mcp.json, mcp_servers/) — stage ONLY each task's explicit
  file list; never `git add tests/`, `data/`, `-A`, or `commit -a`.
- Shell rules: no `$()`, no heredocs containing `#`, no `>` redirection, no `cd&&git`;
  multi-line checks via Write to `/tmp/*.py`.
- Engine runs: Task 4 only (~2.5–3 h); `uptime` first; never concurrent engine jobs.
- Known traps that bind here: pandas CSV loaders need `float_precision="round_trip"` (already
  in the gate loaders — do NOT add new pandas CSV parsing without it); spatial means over
  Baltic bboxes need nanmean + non-finite guards; the O₂ file MUST stay 24 frames
  (CLAUDE.md trap); expected knob factors MUST go through the loader float path
  (`exp(beta*(float(str(tref+dT))-tref))` — plain `exp(beta*dT)` fails by 3 ULP at ΔT=2.9).
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Delta-spec JSON + schema validation test

**Files:**
- Create: `data/baltic/scenarios/b2_literature_deltas.json`
- Create: `tests/test_b2_delta_spec.py`

**Interfaces:**
- Produces: the JSON below VERBATIM (values are review-verified against Meier 2022 Tables
  7/10 — do not alter any number), and a validation module-level loader the builder/harness
  reuse: the test file defines `load_delta_spec(path) -> dict` inline? No — validation logic
  lives in the TEST only this task; Task 2's builder re-reads the JSON with its own minimal
  checks. Arm names `rcp45_bsap`, `rcp45_ref`, `rcp85_bsap`, `rcp85_ref` are the contract
  Tasks 2–4 use.

- [ ] **Step 1: Write the JSON exactly** (Write tool):

```json
{
  "_provenance": "B2 spec 2026-08-29 (docs/superpowers/specs/2026-08-29-baltic-b2-literature-delta-scenarios-design.md). All deltas are CLIMSEA (RCO-SCOBI) ensemble means, 1976-2005 -> 2069-2098, Meier et al. 2022 (doi:10.5194/esd-13-159-2022). Applied RAW on present-day baselines (O2 file: 2024 analysis; knob tref: 1993-2021 mean) -- every arm overstates end-century forcing by the realized 1976-2005->present component (spec decision 5).",
  "reference_periods": {"literature": "1976-2005 -> 2069-2098", "o2_baseline": "2024 CMEMS analysis", "tref_baseline": "1993-2021"},
  "arms": [
    {"name": "rcp45_bsap", "rcp": "RCP4.5", "load": "BSAP",
     "dT_C": 1.9, "dT_source": "Meier2022 Table 7, CLIMSEA RCP4.5, annual mean SST",
     "dO2": {"value_mmol_m3": 26.8, "value_mL_L": 0.6, "referent": "summer_bottom_o2",
             "conversion_mmol_per_mL_L": 44.66,
             "source": "Meier2022 Table 10, CLIMSEA RCP4.5 BSAP, ensemble-mean-SLR"}},
    {"name": "rcp45_ref", "rcp": "RCP4.5", "load": "REF",
     "dT_C": 1.9, "dT_source": "Meier2022 Table 7, CLIMSEA RCP4.5, annual mean SST",
     "dO2": {"value_mmol_m3": 0.0, "value_mL_L": 0.0, "referent": "summer_bottom_o2",
             "conversion_mmol_per_mL_L": 44.66,
             "source": "Meier2022 Table 10, CLIMSEA RCP4.5 REF, ensemble-mean-SLR (sourced zero -- designed null O2 contrast)"}},
    {"name": "rcp85_bsap", "rcp": "RCP8.5", "load": "BSAP",
     "dT_C": 2.9, "dT_source": "Meier2022 Table 7, CLIMSEA RCP8.5, annual mean SST",
     "dO2": {"value_mmol_m3": 17.9, "value_mL_L": 0.4, "referent": "summer_bottom_o2",
             "conversion_mmol_per_mL_L": 44.66,
             "source": "Meier2022 Table 10, CLIMSEA RCP8.5 BSAP, ensemble-mean-SLR"}},
    {"name": "rcp85_ref", "rcp": "RCP8.5", "load": "REF",
     "dT_C": 2.9, "dT_source": "Meier2022 Table 7, CLIMSEA RCP8.5, annual mean SST",
     "dO2": {"value_mmol_m3": -8.9, "value_mL_L": -0.2, "referent": "summer_bottom_o2",
             "conversion_mmol_per_mL_L": 44.66,
             "source": "Meier2022 Table 10, CLIMSEA RCP8.5 REF, ensemble-mean-SLR"}}
  ]
}
```

- [ ] **Step 2: Write the failing validation test** `tests/test_b2_delta_spec.py`:

```python
"""B2 delta-spec schema validation (spec §1): every number cited, one referent,
no dead knobs, conversions self-consistent."""

import json
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent.parent / "data/baltic/scenarios/b2_literature_deltas.json"


def _spec():
    return json.loads(SPEC_PATH.read_text())


def test_arms_and_matrix():
    arms = _spec()["arms"]
    assert [a["name"] for a in arms] == ["rcp45_bsap", "rcp45_ref", "rcp85_bsap", "rcp85_ref"]
    assert all(a["dT_C"] in (1.9, 2.9) for a in arms)
    assert {(a["rcp"], a["load"]) for a in arms} == {
        ("RCP4.5", "BSAP"), ("RCP4.5", "REF"), ("RCP8.5", "BSAP"), ("RCP8.5", "REF")
    }


def test_every_number_cited_and_single_referent():
    for a in _spec()["arms"]:
        assert "Meier2022 Table 7" in a["dT_source"]
        d = a["dO2"]
        assert d["referent"] == "summer_bottom_o2"  # spec decision 4: the only accepted referent
        assert "Meier2022 Table 10" in d["source"]


def test_conversion_self_consistent():
    for a in _spec()["arms"]:
        d = a["dO2"]
        assert abs(d["value_mmol_m3"] - d["value_mL_L"] * d["conversion_mmol_per_mL_L"]) < 0.05


def test_no_dead_knobs():
    for a in _spec()["arms"]:
        for dead in ("ltl_scale", "salinity", "time_slice"):
            assert dead not in a  # spec §1: JSON carries only what machinery consumes


def test_provenance_records_relabel():
    assert "overstates end-century forcing" in _spec()["_provenance"]  # decision 5
```

- [ ] **Step 3:** Run `.venv/bin/python -m pytest tests/test_b2_delta_spec.py -v` — with the
  JSON already written these PASS immediately (this is data validation, not TDD of code; the
  RED phase is skipped by design — say so in the report).
- [ ] **Step 4: Lint + commit**

```bash
.venv/bin/ruff check tests/test_b2_delta_spec.py
git add data/baltic/scenarios/b2_literature_deltas.json tests/test_b2_delta_spec.py
git commit -m "data(baltic): B2 delta spec — Meier 2022 Table 7/10, all cells cited

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Builder `scripts/build_baltic_b2_forcing.py` + tests

**Files:**
- Create: `scripts/build_baltic_b2_forcing.py`, `tests/test_build_baltic_b2_forcing.py`

**Interfaces:**
- Produces (pure, unit-tested): `offset_o2(o2: np.ndarray, wet: np.ndarray, delta: float) ->
  np.ndarray` (adds delta on wet cells only, floors at 0, preserves non-wet values exactly;
  raises on non-finite results in wet cells); `predicted_k_change(o2, wet, k_weights, delta,
  c50=60.0, n=3.0) -> float` (K-weighted mean Hill factor with offset ÷ without, minus 1);
  `write_arm_dir(arm: dict, out_dir: Path, prod_o2_path, grid_path, trefs, betas) -> dict`
  returning `{"series_csv": path, "o2_nc": path|None, "predicted_dK": float}`. The knob series
  reuses C1's conventions by IMPORTING `write_arm_series` from `scripts/baltic_c1_knob_ab.py`
  (importlib-from-path, the established scripts/ idiom). `main()` runs all arms + the
  zero-delta self-check. Task 3 consumes `write_arm_dir`'s dict per arm.
- Wet mask: from `data/baltic/baltic_grid.nc`'s mask variable (read it first —
  `grep -n "mask" osmose/engine/config.py` shows how the engine reads it; 616 ocean cells).
  Hill: import `f_o2_hill` from `osmose/engine/processes/oxygen_function.py` (real function,
  not a reimplementation).

- [ ] **Step 1: Failing tests** (synthetic fixtures; no real NetCDF I/O except tiny temp files
  written by the test via xarray):

```python
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "build_baltic_b2_forcing",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_b2_forcing.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _field():
    o2 = np.full((24, 4, 4), 200.0)
    o2[:, 0, 0] = 30.0     # hypoxic wet cell
    o2[:, 3, 3] = np.nan   # land
    wet = np.ones((4, 4), dtype=bool)
    wet[3, 3] = False
    return o2, wet


def test_offset_wet_only_and_floor():
    o2, wet = _field()
    out = m.offset_o2(o2, wet, -50.0)
    assert out[0, 0, 0] == 0.0                      # floored, not negative
    assert out[0, 1, 1] == 150.0
    assert np.isnan(out[0, 3, 3])                   # land untouched
    out2 = m.offset_o2(o2, wet, 26.8)
    assert out2[0, 0, 0] == 30.0 + 26.8
    assert np.isnan(out2[0, 3, 3])


def test_offset_zero_is_identity():
    o2, wet = _field()
    out = m.offset_o2(o2, wet, 0.0)
    assert np.array_equal(out[:, wet], o2[:, wet])  # exact, bit-level on wet cells


def test_predicted_k_change_signs_and_zero():
    o2, wet = _field()
    k = np.ones((4, 4))
    assert m.predicted_k_change(o2, wet, k, 0.0) == 0.0
    assert m.predicted_k_change(o2, wet, k, 26.8) > 0.0
    assert m.predicted_k_change(o2, wet, k, -8.9) < 0.0


def test_predicted_k_change_uses_real_hill():
    o2, wet = _field()
    k = np.zeros((4, 4)); k[0, 0] = 1.0             # all weight on the hypoxic cell
    from osmose.engine.processes.oxygen_function import f_o2_hill
    expect = f_o2_hill(np.array([56.8]), 60.0, 3.0)[0] / f_o2_hill(np.array([30.0]), 60.0, 3.0)[0] - 1.0
    got = m.predicted_k_change(o2, wet, k, 26.8)
    assert abs(got - expect) < 1e-12
```

  (If `f_o2_hill`'s signature differs — check it FIRST — adapt the expectation line to the
  real signature and say so in the report.)

- [ ] **Step 2:** verify FAIL at import; **Step 3:** implement (pure functions + `write_arm_dir`
  + `main()`; the O₂ copy preserves dims/coords/attrs/dtype via xarray, asserts 24 frames in
  and out; the zero self-check re-reads the written zero-arm file and compares NaN-aware);
  **Step 4:** tests PASS + ruff; **Step 5:** commit
  (`git add scripts/build_baltic_b2_forcing.py tests/test_build_baltic_b2_forcing.py`).

---

### Task 3: Harness `scripts/baltic_b2_scenario_ab.py` + tests

**Files:**
- Create: `scripts/baltic_b2_scenario_ab.py`, `tests/test_baltic_b2_harness_helpers.py`

**Interfaces:**
- Produces: `expected_knob_factor(beta, tref, dT) -> float` = `exp(beta * (float(str(tref + dT))
  - tref))` (the loader float path — spec §4(d)); `arm_overlays(arm_name, artifacts, trefs,
  betas) -> dict` (C1 knob keys + `oxygen.filename` absolute path when the arm has an O₂ file);
  `hill_ordering_ok(arm_o2, base_o2, wet, delta_sign) -> bool` (§4(c), per-cell, uses
  `f_o2_hill`); `run_b2(seeds=(42, 123, 7, 999, 2024)) -> dict`. Arms: baseline, zero,
  + the four from the delta spec. Constants: TREFS/BETAS/ENABLED imported from
  `scripts/baltic_c1_knob_ab.py` (herring-only). The O₂ load-through assert (§4(b)): call the
  engine's oxygen loader directly on each arm's assembled cfg (find it:
  `grep -n "_load_oxygen_data" osmose/engine/simulate.py` — read its signature and return; if
  it is not importable/callable standalone, fall back to re-reading the arm's NetCDF and
  asserting against the builder's array, and report which route you took) and assert equality
  with the builder's written array. Results JSON written to BOTH `/tmp/b2_scenario_report.json`
  and (by Task 4's commit) `docs/diagnostics/baltic_b2_scenario_report.json`.

- [ ] **Step 1: Failing tests** — cover: `expected_knob_factor` non-dyadic case
  (`beta=-0.51, tref=9.670314810741907, dT=2.9` — assert it does NOT equal
  `np.exp(-0.51*2.9)` AND matches the value computed through `float(str(...))`; this pins the
  3-ULP trap), the dyadic identity (`dT=0.0` → exactly 1.0); `arm_overlays` (baseline has no
  knob/oxygen keys; zero + scenario arms carry the C1 keys with full-precision tref strings;
  oxygen.filename present only when the arm has an O₂ artifact); `hill_ordering_ok` on the
  synthetic field of Task 2 (positive delta → factors ≥ baseline everywhere wet, negative →
  ≤, zero → equal). Write the tests concretely following Task 2's importlib idiom.
- [ ] **Step 2:** FAIL at import; **Step 3:** implement `run_b2` on the C1 `run_ab` pattern
  (osmose_demo → reader → per-arm overlays → `PythonEngine().run_in_memory(raw, seed)`;
  builder invoked once into a temp run dir; BLOCKING order: builder zero-check → §4(b)
  load-through per arm → §4(c) Hill ordering per arm → §4(d) knob instrument per arm → engine
  runs → §4(a) zero-arm bit-identity → report). **Step 4:** tests PASS + ruff; **Step 5:**
  commit (`git add scripts/baltic_b2_scenario_ab.py tests/test_baltic_b2_harness_helpers.py`).

---

### Task 4: The run + results doc

**Files:**
- Create: `docs/baltic_b2_scenarios_2026-MM-DD.md` (run date),
  `docs/diagnostics/baltic_b2_scenario_report.json`

- [ ] **Step 1:** `uptime` low; no concurrent engine jobs. Run
  `.venv/bin/python scripts/baltic_b2_scenario_ab.py` via Bash `run_in_background`
  (6 arms × 5 seeds × 50 yr ≈ 2.5–3 h).
- [ ] **Step 2: Gates in order** (any BLOCKING failure = stop, debug, no interpretation):
  builder zero-check; §4(b) load-through; §4(c) Hill ordering; §4(d) knob instrument;
  §4(a) zero-arm bit-identity.
- [ ] **Step 3: Results doc**: the 2×2 table (per assessed stock: final-decade mean per arm,
  delta vs baseline) with the predicted-ΔK column and the ±1.9 % noise-floor column; herring
  declines as reported context; the within-RCP load contrasts for cod_east/flounder; EVERY §4
  label restated (SST proxy direction, summer-only+SLR, uniform-offset+floor asymmetry,
  LTL-at-baseline partial-load world, decision-5 relabel with the realized-component context
  estimate cited, cod_east RV-prescription); load-dominance caveat as the FIRST sentence of
  the O₂ commentary; provenance (arms/seeds/commits, NOT a CI gate); the upgrade-path
  sentence.
- [ ] **Step 4:** Copy the report JSON to docs/diagnostics/; commit exactly:
  `git add docs/baltic_b2_scenarios_2026-*.md docs/diagnostics/baltic_b2_scenario_report.json`

---

## Execution notes

- Tasks 1–3 are CI-safe and fast; Task 4 is the only engine run.
- Precedent files to read before implementing (not paste): `scripts/baltic_c1_knob_ab.py`
  (arm/overlay/identity pattern, constants), `scripts/build_baltic_b2_forcing.py`'s own Task-2
  tests, `osmose/engine/processes/oxygen_function.py` (Hill signature),
  `osmose/engine/simulate.py:_load_oxygen_data` (load-through route).
- The spec is the authority; contradictions STOP the task and surface to the controller.
