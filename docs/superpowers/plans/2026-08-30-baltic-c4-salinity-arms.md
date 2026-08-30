# Baltic C4 Salinity Sensitivity Arms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the five salinity-sensitivity arms (ΔS = 0/−1/−2/−3 + baseline) through the
production salinity gate with sampler-aware instruments and the B2 gate discipline, answering
whether the July coastal regime-shift chain reproduces on the certified config.

**Architecture:** No engine changes. A rationale-carrying delta-spec JSON; a builder cloning
B2's offset pattern for the salinity NetCDF (NaN land convention, mask-AND-finite wet rule)
plus four pre-run instruments computed with engine-loaded maps; a 6-arm harness cloning B2's
with two extra guards (frame count, `.constant` absence); one run + results doc.

**Tech Stack:** Python 3.12, NumPy, xarray, pytest. Always `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-30-baltic-c4-salinity-sensitivity-arms-design.md` —
decisions 1–6 pre-registered; the stated-expectations table is part of the record (do not
re-derive it silently; the builder's printed instruments should REPRODUCE it).

## Global Constraints

- `.venv/bin/python`; ruff clean. EXISTING `data/baltic/` files byte-identical; ADD only
  `data/baltic/scenarios/c4_salinity_sensitivity.json` + docs artifacts; generated fields live
  in run dirs.
- USER-dirty files (runner.py, cli.py, movement_maps.py, .mcp.json, mcp_servers/, 3 test
  files) — explicit `git add` lists only. NOTE movement_maps.py is user-dirty: the builder
  IMPORTS from it (read-only) but no task may EDIT it; if the map-loader import proves
  impossible without edits, that is BLOCKED-and-surface, not a patch.
- Shell rules: no `$()`, no heredocs with `#`, no `>` redirection, no `cd&&git`; `/tmp/*.py`
  via Write.
- Salinity-file facts (review-verified, binding): land is **NaN** (never 0.0 — opposite of the
  O₂ file); 24 frames; float64; wet rule = grid.nc `mask > 0` **AND** finite; 3 finite
  off-mask cells excluded by the AND; floor = stored salinity ≥ 0 (NaN-propagating).
- Map facts (binding): raw CSVs are upside-down vs the field — obtain grids via the engine's
  own loader (`grep -n "_load_csv_grid\|def .*load" osmose/engine/movement_maps.py` — find the
  loader `MovementMapSet` uses and import THAT); CI test pins orientation (zero map-positive
  cells on land per grid mask).
- Engine runs: Task 4 only (~1.6 h); `uptime` first; never concurrent engine jobs.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Delta-spec JSON + validation test

**Files:**
- Create: `data/baltic/scenarios/c4_salinity_sensitivity.json`, `tests/test_c4_delta_spec.py`

- [ ] **Step 1: Write the JSON exactly** (values/rationales are spec-mandated):

```json
{
  "_provenance": "C4 spec 2026-08-30 (docs/superpowers/specs/2026-08-30-baltic-c4-salinity-sensitivity-arms-design.md). MECHANISM-CHARACTERIZATION arms, not projections: every ensemble generation's mean salinity change is ~0 (Meier et al. 2022, doi:10.5194/esd-13-159-2022, Table 8: BalticAPP -0.06, ECOSUPPORT -0.15, CLIMSEA ~0 g/kg SSS); only first-generation extremes reached 'decreases of as much as 45%' (Meier 2006, cited therein). The dS levers are CHOSEN, not cited.",
  "context_citations": {
    "modern_consensus": "Meier et al. 2022 Table 8 + Sect. 3.2.4: 'salinity changes are not robust; i.e. the ensemble spread is larger than the signal'",
    "first_generation_extreme": "Meier 2006 via Meier et al. 2022: 'decreases of as much as 45%'"
  },
  "arms": [
    {"name": "ds_m1", "dS_PSU": -1.0,
     "rationale": "sub-ramp-width lever: characterizes redistribution transmission where exclusions cannot fire (expected TV~0.03, exclusions ~0 on cod_east)"},
    {"name": "ds_m2", "dS_PSU": -2.0,
     "rationale": "two-thirds-ramp lever: expected TV~0.10, exclusions <=0.23% on cod_east"},
    {"name": "ds_m3", "dS_PSU": -3.0,
     "rationale": "full-ramp exclusion-regime lever: baseline cells below 6 PSU reach w=0; approaches the July OFF->ON flip from below; all-zero (map,frame) events reported by the builder"}
  ]
}
```

- [ ] **Step 2: Validation test** `tests/test_c4_delta_spec.py` (mirror
  `tests/test_b2_delta_spec.py`'s structure): arms named `ds_m1/ds_m2/ds_m3` with
  `dS_PSU == -1.0/-2.0/-3.0`; every arm has a non-empty `rationale` and NO `citation` field
  (levers are chosen, not cited — assert absence); both `context_citations` entries contain
  "Meier"; `_provenance` contains "not projections"; no dead knobs (`dT_C`, `dO2`,
  `referent` absent from every arm).
- [ ] **Step 3:** Run (`pytest tests/test_c4_delta_spec.py -v`, ~5 tests PASS — data
  validation, RED skipped by design; say so in the report). Ruff. Commit:
  `git add data/baltic/scenarios/c4_salinity_sensitivity.json tests/test_c4_delta_spec.py`

---

### Task 2: Builder `scripts/build_baltic_c4_forcing.py` + tests

**Files:**
- Create: `scripts/build_baltic_c4_forcing.py`, `tests/test_build_baltic_c4_forcing.py`

**Interfaces:**
- Produces: `offset_salinity(sal: np.ndarray, wet: np.ndarray, dS: float) -> np.ndarray`
  (additive on wet cells, floor at 0.0, NaN cells untouched/propagated, raises on non-finite
  results in wet cells); `ramp_w(sal, s_low=3.0, s_high=6.0) -> np.ndarray` (the production
  ramp, NaN-safe); instrument functions per spec decision 3:
  `tv_distance(map_grid, w_base, w_arm) -> float` (per-frame TV between normalized map·w
  distributions, returned as the 24-frame mean; TV = 0.5·Σ|p−q| over cells with map>0 after
  each is normalized to sum 1; if an arm's map·w sums to 0 for a frame — the all-zero case —
  return `nan` for that frame and record the event),
  `prey_overlap_shift(map_grid, w_base, w_arm, prey_map) -> float` (change in normalized cod
  occupancy mass over prey_map>0 cells, 24-frame mean),
  `excluded_fraction(map_grid, w_base, w_arm) -> float` (map-cell fraction with
  w_base>0 & w_arm==0), `mean_dw(map_grid, w_base, w_arm) -> float` (wiring-only);
  `write_arm_dir(arm, out_dir, prod_sal_path, grid_path) -> {"sal_nc": path, "instruments": {...}, "all_zero_events": [...]}`;
  `main()` runs all arms + zero self-check and PRINTS the instrument table (which must
  reproduce the spec's stated-expectations table to its stated precision — a mismatch is a
  STOP-and-report, not a silent recalibration).
- Maps: load via the engine's own loader (see Global Constraints), for the gated species'
  maps (cod_west + cod_east, all stages listed in `data/baltic/baltic_param-movement.csv`)
  and the prey species' maps (stickleback, perch, pikeperch, smelt).
- Wet mask: `grid.nc mask > 0` AND `np.isfinite(sal[0])`.

- [ ] **Step 1: Failing tests** (synthetic fields; concrete, following
  `tests/test_build_baltic_b2_forcing.py`'s importlib idiom): offset wet-only + NaN-untouched
  + floor-at-0 + zero-identity (exact on wet cells); `ramp_w` values (w(3)=0, w(4.5)=0.5,
  w(6)=1, w(NaN)=NaN); TV=0 and exclusions=0 on a saturation fixture (all cells ≥6 PSU —
  the vacuity case); TV>0 on a mixed fixture with hand-computed value; `excluded_fraction`
  hand-computed on a fixture where dS=-3 zeroes known cells; all-zero event detection (a
  single-cell map at 4 PSU with dS=-3 → nan TV + recorded event); **orientation pin**: load
  ONE real cod map via the engine loader and assert zero map-positive cells on land
  (grid mask) — the test that catches a naive upside-down read.
- [ ] **Step 2:** FAIL at import; **Step 3:** implement; **Step 4:** tests PASS + ruff; run
  `main()` as a smoke check against the real files and confirm the printed instruments
  reproduce the spec's expectations table (report the exact printed values); **Step 5:**
  commit (`git add scripts/build_baltic_c4_forcing.py tests/test_build_baltic_c4_forcing.py`).

---

### Task 3: Harness `scripts/baltic_c4_salinity_ab.py` + tests

**Files:**
- Create: `scripts/baltic_c4_salinity_ab.py`, `tests/test_baltic_c4_harness_helpers.py`

**Interfaces:**
- Arms: `baseline`, `zero`, `ds_m1`, `ds_m2`, `ds_m3`; seeds (42, 123, 7, 999, 2024);
  nyear=50. Overlays: `movement.salinity.field.file` at the arm's absolute path — nothing
  else. Blocking gates in order (spec decision 4): builder zero-check → per-arm `.constant`
  absence assert → per-arm frame-count==24 assert on the written field → three-way
  load-through (engine-loaded field == disk == `offset_salinity(production, wet, dS)`; the
  engine-side array: find `_load_salinity_gate` in osmose/engine/config.py, read how
  `EngineConfig.salinity_field` holds data (`._data` per the review) — import-and-call with
  the arm's cfg like B2 called `_load_oxygen_data`; if not standalone-callable, fall back to
  from_dict + attribute access and report the route) → ramp ordering (w(arm) ≤ w(base) per
  wet cell) → engine runs → zero-arm bit-identity per seed → report.
- Report: per-arm per-species (ALL NINE incl. smelt) final-decade means + seed spreads +
  the builder's instruments; to `/tmp/c4_salinity_report.json`.

- [ ] **Step 1: Failing tests**: overlay construction (baseline has no salinity key
  override; arms carry only the file key; absolute paths); `.constant`-absence guard raises
  on a poisoned cfg; frame-count guard raises on a 23-frame synthetic; ramp-ordering check on
  synthetic fields (negative dS ⇒ ≤, zero ⇒ equal); three-way no-op-write pathological case
  (identical-to-production file + dS≠0 → FAIL) — B2's proven test set adapted.
- [ ] **Step 2:** FAIL at import; **Step 3:** implement on the B2 `run_*` pattern; **Step 4:**
  tests PASS + ruff; **Step 5:** commit
  (`git add scripts/baltic_c4_salinity_ab.py tests/test_baltic_c4_harness_helpers.py`).

---

### Task 4: The run + results doc

**Files:**
- Create: `docs/baltic_c4_salinity_2026-MM-DD.md` (run date),
  `docs/diagnostics/baltic_c4_salinity_report.json`

- [ ] **Step 1:** `uptime` low; run `.venv/bin/python scripts/baltic_c4_salinity_ab.py` via
  `run_in_background` (5 non-baseline+baseline = 6 configs… the arm set is 5 total incl.
  baseline+zero: 5 arms × 5 seeds ≈ 1.6 h — wait: baseline, zero, ds_m1, ds_m2, ds_m3 =
  **5 arms** × 5 seeds = 25 runs).
- [ ] **Step 2: Gates in order** — any blocking failure stops interpretation.
- [ ] **Step 3: Results doc**: the nine-species × five-arm chain table with instruments
  (TV, prey-overlap shift, exclusion fraction) beside stock columns; the graded-vs-flip July
  comparison (chain: cod_east redistribution → stickleback → percids/smelt), stated
  either way; the −3 arm's exclusion regime + all-zero guard status; EVERY decision-6 label
  (not-a-projection sentence, RV confound, cod_west saturated-null, redistributes-never-
  removes framing wherever occupancy metrics appear, single-source climatology, fixed ramp,
  uniform-offset blindness, Java gap deferred); both loader gaps (frame-count wrap; all-zero
  un-gate) as Stage-2 items; provenance + NOT a CI gate.
- [ ] **Step 4:** Copy report JSON to docs/diagnostics/; commit exactly the two files.

---

## Execution notes

- Precedent files to read first: `scripts/build_baltic_b2_forcing.py` +
  `scripts/baltic_b2_scenario_ab.py` (the entire pattern), `osmose/engine/movement_maps.py`
  (map loader — READ ONLY, user-dirty), `osmose/engine/processes/salinity_gate.py` +
  `movement.py:35-60` (ramp + consumption), `osmose/engine/config.py` (`_load_salinity_gate`).
- The spec's stated-expectations table is the builder's acceptance target; contradictions
  STOP the task and surface to the controller.
