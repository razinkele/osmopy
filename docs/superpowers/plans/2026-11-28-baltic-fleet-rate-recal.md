# Baltic Fleet-Rate Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Baltic ICES targets to 2020–2024 and re-solve four fleets' base fishing rates so each fleet's catch matches its ICES catch target, keeping the ecosystem certified.

**Architecture:** A pure, unit-tested solver module (`osmose/calibration/fleet_rates.py`) maps catch vs rate with a coarse sweep and bisects toward the lower-F root; a script drives it against the engines (C# search, Python confirm) and logs every evaluation. Targets come from the existing derivation script with a new window; rates land in the Baltic fishing CSV; certification, SP1b and the ICES validator are re-run.

**Tech Stack:** Python 3.13 (`/opt/micromamba/envs/shiny/bin/python`), NumPy/pandas, the in-process Python and C# engines (`osmose.engine.select.get_engine`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-28-baltic-fleet-rate-recal-design.md`

## Global Constraints

- Target window: **2020–2024** for every windowed ICES target; cod_west 2020–2022 (snapshot pinned to the 2022 assessment).
- Solved fleets: exactly **fsh0 cod_west, fsh8 cod_east, fsh1 herring, fsh3 flounder**, each **to `target_tonnes`** within **±10%**, ≤ **8** bisection evaluations, ≤ **3** passes.
- **fsh2 sprat stays 0.369** (ICES-F anchor). fsh4–fsh7 unchanged.
- Always the **lower-F root**; target above the dome's peak → peak rate + report max attainable catch.
- Catch = `yield` output, **trailing 5-year mean of a 50-yr run**, mean over seeds.
- Search: **C# engine, 2 seeds**; confirm: **Python, cert seeds (42, 123, 7, 999, 2024)**; a fleet off by more than **±15%** on Python is refined on Python (2 seeds).
- Runs use the **SP1-off production base**; SP1b re-solved once at the end.
- Certification "before" = current rates against the **new** targets. Gate: no persisting species lost, in-envelope count ≥ before. **Certification wins**: back off toward band edge, else old rate + report.
- `cod_east` SSB row stays **70 kt (2018–22)**, labelled as the one exception.
- Catch rows are an output of `scripts/derive_ices_targets.py --write`; never hand-edit them.
- One engine job at a time (no concurrent engine runs); a process that has loaded the CLR must not fork afterwards.

## Review Focus

1. **Target above the dome's peak** (cod_east may not reach 1.57 kt by rate alone): the solver must return the peak rate flagged `reachable=False`, not climb into overfishing. → Task 2 test `test_target_above_peak_returns_peak`.
2. **Sweep with no bracket on the lower branch because of seed noise** (catch wiggles, no sign change below the peak): solver must fall back to the closest sweep point and flag it, not raise. → Task 2 test `test_noisy_sweep_without_crossing_returns_closest`.
3. **Zero catch at the current rate** (a collapsed stock or a fleet that catches nothing): log-scale sweep must still work and report `reachable=False` if every point is 0. → Task 2 test `test_all_zero_catch`.
4. **Rate written to the CSV in place, not appended** (Java keeps the first definition of a duplicated key; CLAUDE.md): the apply step must replace the existing line. → Task 4 test `test_apply_rates_replaces_in_place`.
5. **cod_west's 3-year window** in the derivation test: assertions written for 5 years must not silently pass/fail on 3. → Task 1 test `test_cod_west_window_is_three_years`.

---

### Task 1: Targets on the 2020–2024 window

**Files:**
- Modify: `scripts/derive_ices_targets.py` (`WINDOW`, docstring)
- Modify: `tests/test_derive_ices_targets.py` (window constants 2018→2020, 2022→2024)
- Modify (by `--write` + hand for the two SSB rows): `data/baltic/reference/biomass_targets.csv`

**Interfaces:**
- Produces: `biomass_targets.csv` catch rows for 2020–24 (expected, t: cod_west 1428 [115–3256], cod_east 1566 [922–2654], herring 228570 [166998–297546], sprat 273249 [242559–303940], flounder 9520 [5626–13413]); `cod_west` ssb row from 2020–22 tonnes; `cod_east` ssb row unchanged in value, notes say "2018–2022 exception: ICES publishes cod.27.24-32 SSB only as a relative index".

- [ ] **Step 1: Update tests first:** `test_window_is_2018_2022` → `test_window_is_2020_2024` asserting `WINDOW == (2020, 2024)`; replace the hard-coded `2018 <= y <= 2022` filters with `WINDOW`; flounder `per_year` expectation `[2020, 2021, 2022, 2023, 2024]`; add `test_cod_west_window_is_three_years` asserting the cod_west row's notes contain `"(3 yr)"` and its `target_tonnes` ≈ 1428 (rel 1e-3).
- [ ] **Step 2:** Run `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python -m pytest tests/test_derive_ices_targets.py -v` → FAIL on the window tests.
- [ ] **Step 3:** Set `WINDOW = (2020, 2024)`, update the docstring; run `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python scripts/derive_ices_targets.py --write`; then hand-edit the cod_west ssb row (mean/min/max of the snapshot's 2020–2022 SSB, tonnes) and the cod_east ssb row's `source`/`notes`; update the CSV header comment lines (`ices_advice_year`, window wording "2020-2024").
- [ ] **Step 4:** Re-run the test file → PASS; `git diff data/baltic/reference/biomass_targets.csv` shows only the catch rows, the two ssb rows and header lines changed.
- [ ] **Step 5:** Run every test that reads the targets: `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python -m pytest -q -n 12 -k "target or certif or validat"` → green, or each failure explained (a test pinning the old 2018–22 numbers is updated to the new ones, not loosened).
- [ ] **Step 6: Commit** `feat(baltic): ICES targets on the 2020-2024 window`

### Task 2: Solver module

**Files:**
- Create: `osmose/calibration/fleet_rates.py`
- Test: `tests/test_fleet_rates.py`

**Interfaces:**
- Produces:
  - `trailing_catch(yield_df: pd.DataFrame, species: str, n_years: int = 5) -> float` — mean of the last `n_years` rows of the species column.
  - `@dataclass(frozen=True) class Solve: rate: float; catch: float; target: float; reachable: bool; evaluations: list[tuple[float, float]]; note: str`
  - `sweep_rates(current: float, target_ratio: float, n: int = 5) -> list[float]` — log-spaced, spanning `current` and `current * target_ratio` with a factor-2 margin each side.
  - `solve_rate(evaluate: Callable[[float], float], current: float, target: float, *, tol: float = 0.10, max_bisect: int = 8, sweep_n: int = 5) -> Solve` — calls `evaluate(rate) -> catch`.

Algorithm (not determined by the signature): sweep `sweep_rates(current, target / max(evaluate(current), 1e-9))`, sort by rate, find the peak; on the points at or below the peak find the first adjacent pair bracketing `target`; bisect in `log(rate)` until `|catch/target - 1| <= tol` or `max_bisect`. No bracket and target > peak catch → `Solve(rate=peak_rate, reachable=False, note="target above peak")`. No bracket otherwise → closest point, `reachable=False`, note says so.

- [ ] **Step 1: Write the failing tests** with synthetic `evaluate` functions (no engine):
  - `test_solves_on_ascending_limb`: `catch = 1000*r*exp(-r)`, current 0.1, target 300 → `reachable`, `abs(s.catch/300-1) <= 0.10`, `s.rate < 1.0` (the peak).
  - `test_takes_lower_root_when_two_exist`: same curve, current 2.0 (past the peak), target 300 → `s.rate < 1.0`.
  - `test_lowering_rate_past_peak`: current 3.0, target 100 (catch at 3.0 ≈ 149) → `s.rate < 1.0` and catch ≈ 100 (the solver does not assume direction).
  - `test_target_above_peak_returns_peak`: target 500 (peak ≈ 368) → `not s.reachable`, `0.5 < s.rate < 2.0`, `"peak" in s.note`.
  - `test_noisy_sweep_without_crossing_returns_closest`: piecewise curve with no point near the target below the peak → `not s.reachable`, no exception.
  - `test_all_zero_catch`: `evaluate = lambda r: 0.0` → `not s.reachable`.
  - `test_trailing_catch`: 10-row frame, species column 0..9 → `trailing_catch(df, sp) == 7.0`.
  - `test_evaluation_budget`: count calls ≤ `sweep_n + 1 + max_bisect`.
- [ ] **Step 2:** `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python -m pytest tests/test_fleet_rates.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement the module per the Interfaces block.
- [ ] **Step 4:** Tests → PASS; `ruff check` clean.
- [ ] **Step 5: Commit** `feat(calibration): fleet-rate solver (sweep + lower-root bisection)`

### Task 3: Driver script and baseline

**Files:**
- Create: `scripts/recalibrate_fleet_rates.py`
- Test: `tests/test_recalibrate_fleet_rates.py`

**Interfaces:**
- Consumes: `solve_rate`, `trailing_catch`, `Solve` (Task 2); `CERT_SEEDS` from `scripts/baltic_stability_certify.py`; `osmose_demo("baltic", tmp)` + `OsmoseConfigReader` as `certify_python` does.
- Produces:
  - `FLEETS: dict[str, tuple[str, str]] = {"fsh0": ("cod_west", ...), "fsh8": ("cod_east", ...), "fsh1": ("herring", ...), "fsh3": ("flounder", ...)}` (species, rate key `fisheries.rate.base.fshN`).
  - `load_catch_targets(path) -> dict[str, float]` (species → `target_tonnes` of `catch` rows).
  - `make_evaluator(base_cfg, rate_key, species, engine: str, seeds, n_years=50) -> Callable[[float], float]` — sets the rate, runs each seed in memory, returns mean `trailing_catch`.
  - CLI: `--engine {csharp,python}`, `--seeds`, `--years 50`, `--passes 3`, `--fleets fsh0,...`, `--rates-json` (start rates), `--out` (JSON log: every evaluation, every Solve, final rates).
  - `--baseline-only`: one evaluation of current rates, prints catch vs target per fleet.
- C# runs happen in this script's own process (never inside pytest); the script never forks after loading the CLR.

- [ ] **Step 1: Tests (no engine):** `test_load_catch_targets` reads the committed CSV and returns the five catch species with Task 1's targets; `test_fleets_cover_exactly_four` asserts the FLEETS keys and that `fsh2` is absent; `test_make_evaluator_sets_only_its_rate` with a fake engine object capturing the config passed (only its rate key differs from base).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; the pass loop: for each pass, for each fleet, `solve_rate(make_evaluator(...))` from the current rates, update rates; stop early when every fleet is within tol on re-check.
- [ ] **Step 4:** Tests → PASS.
- [ ] **Step 5: Baseline run:** `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python scripts/recalibrate_fleet_rates.py --baseline-only --engine python --seeds 42 123 7 999 2024 --out docs/diagnostics/fleet_recal_baseline.json` → prints the four catches vs targets (expected roughly cod_west 0.45 kt, cod_east 0.57 kt, herring 440 kt, flounder 39 kt).
- [ ] **Step 6: Certification "before":** `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python scripts/baltic_stability_certify.py --params current --out docs/diagnostics/fleet_recal_cert_before.md` → record persists/in-envelope counts.
- [ ] **Step 7: Commit** script, tests, both diagnostics files: `feat(scripts): fleet-rate recalibration driver + 2020-24 baseline`

### Task 4: Solve and apply the rates

**Files:**
- Modify: `data/baltic/baltic_param-fishing.csv` (fsh0, fsh1, fsh3, fsh8 rate lines + provenance comment)
- Modify: `scripts/recalibrate_fleet_rates.py` (`apply_rates(path, rates: dict[str, float]) -> None`, `--apply`)
- Test: `tests/test_recalibrate_fleet_rates.py`

**Interfaces:**
- Consumes: Task 3 CLI and log.
- Produces: `apply_rates` replacing each `fisheries.rate.base.fshN;<value>` line in place, preserving every other line.

- [ ] **Step 1: Test** `test_apply_rates_replaces_in_place`: on a temp copy of the fishing CSV, apply `{"fsh0": 0.5}` → exactly one `fisheries.rate.base.fsh0` line, value `0.5`, all other lines byte-identical.
- [ ] **Step 2:** Run → FAIL; implement; → PASS.
- [ ] **Step 3: Search (background, ~hours):** `--engine csharp --seeds 42 123 --passes 3 --out docs/diagnostics/fleet_recal_search.json`. Expected: four Solves; any `reachable=False` noted for the report.
- [ ] **Step 4: Confirm on Python:** run `--baseline-only --engine python --seeds 42 123 7 999 2024 --rates-json <search final rates>` → each fleet within ±15% of target, else refine that fleet with `--engine python --seeds 42 123 --fleets fshN` and confirm again.
- [ ] **Step 5:** `--apply` the confirmed rates; add a comment line above them: `# 2026-11-28 fleet-rate recal to ICES 2020-2024 catch (docs/baltic_fleet_rate_recal_2026-11-28.md)`.
- [ ] **Step 6: Commit** CSV, log, code: `feat(baltic): fleet rates re-solved to ICES 2020-2024 catch`

### Task 5: Certify, back off if needed, SP1b, suites

**Files:**
- Modify (only if needed): `data/baltic/baltic_param-fishing.csv`
- Modify: `osmose/calibration/larva_recal.py` (`RECAL_RATES`, with new `d0`)
- Modify (only if they pin absolute Baltic biomass and fail): e.g. `tests/test_tutorial_3species.py::_PYRAMID_BOUNDS`

- [ ] **Step 1:** `scripts/baltic_stability_certify.py --params current --out docs/diagnostics/fleet_recal_cert_after.md` → compare with Task 3 Step 6. Pass = no persists ✓→✗, in-envelope count ≥ before.
- [ ] **Step 2 (only on fail):** for the fleet whose stock regressed (or, if unclear, the fleet with the largest rate change), re-solve it with `target` = the band edge nearest its old catch; re-certify. Still failing → restore that fleet's old rate, re-certify, record "unsolved: certification".
- [ ] **Step 3:** `PYTHONPATH=. /opt/micromamba/envs/shiny/bin/python scripts/recalibrate_sp1b.py` → paste the printed `RECAL_RATES` block (with its `d0`) into `osmose/calibration/larva_recal.py`; commit `docs/diagnostics/sp1b_solve.json`.
- [ ] **Step 4:** Full suite `/opt/micromamba/envs/shiny/bin/python -m pytest -q -n 12` and `OSMOSE_REQUIRE_CSHARP=1 ... pytest tests/test_csharp_*.py tests/test_engine_select.py -q -n 8` → green; re-derive any absolute-biomass fixture that fails, following the `_PYRAMID_BOUNDS` pattern (note the re-derivation in the test).
- [ ] **Step 5: Commit** `feat(baltic): certify fleet recal; SP1b re-solved`

### Task 6: ICES re-check and report

**Files:**
- Create: `docs/baltic_fleet_rate_recal_2026-11-28.md`

- [ ] **Step 1:** 50-yr Python runs, cert seeds, written to disk; `scripts/validate_outputs_vs_ices.py --prefix baltic --ices-window 2020-2024 --sms-m2 --json ...` per seed.
- [ ] **Step 2:** Write the report: targets before/after; rates before/after; catch and SSB vs ICES per stock (before = `docs/baltic_ices_check_2026-11-28.md`); certification before/after; SP1b re-solve; unsolved or unreachable fleets with max attainable catch; the cod_east 2018–22 exception.
- [ ] **Step 3: Commit** `docs: Baltic fleet-rate recalibration report`; open the PR (body ends with the session attribution lines).
