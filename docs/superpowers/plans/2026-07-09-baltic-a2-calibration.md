# Baltic A2 calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing Baltic DE calibrator with Chunk A2 depletion on, co-calibrating per-species larval + adult mortality plus one shared zooplankton regrowth rate against the ICES bands, and compare species-in-band against an A2-off baseline.

**Architecture:** Additive changes to `scripts/calibrate_baltic.py`: a grouped-parameter expansion (one DE param → the four depletable resource keys), an A2 param set (phase-1 + the zoo param), an A2 base-config enabler, and a `--a2` CLI flag. The DE machinery, objective, and multiprocessing wrapper are reused unchanged; the non-A2 path stays byte-identical. No engine change (A2 already shipped).

**Tech Stack:** Python 3.11+, numpy, scipy `differential_evolution` (existing). Tests via pytest. Ruff.

## Global Constraints

- **No new dependencies; no engine change.** `ruff check` + `ruff format --check` clean on `scripts/ tests/`.
- **Parity:** without `--a2`, calibration behavior is byte-identical (`expand_param_overrides` produces the same overrides as the prior inline loop for non-sentinel keys). Existing calibrator tests pass unchanged.
- **A2 param wiring (verbatim):** grouped sentinel `species.regrowth.rate.zoo` expands to `species.regrowth.rate.sp{10,11,12,13}` (equal value); phytoplankton `species.regrowth.rate.sp{8,9}` fixed `5.0`; `ltl.depletable.enabled=true`, `ltl.depletable.floor=0.05`.
- **Zoo param bounds/x0:** log10 space, bounds `(-1.0, log10(2.0))` (i.e. rate 0.1–2.0), x0 `log10(0.6)`.
- **Deliverable is a candidate sidecar, NOT a deployed-config overwrite:** `data/baltic/*` is never modified.
- **Compute:** DE is many sims — bound it (small popsize/maxiter, `n_years=15`, `workers>1`); real runs are CLI-only, excluded from CI per `feedback-ci-fragile-emergent-tests`.
- **Test command:** `.venv/bin/python -m pytest tests/test_calibrate_baltic.py -q` (create if absent; else the existing calibrator test module).

---

### Task 1: grouped-param expansion, A2 param set, base-config enabler, `--a2` flag

**Files:**
- Modify: `scripts/calibrate_baltic.py`
- Test: `tests/test_calibrate_baltic_a2.py` (new)

**Interfaces:**
- Produces:
  - `expand_param_overrides(param_keys: list[str], x, use_log_space: bool = True) -> dict[str, str]` — build the override dict from a DE vector, expanding the `species.regrowth.rate.zoo` sentinel to the four `sp{10..13}` keys; non-sentinel keys pass through unchanged.
  - `get_a2_params() -> tuple[list[str], list[tuple[float,float]], list[float]]` — `get_phase1_params()` + the grouped zoo param (17 params).
  - `enable_a2_base_config(base_config: dict) -> dict` — copy with A2 depletion enabled (phyto fixed 5.0).
  - `main`/run path accepts `--a2` (uses `get_a2_params()` + `enable_a2_base_config`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibrate_baltic_a2.py`:

```python
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import calibrate_baltic as cb  # noqa: E402


def test_expand_param_overrides_passthrough_and_logspace():
    keys = ["mortality.additional.larva.rate.sp0", "mortality.additional.rate.sp1"]
    x = np.array([1.0, -2.0])  # log10 -> 10.0, 0.01
    ov = cb.expand_param_overrides(keys, x, use_log_space=True)
    assert ov == {
        "mortality.additional.larva.rate.sp0": str(10.0),
        "mortality.additional.rate.sp1": str(0.01),
    }


def test_expand_param_overrides_zoo_sentinel_expands_to_four_keys():
    keys = ["mortality.additional.larva.rate.sp0", "species.regrowth.rate.zoo"]
    x = np.array([0.0, np.log10(0.6)])  # -> mort 1.0, zoo 0.6
    ov = cb.expand_param_overrides(keys, x, use_log_space=True)
    for r in (10, 11, 12, 13):
        assert ov[f"species.regrowth.rate.sp{r}"] == str(0.6)
    assert "species.regrowth.rate.zoo" not in ov
    assert ov["mortality.additional.larva.rate.sp0"] == str(1.0)


def test_get_a2_params_appends_zoo_param():
    keys, bounds, x0 = cb.get_a2_params()
    base_n = len(cb.get_phase1_params()[0])
    assert len(keys) == base_n + 1
    assert keys[-1] == "species.regrowth.rate.zoo"
    assert bounds[-1] == (-1.0, float(np.log10(2.0)))
    assert abs(x0[-1] - float(np.log10(0.6))) < 1e-12


def test_enable_a2_base_config_sets_keys_without_mutating_input():
    base = {"simulation.time.nyear": "15"}
    out = cb.enable_a2_base_config(base)
    assert out["ltl.depletable.enabled"] == "true"
    assert out["ltl.depletable.floor"] == "0.05"
    assert out["species.regrowth.rate.sp8"] == "5.0"
    assert out["species.regrowth.rate.sp9"] == "5.0"
    assert "ltl.depletable.enabled" not in base  # input untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_a2.py -q`
Expected: FAIL — `AttributeError: module 'calibrate_baltic' has no attribute 'expand_param_overrides'`

- [ ] **Step 3: Add the module-level constants and helpers**

In `scripts/calibrate_baltic.py`, add near the top (after imports / constants):

```python
_ZOO_REGROWTH_SENTINEL = "species.regrowth.rate.zoo"
_ZOO_RESOURCE_INDICES = (10, 11, 12, 13)  # depletable zooplankton + benthos


def expand_param_overrides(param_keys, x, use_log_space: bool = True) -> dict[str, str]:
    """Build the DE override dict, expanding the grouped zoo-regrowth sentinel to the four
    depletable resource keys (equal value). Non-sentinel keys are byte-identical to the prior
    inline construction."""
    overrides: dict[str, str] = {}
    for i, key in enumerate(param_keys):
        val = 10.0 ** x[i] if use_log_space else x[i]
        if key == _ZOO_REGROWTH_SENTINEL:
            for r in _ZOO_RESOURCE_INDICES:
                overrides[f"species.regrowth.rate.sp{r}"] = str(val)
        else:
            overrides[key] = str(val)
    return overrides


def get_a2_params():
    """Phase-1 mortality params + one grouped zooplankton regrowth-rate param (A2)."""
    keys, bounds, x0 = get_phase1_params()
    keys.append(_ZOO_REGROWTH_SENTINEL)
    bounds.append((-1.0, float(np.log10(2.0))))  # rate 0.1 .. 2.0 in log10 space
    x0.append(float(np.log10(0.6)))
    return keys, bounds, x0


def enable_a2_base_config(base_config) -> dict:
    """Copy of base_config with A2 depletion enabled (phytoplankton regrowth fixed fast)."""
    cfg = dict(base_config)
    cfg["ltl.depletable.enabled"] = "true"
    cfg["ltl.depletable.floor"] = "0.05"
    cfg["species.regrowth.rate.sp8"] = "5.0"
    cfg["species.regrowth.rate.sp9"] = "5.0"
    return cfg
```

(`get_a2_params` and `enable_a2_base_config` must be defined AFTER `get_phase1_params`; place them just below it. `expand_param_overrides` + the constants can go near the other module constants.)

- [ ] **Step 4: Refactor the objective wrapper to use the expansion**

In `_ObjectiveWrapper._simulate_and_compute_stats`, replace the inline override loop:

```python
        overrides: dict[str, str] = {}
        for i, key in enumerate(self.param_keys):
            if self.use_log_space:
                val = 10.0 ** x[i]
            else:
                val = x[i]
            overrides[key] = str(val)
```

with:

```python
        overrides = expand_param_overrides(self.param_keys, x, self.use_log_space)
```

- [ ] **Step 5: Add the `--a2` CLI flag and wire it into the run path**

In `main()`'s argument parser, add:

```python
    parser.add_argument("--a2", action="store_true",
                        help="calibrate with Chunk A2 depletion on + a shared zoo regrowth-rate param")
```

Thread `args.a2` into the calibration run function (the one that reads `base_config` and calls
`get_phase1_params()` — locate it with `grep -n "get_phase1_params\|reader.read(BALTIC_CONFIG)"`; it is the
run-path site, not the reporting helpers). At that site, replace the base-config read + param selection so
that when A2 is on:

```python
    base_config = reader.read(BALTIC_CONFIG)
    if a2:
        base_config = enable_a2_base_config(base_config)
        param_keys, bounds, x0 = get_a2_params()
    else:
        param_keys, bounds, x0 = get_phase1_params()
```

(Pass `a2` through from `main()`; default `False` preserves the existing path exactly.)

- [ ] **Step 6: Run tests + verify parity**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_a2.py -q`
Expected: PASS (4 passed).

Run the existing calibrator tests (parity of the non-A2 path):
`.venv/bin/python -m pytest tests/ -k "calibrat" -q`
Expected: PASS, unchanged from before this task.

- [ ] **Step 7: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/calibrate_baltic.py tests/test_calibrate_baltic_a2.py
.venv/bin/ruff format scripts/calibrate_baltic.py tests/test_calibrate_baltic_a2.py
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_a2.py
git commit -m "feat(baltic): --a2 calibration (depletion on + grouped zoo regrowth param)"
```

---

### Task 2: Real-engine — smoke, baseline vs A2-on DE runs, in-band comparison, write-up

**This task is real-engine, CLI-only, NOT CI.** Run after Task 1. Verification: the smoke wires correctly, both DE arms complete, and the results doc + candidate sidecar are written.

**Files:**
- Create: `docs/baltic_a2_calibration_results_2026-07-09.md`, `docs/diagnostics/baltic_a2_calibrated_params.json` (+ the A2-off baseline)

- [ ] **Step 1: Smoke the wiring (fast)**

Run a tiny A2 calibration to confirm the flag, param set, and A2 base config wire without error:

```bash
cd /home/razinka/osmopy
.venv/bin/python scripts/calibrate_baltic.py --a2 --phase 1 --maxiter 2 --popsize 4 --years 5 2>&1 | tail -20
```
Expected: it prints `optimizer=...`, runs a few generations, and finishes with a best vector — no crash, no unknown-key error (the A2 keys are allowlisted). If it errors on the zoo param or A2 keys, fix the wiring before the full runs.

- [ ] **Step 2: A2-OFF baseline DE (control)**

```bash
.venv/bin/python scripts/calibrate_baltic.py --phase 1 --maxiter 25 --popsize 12 --years 15 2>&1 | tee /tmp/baltic_cal_off.log
```
Expected: completes in ~1–3 h; prints the best per-species biomass vs bands and the final objective. Record the best params + which species land in band.

- [ ] **Step 3: A2-ON DE (treatment)**

```bash
.venv/bin/python scripts/calibrate_baltic.py --a2 --phase 1 --maxiter 25 --popsize 12 --years 15 2>&1 | tee /tmp/baltic_cal_a2.log
```
Expected: same budget with depletion on + the zoo param co-optimized. Record the best params (incl. zoo rate) + in-band species.

- [ ] **Step 4: Write the candidate sidecar + results doc**

Extract from both runs' best vectors. Write `docs/diagnostics/baltic_a2_calibrated_params.json` (A2-on best:
per-species larval/adult mortality, zoo rate, per-species mean biomass + in-band bool; and the A2-off best
for comparison). Write `docs/baltic_a2_calibration_results_2026-07-09.md`: the N/8-in-band comparison
(A2-off vs A2-on), the calibrated params, whether cod + the clupeids land in band simultaneously, and the
verdict — a deployable ICES-calibrated Baltic is within reach / partial / cod remains uncalibratable
(→ structural recruitment change needed). **Do not modify `data/baltic/*`.**

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add docs/baltic_a2_calibration_results_2026-07-09.md docs/diagnostics/baltic_a2_calibrated_params.json
git commit -m "docs(baltic): A2 calibration results (2026-07-09) — A2-off vs A2-on in-band comparison"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-09-baltic-a2-calibration-design.md`):
- run existing DE calibrator with A2 on + one grouped zoo rate → Task 1 (`get_a2_params`, `enable_a2_base_config`, `expand_param_overrides`) + Task 2 (`--a2` runs). ✓
- grouped-param expansion (sentinel → sp10–13), phyto fixed 5.0, zoo bounds/x0 → Task 1 Step 3 (verbatim). ✓
- `--a2` CLI toggle, non-A2 path byte-identical → Task 1 Steps 4–5 + parity check Step 6. ✓
- baseline vs treatment, N/8-in-band success metric → Task 2 Steps 2–4. ✓
- candidate sidecar, no deployed-config overwrite → Task 2 Step 4 (explicit). ✓
- bounded compute (small popsize/maxiter, n_years 15) → Task 2 Steps 2–3. ✓
- unit tests CI-safe (helpers), DE runs CLI-only → Task 1 tests + Task 2. ✓

**Placeholder scan:** no TBD/TODO; helper code is complete; the main() wiring gives exact branch code + a grep anchor for the one large-function site.

**Type consistency:** `expand_param_overrides` / `get_a2_params` / `enable_a2_base_config` names match definition and call sites; `_ZOO_REGROWTH_SENTINEL` value (`species.regrowth.rate.zoo`) matches the expansion, `get_a2_params`, and the tests; expanded keys `species.regrowth.rate.sp{10..13}` match the A2 engine config keys from PR #104.
