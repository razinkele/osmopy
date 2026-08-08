# Baltic Depletable-LTL Bounded Recalibration (Phase 1 Contingency) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit the six bounded parameters (zoo/benthos regrowth + four plankton accessibilities) with depletion ON, so the identity-pinned gate can pass and depletion can be adopted — capturing the −75% pikeperch effect without collapsing the assessed tier.

**Architecture:** A new calibration phase `a2r` in `scripts/calibrate_baltic.py` (mirroring the a2/1c precedents), objective restricted to the 7 gate species (indicative species are never tuned against). The A/B harness gains an `--extra-arm` option so the fitted candidate is certified by the same instrument that rejected the prior. Adoption, if the gate passes, follows the original Phase 1 plan's Task 4/5 machinery with fitted values.

**Tech Stack:** Python 3.12, `.venv/bin/python`, scipy `differential_evolution` via the script's existing `isolated_eval_map` (process-isolated workers, `__main__` guard — the repo's spawn-pool rules), pytest, ruff (line 100).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 1 item 4 — bounded recalibration of **regrowth rates and zooplanktivore availability coefficients ONLY**; anything wider aborts.
- Fitted parameters, exactly 6:
  1. `species.regrowth.rate.zoo3` sentinel → `species.regrowth.rate.sp11..sp13` (zooplankton only), log10 bounds (-1.30103, 0.30103) i.e. 0.05–2.0/step, x0 = log10(0.911553421016705).
  2. `species.regrowth.rate.benthos` sentinel → `species.regrowth.rate.sp14`, log10 bounds (-2.0, -0.5) i.e. 0.01–0.316/step (literature P/B ~0.5–3 yr⁻¹ ≈ 0.02–0.12/step sits inside), x0 = log10(0.03).
  3.–6. `species.accessibility2fish.sp9`, `.sp11`, `.sp12`, `.sp13` — same keys and bounds as `get_phase1c_params()` uses (read them from the source; do not invent), x0 = log10(0.8) each (current config value).
- The existing `species.regrowth.rate.zoo` sentinel (expands sp11–14) is NOT touched — historical results must stay interpretable.
- Base config for every eval: production config + `enable_a2_base_config()` (depletion on, floor 0.05, phyto sp9–10 regrowth pinned 5.0). Phyto stays pinned; mortality params stay untouched.
- Objective targets: ONLY the 7 gate species (`cod_west, cod_east, herring, sprat, flounder, perch, stickleback`). `pikeperch`/`smelt` rows are EXCLUDED from the objective (weight-aware doctrine: never tuned against), and `weight_floor=0.5` lifts perch/stickleback so the gate species all bind.
- Final verdict comes only from the 50-yr × 5-seed identity gate via the A/B harness (`REQUIRED_PASS` in `scripts/baltic_depletable_ab.py`), never from the DE loss.
- No concurrent engine jobs outside the DE's own worker pool.
- Adoption (Task 4 here) follows the original plan's mechanics: `docs/superpowers/plans/2026-08-08-baltic-depletable-ltl-phase1.md` Tasks 4–5 (allowlist both copies, overlay file, java-block test update, loading assertions) — with fitted values, and `DEPLETION_KEYS` in the harness updated to the adopted set so the cross-file test still enforces a single source of truth.
- Deferred minors to fix while touching these files (final-review triage): `--out` parent `mkdir(parents=True, exist_ok=True)` in `scripts/baltic_depletable_ab.py`; one-line comment on `jar_version="4.4.1"` in `tests/test_certify_java_pinning.py` (verified deviation — background-species guard, not depletion).
- Tests: `.venv/bin/python -m pytest`; lint `.venv/bin/ruff check scripts/ tests/`.

---

### Task 1: Calibration phase `a2r` + harness `--extra-arm` + deferred minors

**Files:**
- Modify: `scripts/calibrate_baltic.py` (new sentinels, `get_a2r_params()`, phase registry + CLI choice, base-config branch)
- Modify: `scripts/baltic_depletable_ab.py` (`--extra-arm NAME JSON_PATH`, `--skip-default-arms`, `--out` mkdir fix)
- Modify: `tests/test_certify_java_pinning.py` (comment only)
- Test: `tests/test_calibrate_baltic_a2r.py`, extend `tests/test_baltic_depletable_ab.py`

**Interfaces:**
- Consumes: `expand_param_overrides`, `make_objective(..., weight_floor=...)`, `load_targets()`, `enable_a2_base_config`, `_dispatch_optimizer`, phase registry (`elif phase == "1c":` block style) — all in `scripts/calibrate_baltic.py`; `ARM_OFF`, `DEPLETION_KEYS`, `identity_gate`, `make_report` in `scripts/baltic_depletable_ab.py`.
- Produces: CLI `--phase a2r` runnable end-to-end; `scripts/baltic_depletable_ab.py --extra-arm fitted <json> [--skip-default-arms]` where the JSON is a flat `{config_key: value}` dict (the DE result's params); Task 2 runs the former, Task 3 the latter.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibrate_baltic_a2r.py
"""Bounded recalibration phase a2r: 6 params only, gate-species objective, depletion base.

Spec 2026-08-08 §4 Phase 1 item 4: regrowth + zooplanktivore availabilities ONLY —
this file is the guard that the phase stays bounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import calibrate_baltic as cal  # noqa: E402

GATE_SPECIES = {"cod_west", "cod_east", "herring", "sprat", "flounder", "perch", "stickleback"}


def test_a2r_params_exactly_six():
    keys, bounds, x0 = cal.get_a2r_params()
    assert keys == [
        "species.regrowth.rate.zoo3",
        "species.regrowth.rate.benthos",
        "species.accessibility2fish.sp9",
        "species.accessibility2fish.sp11",
        "species.accessibility2fish.sp12",
        "species.accessibility2fish.sp13",
    ]
    assert len(bounds) == len(x0) == 6
    # zoo3: 0.05..2.0 per step in log10; x0 at the carried-over prior
    assert np.isclose(bounds[0][0], np.log10(0.05)) and np.isclose(bounds[0][1], np.log10(2.0))
    assert np.isclose(x0[0], np.log10(0.911553421016705))
    # benthos: 0.01..0.316 per step in log10; x0 at the literature rate
    assert np.isclose(bounds[1][0], -2.0) and np.isclose(bounds[1][1], -0.5)
    assert np.isclose(x0[1], np.log10(0.03))
    # accessibilities: x0 at the current config value 0.8
    for i in (2, 3, 4, 5):
        assert np.isclose(x0[i], np.log10(0.8))


def test_a2r_sentinels_expand_to_current_layout():
    keys, _, x0 = cal.get_a2r_params()
    overrides = cal.expand_param_overrides(keys, x0)
    assert set(overrides) == {
        "species.regrowth.rate.sp11",
        "species.regrowth.rate.sp12",
        "species.regrowth.rate.sp13",
        "species.regrowth.rate.sp14",
        "species.accessibility2fish.sp9",
        "species.accessibility2fish.sp11",
        "species.accessibility2fish.sp12",
        "species.accessibility2fish.sp13",
    }
    # zoo3 must NOT touch benthos; benthos sentinel owns sp14
    assert np.isclose(float(overrides["species.regrowth.rate.sp11"]), 0.911553421016705)
    assert np.isclose(float(overrides["species.regrowth.rate.sp14"]), 0.03)
    # legacy sentinel untouched: still expands sp11..sp14 as before
    legacy = cal.expand_param_overrides(["species.regrowth.rate.zoo"], [np.log10(0.5)])
    assert set(legacy) == {f"species.regrowth.rate.sp{i}" for i in (11, 12, 13, 14)}


def test_a2r_targets_exclude_indicative_overshoots():
    targets = cal.get_a2r_targets()
    names = {t.species for t in targets}
    assert names == GATE_SPECIES
```

Extend `tests/test_baltic_depletable_ab.py` with:

```python
def test_extra_arm_loading(tmp_path):
    import json

    p = tmp_path / "fitted.json"
    json.dump({"ltl.depletable.enabled": "true", "species.regrowth.rate.sp14": "0.05"}, p.open("w"))
    arms = ab.build_arms(extra_arms=[("fitted", str(p))], skip_default=True)
    assert list(arms) == ["off", "fitted"]  # 'off' baseline always present
    assert arms["fitted"]["species.regrowth.rate.sp14"] == "0.05"
    assert arms["off"] == ab.ARM_OFF
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_a2r.py tests/test_baltic_depletable_ab.py -v`
Expected: new tests FAIL (`get_a2r_params`/`get_a2r_targets`/`build_arms` missing); the 9 existing harness tests still PASS.

- [ ] **Step 3: Implement in `scripts/calibrate_baltic.py`**

1. Sentinels next to `_ZOO_REGROWTH_SENTINEL`:

```python
_ZOO3_REGROWTH_SENTINEL = "species.regrowth.rate.zoo3"  # sp11-13 (zooplankton, NOT benthos)
_BENTHOS_REGROWTH_SENTINEL = "species.regrowth.rate.benthos"  # sp14
```

2. Extend `expand_param_overrides` to expand the two new sentinels (mirror the existing
   `species.regrowth.rate.zoo` branch; zoo3 → indices 11–13, benthos → 14). Do not change the
   legacy branch.

3. Phase getter + target filter (near `get_a2_params`):

```python
_A2R_GATE_SPECIES = (
    "cod_west", "cod_east", "herring", "sprat", "flounder", "perch", "stickleback"
)


def get_a2r_targets() -> list[BiomassTarget]:
    """Gate species only: indicative overshoots (pikeperch, smelt) are never tuned against."""
    return [t for t in load_targets() if t.species in _A2R_GATE_SPECIES]


def get_a2r_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Bounded depletion refit (spec 2026-08-08 Phase 1 contingency): 6 params, nothing else.

    Regrowth: zoo grouped sp11-13 (prior = the carried-over pre-split fit), benthos sp14
    separate with literature bounds (P/B ~0.5-3/yr => 0.02-0.12/step; A/B 2026-08-08 showed
    strong benthos sensitivity). Accessibilities: the phase-1c four, seeded at config 0.8.
    """
    keys = [_ZOO3_REGROWTH_SENTINEL, _BENTHOS_REGROWTH_SENTINEL]
    bounds = [(float(np.log10(0.05)), float(np.log10(2.0))), (-2.0, -0.5)]
    x0 = [float(np.log10(0.911553421016705)), float(np.log10(0.03))]
    for sp_idx in (9, 11, 12, 13):
        keys.append(f"species.accessibility2fish.sp{sp_idx}")
        bounds.append(<same bound tuple get_phase1c_params uses for these keys — copy it>)
        x0.append(float(np.log10(0.8)))
    return keys, bounds, x0
```

(Replace the `<...>` placeholder by reading `get_phase1c_params()` and copying its
accessibility bound literal — the test does not pin it, the source precedent governs.)

4. Registry + CLI: add `elif phase == "a2r": keys, bounds, x0 = get_a2r_params()` beside the
   `"1c"` branch; add a `if phase == "a2r":` base-config branch that applies
   `base_config = enable_a2_base_config(base_config)` and swaps targets to `get_a2r_targets()`
   with `weight_floor=0.5` when building the objective (follow how existing phase branches pass
   phase-specific targets/objective kwargs); add `"a2r"` to the `--phase` choices.

5. In `scripts/baltic_depletable_ab.py`: factor arm construction into
   `build_arms(extra_arms=None, skip_default=False, sensitivity=False) -> dict[str, dict[str, str]]`
   — always starts with `{"off": dict(ARM_OFF)}`; default arms `on`/`on-benthoslit` unless
   `skip_default`; each `(name, json_path)` in `extra_arms` loads a flat str→str dict and appends.
   `main()` gains `--extra-arm NAME JSON_PATH` (repeatable, `action="append"`, `nargs=2`) and
   `--skip-default-arms`, and calls `build_arms`. Fix `--out`:
   `Path(args.out).parent.mkdir(parents=True, exist_ok=True)` before `write_text`.

6. In `tests/test_certify_java_pinning.py`, add above the `jar_version="4.4.1"` call:

```python
    # jar_version is required: the Baltic config declares 2 background species, which the
    # guard conservatively blocks when the jar version is unknown — unrelated to depletion.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_a2r.py tests/test_baltic_depletable_ab.py tests/test_certify_java_pinning.py tests/test_calibrate_baltic_a2.py -v`
Expected: all PASS (including the pre-existing a2 tests — the legacy sentinel is untouched).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check scripts/calibrate_baltic.py scripts/baltic_depletable_ab.py tests/test_calibrate_baltic_a2r.py tests/test_baltic_depletable_ab.py tests/test_certify_java_pinning.py
git add -A scripts/calibrate_baltic.py scripts/baltic_depletable_ab.py tests/test_calibrate_baltic_a2r.py tests/test_baltic_depletable_ab.py tests/test_certify_java_pinning.py
git commit -m "feat(baltic): bounded recalibration phase a2r + A/B --extra-arm (Phase 1 contingency)"
```

---

### Task 2: Run the DE (compute checkpoint)

**Files:**
- Create: `data/baltic/calibration_results/a2r_*.json` (checkpoint/results — whatever naming the script's checkpoint callback produces)

**Interfaces:**
- Consumes: `--phase a2r` CLI from Task 1.
- Produces: the best-point params JSON for Task 3. Write the fitted flat dict (merged with the depletion base keys) to `data/baltic/calibration_results/a2r_fitted_params.json`.

- [ ] **Step 1: Launch the DE in the background**

Run: `PYTHONPATH=. .venv/bin/python scripts/calibrate_baltic.py --phase a2r --maxiter 25 --popsize 8 --workers 10`
(adjust flag names to the script's actual CLI — read `--help` first; keep total individuals ≈ 45–50 per the July convergence experience). Expected: ~3–5 h with checkpointing. No other engine jobs while it runs.

- [ ] **Step 2: Extract the fitted point**

From the phase's results JSON, take the **`log10_parameters`** field (NOT `parameters` — that
field is linear-space, and feeding it to the log-space default of `expand_param_overrides` would
double-exponentiate: zoo3 0.9116 → 8.16, silently certifying an unfitted config; review finding
2026-08-08). Expand via `expand_param_overrides(list(d), list(d.values()))` (default
`use_log_space=True`). **Sanity gate before writing:** every expanded value must lie inside its
fitted bounds (zoo3 ∈ [0.05, 2.0], benthos ∈ [0.01, 0.316], accessibilities ∈ [0.1, 0.8]) — abort
extraction on any violation. Merge with `{"ltl.depletable.enabled": "true",
"ltl.depletable.floor": "0.05", "species.regrowth.rate.sp9": "5.0",
"species.regrowth.rate.sp10": "5.0"}`, and write the flat str→str dict to
`data/baltic/calibration_results/a2r_fitted_params.json`. Record the DE loss and generation count
in the ledger. (Do NOT pass `--a2` alongside `--phase a2r` — the flags are independent and the
hybrid is unintended.)

- [ ] **Step 3: Commit the calibration artifacts**

```bash
git add data/baltic/calibration_results/
git commit -m "feat(baltic): a2r bounded-recalibration results (depletion on, gate-species objective)"
```

---

### Task 3: Certify the fitted candidate (decision gate)

**Files:**
- Create: `docs/baltic_depletable_recal_ab_2026-08-08.md` (adjust date)

**Interfaces:**
- Consumes: Task 1's `--extra-arm`; Task 2's `a2r_fitted_params.json`.

- [ ] **Step 1: Run the gate**

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_depletable_ab.py --skip-default-arms --extra-arm fitted data/baltic/calibration_results/a2r_fitted_params.json --out docs/baltic_depletable_recal_ab_2026-08-08.md`
Expected: two arms (off + fitted) × 5 seeds × 50 yr ≈ 35 min.

- [ ] **Step 2: Two-key decision rule**

1. `GATE [off]: PASS` required (baseline sanity — unchanged config, should match the prior A/B).
2. `GATE [fitted]: PASS` → proceed to Task 4 (adoption). Also record pikeperch/smelt tracked
   rows — the scientific payoff claim (overshoot reduction) must be quoted from measurement.
   `GATE [fitted]: FAIL` → STOP. Commit the report recording the negative result; the spec's
   Phase 1 concludes as "tested, not adoptable at bounded scope"; wider refits are out of the
   bounded mandate and need a fresh decision.

- [ ] **Step 3: Commit the report**

```bash
git add docs/baltic_depletable_recal_ab_2026-08-08.md
git commit -m "docs(baltic): a2r fitted-candidate gate verdict"
```

---

### Task 4: Adoption (only on GATE [fitted]: PASS)

Follow the original plan's Task 4 + Task 5 verbatim — `docs/superpowers/plans/2026-08-08-baltic-depletable-ltl-phase1.md` — with these substitutions:

- The overlay `data/baltic/baltic_param-depletion.csv` carries the FITTED values (regrowth sp11–14
  from the fit; phyto sp9–10 = 5.0; floor 0.05; enabled true). Fitted accessibilities REPLACE the
  `species.accessibility2fish.sp{9,11,12,13}` values in `data/baltic/baltic_param-ltl.csv` in
  place (they live there today; two files defining one key is a silent-precedence trap).
- Update `DEPLETION_KEYS` in `scripts/baltic_depletable_ab.py` to the adopted regrowth/floor/enable
  set (and its `test_depletion_keys_exact` expectations) so `test_depletion_raw_keys_match_harness`
  keeps enforcing a single source of truth. Record in the overlay comment that the values are the
  a2r fit (date, results JSON path, DE loss).
- Provenance comments: cite `docs/baltic_depletable_recal_ab_2026-08-08.md` as the gate evidence
  instead of the failed first A/B.
- All other steps (allowlist both copies, master include line, ltl comment, java-block test update,
  loading assertions with fitted values, certification record, CLAUDE.md gotcha) apply unchanged.

---

## Self-review notes

- Bounded mandate enforced structurally: `test_a2r_params_exactly_six` pins the parameter list; the
  objective excludes indicative species (`get_a2r_targets` + test).
- The benthos sentinel separates sp14 from the zoo group precisely because the first A/B measured
  strong benthos sensitivity; its bounds embed the literature range the science review supplied.
- The legacy `zoo` sentinel and all historical results stay untouched (regression-tested via the
  existing a2 tests).
- Single-source-of-truth chain is preserved through adoption: harness constants ↔ overlay file ↔
  loading assertions.
- Task 2/3 compute follows the measured timings from the first A/B run (arm ≈ 17 min), not the
  earlier conservative estimates.
