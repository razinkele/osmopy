# Reproducible Baltic Calibration Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the committed `data/baltic` config *be* a reproducible, calibrated Shepherd baseline that reaches ≥2/8 ICES envelopes with no collapsed species — instead of an uncalibrated config that collapses 6/8 at equilibrium.

**Architecture:** (1) Tighten the Shepherd β bounds so density-dependent recruitment can't under-compensate (β<1 fails to cap overshooters) or, via a short-transient fit, over-crush low-SSB stocks. (2) Re-fit at full 40-year equilibrium so the existing extinction penalty (already 100.0 per collapsed species) actually bites. (3) Write the calibrated parameters into the *tracked* CSVs in place (preserving comments) via a new `apply_calibration.py`, and commit a provenance snapshot. (4) Run the surrogate-Bayesian UQ layer on the β parameters to report identifiability. (5) Verify and update the model report.

**Tech Stack:** Python 3.12, `scripts/calibrate_baltic.py` (scipy differential evolution), `osmose/config` (reader/writer), `scripts/evaluate_calibration_vs_ices.py`, `osmose/calibration/uq` (parallel surrogate-Bayesian layer), pytest.

## Global Constraints

- Run tests with `.venv/bin/python -m pytest`; lint `.venv/bin/ruff check`.
- OSMOSE config keys are lowercase dot-separated; CSVs are `;`-separated; comment lines start with `#`. Preserve file structure and comments on write-back.
- Any script that uses the parallel UQ evaluator MUST set `OMP_NUM_THREADS=1`/`NUMBA_NUM_THREADS=1` per worker and guard top-level code with `if __name__ == "__main__":` (spawn pool). See `docs/superpowers/2026-07-23-uq-real-data-validation.md`.
- Focal species indices: sp0 cod, sp1 herring, sp2 sprat, sp3 flounder, sp4 perch, sp5 pikeperch, sp6 smelt, sp7 stickleback.
- Baseline evidence (fresh 3-seed 40-yr run, this session): phase13 β = cod 1.88, herring 0.76, sprat 0.75, flounder 1.80, perch 1.60, pikeperch 0.50, smelt 2.56, stickleback 1.79 → 1/8 in range, flounder collapsed. Under-compensation (β<1) on pikeperch/herring/sprat; flounder over-crushed by a short-transient fit.

---

### Task 1: Tighten the Shepherd β (shape) bounds

**Files:**
- Modify: `scripts/calibrate_baltic.py:801` (inside `get_phase13_shepherd_params`)
- Test: `tests/test_calibrate_baltic_bounds.py` (create)

**Interfaces:**
- Consumes: `get_phase13_shepherd_params() -> (keys: list[str], bounds: list[tuple[float,float]], x0: list[float])` — bounds in log10 space (objective applies `10**x`).
- Produces: the same signature; only the β (`stock.recruitment.shape.sp*`) bounds change from `(log10(0.3), log10(5.0))` to `(log10(1.0), log10(3.0))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate_baltic_bounds.py
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from calibrate_baltic import get_phase13_shepherd_params  # noqa: E402


def test_shepherd_beta_bounds_forbid_undercompensation_and_overcrush():
    keys, bounds, _ = get_phase13_shepherd_params()
    beta = [(k, b) for k, b in zip(keys, bounds) if k.startswith("stock.recruitment.shape.")]
    assert len(beta) == 8
    for k, (lo, hi) in beta:
        # lower bound >= 1.0 (no under-compensation), upper <= 3.0 (no extreme over-crush)
        assert math.isclose(10**lo, 1.0, rel_tol=1e-6), f"{k} lower {10**lo} != 1.0"
        assert math.isclose(10**hi, 3.0, rel_tol=1e-6), f"{k} upper {10**hi} != 3.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_bounds.py -v`
Expected: FAIL (current lower bound is 0.3, upper 5.0).

- [ ] **Step 3: Change the bound line**

In `scripts/calibrate_baltic.py`, in `get_phase13_shepherd_params`, change:
```python
        shape_bounds.append((np.log10(0.3), np.log10(5.0)))
```
to:
```python
        shape_bounds.append((np.log10(1.0), np.log10(3.0)))  # >=1 forbids under-compensation; <=3 avoids over-crush
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_bounds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_bounds.py
git commit -m "fix(calib): tighten Shepherd beta bounds to [1.0, 3.0] (forbid under-compensation / over-crush)"
```

---

### Task 2: `apply_calibration.py` — write calibrated params into tracked CSVs

**Files:**
- Create: `scripts/apply_calibration.py`
- Test: `tests/test_apply_calibration.py`

**Interfaces:**
- Produces:
  - `set_key(path: Path, key: str, value) -> None` — set/append `key;value` in a `;`-separated OSMOSE CSV, in place, preserving other lines and comments.
  - `apply_calibration(results_path: Path, config_dir: Path) -> None` — switch sp0–7 to Shepherd SR and write every param from `json.load(results_path)["parameters"]` into its owning tracked CSV.
- Consumes: results JSON shape `{"parameters": {key: value, ...}}` (e.g. `data/baltic/calibration_results/phase13_results.json`).

Routing (key prefix → tracked file under `config_dir`):
`stock.recruitment.` → `baltic_param-reproduction.csv`; `mortality.additional.` → `baltic_param-additional-mortality.csv`; `fisheries.rate.base.` → `baltic_param-fishing.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apply_calibration.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from apply_calibration import set_key  # noqa: E402


def test_set_key_updates_existing_line_preserving_comments(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("# a comment\nstock.recruitment.type.sp0;beverton_holt\nother.key;9\n")
    set_key(f, "stock.recruitment.type.sp0", "shepherd")
    lines = f.read_text().splitlines()
    assert "# a comment" in lines  # comment preserved
    assert "stock.recruitment.type.sp0;shepherd" in lines
    assert "other.key;9" in lines  # untouched


def test_set_key_appends_when_absent(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("existing;1\n")
    set_key(f, "stock.recruitment.shape.sp2", "1.5")
    assert "stock.recruitment.shape.sp2;1.5" in f.read_text().splitlines()
    assert "existing;1" in f.read_text().splitlines()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_apply_calibration.py -v`
Expected: FAIL with ImportError (module not created).

- [ ] **Step 3: Write the implementation**

```python
# scripts/apply_calibration.py
"""Apply a calibration results JSON to the tracked Baltic config CSVs, in place.

Switches all 8 focal species to Shepherd stock-recruitment and writes every
calibrated mortality / fishing / recruitment parameter into its owning CSV,
editing only the affected key lines so comments and structure are preserved.
Run: .venv/bin/python scripts/apply_calibration.py <results.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CONFIG_DIR = Path("data/baltic")
_FILE_FOR = {
    "stock.recruitment.": "baltic_param-reproduction.csv",
    "mortality.additional.": "baltic_param-additional-mortality.csv",
    "fisheries.rate.base.": "baltic_param-fishing.csv",
}


def _file_for(key: str, config_dir: Path) -> Path:
    for prefix, fname in _FILE_FOR.items():
        if key.startswith(prefix):
            return config_dir / fname
    raise KeyError(f"no tracked CSV owns key {key!r}")


def set_key(path: Path, key: str, value) -> None:
    """Set ``key;value`` in a ``;``-separated OSMOSE CSV, in place; append if absent."""
    lines = path.read_text().splitlines() if path.exists() else []
    out, found = [], False
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and ";" in s and s.split(";", 1)[0].strip().lower() == key.lower():
            out.append(f"{key};{value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key};{value}")
    path.write_text("\n".join(out) + "\n")


def apply_calibration(results_path: Path, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
    params = json.loads(Path(results_path).read_text())["parameters"]
    repro = config_dir / _FILE_FOR["stock.recruitment."]
    for i in range(8):
        set_key(repro, f"stock.recruitment.type.sp{i}", "shepherd")
    for key, val in params.items():
        set_key(_file_for(key, config_dir), key, val)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a calibration JSON to tracked Baltic CSVs")
    ap.add_argument("results_json")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    args = ap.parse_args()
    cfg_dir = Path(args.config_dir)
    apply_calibration(Path(args.results_json), cfg_dir)

    from osmose.config import OsmoseConfigReader  # roundtrip check

    cfg = OsmoseConfigReader().read(cfg_dir / "baltic_all-parameters.csv")
    params = json.loads(Path(args.results_json).read_text())["parameters"]
    for key, val in params.items():
        got = cfg.get(key.lower())
        assert got is not None and abs(float(got) - float(val)) < 1e-6, f"{key}: {got!r} != {val}"
    print(f"applied {len(params)} params + set 8x shepherd type; roundtrip OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_apply_calibration.py -v`
Expected: PASS.

- [ ] **Step 5: Add a roundtrip integration test**

```python
# append to tests/test_apply_calibration.py
import json
from apply_calibration import apply_calibration  # noqa: E402
from osmose.config import OsmoseConfigReader  # noqa: E402


def test_apply_calibration_roundtrips_through_reader(tmp_path):
    # minimal config dir with the three target CSVs + a master include
    cfg = tmp_path
    (cfg / "baltic_param-reproduction.csv").write_text("stock.recruitment.type.sp0;beverton_holt\n")
    (cfg / "baltic_param-additional-mortality.csv").write_text("mortality.additional.rate.sp0;0.1\n")
    (cfg / "baltic_param-fishing.csv").write_text("fisheries.rate.base.sp0;0.2\n")
    results = cfg / "r.json"
    results.write_text(json.dumps({"parameters": {
        "mortality.additional.rate.sp0": 3.7,
        "fisheries.rate.base.sp0": 0.077,
        "stock.recruitment.shape.sp0": 1.88,
        "stock.recruitment.ssbhalf.sp0": 120000.0,
    }}))
    apply_calibration(results, cfg)
    repro = (cfg / "baltic_param-reproduction.csv").read_text()
    assert "stock.recruitment.type.sp0;shepherd" in repro
    assert "stock.recruitment.shape.sp0;1.88" in repro
    mort = (cfg / "baltic_param-additional-mortality.csv").read_text()
    assert "mortality.additional.rate.sp0;3.7" in mort
```

- [ ] **Step 6: Run and commit**

Run: `.venv/bin/python -m pytest tests/test_apply_calibration.py -v` → PASS
```bash
git add scripts/apply_calibration.py tests/test_apply_calibration.py
git commit -m "feat(calib): apply_calibration.py — write calibrated params into tracked CSVs in place"
```

---

### Task 3: Re-fit at equilibrium, apply, and verify acceptance

**Files:**
- Run: `scripts/calibrate_baltic.py` (phase 13), `scripts/apply_calibration.py`, `scripts/evaluate_calibration_vs_ices.py`
- Create: `data/baltic/calibration_results/phase13_equilibrium.json` (the new fit; un-gitignore in Task 5)

**Interfaces:**
- Consumes: tightened bounds (Task 1), `apply_calibration` (Task 2).
- Produces: a re-fit results JSON; an updated tracked `data/baltic` config.

- [ ] **Step 1: Launch the equilibrium re-fit** (~4.5 h unattended, background)

```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OSMOSE_DE_WORKERS=16 \
  .venv/bin/python scripts/calibrate_baltic.py --phase 13 --optimizer de --seeds 3 \
  --years 40 --popsize-mult 5 --warm-start data/baltic/calibration_results/phase12_results.json \
  --skip-warm-start-keys mortality.additional.rate.sp0 --patience 20 --wall-clock-cap-h 4 \
  --checkpoint-every 5
```
Expected: writes a results JSON under `data/baltic/calibration_results/` (name per the script's convention). Note the exact path it writes.

- [ ] **Step 2: Apply the fit to the tracked config**

```bash
.venv/bin/python scripts/apply_calibration.py <path-to-refit-results.json>
```
Expected: `applied N params + set 8x shepherd type; roundtrip OK`.

- [ ] **Step 3: Acceptance — fresh run of the COMMITTED config must not collapse**

Reuse `scratchpad/run_present_fit.py` (or re-run `evaluate_calibration_vs_ices.py --params <refit.json> --mode shepherd --years 40 --seed 0/1/2`) and assert: **≥2/8 in range AND zero species with biomass ≤ 1 t** (flounder alive). If flounder still collapses, lower its β upper bound specifically (per-species cap in Task 1) and re-fit; do not hand-tune the committed CSV.

- [ ] **Step 4: Commit the updated config**

```bash
git add data/baltic/baltic_param-reproduction.csv data/baltic/baltic_param-additional-mortality.csv data/baltic/baltic_param-fishing.csv
git commit -m "feat(baltic): commit equilibrium-calibrated Shepherd baseline (>=2/8 ICES, no collapse)"
```

---

### Task 4: Surrogate-Bayesian identifiability check on the β parameters

**Files:**
- Create: `scripts/uq_beta_identifiability.py`

**Interfaces:**
- Consumes: `osmose.calibration.uq` (`make_engine_evaluator(..., n_workers>1)`, `run_surrogate_bayes`), the committed calibrated config.
- Produces: a printed per-β report of concentration (posterior SD / prior SD) and centeredness, flagging prior-dominated (fragile) β's.

- [ ] **Step 1: Write the identifiability script**

Self-consistency setup around the calibrated β\* (mirror `scratchpad/selfconsist_run2.py`): vary `stock.recruitment.shape.sp{i}` for the assessed species in a tight box around the fitted values, engine-generate targets at β\*, run `run_surrogate_bayes` with the pass-through gate, and report `post_sd / box_sd` (concentration) and `|post_mean − β*| / post_sd` (centeredness) per β. Set `OMP_NUM_THREADS=1`/`NUMBA_NUM_THREADS=1`; guard with `if __name__ == "__main__":`; save the design to disk.

- [ ] **Step 2: Run it and record which β's are identifiable vs prior-dominated**

Run (background, ~30–60 min): `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -u scripts/uq_beta_identifiability.py`
Expected: a concentration/centeredness table. β's with concentration ≈ 1.0 are prior-dominated → their bounds/values are weakly constrained by the data and should be pinned or re-weighted rather than trusted.

- [ ] **Step 3: Commit**

```bash
git add scripts/uq_beta_identifiability.py
git commit -m "feat(uq): beta identifiability check on the calibrated Shepherd baseline"
```

---

### Task 5: Commit provenance snapshot and update the model report

**Files:**
- Modify: `.gitignore` (un-ignore the one named snapshot), add `data/baltic/calibration_results/phase13_equilibrium.json`
- Modify: `docs/baltic_model_report_2026-07-24.docx` (regenerate via `scratchpad/build_report.py`), `docs/superpowers/2026-07-23-uq-real-data-validation.md` (note the baseline is fixed)

- [ ] **Step 1: Un-gitignore the named snapshot**

In `.gitignore`, below the `data/baltic/calibration_results/` line add:
```
!data/baltic/calibration_results/phase13_equilibrium.json
```
Then `git add -f data/baltic/calibration_results/phase13_equilibrium.json`.

- [ ] **Step 2: Regenerate the model report from the now-committed calibrated config**

Re-run the fresh-calibrated-fit + report build (`scratchpad/run_calibrated_fit.py` then `scratchpad/build_report.py`). Confirm §3.3 now reads "committed config is the calibrated baseline" (edit `build_report.py`'s §3.3 text accordingly) and the in-range count reflects the new fit.

- [ ] **Step 3: Commit**

```bash
git add .gitignore data/baltic/calibration_results/phase13_equilibrium.json docs/baltic_model_report_2026-07-24.docx docs/superpowers/2026-07-23-uq-real-data-validation.md
git commit -m "docs(baltic): commit reproducible calibration snapshot; report reflects fixed baseline"
```

---

## Self-Review

- **Spec coverage:** Task 1 = tighten β bounds; Task 2 = write-back tooling (reproducibility); Task 3 = equilibrium re-fit + apply + acceptance (the ≥2/8, no-collapse criterion); Task 4 = UQ identifiability; Task 5 = provenance snapshot + report update. The scope's "extinction penalty" decision is resolved in the plan preamble (already 100.0; equilibrium fit makes it bite) — no separate task needed.
- **Placeholder scan:** Task 1 and Task 2 carry complete code and exact commands. Task 3/4/5 are execution/run tasks whose commands are exact; Task 4's script mirrors an existing scratchpad script (referenced, not re-pasted, because it is an existing artifact in this repo).
- **Type consistency:** `set_key(path, key, value)` and `apply_calibration(results_path, config_dir)` are used consistently across Task 2's steps and Task 3's apply step. `get_phase13_shepherd_params` signature matches its use in Task 1.
- **Open risk:** if the β upper bound of 3.0 does not prevent flounder collapse at equilibrium, Task 3 Step 3 routes back to a per-species β cap in Task 1 (documented, not hand-tuning the CSV).
