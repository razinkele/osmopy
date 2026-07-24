# SP-A — Baltic Stability Recalibration (params-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate `data/baltic` free parameters so all 8 focal species persist within their ICES envelopes over 50 yr+ (bounded equilibrium) instead of collapsing to herring+sprat by ~yr30 — or produce the named species that parameters alone cannot stabilise (the SP-B grid-resolution gate).

**Architecture:** Add a pure `stability_penalty()` function; fold it into the *existing* envelope-aware Baltic calibration objective (`scripts/calibrate_baltic.py`, whose legacy `w_stability` cv/trend terms are zeroed when stability is on) as an **ε-constraint** scalar `ICES_loss + Λ·max(0, stability − ε)`; trace the Pareto front by an ε-sweep over the existing single-objective `surrogate_assisted_de`; certify the chosen front point at 50 yr × 5 seeds on both engines before writing `data/baltic`.

**Tech Stack:** Python 3.12, NumPy/pandas, the in-repo calibration package (`osmose/calibration/`), `PythonEngine.run_in_memory`, pytest. No new dependencies.

## Global Constraints

- Parameters-only — **no grid, map, LTL-forcing, or fishing-policy changes** (those are SP-B/SP-C).
- The existing Baltic ICES objective in `scripts/calibrate_baltic.py` is already envelope-aware (zero inside `[lower, upper]`, `log10²` distance outside) — **reuse it; do not switch to `objectives.biomass_rmse` (point-only)**.
- `calibrate_baltic.py`'s objective **already carries** `w_stability` cv/trend penalties (default 5.0). **Zero them when stability is enabled** (`--epsilon` finite) so the new commensurate term isn't double-counted.
- Trace the Pareto front by the **ε-constraint** scalar `ICES + Λ·max(0, Stability − ε)` (sweep ε), **not** a plain weighted sum (which recovers only the convex hull and misses near-threshold front points).
- The persistence penalty is a **smooth** log10-below-floor term (commensurate with the ICES `log10²` scale), **not** a flat step that would swamp the ICES error.
- **Reuse the existing `--years`/`--seeds`**; do **not** add `--eval-years` (it collides with `--years`, default 40).
- Per-objective seed aggregation: **worst-seed** for the stability term, **mean** for the ICES term (`validate_multiseed` reduces one scalar per call → call it twice, once per component).
- The stability "late window" is **relative** (final ~third / ~10 yr of whatever horizon is run) — never an absolute year (the in-loop proxy is 35 yr, certification is 50 yr).
- Targets + weights come from `data/baltic/reference/biomass_targets.csv` (cod/herring/sprat 1.0; flounder 0.5; smelt 0.3; perch/pikeperch/stickleback 0.2). Stickleback is documented boom-bust → its variability is not penalised.
- Any write to `data/baltic` must round-trip faithfully through the native-4.4.0 reader/writer (a *value* write→read→write check — **not** the pre/post-cutover `native_440_parity.py` baseline gate, which recalibration would fail by design).
- Run Python with `.venv/bin/python`; lint with `.venv/bin/ruff check`.

---

### Task 1: Phase 0 diagnostic — identify the collapse driver

**Files:**
- Create: `scripts/baltic_stability_diagnostic.py`
- Create: `docs/baltic_stability_diagnostic_2026-07-01.md` (the finding note)

**Interfaces:**
- Consumes: `osmose.demo.osmose_demo`, `osmose.config.reader.OsmoseConfigReader`, `osmose.engine.PythonEngine`, `osmose.results.OsmoseResults`.
- Produces: a written diagnostic note naming (a) which species collapses first, (b) the dominant mortality term in its decline, (c) the confirmed free-parameter set for Task 3.

- [ ] **Step 1: Write the diagnostic script.** It must: load Baltic via `osmose_demo("baltic", tmp)`, set `simulation.time.nyear=50`, run `PythonEngine().run(cfg, output_dir, seed=s)` for `s in (42, 123, 7)` to a temp dir, then read `OsmoseResults(output_dir)` and extract per-species annual biomass and the per-species mortality decomposition from the `mortalityRate-{sp}` frames. For each seed, print: the year each focal species first drops below `0.1 × ICES-lower`, and for the earliest-collapsing species, the mean share of each mortality cause (predation/starvation/additional/fishing) in the 5 yr before its collapse.

```python
# scripts/baltic_stability_diagnostic.py  (skeleton — fill the extraction against real columns)
import sys, tempfile
from pathlib import Path
import numpy as np
from osmose.demo import osmose_demo
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults

FOCAL = ["cod","herring","sprat","flounder","perch","pikeperch","smelt","stickleback"]
LOWER = {"cod":60000,"herring":800000,"sprat":800000,"flounder":20000,
         "perch":8000,"pikeperch":4000,"smelt":20000,"stickleback":50000}

def main():
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg["simulation.time.nyear"] = "50"
    for seed in (42, 123, 7):
        outdir = tmp / f"out{seed}"; outdir.mkdir()
        PythonEngine().run(cfg, output_dir=outdir, seed=seed)
        r = OsmoseResults(outdir)
        bio = r.biomass()  # wide: Time + per-species columns
        print(f"--- seed {seed}: first year below 0.1*ICES-lower ---")
        for sp in FOCAL:
            v = bio[sp].values
            below = np.where(v < 0.1 * LOWER[sp])[0]
            print(f"  {sp:12s}: {'yr '+str(int(below[0])) if len(below) else 'persists'}")
        # mortality decomposition for the earliest collapser via r.mortality(sp) — the real accessor
        # (osmose/results.py:432-434, returns a (cause, stage) MultiIndex frame). Print the mean share
        # of each cause (predation/starvation/additional/fishing) over the 5 steps before its collapse.
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it.** `PYTHONPATH=. .venv/bin/python scripts/baltic_stability_diagnostic.py` — expect 3 seeds of first-collapse years + the mortality breakdown. Runtime ~10–15 min.
- [ ] **Step 3: Write the finding note** `docs/baltic_stability_diagnostic_2026-07-01.md`: the keystone species, the dominant mortality cause, **why the existing `w_stability` cv/trend penalty (`calibrate_baltic.py:228-235`) failed to prevent the collapse** (e.g. the measurement window is shorter than the collapse onset, or the `cv>0.2`/`trend>0.05` thresholds never trip on the slow drift), and the resulting **confirmed free-parameter set** — the `configure.py` baseline (`mortality.additional.rate`, `…additional.larva.rate`, `…starvation.rate.max`, `predation.ingestion.rate.max`) plus, **only if** recruitment is implicated, `stock.recruitment.ssbhalf.sp{i}` / `stock.recruitment.shape.sp{i}` (percids) / `species.relativefecundity.sp{i}`.
- [ ] **Step 4: Commit.**
```bash
git add scripts/baltic_stability_diagnostic.py docs/baltic_stability_diagnostic_2026-07-01.md
git commit -m "diag(baltic): Phase 0 collapse-driver diagnostic + finding note"
```

---

### Task 2: `stability_penalty` pure function + unit tests

**Files:**
- Create: `osmose/calibration/stability.py`
- Create: `tests/test_stability_penalty.py`

**Interfaces:**
- Consumes: a WIDE biomass `pd.DataFrame` (a `Time`/`time` column + one numeric column per species) and the `BiomassTarget` list (`species`, `target`, `lower`, `upper`, `weight`).
- Produces: `stability_penalty(biomass, targets, *, phi=0.1, boombust=frozenset({"stickleback"}), warmup_frac=0.2) -> float`. Returns 0.0 for a flat-in-envelope trajectory; large for extinction/decline/explosion. Pure + side-effect-free (so the picklable objective wrapper that calls it stays picklable).

- [ ] **Step 1: Write the failing tests.**
```python
# tests/test_stability_penalty.py
import numpy as np, pandas as pd, pytest
from osmose.calibration.stability import stability_penalty

class T:  # minimal BiomassTarget stand-in (species, lower, upper, weight)
    def __init__(s, species, lower, upper, weight): s.species=species; s.lower=lower; s.upper=upper; s.weight=weight; s.target=(lower+upper)/2

def _wide(series: dict, n=50):
    return pd.DataFrame({"Time": np.arange(n), **{k: np.asarray(v, float) for k, v in series.items()}})

TGT = [T("cod", 60000, 250000, 1.0)]

def test_flat_in_envelope_is_zero():
    bio = _wide({"cod": np.full(50, 120000.0)})
    assert stability_penalty(bio, TGT) == pytest.approx(0.0, abs=1e-6)

def test_collapse_is_heavily_penalised():
    bio = _wide({"cod": np.linspace(120000, 1.0, 50)})  # collapses below the floor
    assert stability_penalty(bio, TGT) > 1.0

def test_sub_collapse_decline_tracks_trend():
    # both stay ABOVE the persistence floor (0.1*lo=6000) -> trend/envelope, not persistence, drive it
    gentle = _wide({"cod": np.linspace(120000, 80000, 50)})
    steep = _wide({"cod": np.linspace(120000, 12000, 50)})
    assert stability_penalty(steep, TGT) > stability_penalty(gentle, TGT)

def test_persistence_floor_isolated():
    # held just above the floor vs dipping below it -> the persistence term is what differs
    lo = 60000
    alive = _wide({"cod": np.full(50, 0.2 * lo)})  # below envelope but above the 0.1*lo floor
    extinct = _wide({"cod": np.concatenate([np.full(25, 0.2 * lo), np.full(25, 0.05 * lo)])})
    assert stability_penalty(extinct, TGT) > stability_penalty(alive, TGT)

def test_explosion_is_penalised():
    bio = _wide({"cod": np.linspace(120000, 1e7, 50)})
    assert stability_penalty(bio, TGT) > 1.0

def test_boombust_stickleback_not_punished_for_variance():
    tgt = [T("stickleback", 50000, 500000, 0.2)]
    osc = _wide({"stickleback": 200000 + 150000*np.sin(np.arange(50))})  # in-envelope oscillation
    assert stability_penalty(osc, tgt) < 0.5  # variability not charged for boom-bust species

def test_weight_scales_penalty():
    hi = [T("cod", 60000, 250000, 1.0)]; lo = [T("cod", 60000, 250000, 0.2)]
    bio = _wide({"cod": np.linspace(120000, 1.0, 50)})
    assert stability_penalty(bio, hi) > stability_penalty(bio, lo)
```

- [ ] **Step 2: Run to verify failure.** `PYTHONPATH=. .venv/bin/python -m pytest tests/test_stability_penalty.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `stability_penalty`.**
```python
# osmose/calibration/stability.py
"""Bounded-equilibrium instability penalty for SP-A Baltic stability calibration.

0.0 = bounded & in-envelope; grows with extinction, drift (trend), envelope violation, and
boom-bust variability. Pure function so the picklable calibration objective can call it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_W_PERSIST, _W_ENVELOPE, _W_TREND, _W_VAR = 10.0, 1.0, 3.0, 1.0

def _series(biomass: pd.DataFrame, sp: str) -> np.ndarray | None:
    if sp not in biomass.columns:
        return None
    return np.clip(np.asarray(biomass[sp].values, float), 1e-9, None)

def stability_penalty(
    biomass: pd.DataFrame,
    targets,
    *,
    phi: float = 0.1,
    boombust = frozenset({"stickleback"}),
    warmup_frac: float = 0.2,
) -> float:
    """Scalar instability penalty over the post-warmup window (0 = bounded-stable)."""
    n = len(biomass)
    if n < 5:
        return float("inf")
    start = int(round(warmup_frac * n))
    late = max(start + 1, n - 10)  # relative final-decade
    total = 0.0
    for t in targets:
        v = _series(biomass, t.species)
        if v is None:
            continue
        win = v[start:]
        lo, hi, w = float(t.lower), float(t.upper), float(t.weight)
        # persistence: SMOOTH log10-distance of the window-min BELOW the floor phi*lo (commensurate
        # with the ICES log10^2 error; 0 if above the floor — no flat step that would swamp ICES)
        wmin = float(win.min()); floor = phi * lo
        persist = float(np.log10(floor / wmin) ** 2) if wmin < floor else 0.0
        # envelope: fraction of window outside [lo,hi] + final-decade mean outside
        frac_out = float(np.mean((win < lo) | (win > hi)))
        late_mean = float(np.mean(v[late:]))
        late_out = 0.0 if lo <= late_mean <= hi else float(np.log10(max(late_mean, 1e-9) / np.clip(late_mean, lo, hi)) ** 2)
        envelope = frac_out + late_out
        # trend: |slope| of log10-biomass; MAX of full-window and late-window (final third) slopes so a
        # config that holds flat then tips in the last years is not averaged into a near-zero slope
        def _slope(a):
            return float(np.polyfit(np.arange(len(a), dtype=float), np.log10(a), 1)[0]) if len(a) >= 3 else 0.0
        third = max(3, len(win) // 3)
        trend = max(abs(_slope(win)), abs(_slope(win[-third:])))
        # variability: CV, not charged for documented boom-bust species
        mean = float(np.mean(win))
        cv = float(np.std(win) / mean) if mean > 0 else 0.0
        variability = 0.0 if t.species in boombust else cv
        total += w * (_W_PERSIST * persist + _W_ENVELOPE * envelope + _W_TREND * trend + _W_VAR * variability)
    return float(total)
```

- [ ] **Step 4: Run to verify pass.** `PYTHONPATH=. .venv/bin/python -m pytest tests/test_stability_penalty.py -q` → all pass. Then `.venv/bin/ruff check osmose/calibration/stability.py tests/test_stability_penalty.py`.
- [ ] **Step 5: Commit.**
```bash
git add osmose/calibration/stability.py tests/test_stability_penalty.py
git commit -m "feat(calibration): stability_penalty (persistence+envelope+trend+variability)"
```

---

### Task 3: Scalarized objective + CLI in `calibrate_baltic.py`

**Files:**
- Modify: `scripts/calibrate_baltic.py` (the objective `__call__` ~line 183-220; argparse block)
- Test: `tests/test_calibrate_baltic_stability.py`

**Interfaces:**
- Consumes: `stability_penalty` (Task 2); the existing envelope-aware ICES error **and** the legacy
  `w_stability` cv/trend penalties already inside `calibrate_baltic.py`'s `_ObjectiveWrapper`.
- Produces: an objective that, when stability is enabled, returns the **ε-constraint** scalar
  `ices_loss + Λ·max(0, Stability − ε)` (`Λ` a large fixed constant, e.g. `1e3`) with **mean-over-seeds**
  ICES and **worst-over-seeds** Stability; each run records `ices_loss` and `stability` **separately**.
  Reuses the existing `--years` (eval horizon, default 40) and `--seeds`; adds `--epsilon <float>`
  (default `inf` = stability OFF = exact legacy score). The legacy `w_stability` cv/trend terms are
  **zeroed when `--epsilon` is finite** (the new commensurate term supersedes them — see spec premise).

- [ ] **Step 1: Write the failing test** (`--epsilon inf` reproduces the legacy score; a finite ε
  raises the score for an unstable candidate). Build the `_ObjectiveWrapper` directly (import via
  `importlib` from `scripts/calibrate_baltic.py`), evaluate one param vector with `epsilon=float("inf")`
  and with a finite `epsilon`, on a SHORT run; assert `score(finite_eps) >= score(inf)` and that with
  `epsilon=inf` it equals the pre-change legacy score within 1e-9. Mark `@pytest.mark.slow`.
```python
# tests/test_calibrate_baltic_stability.py
import importlib.util, pathlib, numpy as np, pytest
def _load():
    s = importlib.util.spec_from_file_location("cb", pathlib.Path("scripts/calibrate_baltic.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

@pytest.mark.slow
def test_epsilon_inf_is_legacy_and_finite_eps_raises_unstable_score():
    cb = _load()
    # build two objectives over the SAME targets/params/seed: stability off vs a tight epsilon.
    # (Use cb's objective factory / _ObjectiveWrapper constructor — see its __init__ signature.)
    off  = cb.make_objective(epsilon=float("inf"), years=8, seeds=(42,))      # stability OFF
    tight = cb.make_objective(epsilon=0.0,         years=8, seeds=(42,))      # demand near-perfect stability
    x = cb.default_x0()
    assert tight(x) >= off(x)
```
(Implementer: adapt the exact factory/constructor names to `calibrate_baltic.py`'s API; the contract
asserted is what matters — `epsilon=inf` ⇒ legacy, finite ε ⇒ ≥.)

- [ ] **Step 2: Run to verify failure.** `PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibrate_baltic_stability.py -q` → FAIL.
- [ ] **Step 3a: Thread the WIDE biomass frame out of `run_simulation`** (prerequisite — the gap the review found). `run_simulation` (calibrate_baltic.py:89) reads `bio = results.biomass()` (line 115) but returns ONLY the summary `species_stats` dict (line 147); the WIDE frame is discarded when the `TemporaryDirectory`/`results` close. Change it to also return the frame, e.g. `return species_stats, bio.copy()`; update `_simulate_and_compute_stats` (line ~240) and `_ObjectiveWrapper.__call__` (line ~183) to receive and pass it.
- [ ] **Step 3b: Add `--epsilon` + thread `--years`.** Add `epsilon: float = float("inf")` to `_ObjectiveWrapper.__init__`; wire `--epsilon` in argparse; **reuse the existing `--years`/`--seeds`** (do NOT add `--eval-years` — it collides with `--years`, default 40). When `epsilon` is finite, set the legacy `w_stability` to `0.0` for this objective (avoid double-counting the cv/trend penalties).
- [ ] **Step 3c: Compute the ε-constraint scalar.** In `__call__`, after the existing envelope `ices_loss` loop, compute `stab = stability_penalty(bio, self.targets)`; return `ices_loss + 1e3 * max(0.0, stab - self.epsilon)`. With `epsilon=inf` the hinge is 0 → exact legacy score.
- [ ] **Step 3d: Multiseed — two aggregations.** `validate_multiseed` reduces ONE scalar per call, so call it **twice**: `make_ices(seed)` (objective returns only `ices_loss`) → take `result["mean"]`; `make_stab(seed)` (objective returns only `stab`) → take `result["worst_value"]`. Final score `= mean_ices + 1e3*max(0.0, worst_stab - epsilon)`.
- [ ] **Step 3e: Record components separately.** Persist `ices_loss` and `stab` as distinct fields in the per-run payload (`_save_run_for_de`) so the Task-4 sweep reads the true front, not just the summed score.
- [ ] **Step 4: Run to verify pass + regression.** `PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibrate_baltic_stability.py -q` and the existing calibration tests `tests/test_*calibrat*` → pass (legacy score unchanged at `epsilon=inf`). Lint.
- [ ] **Step 5: Commit.**
```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_stability.py
git commit -m "feat(calibration): scalarized ICES+lambda*stability objective + CLI"
```

---

### Task 4: λ-sweep driver + integration smoke

**Files:**
- Create: `scripts/baltic_stability_sweep.py`
- Test: `tests/test_baltic_stability_sweep_smoke.py`

**Interfaces:**
- Consumes: `calibrate_baltic.py` (the λ-aware objective + its `surrogate_assisted_de` call), `osmose.calibration.surrogate_de.surrogate_assisted_de` (single-objective), `validate_multiseed`.
- Produces: a script that runs the calibration once per **ε** in a loose→tight grid (e.g. `[inf, 5.0, 2.0, 1.0, 0.5, 0.2]`), reading the **separately-recorded** `ices_loss` and `stability` (Task 3e) from each solve, records `{epsilon, params, ices_loss, stability, per_species_summary}` to `data/baltic/reference/stability_sweep.json`, and prints the front. A `--smoke` flag runs one ε with a tiny eval budget + 5-yr horizon for CI.

- [ ] **Step 1: Write the smoke test.**
```python
# tests/test_baltic_stability_sweep_smoke.py
import subprocess, sys, json, pathlib, pytest

@pytest.mark.slow
def test_sweep_smoke_runs_end_to_end(tmp_path):
    out = tmp_path / "sweep.json"
    r = subprocess.run([sys.executable, "scripts/baltic_stability_sweep.py", "--smoke", "--out", str(out)],
                       capture_output=True, text=True, timeout=600, env={"PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data and "epsilon" in data[0] and "stability" in data[0] and "ices_loss" in data[0]
```
- [ ] **Step 2: Run to verify failure** → FAIL (script missing).
- [ ] **Step 3: Implement** `scripts/baltic_stability_sweep.py`: parse `--smoke`/`--out`/`--epsilons`/`--years`/`--seeds`; for each ε, invoke the Task-3 calibration (`subprocess` `calibrate_baltic.py --epsilon E --years Y --seeds N`), read the best params + its separately-recorded `ices_loss`/`stability`/per-species summary from the run payload, append to the list, write JSON. `--smoke` → `--epsilons 1.0 --years 5` and the smallest `surrogate_assisted_de` budget (`n_initial≈8, n_iterations=1, n_topk=4`).
- [ ] **Step 4: Run to verify pass** (`-m slow`). Lint.
- [ ] **Step 5: Commit.**
```bash
git add scripts/baltic_stability_sweep.py tests/test_baltic_stability_sweep_smoke.py
git commit -m "feat(calibration): lambda-sweep stability driver + smoke test"
```

---

### Task 5: Certification harness (50 yr × 5 seeds × both engines)

**Files:**
- Create: `scripts/baltic_stability_certify.py`
- Reuse: the persistence/envelope helpers from `osmose/calibration/stability.py` (Task 2), the ICES
  envelopes from `data/baltic/reference/biomass_targets.csv`, `osmose.java_background_staging`, and the
  Java runner path. **(NOT `scripts/validate_baltic_vs_ices.py`** — that is a spatial grid-mask
  validator over 4 species needing network/MCP access, unrelated to this biomass-time-series cert table.)

**Interfaces:**
- Consumes: a chosen params dict (from the sweep JSON) or a candidate `data/baltic` copy.
- Produces: a per-species table — for each of 8 species: `min_biomass`, `late_decade_mean`, `persists?` (`min > 0.1·lower`), `in_envelope?` (`lower ≤ late_mean ≤ upper`), on **Python** (50 yr × seeds 42/123/7/999/2024) and a **Java** cross-check (single seed, staged via `java_background_staging`). Writes `docs/baltic_stability_certification_2026-07-01.md` with the table + the verdict (N/8 stable-in-ICES; named failures → SP-B gate).

- [ ] **Step 1: Implement** the harness: apply the params to a staged Baltic config, run Python `run_in_memory` 50 yr for each seed, compute the table via the same persistence/envelope rules as `stability_penalty` (import the helpers), then a single Java run (reuse the C2 staging) and compare survivor sets. No new unit test — this is a reporting harness; correctness is the recorded table.
- [ ] **Step 2: Run** on the current (un-recalibrated) params first as a **baseline sanity check** — it must reproduce the known collapse (2/8 Python), proving the harness detects instability. `PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current`.
- [ ] **Step 3: Commit.**
```bash
git add scripts/baltic_stability_certify.py
git commit -m "feat(calibration): 50yr x multiseed x both-engine stability certification harness"
```

---

### Task 6: Run the calibration, certify, and write (or gate)

**Files:**
- Modify (conditional): `data/baltic/*` (only the recalibrated parameter values), `docs/baltic_stability_certification_2026-07-01.md`

**Interfaces:**
- Consumes: Tasks 3–5. Produces: either a recalibrated, parity-preserving `data/baltic`, or the SP-B gate note.

- [ ] **Step 1: Run the full sweep** (overnight–multi-day, or HPC): `PYTHONPATH=. .venv/bin/python scripts/baltic_stability_sweep.py --eval-years 35 --seeds 3 --out data/baltic/reference/stability_sweep.json`.
- [ ] **Step 2: Certify** the best in-envelope front point: `scripts/baltic_stability_certify.py --params <best-from-sweep>`. Read the per-species table.
- [ ] **Step 3a — if all 8 persist & in-envelope:** apply the params to `data/baltic`, then **verify the written config round-trips faithfully** — read it back with `OsmoseConfigReader`, re-write via the native-4.4.0 writer, and assert the *changed parameter values* survive a write→read→write cycle. (Do **NOT** use `scripts/native_440_parity.py`: its CLI is two-phase `capture <name>` / `gate <name>`, its tolerance is `1e-9` not "bit-exact", and it gates the recalibrated config against the *pre-cutover 4.3.3 baseline* — recalibration deliberately changes values, so that gate would fail by design.) Also confirm `data/baltic` default `nyear` still runs a healthy ecosystem. Commit the recalibrated config + certification note.
- [ ] **Step 3b — if < 8/8:** do **not** modify `data/baltic`. Record the named failing species + the structural-vs-tunable evidence (did sweeping their params move them?) in the certification note as the **SP-B decision gate** input. Commit the note only.
- [ ] **Step 4: Commit.**
```bash
git add data/baltic docs/baltic_stability_certification_2026-07-01.md   # 3a
# or
git add docs/baltic_stability_certification_2026-07-01.md               # 3b (gate only)
git commit -m "calib(baltic): SP-A stability recalibration result + certification"
```

---

## Notes for the executor

- Tasks 1, 5, 6 are **research/harness** tasks (run + record), not pure TDD — their deliverable is a written finding/table, not a green assert. Tasks 2–4 are TDD with real tests.
- The long calibration (Task 6 Step 1) is the only multi-hour step; everything else is minutes. Consider running Task 6 Step 1 in the background / on the HPC container.
- `@pytest.mark.slow` the sim-running tests so the default suite stays fast.
