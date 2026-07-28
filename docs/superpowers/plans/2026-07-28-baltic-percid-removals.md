# Baltic Percid Missing-Removals — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the Baltic model's percid/smelt overshoot (perch ~2×, pikeperch ~90×, smelt ~5× over ICES envelopes) by adding two scientifically-grounded removals the model currently omits — realistic percid fishing mortality (recreational + coastal, under-reported to ICES) and cormorant predation — without regressing the well-assessed stocks.

**Architecture:** On the aggregate 8-species 5/8 baseline (branch off `646a36d`), set fixed elevated percid F, add a tunable cormorant predator column to the accessibility matrix, expose cormorant biomass/ingestion as calibration levers, gate the whole thing behind one cheap forward-sim feasibility check, then a warm-started re-calibration with a pre-registered acceptance bar and revert rule.

**Tech Stack:** Python 3.12; OSMOSE config CSVs; `scripts/calibrate_baltic.py`, `apply_calibration.py`, `baltic_stability_certify.py`; pytest; `.venv/bin/python`.

**Design:** `docs/superpowers/specs/2026-07-28-baltic-fishing-forced-cod-topdown-control-design.md`.

## Global Constraints

- All work on a branch off commit `646a36d` (aggregate 8-species baseline). `master` keeps the disaggregation experiment untouched. **Do NOT modify `master`'s config.**
- Predator indices on `646a36d`: **sp14 = GreySeal, sp15 = Cormorant** (there is NO sp16). Every cormorant key uses **sp15**. (Master's disaggregated config uses sp16 — do not copy those.)
- Run tests: `.venv/bin/python -m pytest`. Lint: `.venv/bin/ruff check`. Engine runs: `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1` per process.
- Honest bar: perch (~2×) is plausibly closable; **pikeperch (~90×) very likely is not** (cormorants reach only juvenile pikeperch; the gap is partly coarse-grid structural). Record what moves; do not force envelope-membership for pikeperch.
- Percid F is **fixed** (not a calibration free param) so it is not optimised away.

---

### Task 1: Branch + verify the clean baseline

**Files:** none created; establishes the working config state.

**Interfaces:**
- Produces: a verified 8-species aggregate baseline (5/8 in-envelope, obj ~2.33, cod ~64 kt) recorded as the no-regression reference.

- [ ] **Step 1: Create the branch off the aggregate baseline**

```bash
git -C /home/razinka/osmopy checkout -b baltic-percid-removals 646a36d
```

- [ ] **Step 2: Restore the 5/8 calibrated params**

```bash
cd /home/razinka/osmopy && .venv/bin/python scripts/apply_calibration.py data/baltic/calibration_results/phase13_equilibrium.json
```
Expected: `applied 39 params + set 8x shepherd type; roundtrip OK` (8 species — confirms the 8-species-era artifact applies on the 8-species config).

- [ ] **Step 3: Certify the baseline and record the reference table**

```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --years 50 --out docs/baltic_percid_baseline_2026-07-28.md
```
Expected: cod in-envelope (~61–68 kt), herring/sprat/flounder/stickleback in-envelope, perch/pikeperch/smelt OVER. Record the per-species final-decade means — these are the no-regression reference for Task 5.

- [ ] **Step 4: Commit the baseline reference**

```bash
git add docs/baltic_percid_baseline_2026-07-28.md && git commit -m "chore(baltic): verified 5/8 baseline reference for percid-removals branch"
```

### Task 2: Realistic percid fishing F (Lever A)

**Files:**
- Create: `data/baltic/reference/percid_removal_provenance.md`
- Modify: `data/baltic/baltic_param-fishing.csv`
- Test: `tests/test_baltic_percid_removals.py`

**Interfaces:**
- Produces: fixed elevated `fisheries.rate.base.fsh4` (perch) and `.fsh5` (pikeperch) representing total (commercial + recreational) coastal removal.

- [ ] **Step 1: Write the provenance note**

Create `data/baltic/reference/percid_removal_provenance.md` documenting the chosen F: perch F = 0.40, pikeperch F = 0.50 (total commercial + recreational coastal removal; real Baltic coastal percid F ~0.3–0.6, recreational ≈/> commercial). Cite Hansson et al. 2018 (ICES JMS 75(3):999) and the Baltic pikeperch status reviews. State these REPLACE the calibration artifacts (perch 0.029, pikeperch 0.0095) and are held fixed.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_baltic_percid_removals.py
from osmose.config import OsmoseConfigReader

def test_percid_fishing_F_is_elevated_to_coastal_levels():
    cfg = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
    assert float(cfg["fisheries.rate.base.fsh4"]) >= 0.3   # perch, coastal+recreational
    assert float(cfg["fisheries.rate.base.fsh5"]) >= 0.3   # pikeperch
```

- [ ] **Step 3: Run test → FAIL**

Run: `.venv/bin/python -m pytest tests/test_baltic_percid_removals.py::test_percid_fishing_F_is_elevated_to_coastal_levels -q`
Expected: FAIL (current fsh4≈0.029, fsh5≈0.0095).

- [ ] **Step 4: Set the elevated percid F**

In `data/baltic/baltic_param-fishing.csv`, set `fisheries.rate.base.fsh4;0.40` and `fisheries.rate.base.fsh5;0.50` (edit the existing rows).

- [ ] **Step 5: Run test → PASS**, then commit

```bash
.venv/bin/python -m pytest tests/test_baltic_percid_removals.py -q
git add data/baltic/baltic_param-fishing.csv data/baltic/reference/percid_removal_provenance.md tests/test_baltic_percid_removals.py
git commit -m "feat(baltic): realistic percid fishing F (recreational + coastal, fixed)"
```

### Task 3: Cormorant predation — matrix column + calibratable levers (Lever B)

**Files:**
- Modify: `data/baltic/predation-accessibility.csv`, `data/baltic/baltic_param-background.csv`, `scripts/apply_calibration.py`, `scripts/calibrate_baltic.py`
- Test: `tests/test_baltic_percid_removals.py`, `tests/test_apply_calibration.py`

**Interfaces:**
- Consumes: engine facts — background predators predate at accessibility 1.0 unless given a matrix column; `species.biomass.multiplier.sp15` scales cormorant biomass; `predation.ingestion.rate.max.sp15` caps its consumption.
- Produces: a `Cormorant` predator column in the matrix; a `species.biomass.multiplier.sp15` config key; `apply_calibration` routing for the two new key families; cormorant biomass/ingestion in the calibration free-param set.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_baltic_percid_removals.py
import pandas as pd

def test_cormorant_is_a_predator_column_shaped_toward_percids():
    df = pd.read_csv("data/baltic/predation-accessibility.csv", sep=";", index_col=0)
    assert "Cormorant" in df.columns              # predator column added
    # cormorant preys harder on perch than on herring (shaped toward percids)
    assert df.loc["perch", "Cormorant"] > df.loc["herring", "Cormorant"]

def test_cormorant_biomass_multiplier_present():
    from osmose.config import OsmoseConfigReader
    cfg = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
    assert "species.biomass.multiplier.sp15" in cfg   # cormorant = sp15 on this base

def test_phase13_free_set_fixes_percid_F_and_frees_cormorant():
    # The plan's most load-bearing invariant: percid F is FIXED (not in the DE free
    # set) and the cormorant levers ARE free. (sys.path import, not importlib — the
    # module's @dataclass resolves __module__ against sys.modules.)
    import sys
    sys.path.insert(0, "scripts")
    import calibrate_baltic as cb
    keys, bounds, x0 = cb.get_phase13_shepherd_params()
    assert "fisheries.rate.base.fsh4" not in keys   # perch F fixed, not optimised
    assert "fisheries.rate.base.fsh5" not in keys   # pikeperch F fixed
    assert "species.biomass.multiplier.sp15" in keys       # cormorant levers present
    assert "predation.ingestion.rate.max.sp15" in keys
    assert len(keys) == len(bounds) == len(x0)
```
This test **fails before Step 6** (fsh4/fsh5 present, cormorant keys absent) and is the guard that makes "Run tests → PASS" (Step 7) actually verify Lever A stays fixed.

```python
# add to tests/test_apply_calibration.py
def test_file_for_routes_background_predation_keys():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("ac", "scripts/apply_calibration.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
    d = pathlib.Path("data/baltic")
    assert ac._file_for("species.biomass.multiplier.sp15", d).name == "baltic_param-background.csv"
    assert ac._file_for("predation.ingestion.rate.max.sp15", d).name == "baltic_param-background.csv"
```

- [ ] **Step 2: Run tests → FAIL**

Run: `.venv/bin/python -m pytest tests/test_baltic_percid_removals.py tests/test_apply_calibration.py::test_file_for_routes_background_predation_keys -q`
Expected: FAIL (no Cormorant column, no multiplier key, `_file_for` KeyError).

- [ ] **Step 3: Add the Cormorant predator column to the matrix**

Add a `Cormorant` column to `data/baltic/predation-accessibility.csv` (prey rows unchanged; the loader treats predator columns independently, so non-square is fine). Per-prey accessibility shaping cormorant onto percids/forage without over-cropping the high-weight stocks:
`perch;0.6, pikeperch;0.4, herring;0.15, sprat;0.15, smelt;0.25, stickleback;0.15, cod;0.05, flounder;0.1, all LTL rows;0`.

- [ ] **Step 4: Add the cormorant biomass multiplier + raise ingestion toward physiology**

In `data/baltic/baltic_param-background.csv`, add `species.biomass.multiplier.sp15;2.0` (count-based standing-stock anchor) and set `predation.ingestion.rate.max.sp15;70.0` (a 2 kg bird eating ~400–500 g/day ≈ 70/yr; was 40).

- [ ] **Step 5: Route the new key families in apply_calibration**

In `scripts/apply_calibration.py`, add to `_FILE_FOR`:
```python
    "species.biomass.multiplier.": "baltic_param-background.csv",
    "predation.ingestion.rate.max.": "baltic_param-background.csv",
```
Add a comment: `predation.ingestion.rate.max.` also matches focal sp0–7 (in `baltic_param-predation.csv`); safe ONLY while just sp15 is a free param — guard if a focal ingestion is ever freed.

- [ ] **Step 6: Wire cormorant levers in + remove percid F from the free set (LOAD-BEARING)**

In `scripts/calibrate_baltic.py` `get_phase13_shepherd_params()`, immediately **BEFORE** the final `return keys, bounds, x0` — i.e. AFTER the `keys = keys1 + keys2 + ssbhalf_keys + shape_keys` assembly line (the working lists do not exist until then; placing this earlier raises UnboundLocalError at call time):
```python
    # Cormorant top-down levers (Lever B); x0 = the Task-4 max-grounded gate values
    keys += ["species.biomass.multiplier.sp15", "predation.ingestion.rate.max.sp15"]
    bounds += [(np.log10(1.0), np.log10(3.0)), (np.log10(40.0), np.log10(80.0))]
    x0 += [np.log10(3.0), np.log10(80.0)]
    # Lever A (percid F) is FIXED, not optimised — drop fsh4/fsh5 by NAME here.
    # Do NOT edit the shared get_phase2_params (it also feeds phases 2/12).
    _drop = {"fisheries.rate.base.fsh4", "fisheries.rate.base.fsh5"}
    _keep = [i for i, k in enumerate(keys) if k not in _drop]
    keys, bounds, x0 = [keys[i] for i in _keep], [bounds[i] for i in _keep], [x0[i] for i in _keep]
    assert len(keys) == len(bounds) == len(x0)
    assert not (_drop & set(keys)), "percid F must stay fixed, not a free param"
```
**Why this is the single most load-bearing edit:** on `646a36d`, `get_phase2_params` unconditionally frees `fisheries.rate.base.fsh0..fsh7`, so fsh4/fsh5 ARE in the phase-13 free set. If left free, `apply_warm_start` seeds them from `phase13_equilibrium.json` (0.029/0.0095), the DE optimises them per-evaluation (overriding Task 2's fixed 0.40/0.50 via `expand_param_overrides` on every sim), and `apply_calibration` then erases the fixed values from the FINAL config too — the whole 4–8h run certifies a config where Lever A was **never active**, with no error and no failing test.

- [ ] **Step 7: Run tests → PASS**, then commit

```bash
.venv/bin/python -m pytest tests/test_baltic_percid_removals.py tests/test_apply_calibration.py -q
git add data/baltic/predation-accessibility.csv data/baltic/baltic_param-background.csv scripts/apply_calibration.py scripts/calibrate_baltic.py tests/
git commit -m "feat(baltic): tunable cormorant predation on percids (matrix column + levers + routing)"
```

### Task 4: Mandatory cheap feasibility gate (Step 0 — go/no-go)

**Files:** Create `scripts/percid_feasibility_gate.py`

**Interfaces:**
- Consumes: the Task 1–3 config (fixed percid F, cormorant column, levers).
- Produces: a GO/NO-GO on whether maxed grounded levers move the percids without regressing the well-assessed stocks — gating the 4–8 h calibration.

- [ ] **Step 1: Write the gate script (with lever attribution)**

`scripts/percid_feasibility_gate.py`: load `baltic_all-parameters.csv` (percid F already fixed) and run **three** 50-yr `PythonEngine` sims to *attribute* the percid response to each lever, not just detect it:
- **both:** cormorant `species.biomass.multiplier.sp15=3.0`, `predation.ingestion.rate.max.sp15=80.0` (max grounded);
- **F-only:** cormorant multiplier=0 (cormorant off) — isolates Lever A;
- **cormorant-only:** percid F reset to the baseline 0.029/0.0095 — isolates Lever B.

Print per-species final-decade mean for each run vs the Task-1 baseline and the ICES envelopes; compute realized cormorant consumption (total + perch-specific) vs the Hansson ~2×-fishery-for-perch anchor. (Seed 42 single-seed — this is a cheap SCREEN; the authoritative no-regression check is Task 5's 5-seed certification.)

- [ ] **Step 2: Run the gate**

```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/percid_feasibility_gate.py
```

- [ ] **Step 3: Go/no-go decision (record in the branch)**

- GO if: perch and/or smelt move materially toward their envelopes, **each moving lever is causally responsible** (the F-only and cormorant-only contrasts show a non-trivial share attributable to each — so a passing perch result is not silently all-F or all-cormorant), AND cod/herring/sprat/flounder/stickleback stay in-envelope, AND realized cormorant consumption is within the documented Hansson budget (a cormorant consuming ≫ its budget means the accessibility column is mis-shaped — see should-fix, tune it down before proceeding).
- NO-GO if: nothing moves, or the well-assessed stocks drop out of envelope, or the perch response is entirely one lever with the other inert (revisit that lever, don't launch the full run). → Record the finding in `docs/baltic_percid_baseline_2026-07-28.md`, STOP, do not run Task 5.

- [ ] **Step 4: Commit the gate + result**

```bash
git add scripts/percid_feasibility_gate.py docs/baltic_percid_baseline_2026-07-28.md
git commit -m "feat(baltic): percid-removals feasibility gate + go/no-go result"
```

### Task 5: Scoped re-calibration + validation (only if Task 4 = GO)

**Files:** none created; produces `calibration_results/phase13_results.json` + a certification doc.

- [ ] **Step 1: Run the warm-started re-calibration**

```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OSMOSE_DE_WORKERS=8 .venv/bin/python scripts/calibrate_baltic.py --phase 13 --years 40 --seeds 1 --maxiter 40 --popsize-mult 4 --patience 15 --wall-clock-cap-h 6 --warm-start data/baltic/calibration_results/phase13_equilibrium.json --isolated-eval --sim-timeout 300 --checkpoint-every 3
```
(percid F fixed; cormorant biomass/ingestion free with x0 from Task 4.)

- [ ] **Step 2: Apply + certify against the bar**

```bash
.venv/bin/python scripts/apply_calibration.py data/baltic/calibration_results/phase13_results.json
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --years 50 --out docs/baltic_percid_removals_certification_2026-07-28.md
```

- [ ] **Step 3: Evaluate against the pre-registered bar + revert rule**

- PASS if: perch overshoot reduced to ≤ (envelope-upper × 2) or better; smelt improved; every well-assessed stock stays in its baseline envelope; realized cormorant consumption ≈ the Hansson budget. Record the pikeperch reduction achieved (expected small — honest).
- REVERT if: any well-assessed stock drops below its baseline envelope, or obj > baseline 2.33 + margin. → `git checkout` the config, restore the baseline, and document the finding as the structural limit.

- [ ] **Step 4: Insensitivity test for the pikeperch structural claim**

Re-run the Task-4 gate at the top of the defensible lever range; if the residual pikeperch overshoot persists, document it as the coarse-grid structural limit (do not attribute to lever weakness).

- [ ] **Step 5: Document + commit**

Write the outcome (what moved, cormorant consumption vs Hansson, perch/smelt result, pikeperch residual + insensitivity) and commit the certified config or the revert + finding.

---

## Self-Review

- **Spec coverage:** Lever A (percid F) → Task 2; Lever B (cormorant column + biomass/ingestion + routing) → Task 3; Step-0 gate → Task 4; scoped re-calibration + bar + revert + insensitivity → Task 5; base restore/verify → Task 1. All spec §§4–8 map to a task.
- **Placeholder scan:** the only deferred value (percid F) is fixed to concrete numbers (perch 0.40, pikeperch 0.50) with a provenance task; no TBD/TODO.
- **Type/name consistency:** `sp15` = Cormorant throughout (Tasks 3–4); `fsh4`/`fsh5` = perch/pikeperch fisheries; `_FILE_FOR` / `_file_for` used consistently with the actual `apply_calibration.py` names; the free-param append matches `get_phase13_shepherd_params`'s existing structure.
- **Hardened by a 3-dimension deep-review workflow (2026-07-28).** Blocking fixes applied: Task 3 Step 6's percid-F removal is now concrete name-based filter code (not conditional prose) placed after the list assembly, with an assertion; a `test_phase13_free_set_fixes_percid_F_and_frees_cormorant` test guards the invariant (fsh4/fsh5 absent, cormorant levers present) so "Run tests → PASS" actually verifies Lever A stays fixed. Non-blocking hardening: the Task-4 gate now runs F-only / cormorant-only contrasts to *attribute* perch movement (so Lever B can't mask a skipped Lever A) and flags a mis-shaped accessibility column; cormorant x0 aligned to the gate's 3.0/80.0; single-seed gate documented as a cheap screen vs the 5-seed certification.
- **Residual (accepted):** the cormorant accessibility column (perch 0.6 vs herring/sprat 0.15) is a fixed authored shape, not calibrated — if the Task-4 attribution shows cormorant perch-removal is negligible against the forage-fish biomass advantage, tune the forage-fish accessibilities down (~0.01–0.03) before the full run, or accept that perch control rests on Lever A (F) alone.
