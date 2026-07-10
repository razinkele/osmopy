# Chunk C — clupeid→cod-egg predation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-only clupeid→cod-egg predation lever (cod-as-prey accessible to herring/sprat) and test — via the warm-start regime-shift sweep across accessibility strengths — whether it creates a cod↔sprat regime-shift bistability.

**Architecture:** A pure helper generates a variant predation-accessibility matrix (deployed matrix + cod→herring/sprat = X). The existing warm-start regime-shift sweep is reused unchanged; Chunk C is applied globally per sweep by overriding `predation.accessibility.file` in the base config to point at the variant. The size-ratio window (herring/sprat `[5,500]`) auto-restricts the new predation to egg/larval cod, so no engine change and no explicit stage row. Real Baltic runs stay CLI-only.

**Tech Stack:** Python 3.11+, pandas (already an engine dep), stdlib. Tests via pytest with the existing fake-runner pattern. Ruff for lint/format.

## Global Constraints

- **No new dependencies; no engine change.** `ruff check` + `ruff format --check` clean on `scripts/ tests/`.
- **The deployed `data/baltic/predation-accessibility.csv` is NEVER modified.** Chunk C lives only in generated variant CSVs selected per-run via the `predation.accessibility.file` override.
- **Matrix layout (verbatim):** semicolon-separated, prey **rows** × predator **columns**, read by `pd.read_csv(sep=";", index_col=0)`. Predator column order: `cod;herring;sprat;flounder;perch;pikeperch;smelt;stickleback;<6 LTL>`. The **cod** prey row is `cod;0.05;0;0;0;0;0.05;0.05;0;...`. Chunk C sets the **cod** row's **herring** and **sprat** cells to X; every other cell unchanged (cod→cod cannibalism 0.05 untouched).
- **Path resolution:** `predation.accessibility.file` resolves via `osmose.engine.path_resolution.resolve_data_path(key, config_dir)`, which returns an **absolute** path as-is if it exists (rejects `..`). Config dir is carried in the config key `_osmose.config.dir`.
- **Chunk C is a REAL treatment (not inert):** existing byte-identical-off parity does not apply here; the lever is only ever active when `--chunk-c-strength` is passed.
- **Species→index / sizes (verbatim):** sp0 cod (egg 0.15 cm, Linf 110), sp1 herring (Linf 27), sp2 sprat (Linf 16); herring/sprat size-ratio window `[5,500]`.
- **CI discipline:** real emergent Baltic runs are CLI-only, excluded from CI (`feedback-ci-fragile-emergent-tests`). New automated tests use fixtures / fake runners only.
- **Test command:** `.venv/bin/python -m pytest tests/test_chunkc_accessibility.py tests/test_baltic_bistability_chunk0.py -q`

---

### Task 1: `write_chunkc_matrix` helper

**Files:**
- Create: `scripts/chunkc_accessibility.py`
- Test: `tests/test_chunkc_accessibility.py`

**Interfaces:**
- Produces: `write_chunkc_matrix(deployed_csv: str, strength: float, out_path: str) -> str` — reads the deployed accessibility CSV, sets the cod prey row's herring and sprat predator cells to `strength`, writes the variant to `out_path` (semicolon-separated, same layout), returns `out_path`. Raises `KeyError` if the `cod` prey row or `herring`/`sprat` predator columns are absent.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chunkc_accessibility.py`:

```python
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import chunkc_accessibility as cc  # noqa: E402

_DEPLOYED = (
    "v Prey / Predator >;cod;herring;sprat;smelt\n"
    "cod;0.05;0;0;0.05\n"
    "herring;0.4;0;0;0\n"
    "sprat;0.4;0;0;0\n"
    "smelt;0.1;0.2;0.2;0\n"
)


def _write_deployed(tmp_path) -> str:
    p = tmp_path / "deployed.csv"
    p.write_text(_DEPLOYED)
    return str(p)


def test_write_chunkc_matrix_sets_only_cod_to_clupeids(tmp_path):
    dep = _write_deployed(tmp_path)
    out = str(tmp_path / "chunkc.csv")
    assert cc.write_chunkc_matrix(dep, 0.3, out) == out
    d = pd.read_csv(dep, sep=";", index_col=0)
    v = pd.read_csv(out, sep=";", index_col=0)
    # cod -> herring and cod -> sprat set to 0.3
    assert v.loc["cod", "herring"] == 0.3
    assert v.loc["cod", "sprat"] == 0.3
    # cod cannibalism and every other cell unchanged
    changed = {(r, c) for r in v.index for c in v.columns if v.loc[r, c] != d.loc[r, c]}
    assert changed == {("cod", "herring"), ("cod", "sprat")}


def test_write_chunkc_matrix_missing_labels_raises(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("v Prey / Predator >;cod;flounder\ncod;0.05;0\nflounder;0.1;0\n")
    with pytest.raises(KeyError):
        cc.write_chunkc_matrix(str(p), 0.2, str(tmp_path / "o.csv"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_chunkc_accessibility.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chunkc_accessibility'`

- [ ] **Step 3: Implement `scripts/chunkc_accessibility.py`**

```python
"""Generate a Chunk-C predation-accessibility matrix: clupeid->cod-egg predation.

Reads the deployed OSMOSE accessibility CSV (prey rows x predator cols) and writes a
variant with cod-as-prey accessible to herring and sprat at `strength`; every other cell
is unchanged. The herring/sprat size-ratio window ([5,500]) restricts this predation to
egg/larval cod automatically, so no explicit prey stage is needed. See
docs/superpowers/specs/2026-07-09-baltic-chunkc-clupeid-cod-egg-predation-design.md.
"""

from __future__ import annotations

import argparse

import pandas as pd

_PREY = "cod"
_PREDATORS = ("herring", "sprat")


def write_chunkc_matrix(deployed_csv: str, strength: float, out_path: str) -> str:
    """cod->herring and cod->sprat set to `strength`; all other cells identical to deployed."""
    df = pd.read_csv(deployed_csv, sep=";", index_col=0)
    if _PREY not in df.index:
        raise KeyError(f"prey row {_PREY!r} not in accessibility matrix {deployed_csv}")
    missing = [p for p in _PREDATORS if p not in df.columns]
    if missing:
        raise KeyError(f"predator column(s) {missing} not in accessibility matrix {deployed_csv}")
    for pred in _PREDATORS:
        df.loc[_PREY, pred] = float(strength)
    df.to_csv(out_path, sep=";")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a Chunk-C clupeid->cod-egg accessibility matrix")
    ap.add_argument("--deployed", required=True, help="deployed predation-accessibility.csv")
    ap.add_argument("--strength", type=float, required=True, help="cod->herring/sprat accessibility")
    ap.add_argument("--out", required=True, help="output variant CSV path")
    args = ap.parse_args(argv)
    print(write_chunkc_matrix(args.deployed, args.strength, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chunkc_accessibility.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/chunkc_accessibility.py tests/test_chunkc_accessibility.py
.venv/bin/ruff format scripts/chunkc_accessibility.py tests/test_chunkc_accessibility.py
git add scripts/chunkc_accessibility.py tests/test_chunkc_accessibility.py
git commit -m "feat(baltic): Chunk-C accessibility-matrix generator (clupeid->cod-egg)"
```

---

### Task 2: harness CLI wiring — `--chunk-c-strength`

**Files:**
- Modify: `scripts/baltic_bistability_chunk0.py` (add `chunkc_output_name` + `_deployed_accessibility_csv` near the other loaders ~after `_load_targets`; add the CLI arg and the chunk-c branch in `main`)
- Test: `tests/test_baltic_bistability_chunk0.py` (append)

**Interfaces:**
- Consumes: `write_chunkc_matrix` (Task 1), `run_bistability_sweep` / `cod_dominated_seeding` / `clupeid_dominated_seeding` / `_clupeid_targets_from` (existing), `_DIAG_DIR`, `_default_runner`.
- Produces:
  - `chunkc_output_name(strength: float) -> str` — `f"baltic_chunkc_regime-shift_s{strength:g}.json"`.
  - `_deployed_accessibility_csv(base_config) -> str` — resolve the deployed `predation.accessibility.file` to an absolute path via `resolve_data_path`.
  - `main` accepts `--chunk-c-strength FLOAT [FLOAT ...]`: for each strength, generate a variant matrix, override the base config's accessibility file, run the warm-start regime-shift sweep, write `docs/diagnostics/<chunkc_output_name>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baltic_bistability_chunk0.py`:

```python
# ---------------------------------------------------------------- Chunk C CLI
def test_chunkc_output_name():
    assert c0.chunkc_output_name(0.2) == "baltic_chunkc_regime-shift_s0.2.json"
    assert c0.chunkc_output_name(0.4) == "baltic_chunkc_regime-shift_s0.4.json"


def test_cli_chunk_c_writes_variant_and_runs_sweep(tmp_path, monkeypatch):
    import pandas as pd

    dep = tmp_path / "predation-accessibility.csv"
    dep.write_text(
        "v Prey / Predator >;cod;herring;sprat;smelt\n"
        "cod;0.05;0;0;0.05\n"
        "herring;0.4;0;0;0\n"
        "sprat;0.4;0;0;0\n"
        "smelt;0.1;0.2;0.2;0\n"
    )
    tgts = [
        Tgt("cod", 120_000, 60_000, 250_000),
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]
    captured = {}

    def fake_runner(config, overrides, n_years, seed):
        captured["accessibility_file"] = config.get("predation.accessibility.file")
        return _stats(cod=120_000, herring=400_000, sprat=300_000)

    monkeypatch.setattr(
        c0,
        "read_base_config",
        lambda: {"predation.accessibility.file": str(dep), "_osmose.config.dir": str(tmp_path)},
    )
    monkeypatch.setattr(c0, "read_base_larva_rates", lambda cfg, n_focal=8: {0: 15.0})
    monkeypatch.setattr(c0, "_load_targets", lambda: tgts)
    monkeypatch.setattr(c0, "_default_runner", fake_runner)
    monkeypatch.setattr(c0, "_DIAG_DIR", tmp_path)

    rc = c0.main(["--chunk-c-strength", "0.2", "--smoke"])
    assert rc == 0
    variant = tmp_path / "predation-accessibility-chunkc-s0.2.csv"
    assert variant.exists()
    v = pd.read_csv(str(variant), sep=";", index_col=0)
    assert v.loc["cod", "herring"] == 0.2 and v.loc["cod", "sprat"] == 0.2
    # the sweep ran against the variant matrix, not the deployed one
    assert captured["accessibility_file"] == str(variant.resolve())
    assert (tmp_path / c0.chunkc_output_name(0.2)).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -k "chunkc_output_name or chunk_c_writes" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'chunkc_output_name'`

- [ ] **Step 3: Add the two helpers**

In `scripts/baltic_bistability_chunk0.py`, after `_load_targets` (the module-level loaders block), add:

```python
def chunkc_output_name(strength: float) -> str:
    return f"baltic_chunkc_regime-shift_s{strength:g}.json"


def _deployed_accessibility_csv(base_config) -> str:
    from osmose.engine.path_resolution import resolve_data_path

    key = base_config.get("predation.accessibility.file", "")
    path = resolve_data_path(key, base_config.get("_osmose.config.dir", ""))
    if path is None:
        raise FileNotFoundError(f"could not resolve deployed accessibility file {key!r}")
    return str(path)
```

- [ ] **Step 4: Add the CLI argument**

In `main`, after the `--preflight` argument:

```python
    ap.add_argument("--chunk-c-strength", type=float, nargs="+", default=None)
```

- [ ] **Step 5: Add the chunk-c branch in `main`**

Insert immediately after the `if args.preflight:` block returns and BEFORE `if args.warmstart:`:

```python
    if args.chunk_c_strength:
        from chunkc_accessibility import write_chunkc_matrix

        clup = _clupeid_targets_from(targets)
        deployed_csv = _deployed_accessibility_csv(base_config)
        for strength in args.chunk_c_strength:
            variant = (_DIAG_DIR / f"predation-accessibility-chunkc-s{strength:g}.csv").resolve()
            write_chunkc_matrix(deployed_csv, strength, str(variant))
            cfg = dict(base_config)
            cfg["predation.accessibility.file"] = str(variant)
            out_path = _DIAG_DIR / chunkc_output_name(strength)
            result = run_bistability_sweep(
                scales,
                cfg,
                base_rates,
                cod_bands,
                seeds,
                runner=_default_runner,
                n_years=years,
                ic_a=cod_dominated_seeding,
                ic_b=clupeid_dominated_seeding,
                warmstart=True,
                contrast="regime-shift",
                clupeid_targets=clup,
                on_point=lambda payload, p=out_path: p.write_text(json.dumps(payload, indent=2)),
            )
            print(f"\n=== CHUNK C (cod->clupeid accessibility {strength:g}) ===")
            for pt in result["points"]:
                print(f"  larva x{pt['scale']:<5} outcome={pt['outcome']}")
            print(f"VERDICT: {result['verdict']}")
            out_path.write_text(json.dumps(result, indent=2))
        return 0
```

- [ ] **Step 6: Run the new tests AND the full harness suite**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (42 passed — 40 existing + 2 new)

- [ ] **Step 7: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
.venv/bin/ruff format scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): --chunk-c-strength runs regime-shift sweep with clupeid->cod-egg predation"
```

---

### Task 3: Real-engine de-risk, strength sweep, ICES check, write-up

**This task is real-engine, CLI-only, NOT CI.** Run after Tasks 1–2 pass. Verification is that the de-risk gate passes, the sweeps complete, and the results doc is written from the JSONs.

**Files:**
- Create: `docs/baltic_chunkc_results_2026-07-09.md`
- Produces (real outputs): `docs/diagnostics/baltic_chunkc_regime-shift_s{0.1,0.2,0.4}.json` and their variant CSVs.

- [ ] **Step 1: De-risk gate — confirm clupeids realize cod-egg predation (STOP if not)**

Run this comparison at a cod-**established** larva scale (×0.1), where egg predation can show an effect (at ×1.0 cod is already collapsed):

```bash
cd /home/razinka/osmopy
.venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "scripts")
import baltic_bistability_chunk0 as c0
from chunkc_accessibility import write_chunkc_matrix
from calibrate_baltic import run_simulation

base = c0.read_base_config()
rates = c0.read_base_larva_rates(base)
driver = c0.larva_scale_override(0.1, rates)  # cod established (overshoot regime)
dep = c0._deployed_accessibility_csv(base)
variant = os.path.abspath("docs/diagnostics/predation-accessibility-chunkc-s0.4.csv")
write_chunkc_matrix(dep, 0.4, variant)

off = run_simulation(base, {**driver}, n_years=15, seed=0)
on = run_simulation(base, {**driver, "predation.accessibility.file": variant}, n_years=15, seed=0)
print(f"cod off={off.get('cod_mean'):.0f}  cod on={on.get('cod_mean'):.0f}")
print(f"herring off={off.get('herring_mean'):.0f}  on={on.get('herring_mean'):.0f}")
print(f"sprat off={off.get('sprat_mean'):.0f}  on={on.get('sprat_mean'):.0f}")
PY
```
Expected: `cod on` is **measurably lower** than `cod off` (clupeids eating cod eggs suppresses cod). If `cod on ≈ cod off` (no effect), **STOP**: egg/larval cod is not being reached by clupeid predation at run time (size-window edge or eggs handled outside the predation loop). Record this as a finding in the results doc and do **not** run the sweep — Chunk C would then need engine work, which is out of scope for this plan.

- [ ] **Step 2: Strength sweep (headline X = 0.2 first, 25-year horizon)**

```bash
.venv/bin/python scripts/baltic_bistability_chunk0.py --chunk-c-strength 0.2 --years 25 2>&1 | tee /tmp/chunkc_s0.2.log
```
Expected: `=== CHUNK C (cod->clupeid accessibility 0.2) ===` prints per-scale outcomes and a VERDICT; `docs/diagnostics/baltic_chunkc_regime-shift_s0.2.json` written. ~2 h wall clock (30 real 25-year runs).

- [ ] **Step 3: Fill in the strength sweep (X = 0.1 and 0.4)**

```bash
.venv/bin/python scripts/baltic_bistability_chunk0.py --chunk-c-strength 0.1 0.4 --years 25 2>&1 | tee /tmp/chunkc_s0.1_0.4.log
```
Expected: two more JSONs written. (Run only if Step 2 is informative — e.g. skip if X=0.2 already shows a clear regime shift or a clear null.)

- [ ] **Step 4: ICES calibration check per strength**

For each swept X, compare the deployed config (larva ×1.0, egg-only) with Chunk C on vs the control, against ICES bands:

```bash
.venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "scripts")
import baltic_bistability_chunk0 as c0
from calibrate_baltic import run_simulation, load_targets

base = c0.read_base_config()
bands = {t.species: (t.lower, t.target, t.upper) for t in load_targets()}
off = run_simulation(base, {}, n_years=25, seed=0)
for X in (0.1, 0.2, 0.4):
    variant = os.path.abspath(f"docs/diagnostics/predation-accessibility-chunkc-s{X:g}.csv")
    on = run_simulation(base, {"predation.accessibility.file": variant}, n_years=25, seed=0)
    print(f"X={X}: cod {off.get('cod_mean'):.0f}->{on.get('cod_mean'):.0f} "
          f"herring {off.get('herring_mean'):.0f}->{on.get('herring_mean'):.0f} "
          f"sprat {off.get('sprat_mean'):.0f}->{on.get('sprat_mean'):.0f}  (cod band {bands['cod']})")
PY
```
Record whether egg predation moves the deployed calibration toward or away from the ICES cod band.

- [ ] **Step 5: Write the results doc**

Create `docs/baltic_chunkc_results_2026-07-09.md` mirroring `docs/baltic_chunk0_warmstart_results_2026-07-09.md`. Include: the de-risk outcome; a per-X regime-shift table (larva scale → cod bands both arms, clupeid biomasses, clupeid gap, outcome) and verdict from each JSON; the ICES check; and the honest interpretation — **created bistability** (a determinate `regime-shift` at some (X, scale)) or **negative** (monostable at all X → the cultivation-depensation lever as implemented does not create an alternative state; next lever = Chunk A2 depletable plankton).

- [ ] **Step 6: Commit results + diagnostics**

```bash
cd /home/razinka/osmopy
git add docs/baltic_chunkc_results_2026-07-09.md \
        docs/diagnostics/baltic_chunkc_regime-shift_s*.json \
        docs/diagnostics/predation-accessibility-chunkc-s*.csv
git commit -m "docs(baltic): Chunk C clupeid->cod-egg predation results (2026-07-09)"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-09-baltic-chunkc-clupeid-cod-egg-predation-design.md`):
- variant-matrix generator (cod→herring/sprat = X, rest unchanged) → Task 1 `write_chunkc_matrix`. ✓
- config-only application via `predation.accessibility.file` override, no sweep-signature change → Task 2 chunk-c branch sets `cfg["predation.accessibility.file"]`. ✓
- CLI `--chunk-c-strength X [X ...]`, per-strength output names → Task 2 `--chunk-c-strength` + `chunkc_output_name`. ✓
- reuse warm-start regime-shift sweep + directional verdict → Task 2 calls `run_bistability_sweep(..., warmstart=True, contrast="regime-shift", ...)`. ✓
- deployed matrix never modified → Task 1 writes to a separate `out_path`; Task 2 writes variants to `_DIAG_DIR`; the deployed CSV is only read. ✓
- risk 1 (override resolution) → verified by the Task 2 CLI test (`captured["accessibility_file"] == variant`) and Task 3 Step 1 real run. ✓
- risk 2 (predation actually reaches cod eggs) → Task 3 Step 1 de-risk STOP gate. ✓
- 25-year horizon → Task 3 Steps 2–3 `--years 25`. ✓
- ICES check → Task 3 Step 4. ✓
- outputs (JSONs + results doc) → Task 3 Steps 5–6. ✓
- unit tests CI-safe (fixtures / fake runner) → Tasks 1–2. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every command shows expected output and a STOP condition where relevant.

**Type consistency:** `write_chunkc_matrix`, `chunkc_output_name`, `_deployed_accessibility_csv` are named identically at definition and call sites. The variant path passed to `cfg["predation.accessibility.file"]` (Task 2) matches the path asserted in the CLI test and generated by `write_chunkc_matrix`. The `{strength:g}` format is used identically for the variant CSV name and `chunkc_output_name`.
