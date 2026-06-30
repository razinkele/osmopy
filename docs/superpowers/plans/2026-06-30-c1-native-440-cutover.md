# C1 — Native OSMOSE 4.4.0 cutover of the bundled Python stack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline — subagent dispatch is spend-limited this month) or subagent-driven-development if available. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Bundled configs (minus BoB) become native `osmose.version 4.4.1` on disk; write/jar/calibration defaults flip to 4.4.1; the 4.3.3 `canonicalize` path is demoted to a legacy adapter. Python-engine behaviour is unchanged — every config conversion is gated by a round-trip parity check.

**Architecture:** First make `to_target_keys`'s larval-rate transform source-version-aware (so native-4.4.0 configs don't double-scale on the Java write path). Then build a parity harness, capture baselines from the current 4.3.3 sources, convert each config (EEC canary first), flip the defaults, and demote the legacy framing. Spec: `docs/superpowers/specs/2026-06-30-c1-native-440-cutover-design.md`.

**Tech Stack:** Python 3.12, the Python engine (`PythonEngine.run_in_memory`), pytest, the 4.4.1/4.3.3 jars.

## Global Constraints

- **Main venv ABSOLUTE path** `/home/razinka/osmose/osmose-python/.venv/bin/python` — NO `.venv` in the worktree. Tests/lint via that path. (`ruff check` + `ruff format --check` on `osmose/ ui/ tests/ scripts/`.)
- **Python-engine behaviour MUST NOT change** — gated by the §4.3 round-trip parity (max relative diff **< 1e-9**, fixed RNG, 3-yr run) per config. Bit-exact where the rate doesn't apply.
- **Larval-rate transform is source×target aware**: net factor `ndt^[target≥4.4] ÷ ndt^[source≥4.4]`. The key `mortality.additional.larva.rate.spN` is version-stable (only its value-unit changed).
- **Native source KEEPS `species.lmax`/`species.beta`** (engine reads them: `config.py:473`/`1976`). The Java write path still drops them (idempotent).
- **In scope:** `data/eec_full`, `data/minimal`, `data/baltic`, `data/baltic_ev`. **Excluded:** `data/examples` (BoB) — verified no shared CSV includes with EEC.
- **Target version string = `"4.4.1"`** (matches the jar). `canonicalize`/`migrate_config` operate on `"4.4.0"` boundary — unchanged.
- Builds on sub-project A (master `072b388`); the Baltic Java-only staging stays in `scripts/baltic_440_smoke.py`, NOT source.

---

### Task 1: Source-aware larval-rate transform in `to_target_keys`

**Files:** Modify `osmose/config/aliases.py` (`to_target_keys`, ~line 1-32 of the function). Test: `tests/test_config_aliases.py` (append).

**Interfaces:** `to_target_keys(cfg, target_version)` unchanged signature; behaviour: rate factor now depends on `cfg["osmose.version"]` (source) AND `target_version`.

- [ ] **Step 1: Failing tests** — append to `tests/test_config_aliases.py`:
```python
import pytest
from osmose.config.aliases import to_target_keys

_RK = "mortality.additional.larva.rate.sp0"

@pytest.mark.parametrize("src,tgt,factor", [
    ("4.3.3", "4.4.1", 24.0),    # write native: ×ndt
    ("4.4.1", "4.4.1", 1.0),     # native->native: NO double-scale
    ("4.4.1", "4.3.3", 1/24.0),  # native->legacy jar: ÷ndt
    ("4.3.3", "4.3.3", 1.0),     # legacy->legacy: unchanged
])
def test_larva_rate_source_aware(src, tgt, factor):
    cfg = {"osmose.version": src, "simulation.time.ndtperyear": "24", _RK: "2.4"}
    out = to_target_keys(dict(cfg), target_version=tgt)
    assert float(out[_RK]) == pytest.approx(2.4 * factor, rel=1e-12)

def test_native_source_keeps_emit_idempotent_no_double_scale():
    # a native-4.4.0 config with resource + background species, staged for 4.4.1, must not double-scale
    cfg = {"osmose.version": "4.4.1", "simulation.time.ndtperyear": "24", _RK: "2.4",
           "species.type.sp9": "background", "species.name.sp9": "Seal"}
    out = to_target_keys(dict(cfg), target_version="4.4.1")
    assert float(out[_RK]) == pytest.approx(2.4, rel=1e-12)        # unchanged
    assert out["species.multiplier.sp9"] == "1"                    # emit still idempotent
```

- [ ] **Step 2: Run — verify fail**
`PYTHONPATH=. .venv/bin/python -m pytest tests/test_config_aliases.py -k "source_aware or double_scale" -v` → FAIL (currently ×ndt regardless of source).

- [ ] **Step 3: Implement** in `osmose/config/aliases.py::to_target_keys`. Replace the unconditional `_migrate_larva_rate(result, _ndtperyear(result) or 1.0, ...)` with a source-aware net factor applied in BOTH branches:
```python
    from osmose.demo import _version_tuple
    result = dict(cfg)
    src_ge = _numeric_version(cfg.get("osmose.version", "4.3.3")) >= _version_tuple("4.4.0")
    tgt_ge = _numeric_version(target_version) >= _version_tuple("4.4.0")
    ndt = _ndtperyear(result) or 1.0
    # net larval-rate factor: ndt^[target>=4.4] / ndt^[source>=4.4]  (key name is version-stable)
    rate_factor = (ndt if tgt_ge else 1.0) / (ndt if src_ge else 1.0)

    if tgt_ge:
        result = _drop_4_4_0_removed_keys(result)            # Java drops lmax/beta; no-op if absent
        if rate_factor != 1.0:
            result = _migrate_larva_rate(result, rate_factor, warn_bydt=True)
        result = _emit_resource_biomass_forcing(result)
        result = _emit_background_species_keys(result)
        result["osmose.version"] = target_version
        return result
    # reverse branch (target < 4.4.0): un-scale a native source, then reverse the renames
    if rate_factor != 1.0:                                    # native 4.4 -> 4.3.3 jar: ÷ndt
        result = _migrate_larva_rate(result, rate_factor, warn_bydt=False)
    for new_prefix in sorted(_INVERSE_440, key=len, reverse=True):
        ...  # unchanged reverse-rename loop
    result["osmose.version"] = target_version
    return result
```

- [ ] **Step 4: Run — verify pass** `pytest tests/test_config_aliases.py -q` → PASS (new + existing; `test_from_dict_warn_mode_clean_on_example_configs` stays warning-free).

- [ ] **Step 5: Commit** `git add osmose/config/aliases.py tests/test_config_aliases.py && git commit -m "fix(aliases): make to_target_keys larval-rate transform source-version-aware (no double-scale on native 4.4.0)"`

---

### Task 2: Round-trip parity harness + capture baselines (BEFORE any conversion)

**Files:** Create `scripts/native_440_parity.py`. Test: `tests/test_native_440_parity.py`.

**Interfaces:** `run_outputs(config_dir, years=3, seed=42) -> dict[str, np.ndarray]` (fixed-RNG engine biomass/abundance/yield arrays); `max_rel_diff(a, b) -> float`; `capture_baseline(name)` / `gate(name)`.

- [ ] **Step 1: Failing test** — `tests/test_native_440_parity.py`:
```python
def test_run_outputs_deterministic():
    from scripts.native_440_parity import run_outputs, max_rel_diff
    a = run_outputs("data/minimal", years=1)
    b = run_outputs("data/minimal", years=1)
    assert max_rel_diff(a["biomass"], b["biomass"]) == 0.0   # fixed RNG -> bit-reproducible
```

- [ ] **Step 2: Implement `scripts/native_440_parity.py`**:
```python
"""Round-trip parity: Python-engine outputs of a config, with fixed RNG, for the 4.4.0 cutover."""
from pathlib import Path
import json
import numpy as np
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path("/home/razinka/osmose/osmose-python")
BASELINE = ROOT / "scripts" / "_parity_baselines"   # gitignored scratch
CONFIGS = {"eec_full": "eec_all-parameters.csv", "minimal": "osm_all-parameters.csv",
           "baltic": "baltic_all-parameters.csv", "baltic_ev": "baltic_ev_all-parameters.csv"}

def run_outputs(config_dir, years=3, seed=42):
    master = next(p for p in (Path(config_dir).glob("*all-parameters*.csv")))
    raw = dict(OsmoseConfigReader().read(str(master)))
    raw["simulation.time.nyear"] = str(years)
    raw["simulation.rng.fixed"] = "true"
    res = PythonEngine().run_in_memory(raw, seed=seed)  # seed param drives determinism
    # OsmoseResults accessors are METHODS (results.py:416/424/428): biomass(), abundance(),
    # yield_biomass() -- NOT attributes, and "yield" is yield_biomass.
    out = {}
    for name, fn in (("biomass", res.biomass), ("abundance", res.abundance),
                     ("yield", res.yield_biomass)):
        try:
            df = fn()
            out[name] = np.asarray(df.to_numpy(), dtype=float)
        except Exception:  # accessor absent for this config -> skip that metric
            pass
    return out

def max_rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    denom = np.maximum(np.abs(a), 1e-30)
    return float(np.nanmax(np.abs(a - b) / denom)) if a.size else 0.0

def capture_baseline(name):
    BASELINE.mkdir(parents=True, exist_ok=True)
    out = run_outputs(ROOT / "data" / name)
    np.savez(BASELINE / f"{name}.npz", **out)
    print(f"baseline captured: {name} ({list(out)})")

def gate(name, tol=1e-9):
    base = np.load(BASELINE / f"{name}.npz")
    now = run_outputs(ROOT / "data" / name)
    worst = max(max_rel_diff(base[k], now[k]) for k in base.files)
    print(f"{name}: max_rel_diff={worst:.2e} {'PASS' if worst < tol else 'FAIL'} (tol={tol})")
    assert worst < tol, f"{name} parity FAILED: {worst:.2e} >= {tol}"

if __name__ == "__main__":
    import sys
    cmd, name = sys.argv[1], sys.argv[2]
    capture_baseline(name) if cmd == "capture" else gate(name)
```
Add `scripts/_parity_baselines/` to `.gitignore`.

- [ ] **Step 3: Verify the harness + CAPTURE ALL 4 BASELINES (from the current 4.3.3 sources — must be done before Task 3/4 changes any source):**
`pytest tests/test_native_440_parity.py -q` then for each name in eec_full, minimal, baltic, baltic_ev: `PYTHONPATH=. .venv/bin/python scripts/native_440_parity.py capture <name>`. Confirm 4 `.npz` files exist.

- [ ] **Step 4: Commit** `git add scripts/native_440_parity.py tests/test_native_440_parity.py .gitignore && git commit -m "feat(parity): round-trip parity harness for the 4.4.0 cutover + capture 4.3.3 baselines"`

---

### Task 3: Conversion tool + convert EEC (canary)

**Files:** Create `scripts/migrate_bundled_to_440.py`. Modify `data/eec_full/*`.

**Interfaces:** `convert_config(config_dir)` — rewrites each sub-file to native 4.4.0 in place, preserving structure.

- [ ] **Step 1: Implement `scripts/migrate_bundled_to_440.py`** — per-file rewrite (reader tracks no key→file origin, so process each `.csv` independently):
  - Read the merged config once (for `ndt` + the `osmose.version` guard: skip the whole config if already ≥4.4.0).
  - A **per-key prefix-aware forward-rename** helper (mirror `to_target_keys`'s inverse loop, forward direction — `RENAMES_440` is a longest-prefix-first OLD→NEW map; the larva-rate key is NOT in it):
    ```python
    def _rename_forward(key):
        for old in sorted(RENAMES_440, key=len, reverse=True):
            if key == old or key.startswith(old + "."):
                return RENAMES_440[old] + key[len(old):]
        return key
    ```
  - For each `*.csv` sub-file: read raw lines; for each `key;value` line, `key=_rename_forward(key)`; if the key matches `_LARVA_RATE_RE`, scale `value ×ndt`; write back **preserving the line's original separator, key case, blank lines and `#` comments** (rewrite only `key;value` lines).
  - Append `_emit_resource_biomass_forcing` output (resource species) to the sub-file that defines those species (`species`/`ltl`), only for keys not already present.
  - **Do NOT drop `species.lmax`/`species.beta`** (keep — Python engine reads them; the Java write path drops them later via `to_target_keys`).
  - Stamp `osmose.version ; 4.4.1` in the master (replace the existing `osmose.version` line).
  - Guard: refuse to touch `data/examples` (BoB) — assert the target dir is one of the 4 in-scope.

- [ ] **Step 2: Convert EEC + GATE**
`PYTHONPATH=. .venv/bin/python scripts/migrate_bundled_to_440.py data/eec_full` then `PYTHONPATH=. .venv/bin/python scripts/native_440_parity.py gate eec_full`.
Expected: `eec_full: max_rel_diff=... PASS (tol=1e-9)`. If FAIL → investigate (likely a dropped key the engine reads, or a rate mis-scale) BEFORE proceeding; revert `data/eec_full` via git if needed.

- [ ] **Step 3: Confirm native + version** `grep -c "osmose.version ; 4.4.1" data/eec_full/eec_all-parameters.csv` = 1; spot-check a renamed key is 4.4.0-spelled.

- [ ] **Step 4: Commit** `git add scripts/migrate_bundled_to_440.py data/eec_full && git commit -m "feat(440): convert EEC to native 4.4.0 (canary, round-trip parity PASS)"`

---

### Task 4: Convert minimal, baltic, baltic_ev (gated rollout)

**Files:** Modify `data/minimal/*`, `data/baltic/*`, `data/baltic_ev/*`.

- [ ] **Step 1: Convert + gate each** — for name in minimal, baltic, baltic_ev:
`PYTHONPATH=. .venv/bin/python scripts/migrate_bundled_to_440.py data/<name>` then `... native_440_parity.py gate <name>`. Each must PASS < 1e-9. If a config FAILs, stop and diagnose (do not loosen the gate without cause).

- [ ] **Step 2: A-harness regression (Baltic)** — `PYTHONPATH=. .venv/bin/python scripts/baltic_440_smoke.py` must still exit 0 + predators feed (native-4.4.0 Baltic source + Task 1's source-aware rate = NO double-scale when the harness stages at 4.4.1). If it regresses, Task 1's fix is the suspect.

- [ ] **Step 3: Commit** `git add data/minimal data/baltic data/baltic_ev && git commit -m "feat(440): convert minimal/baltic/baltic_ev to native 4.4.0 (round-trip parity PASS, A-harness green)"`

---

### Task 5: Flip write/jar/calibration defaults to 4.4.1

**Files:** Modify `osmose/config/aliases.py`, `ui/state.py:42`, `ui/pages/run.py:366`, `osmose/calibration/problem.py:475`, `osmose/config/writer.py:63`, `osmose/demo.py:315`. Test: `tests/test_config_aliases.py`.

- [ ] **Step 1: Failing test** for the helper:
```python
def test_target_version_for_jar():
    from osmose.config.aliases import target_version_for_jar, DEFAULT_TARGET_VERSION
    assert target_version_for_jar("osmose-java/osmose-4.4.1-jar-with-dependencies.jar") == "4.4.1"
    assert target_version_for_jar("osmose-java/osmose_4.3.3-jar-with-dependencies.jar") == "4.3.3"
    assert target_version_for_jar("weird.jar") == DEFAULT_TARGET_VERSION == "4.4.1"
```

- [ ] **Step 2: Implement** in `aliases.py`:
```python
import re
DEFAULT_TARGET_VERSION = "4.4.1"
def target_version_for_jar(jar_path) -> str:
    m = re.search(r"(\d+\.\d+\.\d+)", str(jar_path))
    return m.group(1) if m else DEFAULT_TARGET_VERSION
```
Then wire: `ui/state.py:42` default jar → `osmose-java/osmose-4.4.1-jar-with-dependencies.jar`; `run.py:366` → `write_temp_config(config, work_dir, source_dir, key_case_map=..., target_version=target_version_for_jar(jar_path))`; `problem.py:475` → `to_target_keys(dict(overrides), target_version=target_version_for_jar(self.jar_path))`; `aliases.py:229` / `writer.py:63` / `demo.py:315` defaults → `DEFAULT_TARGET_VERSION` (import it; `demo.migrate_config` stays `"4.3.3"`-chain — only the `writer`/`to_target_keys` write defaults flip).

- [ ] **Step 3: Run** `pytest tests/test_config_aliases.py -q` PASS. Sanity: `target_version_for_jar` returns 4.3.3 for the legacy jar so picking it still reverse-maps.

- [ ] **Step 4: Commit** `git add -A && git commit -m "feat(440): flip default write-target/jar/calibration to 4.4.1 (derive target from selected jar)"`

---

### Task 6: Demote the legacy 4.3.3 framing + deprecation log

**Files:** Modify `osmose/config/aliases.py` (`canonicalize_config`), `osmose/engine/timeseries.py` (docstring), `osmose/engine/config_validation.py` (allowlist notes). Test: `tests/test_config_aliases.py`.

- [ ] **Step 1: Failing test** — loading a legacy 4.3.3 config emits a one-time deprecation warning:
```python
def test_canonicalize_warns_on_legacy_keys(recwarn):
    from osmose.config.aliases import canonicalize_config
    cfg, dep = canonicalize_config({"osmose.version": "4.3.3", "mortality.natural.larva.rate.sp0": "1"})  # an OLD-spelled key
    assert dep  # deprecated old keys returned
    assert any("4.3.3" in str(w.message) or "deprecat" in str(w.message).lower() for w in recwarn.list)
```
(Use a real RENAMES_440 OLD key for the input — pick one from the map.)

- [ ] **Step 2: Implement** — in `canonicalize_config`, when `deprecated` is non-empty, `warnings.warn(...)` a one-time legacy-config notice; reframe its docstring as a legacy 4.3.3 adapter. Update `timeseries.py`'s "Matches Java OSMOSE 4.3.3" to note 4.4.x compatibility (version-stable semantics — no behaviour change). Finalize `config_validation.py`'s 4.4.0-canonical allowlist comment.

- [ ] **Step 3: Run** `pytest tests/test_config_aliases.py -q` PASS; `test_from_dict_warn_mode_clean_on_example_configs` still warning-free (the bundled configs are now native 4.4.0 → no deprecation warning fires on them).

- [ ] **Step 4: Commit** `git add -A && git commit -m "refactor(440): demote canonicalize to a legacy 4.3.3 adapter (+ one-time deprecation log); clean residual 4.3.3 framing"`

---

### Task 7: Test sweep + full-suite green

**Files:** Modify affected tests under `tests/`. Add a legacy-4.3.3 fixture if none survives.

- [ ] **Step 1: Find the breakage** — `PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider` (full suite). Expect failures in tests asserting `osmose.version 4.3.3` or 4.3.3-spelled keys on the now-native bundled configs (~6–13).

- [ ] **Step 2: Update each** — flip the expected version/keys to 4.4.x where the test asserts the bundled config's representation. **Keep a legacy-4.3.3 config fixture** (e.g. a small inline dict or a `tests/data/legacy_433.csv`) so the `canonicalize` adapter + the reverse-map stay covered — do NOT convert that fixture.

- [ ] **Step 3: Run — full green** `pytest -q -p no:cacheprovider` → all pass. `cross_engine_parity_440.py` (EEC Java 4.4.1) still works (now reads native-4.4.0 EEC; `target_version_for_jar` keeps the 4.4.1 path).

- [ ] **Step 4: Lint** `.venv/bin/ruff check osmose/ ui/ tests/ scripts/ && .venv/bin/ruff format --check osmose/ ui/ tests/ scripts/` clean.

- [ ] **Step 5: Commit** `git add -A && git commit -m "test(440): update bundled-config assertions to native 4.4.x; keep a legacy-4.3.3 fixture"`

---

## Notes for the executor
- **Order is load-bearing**: Task 1 (source-aware rate) BEFORE any conversion, and Task 2 captures baselines BEFORE Task 3/4 mutate sources. If baselines are captured after a source is converted, the gate is meaningless.
- **Parity FAIL is a stop signal**, not a tolerance-to-loosen: investigate the dropped-key/rate cause first. The expected drivers are H1 (rate) and H2 (lmax/beta) — both handled, so a FAIL means a real regression.
- Inline execution (executing-plans) — subagent dispatch is spend-limited. Use the main-venv absolute path; never create a `.venv` in the worktree.
- The exact `OsmoseResults` accessor (`res.biomass` vs a method) is resolved at Task 2 impl — adapt `run_outputs` to whatever the API exposes; the comparison logic is the contract.
