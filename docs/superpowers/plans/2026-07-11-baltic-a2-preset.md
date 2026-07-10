# Baltic A2-calibrated preset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the converged A2 (depletable-plankton) Baltic calibration as a Python-engine-only bundled demo preset `baltic_a2`, selectable in the UI model picker.

**Architecture:** A DRY overlay on `data/baltic/`. `data/baltic_a2/` holds three small text files (a master + two delta CSVs); the generator copies baltic's full config then overlays the deltas, so no NetCDFs are duplicated. Depletion is Python-only, gated by a new `java_engine_block_reason` check.

**Tech Stack:** Python 3, `osmose.demo` (registry + generators), `osmose.config.reader.OsmoseConfigReader`, `osmose.runner.java_engine_block_reason`, `osmose.engine.PythonEngine`, pytest.

## Global Constraints

- **Do not modify `data/baltic/`** — it must stay byte-identical. Verify with `git status data/baltic/` at the end.
- **Do not re-run the DE or tune** — bake the existing `a2_on_converged` values verbatim.
- **No CI gate on emergent simulation outcomes** (biomass bands) — non-reproducible across CI cores. Emergent validation is a local, documented run only.
- **Honest framing everywhere the preset is named:** best-achievable, 3/8 in-band, NOT "ICES-calibrated". Pelagics + stickleback in-band, cod ~2.5× over, coastal percids structurally over.
- Config includes resolve as `master.parent / value` with an `is_relative_to(config_dir)` guard (`osmose/config/reader.py:124`): **bare basenames only**, and every referenced basename must exist flat in the generated `config/`.
- Species indices: sp0 cod, sp1 herring, sp2 sprat, sp3 flounder, sp4 perch, sp5 pike-perch, sp6 smelt, sp7 stickleback. LTL resources sp8 Diatoms, sp9 Dinoflagellates, sp10 Microzoo, sp11 Mesozoo, sp12 Macrozoo, sp13 Benthos.
- Run tests with `.venv/bin/python -m pytest`.

---

## File Structure

- **Create** `data/baltic_a2/baltic_a2_all-parameters.csv` — master; baltic's includes with `mortality.additional` redirected + `a2.depletion` added.
- **Create** `data/baltic_a2/baltic_a2_param-additional-mortality.csv` — 16 converged mortality values.
- **Create** `data/baltic_a2/baltic_a2_param-depletion.csv` — 10 depletion/regrowth keys.
- **Modify** `osmose/demo.py` — add `_generate_baltic_a2`, register in `list_demos()`, `DEMO_INFO`, `osmose_demo` generators dict.
- **Modify** `osmose/runner.py` — add the `ltl.depletable.enabled` guard to `java_engine_block_reason`.
- **Create** `tests/test_baltic_a2_demo.py` — CI-safe unit tests.
- **Modify** `docs/baltic_a2_calibration_results_2026-07-09.md` — add "Deployed as `baltic_a2` preset" section with local-validation numbers.

---

## Task 1: The `data/baltic_a2/` delta files

**Files:**
- Create: `data/baltic_a2/baltic_a2_all-parameters.csv`
- Create: `data/baltic_a2/baltic_a2_param-additional-mortality.csv`
- Create: `data/baltic_a2/baltic_a2_param-depletion.csv`
- Test: `tests/test_baltic_a2_demo.py`

**Interfaces:**
- Produces: three static CSVs consumed by Task 2's generator. The master references baltic sub-CSV basenames (`baltic_param-*.csv`) plus `baltic_a2_param-additional-mortality.csv` and `baltic_a2_param-depletion.csv`.

- [ ] **Step 1: Write the failing test** (`tests/test_baltic_a2_demo.py`)

```python
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import demo_info, list_demos, osmose_demo
from osmose.runner import java_engine_block_reason

DATA = Path(__file__).resolve().parent.parent / "data"
BALTIC_A2_DIR = DATA / "baltic_a2"

EXPECTED_MORTALITY = {
    "mortality.additional.larva.rate.sp0": "1.8495054614929225",
    "mortality.additional.larva.rate.sp1": "0.6091614461276307",
    "mortality.additional.larva.rate.sp2": "1.7574285062912955",
    "mortality.additional.larva.rate.sp3": "0.3277205467582994",
    "mortality.additional.larva.rate.sp4": "5.024141712395672",
    "mortality.additional.larva.rate.sp5": "1.1869723413415985",
    "mortality.additional.larva.rate.sp6": "0.3791432328547528",
    "mortality.additional.larva.rate.sp7": "0.27314862986759136",
    "mortality.additional.rate.sp0": "4.288045380663061",
    "mortality.additional.rate.sp1": "0.2636287453341465",
    "mortality.additional.rate.sp2": "0.003071941136699811",
    "mortality.additional.rate.sp3": "0.0045211280482306045",
    "mortality.additional.rate.sp4": "0.005680413608708062",
    "mortality.additional.rate.sp5": "0.855951786667689",
    "mortality.additional.rate.sp6": "0.0036156979635421347",
    "mortality.additional.rate.sp7": "0.19494616193531136",
}
EXPECTED_DEPLETION = {
    "ltl.depletable.enabled": "true",
    "ltl.depletable.floor": "0.05",
    "species.regrowth.rate.sp8": "5.0",
    "species.regrowth.rate.sp9": "5.0",
    "species.regrowth.rate.sp10": "1.0580953986747008",
    "species.regrowth.rate.sp11": "1.0580953986747008",
    "species.regrowth.rate.sp12": "1.0580953986747008",
    "species.regrowth.rate.sp13": "1.0580953986747008",
}


def _parse_csv(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition(";")
        d[k.strip()] = v.strip()
    return d


def test_a2_mortality_deltas_exact():
    got = _parse_csv(BALTIC_A2_DIR / "baltic_a2_param-additional-mortality.csv")
    assert got == EXPECTED_MORTALITY


def test_a2_depletion_deltas_exact():
    got = _parse_csv(BALTIC_A2_DIR / "baltic_a2_param-depletion.csv")
    assert got == EXPECTED_DEPLETION
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py::test_a2_mortality_deltas_exact tests/test_baltic_a2_demo.py::test_a2_depletion_deltas_exact -v`
Expected: FAIL (files do not exist → FileNotFoundError).

- [ ] **Step 3: Create `data/baltic_a2/baltic_a2_param-additional-mortality.csv`**

```
# Baltic Sea (A2-calibrated) additional mortality — CONVERGED DE values (a2_on_converged).
# Replaces the R18 values used by the baltic demo. Co-calibrated WITH depletable plankton on
# (see baltic_a2_param-depletion.csv); these are far lower than baltic's R18 larval rates
# because depletion, not extreme larval mortality, brakes over-production under A2.
# Source: docs/diagnostics/baltic_a2_calibrated_params.json -> a2_on_converged.params
# (objective 2.68 multi-seed; DE via scripts/calibrate_baltic.py --a2 --isolated-eval).
# Larval mortality (year^-1)
mortality.additional.larva.rate.sp0;1.8495054614929225
mortality.additional.larva.rate.sp1;0.6091614461276307
mortality.additional.larva.rate.sp2;1.7574285062912955
mortality.additional.larva.rate.sp3;0.3277205467582994
mortality.additional.larva.rate.sp4;5.024141712395672
mortality.additional.larva.rate.sp5;1.1869723413415985
mortality.additional.larva.rate.sp6;0.3791432328547528
mortality.additional.larva.rate.sp7;0.27314862986759136
# Adult additional mortality (year^-1)
mortality.additional.rate.sp0;4.288045380663061
mortality.additional.rate.sp1;0.2636287453341465
mortality.additional.rate.sp2;0.003071941136699811
mortality.additional.rate.sp3;0.0045211280482306045
mortality.additional.rate.sp4;0.005680413608708062
mortality.additional.rate.sp5;0.855951786667689
mortality.additional.rate.sp6;0.0036156979635421347
mortality.additional.rate.sp7;0.19494616193531136
```

- [ ] **Step 4: Create `data/baltic_a2/baltic_a2_param-depletion.csv`**

```
# Chunk A2 depletable-plankton keys for the baltic_a2 preset.
# Phytoplankton (sp8/sp9) regrow fast (fixed 5.0, ~non-limiting); depletable zooplankton +
# benthos (sp10-13) use the converged zoo regrowth rate. Python engine only.
# Source: enable_a2_base_config() + a2_on_converged zoo regrowth (scripts/calibrate_baltic.py).
ltl.depletable.enabled;true
ltl.depletable.floor;0.05
species.regrowth.rate.sp8;5.0
species.regrowth.rate.sp9;5.0
species.regrowth.rate.sp10;1.0580953986747008
species.regrowth.rate.sp11;1.0580953986747008
species.regrowth.rate.sp12;1.0580953986747008
species.regrowth.rate.sp13;1.0580953986747008
```

- [ ] **Step 5: Create `data/baltic_a2/baltic_a2_all-parameters.csv`**

```
# Baltic Sea (A2-calibrated) OSMOSE configuration - Main parameter file
# Overlay on the baltic demo: depletable plankton (Chunk A2) + converged DE mortality.
# BEST-ACHIEVABLE community fit (3/8 in-band, objective 2.68), NOT a full ICES calibration:
# herring, sprat, stickleback land in-band; cod ~2.5x over; coastal percids structurally over.
# All baltic_param-*.csv includes are resolved from the generated config/ dir (see osmose.demo).
# See docs/baltic_a2_calibration_results_2026-07-09.md.

osmose.configuration.simulation;baltic_param-simulation.csv
osmose.configuration.species;baltic_param-species.csv
osmose.configuration.grid;baltic_param-grid.csv
osmose.configuration.output;baltic_param-output.csv
osmose.configuration.mortality.predation;baltic_param-predation.csv
osmose.configuration.mortality.fishing;baltic_param-fishing.csv
osmose.configuration.mortality.starvation;baltic_param-starvation.csv
osmose.configuration.mortality.additional;baltic_a2_param-additional-mortality.csv
osmose.configuration.reproduction;baltic_param-reproduction.csv
osmose.configuration.movement;baltic_param-movement.csv
osmose.configuration.plankton;baltic_param-ltl.csv
osmose.configuration.background;baltic_param-background.csv
osmose.configuration.initialization;baltic_param-init-pop.csv
osmose.configuration.migration;baltic_param-out-mortality.csv
osmose.configuration.a2.depletion;baltic_a2_param-depletion.csv

mortality.subdt;10

simulation.nschool.sp0;50
simulation.nschool.sp1;60
simulation.nschool.sp2;60
simulation.nschool.sp3;40
simulation.nschool.sp4;30
simulation.nschool.sp5;30
simulation.nschool.sp6;30
simulation.nschool.sp7;40

osmose.version;4.4.1
```

Note: this master's non-a2 include lines are copied verbatim from `data/baltic/baltic_all-parameters.csv` (keep them in sync). The only differences: `mortality.additional` → `baltic_a2_param-additional-mortality.csv`, and the added `a2.depletion` line.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py::test_a2_mortality_deltas_exact tests/test_baltic_a2_demo.py::test_a2_depletion_deltas_exact -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add data/baltic_a2/ tests/test_baltic_a2_demo.py
git commit -m "feat(baltic): baltic_a2 preset delta CSVs (converged A2 mortality + depletion)"
```

---

## Task 2: Generator + registration

**Files:**
- Modify: `osmose/demo.py` (add `_generate_baltic_a2`; edit `list_demos`, `DEMO_INFO`, `osmose_demo`)
- Test: `tests/test_baltic_a2_demo.py`

**Interfaces:**
- Consumes: Task 1's three CSVs via `_bundled_data_dir("baltic_a2")`.
- Produces: `osmose_demo("baltic_a2", out) -> {"config_file": <config/baltic_a2_all-parameters.csv>, "output_dir": <output/>}`. `demo_info("baltic_a2")` with engine `"Python"`. `"baltic_a2"` in `list_demos()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_baltic_a2_demo.py`)

```python
def test_a2_registered_python_only():
    assert "baltic_a2" in list_demos()
    info = demo_info("baltic_a2")
    assert info is not None
    for field in ("title", "region", "species", "resources", "engine", "summary"):
        assert info.get(field), f"DEMO_INFO['baltic_a2'] missing {field}"
    assert info["engine"] == "Python"
    assert "a2" in info["title"].lower() or "calibrat" in info["title"].lower()


def test_a2_generates_and_loads(tmp_path):
    out = osmose_demo("baltic_a2", tmp_path)
    cfg = Path(out["config_file"])
    assert cfg.name == "baltic_a2_all-parameters.csv" and cfg.exists()
    # Overlay must NOT duplicate NetCDFs: baltic_a2 dir is text-only.
    assert not any(p.suffix == ".nc" for p in BALTIC_A2_DIR.iterdir())
    # Loads cleanly through the reader (proves basename includes resolve after overlay).
    loaded = dict(OsmoseConfigReader().read(str(cfg)))
    for key, val in {**EXPECTED_MORTALITY, **EXPECTED_DEPLETION}.items():
        assert key in loaded, f"missing {key} in loaded config"
        assert float(loaded[key]) == float(val), f"{key}: {loaded[key]} != {val}"
    assert loaded["simulation.time.nyear"] == "15"  # inherited from baltic


def _includes(path: Path) -> dict[str, str]:
    return {
        k: v for k, v in _parse_csv(path).items() if k.startswith("osmose.configuration.")
    }


def test_a2_master_includes_parity(tmp_path):
    baltic_inc = _includes(DATA / "baltic" / "baltic_all-parameters.csv")
    a2_inc = _includes(BALTIC_A2_DIR / "baltic_a2_all-parameters.csv")
    # Same include KEYS plus the one new depletion include.
    assert set(a2_inc) == set(baltic_inc) | {"osmose.configuration.a2.depletion"}
    # Every include TARGET basename exists in the generated config dir.
    out = osmose_demo("baltic_a2", tmp_path)
    cfgdir = Path(out["config_file"]).parent
    for target in a2_inc.values():
        assert (cfgdir / target).exists(), f"include target missing: {target}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py -k "registered or generates or parity" -v`
Expected: FAIL (`baltic_a2` not in `list_demos()`; `osmose_demo` raises `ValueError: Unknown scenario`).

- [ ] **Step 3: Add `baltic_a2` to `list_demos()`** (`osmose/demo.py:102-104`)

```python
def list_demos() -> list[str]:
    """List available demo scenarios."""
    return ["baltic", "baltic_a2", "bay_of_biscay", "eec", "eec_full", "minimal", "benguela"]
```

- [ ] **Step 4: Add the `DEMO_INFO["baltic_a2"]` entry** (insert after the `"baltic"` block, `osmose/demo.py:145`)

```python
    "baltic_a2": {
        "title": "Baltic Sea (A2-calibrated)",
        "region": "Central/Eastern Baltic",
        "species": "8 focal species",
        "resources": "6 LTL (depletable plankton) + 2 background groups",
        "engine": "Python",
        "summary": "The Baltic demo with depletable plankton (Chunk A2) and the converged DE "
        "mortality calibration. Best-achievable community fit, NOT fully ICES-calibrated: "
        "herring, sprat and stickleback land in-band and cod sits just above band, while the "
        "coastal percids (perch/pike-perch) stay structurally over at this grid resolution. A2 "
        "compresses the A2-off overshoot (17-400x) down to near-band. Python engine only "
        "(depletable plankton has no Java equivalent).",
    },
```

- [ ] **Step 5: Register the generator in `osmose_demo`** (`osmose/demo.py:184-191`)

```python
    generators = {
        "baltic": _generate_baltic,
        "baltic_a2": _generate_baltic_a2,
        "bay_of_biscay": _generate_bay_of_biscay,
        "eec": _generate_eec,
        "eec_full": _generate_eec_full,
        "minimal": _generate_minimal,
        "benguela": _generate_benguela,
    }
```

- [ ] **Step 6: Add `_generate_baltic_a2`** (insert immediately after `_generate_baltic`, `osmose/demo.py:222`)

```python
def _generate_baltic_a2(output_dir: Path) -> dict:
    """Generate the A2-calibrated Baltic preset (depletable plankton + converged mortality).

    A thin overlay on the baltic demo: copy baltic's full config (grid/forcing/maps/sub-CSVs),
    then overlay the three baltic_a2 delta files (master + a2 mortality + a2 depletion). No
    NetCDFs are duplicated. Python-engine only (depletion has no Java equivalent).
    """
    data_dir = _bundled_data_dir("baltic")
    a2_dir = _bundled_data_dir("baltic_a2")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None and a2_dir is not None:
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
        shutil.copytree(a2_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "baltic_a2_all-parameters.csv").write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 15\n"
            "simulation.nspecies ; 8\n"
            "simulation.nresource ; 6\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "baltic_a2_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py -k "registered or generates or parity" -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add osmose/demo.py tests/test_baltic_a2_demo.py
git commit -m "feat(baltic): register baltic_a2 demo preset (overlay generator + picker metadata)"
```

---

## Task 3: Java-engine guard

**Files:**
- Modify: `osmose/runner.py` (`java_engine_block_reason`)
- Test: `tests/test_baltic_a2_demo.py`

**Interfaces:**
- Consumes: `java_engine_block_reason(config: dict, jar_version=None) -> str | None` — existing signature, operates on the flattened config.
- Produces: returns a non-None Python-only reason when `config["ltl.depletable.enabled"]` is truthy.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_baltic_a2_demo.py`)

```python
def test_a2_blocks_java_engine(tmp_path):
    out = osmose_demo("baltic_a2", tmp_path)
    loaded = dict(OsmoseConfigReader().read(str(out["config_file"])))
    # Pin jar 4.4.1 so the nbackground path is NOT what blocks it (baltic_a2 inherits
    # nbackground=2 with GreySeal/Cormorant staging, which 4.4.1 supports -> that path
    # returns None). The ONLY thing that must block it is the depletable-plankton guard.
    reason = java_engine_block_reason(loaded, jar_version="4.4.1")
    assert reason is not None
    assert "depletable" in reason.lower()


def test_java_guard_depletion_check_direct():
    assert java_engine_block_reason({"ltl.depletable.enabled": "true"}) is not None
    # Non-depletable config with no background is still Java-runnable (regression guard).
    assert java_engine_block_reason({"ltl.depletable.enabled": "false"}) is None
    assert java_engine_block_reason({}) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py -k "java" -v`
Expected: FAIL — pre-fix, `test_a2_blocks_java_engine` returns `None` (jar 4.4.1 + supported staging → the nbackground path returns None and there is no depletion guard yet), and `test_java_guard_depletion_check_direct` returns `None` for the depletable-only config.

- [ ] **Step 3: Add the guard** (insert at the start of `java_engine_block_reason`, `osmose/runner.py:28`, before the benguela check)

```python
    if str(config.get("ltl.depletable.enabled", "")).strip().lower() == "true":
        return (
            "This configuration uses depletable plankton (ltl.depletable.enabled), a "
            "Python-engine feature with no Java-jar equivalent. Run it on the Python engine."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py -k "java" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full new test file + the runner/demo regression tests**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py tests/test_benguela_demo.py -v && .venv/bin/python -m pytest tests/ -k "runner or java_engine or demo" -q`
Expected: PASS (no regressions in existing demo / java-guard tests).

- [ ] **Step 6: Commit**

```bash
git add osmose/runner.py tests/test_baltic_a2_demo.py
git commit -m "feat(runner): block Java engine for depletable-plankton configs (baltic_a2)"
```

---

## Task 4: Local validation + results-doc update

**Files:**
- Modify: `docs/baltic_a2_calibration_results_2026-07-09.md`
- (No test — this is a documented local run, deliberately NOT a CI gate.)

**Interfaces:**
- Consumes: `osmose_demo("baltic_a2", …)` + `PythonEngine().run_in_memory(raw, seed)`.

- [ ] **Step 1: Run the preset locally (two seeds, nyear 15)**

Run:
```bash
.venv/bin/python -c "
import tempfile, numpy as np
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
BANDS = {'cod':(60e3,250e3),'herring':(0.8e6,3e6),'sprat':(0.8e6,2.5e6),
         'flounder':(20e3,100e3),'perch':(8e3,50e3),'pikeperch':(4e3,25e3),
         'smelt':(20e3,120e3),'stickleback':(50e3,500e3)}
tmp = Path(tempfile.mkdtemp())
raw = dict(OsmoseConfigReader().read(str(osmose_demo('baltic_a2', tmp)['config_file'])))
for seed in (42, 123):
    b = PythonEngine().run_in_memory(raw, seed=seed).biomass()
    last = b.iloc[-10:].mean()
    print(f'--- seed {seed} ---')
    for sp,(lo,hi) in BANDS.items():
        v = float(last.get(sp, float('nan')))
        st = 'in' if lo<=v<=hi else ('over %.1fx'%(v/hi) if v>hi else 'under')
        print(f'  {sp:12s} {v:12.0f}  [{lo:.0f}-{hi:.0f}]  {st}')
"
```
Expected: runs to completion (no crash); community lands near the documented bands — pelagics + stickleback roughly in-band, cod ~2.5×, coastal percids over. Capture the printed numbers.

- [ ] **Step 2: Add a "Deployed as `baltic_a2` preset" section** to `docs/baltic_a2_calibration_results_2026-07-09.md`

Append a section stating: the converged config is now bundled as the `baltic_a2` demo preset (Python-engine only; DRY overlay on `data/baltic`, no NetCDF duplication; Java-gated on `ltl.depletable.enabled`). Paste the Step-1 per-seed table. Reiterate the honest framing (best-achievable, 3/8, percids structural). Note single-run variance vs. the multi-seed reference numbers.

- [ ] **Step 3: Verify `data/baltic/` is byte-unchanged**

Run: `git status --porcelain data/baltic/`
Expected: **empty output** (no modifications to the deployed config).

- [ ] **Step 4: Full CI-safe test sweep + lint**

Run: `.venv/bin/python -m pytest tests/test_baltic_a2_demo.py -v && .venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/`
Expected: all tests PASS; ruff clean (check + format). Fix any lint issues and re-run.

- [ ] **Step 5: Commit**

```bash
git add docs/baltic_a2_calibration_results_2026-07-09.md
git commit -m "docs(baltic): record baltic_a2 preset local validation + deployment note"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** every spec section maps to a task — delta files (T1), generator+registration (T2), Java guard (T3), local validation + honest-framing doc (T4). All 6 spec tests are placed (test1→T2, test2/3→T1+T2, test4→T2, test5→T2, test6→T3).
- **Placeholder scan:** no TBD/TODO; all CSV contents, code, and commands are literal.
- **Type consistency:** `_parse_csv`/`_includes` helpers, `EXPECTED_MORTALITY`/`EXPECTED_DEPLETION` constants, and `osmose_demo`/`java_engine_block_reason`/`OsmoseConfigReader().read` signatures are consistent across tasks.
- **TDD-red integrity (resolved):** `test_a2_blocks_java_engine` pins `jar_version="4.4.1"` so the nbackground path returns `None` (baltic_a2 inherits supported GreySeal/Cormorant staging) — making the depletion guard the *only* thing that can block it, a clean red→green. Without the pin it would spuriously pass pre-fix via the background message.
