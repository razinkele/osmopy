# Baltic Depletable LTL (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable depletable plankton (grazing feedback with logistic regrowth) in the production 9-species Baltic config, via an A/B-measure-first protocol gated by an identity-pinned certification.

**Architecture:** No engine changes — the depletable mode already exists (`osmose/engine/resources.py`, keyed `ltl.depletable.enabled`). We add (1) an A/B harness script reusing `certify_python` from the certifier, (2) a Java-arm pinning helper in the certifier (depletion is Python-only), (3) on gate PASS, a `baltic_param-depletion.csv` overlay in `data/baltic` mirroring the existing `data/baltic_a2` pattern, with loading-assertion tests.

**Tech Stack:** Python 3.12, `.venv/bin/python`, pytest, ruff (line length 100). Engine runs via `scripts/baltic_stability_certify.py` internals.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 1.
- Starting keys, exactly (spec §4 Phase 1 item 2; provenance `data/baltic/calibration_results/phase1_results.json` and `enable_a2_base_config` in `scripts/calibrate_baltic.py`):
  `ltl.depletable.enabled=true`, `ltl.depletable.floor=0.05`,
  `species.regrowth.rate.sp9=5.0`, `species.regrowth.rate.sp10=5.0`,
  `species.regrowth.rate.sp11..sp14=0.911553421016705`.
- Identity-pinned gate: `cod_west, cod_east, herring, sprat, flounder, perch, stickleback` must each have `persists AND in_envelope` (worst case across seeds). `pikeperch`/`smelt` are reported, never gated, never tuned against.
- Certification protocol: 50 years, seeds `42 123 7 999 2024`, `scripts/baltic_stability_certify.py`.
- The 9-species production resources are **sp9–sp14** (Diatoms, Dinoflagellates, Micro/Meso/Macrozooplankton, Benthos). The `baltic_a2` demo is the OLD 8-species layout (resources sp8–13) — do not copy its indices.
- `--java` certification arms pin `ltl.depletable.enabled=false` and say so (runner guard `osmose/runner.py:java_engine_block_reason` blocks Java otherwise).
- All config keys used here are already in the validation allowlist (the `baltic_a2` strict-validation tests prove it) — no `config_validation.py` changes.
- Run tests with `.venv/bin/python -m pytest`; lint with `.venv/bin/ruff check scripts/ tests/` and `format`.
- D3 (resource-pool seasonality) and F2 (focal-prey seasonal accessibility) are NOT in this plan.

---

### Task 1: A/B harness script

**Files:**
- Create: `scripts/baltic_depletable_ab.py`
- Test: `tests/test_baltic_depletable_ab.py`

**Interfaces:**
- Consumes: `certify_python(params: dict[str,str], n_years: int, seeds) -> dict` and `_print_table(engine: str, table: dict) -> int` from `scripts/baltic_stability_certify.py`; certify table shape per species: `{"persists": bool, "in_envelope": bool, "min_biomass": float, "late_mean_range": [lo, hi]}`.
- Produces: `DEPLETION_KEYS: dict[str,str]`, `REQUIRED_PASS: tuple[str,...]`, `identity_gate(table) -> tuple[bool, list[str]]`, `make_report(tables: dict[str,dict], years: int, seeds: list[int]) -> str` — Task 3 runs this script; Task 4 copies `DEPLETION_KEYS` values into the config.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baltic_depletable_ab.py
"""A/B harness for depletable LTL (spec 2026-08-08 Phase 1): keys, gate, report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baltic_depletable_ab as ab  # noqa: E402

FITTED_ZOO = "0.911553421016705"


def _row(persists=True, in_env=True, mean=1000.0):
    return {
        "persists": persists,
        "in_envelope": in_env,
        "min_biomass": mean / 10,
        "late_mean_range": [mean * 0.95, mean * 1.05],
    }


def _table(**overrides):
    species = [
        "cod_west", "cod_east", "herring", "sprat", "flounder",
        "perch", "pikeperch", "smelt", "stickleback",
    ]
    t = {sp: _row() for sp in species}
    t.update(overrides)
    return t


def test_depletion_keys_exact():
    assert ab.DEPLETION_KEYS == {
        "ltl.depletable.enabled": "true",
        "ltl.depletable.floor": "0.05",
        "species.regrowth.rate.sp9": "5.0",
        "species.regrowth.rate.sp10": "5.0",
        "species.regrowth.rate.sp11": FITTED_ZOO,
        "species.regrowth.rate.sp12": FITTED_ZOO,
        "species.regrowth.rate.sp13": FITTED_ZOO,
        "species.regrowth.rate.sp14": FITTED_ZOO,
    }


def test_required_pass_is_identity_pinned():
    assert ab.REQUIRED_PASS == (
        "cod_west", "cod_east", "herring", "sprat", "flounder", "perch", "stickleback"
    )


def test_identity_gate_passes_clean_table():
    ok, failures = ab.identity_gate(_table())
    assert ok and failures == []


def test_identity_gate_fails_on_required_species():
    ok, failures = ab.identity_gate(_table(perch=_row(in_env=False)))
    assert not ok and failures == ["perch"]


def test_identity_gate_ignores_indicative_failures():
    ok, failures = ab.identity_gate(
        _table(pikeperch=_row(in_env=False), smelt=_row(persists=False))
    )
    assert ok and failures == []


def test_make_report_contains_arms_gate_and_deltas():
    tables = {"off": _table(), "on": _table(herring=_row(mean=800.0))}
    rep = ab.make_report(tables, years=50, seeds=[42, 123])
    assert "herring" in rep and "GATE" in rep and "off" in rep and "on" in rep
    assert "-20.0%" in rep  # 800 vs 1000 midpoint delta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_depletable_ab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'baltic_depletable_ab'`

- [ ] **Step 3: Write the script**

```python
# scripts/baltic_depletable_ab.py
"""A/B: depletable LTL off vs on, under current production parameters (spec 2026-08-08 Phase 1).

Runs certify_python for both arms and reports per-species final-decade deltas plus the
identity-pinned gate verdict. Measure first, certify second — this script issues NO adoption
verdict on its own; a human reads the report (spec: A/B before any certification verdict).

    PYTHONPATH=. .venv/bin/python scripts/baltic_depletable_ab.py --out docs/baltic_depletable_ab_2026-08-08.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baltic_stability_certify import CERT_SEEDS, _print_table, certify_python  # noqa: E402

_FITTED_ZOO = "0.911553421016705"  # phase1_results.json sentinel species.regrowth.rate.zoo
A2_SENSITIVITY_ZOO_RATE = "1.0580953986747008"  # a2_on_converged (8-species co-fit; stale here)

DEPLETION_KEYS = {
    "ltl.depletable.enabled": "true",
    "ltl.depletable.floor": "0.05",
    "species.regrowth.rate.sp9": "5.0",   # Diatoms — phyto pinned near-reset (enable_a2_base_config)
    "species.regrowth.rate.sp10": "5.0",  # Dinoflagellates
    "species.regrowth.rate.sp11": _FITTED_ZOO,  # Microzooplankton
    "species.regrowth.rate.sp12": _FITTED_ZOO,  # Mesozooplankton
    "species.regrowth.rate.sp13": _FITTED_ZOO,  # Macrozooplankton
    "species.regrowth.rate.sp14": _FITTED_ZOO,  # Benthos — WAS in the fitted group (spec finding)
}

REQUIRED_PASS = ("cod_west", "cod_east", "herring", "sprat", "flounder", "perch", "stickleback")
TRACKED_ONLY = ("pikeperch", "smelt")


def identity_gate(table: dict) -> tuple[bool, list[str]]:
    """Identity-pinned gate: every REQUIRED_PASS species persists AND is in envelope."""
    failures = [
        sp for sp in REQUIRED_PASS
        if not (table[sp]["persists"] and table[sp]["in_envelope"])
    ]
    return (not failures, failures)


def _mid(row: dict) -> float:
    lo, hi = row["late_mean_range"]
    return (lo + hi) / 2.0


def make_report(tables: dict[str, dict], years: int, seeds: list[int]) -> str:
    arms = list(tables)
    base = tables[arms[0]]
    lines = [
        "# Depletable LTL A/B (Phase 1, spec 2026-08-08)",
        "",
        f"**Arms:** {', '.join(arms)} · **horizon:** {years} yr · **seeds:** {list(seeds)}",
        "",
        "| species | " + " mid (t) | ".join(arms) + " mid (t) | delta vs " + arms[0] + " | gated |",
        "|---|" + "---|" * (len(arms) + 2),
    ]
    for sp in base:
        mids = [_mid(tables[a][sp]) for a in arms]
        delta = (mids[-1] - mids[0]) / mids[0] * 100 if mids[0] else float("nan")
        gated = "yes" if sp in REQUIRED_PASS else "tracked only"
        cells = " | ".join(f"{m:,.0f}" for m in mids)
        lines.append(f"| {sp} | {cells} | {delta:+.1f}% | {gated} |")
    for arm in arms:
        ok, failures = identity_gate(tables[arm])
        verdict = "PASS" if ok else f"FAIL ({', '.join(failures)})"
        lines.append("")
        lines.append(f"**GATE [{arm}]: {verdict}** (required: {', '.join(REQUIRED_PASS)})")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CERT_SEEDS))
    ap.add_argument("--sensitivity", action="store_true",
                    help="add a third arm with the a2_on_converged zoo rate (8-species co-fit)")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    arms: dict[str, dict[str, str]] = {"off": {}, "on": dict(DEPLETION_KEYS)}
    if args.sensitivity:
        sens = dict(DEPLETION_KEYS)
        for sp in ("sp11", "sp12", "sp13", "sp14"):
            sens[f"species.regrowth.rate.{sp}"] = A2_SENSITIVITY_ZOO_RATE
        arms["on-a2conv"] = sens

    tables = {}
    for name, params in arms.items():
        print(f"\n=== arm: {name} ===")
        tables[name] = certify_python(params, args.years, args.seeds)
        _print_table(f"Python[{name}]", tables[name])

    report = make_report(tables, args.years, args.seeds)
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_depletable_ab.py -v`
Expected: 6 PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check scripts/baltic_depletable_ab.py tests/test_baltic_depletable_ab.py && .venv/bin/ruff format scripts/baltic_depletable_ab.py tests/test_baltic_depletable_ab.py
git add scripts/baltic_depletable_ab.py tests/test_baltic_depletable_ab.py
git commit -m "feat(baltic): depletable-LTL A/B harness with identity-pinned gate (Phase 1)"
```

---

### Task 2: Java-arm pinning in the certifier

**Files:**
- Modify: `scripts/baltic_stability_certify.py` (function `certify_java`, after `cfg.update(params)` at ~line 196)
- Test: `tests/test_certify_java_pinning.py`

**Interfaces:**
- Consumes: `osmose.runner.java_engine_block_reason(config) -> str | None` (returns a reason string when `ltl.depletable.enabled=true`).
- Produces: `pin_java_incompatible(cfg: dict) -> tuple[dict, list[str]]` in `baltic_stability_certify.py` — copy of cfg with Python-only flags forced off, plus the list of keys that were pinned.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_certify_java_pinning.py
"""--java certification arms pin Python-only depletion off (spec 2026-08-08 §4 Phase 1 parity)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baltic_stability_certify as cert  # noqa: E402
from osmose.runner import java_engine_block_reason  # noqa: E402


def test_pin_flips_enabled_true():
    cfg, pinned = cert.pin_java_incompatible({"ltl.depletable.enabled": "true", "x": "1"})
    assert cfg["ltl.depletable.enabled"] == "false"
    assert pinned == ["ltl.depletable.enabled"]
    assert cfg["x"] == "1"


def test_pin_noop_when_absent_or_false():
    for base in ({}, {"ltl.depletable.enabled": "false"}):
        cfg, pinned = cert.pin_java_incompatible(dict(base))
        assert pinned == []
        assert cfg.get("ltl.depletable.enabled", "false") == "false"


def test_pinned_cfg_passes_runner_guard():
    cfg, _ = cert.pin_java_incompatible({"ltl.depletable.enabled": "true"})
    assert java_engine_block_reason(cfg) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_certify_java_pinning.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'pin_java_incompatible'`

- [ ] **Step 3: Implement the helper and wire it into certify_java**

Add near the top of `scripts/baltic_stability_certify.py` (after `CERT_SEEDS`):

```python
# Python-only features with no Java-jar equivalent: the Java cross-check arm runs with these
# pinned off and SAYS so, keeping the arm runnable while labelling the divergence
# (spec 2026-08-08 §4 Phase 1; runner.java_engine_block_reason blocks Java otherwise).
JAVA_INCOMPATIBLE_PINS = {"ltl.depletable.enabled": "false"}


def pin_java_incompatible(cfg: dict) -> tuple[dict, list[str]]:
    """Copy of ``cfg`` with Python-only flags forced off, plus the list of keys pinned."""
    out = dict(cfg)
    pinned = []
    for key, off_value in JAVA_INCOMPATIBLE_PINS.items():
        if str(out.get(key, "")).strip().lower() == "true":
            out[key] = off_value
            pinned.append(key)
    return out, pinned
```

In `certify_java`, immediately after `cfg.update(params)  # bake the recalibrated params...`:

```python
    cfg, pinned = pin_java_incompatible(cfg)
    if pinned:
        print(f"(Java arm: pinned off Python-only features: {', '.join(pinned)})")
```

- [ ] **Step 4: Run the new tests and the existing certifier tests**

Run: `.venv/bin/python -m pytest tests/test_certify_java_pinning.py tests/test_certify_weight_aware.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check scripts/baltic_stability_certify.py tests/test_certify_java_pinning.py
git add scripts/baltic_stability_certify.py tests/test_certify_java_pinning.py
git commit -m "feat(certify): --java arm pins Python-only depletion off and labels it"
```

---

### Task 3: Run the A/B (compute checkpoint + decision gate)

**Files:**
- Create: `docs/baltic_depletable_ab_2026-08-08.md` (script output; adjust date to run date)

**Interfaces:**
- Consumes: `scripts/baltic_depletable_ab.py` CLI from Task 1.
- Produces: the A/B report — Task 4 proceeds ONLY if `GATE [on]: PASS` appears in it.

- [ ] **Step 1: Run the A/B**

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_depletable_ab.py --out docs/baltic_depletable_ab_2026-08-08.md`
Expected: two arms × 5 seeds × 50 yr ≈ 1.5–2 h serial (one historical certification ≈ 50 min). Run in background/tmux; do not run other engine jobs concurrently (spawn-pool constraint).

- [ ] **Step 2: Read the report and decide**

Read `docs/baltic_depletable_ab_2026-08-08.md`.
- `GATE [on]: PASS` → continue to Step 3 and Task 4.
- `GATE [on]: FAIL (...)` → STOP the plan here. Commit the report with a `docs(baltic):` message stating which species left the gate and by how much. The contingency (bounded recalibration of `species.regrowth.rate.sp11..sp14` and zooplanktivore availability coefficients only — spec §4 Phase 1 item 4) is a separate follow-up plan; adoption does not proceed.

- [ ] **Step 3: Commit the report**

```bash
git add docs/baltic_depletable_ab_2026-08-08.md
git commit -m "docs(baltic): depletable-LTL A/B — per-species deltas and identity-gate verdict"
```

---

### Task 4: Adoption — depletion overlay in `data/baltic` (only after GATE [on]: PASS)

**Files:**
- Create: `data/baltic/baltic_param-depletion.csv`
- Modify: `data/baltic/baltic_all-parameters.csv` (add one include line next to `osmose.configuration.plankton;baltic_param-ltl.csv`)
- Modify: `data/baltic/baltic_param-ltl.csv` (one pointer comment)
- Test: `tests/test_baltic_depletion_config.py`

**Interfaces:**
- Consumes: `DEPLETION_KEYS` values from Task 1 (must match byte-for-byte); `OsmoseConfigReader`, `ResourceState`, `Grid` from the engine.
- Produces: the production config with depletion enabled — Task 5 certifies it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baltic_depletion_config.py
"""Production Baltic depletion overlay: raw keys exact, engine loads them (spec 2026-08-08 Phase 1).

Guards the silent wrong-key-family no-op: the legacy ltl.regrowth.rate.rsc{i} keys are ignored on
the species-type loading path and pass validation warning-free, so only a functional assertion on
ResourceSpeciesInfo.regrowth_rate proves the config actually reaches the engine.
"""

from __future__ import annotations

from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState

DATA = Path(__file__).resolve().parents[1] / "data" / "baltic"
FITTED_ZOO = "0.911553421016705"

EXPECTED_RAW = {
    "ltl.depletable.enabled": "true",
    "ltl.depletable.floor": "0.05",
    "species.regrowth.rate.sp9": "5.0",
    "species.regrowth.rate.sp10": "5.0",
    "species.regrowth.rate.sp11": FITTED_ZOO,
    "species.regrowth.rate.sp12": FITTED_ZOO,
    "species.regrowth.rate.sp13": FITTED_ZOO,
    "species.regrowth.rate.sp14": FITTED_ZOO,
}


def _raw_pairs(path: Path) -> dict[str, str]:
    pairs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(";")
        pairs[key.strip()] = value.strip()
    return pairs


def test_master_includes_depletion_file():
    text = (DATA / "baltic_all-parameters.csv").read_text()
    assert "osmose.configuration.depletion;baltic_param-depletion.csv" in text


def test_depletion_raw_keys_exact():
    assert _raw_pairs(DATA / "baltic_param-depletion.csv") == EXPECTED_RAW


def test_engine_loads_regrowth_rates(tmp_path):
    res = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    # Grid dims are irrelevant to the metadata assertion (forcing regrids if needed).
    rs = ResourceState(config=cfg, grid=Grid.from_dimensions(ny=40, nx=50))
    assert rs.depletable is True
    assert rs.depletable_floor == 0.05
    rates = {s.name: s.regrowth_rate for s in rs.species}
    assert rates == {
        "Diatoms": 5.0,
        "Dinoflagellates": 5.0,
        "Microzooplankton": float(FITTED_ZOO),
        "Mesozooplankton": float(FITTED_ZOO),
        "Macrozooplankton": float(FITTED_ZOO),
        "Benthos": float(FITTED_ZOO),
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_depletion_config.py -v`
Expected: 3 FAIL (missing include line, missing file, `rs.depletable is False`)

- [ ] **Step 3: Create the overlay file**

`data/baltic/baltic_param-depletion.csv`:

```
# Depletable plankton for the production 9-species Baltic config (Phase 1,
# docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md).
# Python engine only — no Java-jar equivalent; certify --java pins ltl.depletable.enabled=false.
#
# Provenance: phase-1 joint fit (data/baltic/calibration_results/phase1_results.json) fitted
# species.regrowth.rate.zoo = 0.911553421016705 over sp11-sp14 (zooplankton AND benthos);
# phytoplankton (sp9-sp10) pinned near-reset at 5.0 per enable_a2_base_config()
# (scripts/calibrate_baltic.py). Adopted after the depletable on/off A/B
# (docs/baltic_depletable_ab_2026-08-08.md) passed the identity-pinned gate.
#
# KNOWN APPROXIMATION (spec §4 Phase 1 item 3): the CMEMS/ERGOM forcing is a POST-GRAZING
# standing-stock climatology whose closure already implicitly includes fish predation; using it
# as carrying capacity K while ALSO deducting explicit OSMOSE grazing double-counts removal.
# Accepted and documented here; compensation (ungrazed-K reinterpretation) is deferred to
# Phase 3, which requires it anyway.

ltl.depletable.enabled;true
ltl.depletable.floor;0.05
species.regrowth.rate.sp9;5.0
species.regrowth.rate.sp10;5.0
species.regrowth.rate.sp11;0.911553421016705
species.regrowth.rate.sp12;0.911553421016705
species.regrowth.rate.sp13;0.911553421016705
species.regrowth.rate.sp14;0.911553421016705
```

- [ ] **Step 4: Wire it into the master config and the LTL file comment**

In `data/baltic/baltic_all-parameters.csv`, directly after the line
`osmose.configuration.plankton;baltic_param-ltl.csv`, add:

```
osmose.configuration.depletion;baltic_param-depletion.csv
```

In `data/baltic/baltic_param-ltl.csv`, append to the header comment block:

```
# NOTE: depletion (grazing feedback) is configured in baltic_param-depletion.csv; the NetCDF
# biomass below acts as carrying capacity K, not prescribed biomass, when depletion is enabled.
```

- [ ] **Step 5: Run the tests and the config-validation integration test**

Run: `.venv/bin/python -m pytest tests/test_baltic_depletion_config.py tests/test_engine_config_validation.py -v`
Expected: all PASS, and the validation test stays warning-free (keys are already allowlisted).

- [ ] **Step 6: Commit**

```bash
git add data/baltic/baltic_param-depletion.csv data/baltic/baltic_all-parameters.csv data/baltic/baltic_param-ltl.csv tests/test_baltic_depletion_config.py
git commit -m "feat(baltic): adopt depletable LTL in the production config (Phase 1, A/B-gated)"
```

---

### Task 5: Certification record and closeout

**Files:**
- Create: `docs/baltic_depletable_certification_2026-08-08.md` (certifier output; adjust date)
- Modify: `CLAUDE.md` (one gotcha line)

**Interfaces:**
- Consumes: the adopted config from Task 4; standard certifier CLI.

- [ ] **Step 1: Run the standard certification on the adopted config**

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --out docs/baltic_depletable_certification_2026-08-08.md`
Expected: ≈ 50 min. The report's ASSESSED verdict must read 5/5, and perch + stickleback must show `persists ✓ / in-envelope ✓` (identity gate re-confirmed on the committed config, not just on the A/B overlay).

- [ ] **Step 2: If the certification contradicts the A/B, stop**

The A/B (Task 3) and this run use the same code path, so disagreement means the config files do not reproduce `DEPLETION_KEYS` — re-check Task 4 rather than re-tuning anything.

- [ ] **Step 3: Add the CLAUDE.md gotcha**

Append to the Gotchas section of `CLAUDE.md`:

```
- **Depletable LTL is ON in `data/baltic`** (Phase 1, 2026-08-08): the CMEMS forcing acts as
  carrying capacity, not prescribed biomass. Python-only — `certify --java` pins
  `ltl.depletable.enabled=false` and labels it. Regrowth keys are `species.regrowth.rate.sp{9..14}`
  (the `ltl.regrowth.rate.rsc{i}` family is silently ignored on this config).
```

- [ ] **Step 4: Full suite + lint**

Run: `.venv/bin/python -m pytest` and `.venv/bin/ruff check osmose/ ui/ tests/ scripts/`
Expected: suite green (~4,400 tests), lint clean.

- [ ] **Step 5: Commit**

```bash
git add docs/baltic_depletable_certification_2026-08-08.md CLAUDE.md
git commit -m "docs(baltic): certify depletable-LTL production config; record gotcha"
```

---

## Self-review notes

- Spec coverage: Phase 1 items 1 (A/B first → Tasks 1+3), 2 (exact fitted starting keys → Tasks 1+4), 3 (documented bias → Task 4 Step 3 comment), 4 (contingency → Task 3 Step 2 stop rule), parity pinning (→ Task 2), loading-assertion test (→ Task 4), identity-pinned gate (→ Tasks 1, 3, 5). D3/F2 intentionally excluded (spec marks D3 optional; excluded here for YAGNI).
- The `baltic_a2` demo is 8-species; all indices here are the 9-species sp9–sp14 layout (Global Constraints warn the implementer).
- Type consistency: certify table shape used by `identity_gate`/`make_report` matches `certify_python`'s output dict exactly; `EXPECTED_RAW` in Task 4 equals `DEPLETION_KEYS` in Task 1 byte-for-byte.
