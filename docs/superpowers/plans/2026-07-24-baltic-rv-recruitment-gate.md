# Baltic RV Recruitment Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Enable and validate the reproductive-volume (RV) recruitment gate for Baltic cod — the population-level lever the SP-A branch's three negative experiments (params, finer grid, spawning maps) all pointed to — and measure whether gating cod recruitment on the RV series moves the committed 5/8 baseline toward long-horizon stability.

**Architecture:** The RV gate is **already implemented** on master (`osmose/engine/processes/recruitment_gate.py`, the `reproduction.py` hook, `_load_rv_gate` in `config.py`, RV egg-survival in `natural.py`). It is inert unless enabled. This plan supplies the missing pieces: (1) an RV time-series data file, (2) config wiring for cod, (3) validation on the Baltic 5/8 config via the reconciled stability certifier. No engine code is written — this is data + config + validation. If the gate mechanism itself needs a fix, that's out of scope (file a separate bug).

**Tech Stack:** Python 3.12; the existing RV-gate engine code; `scripts/baltic_stability_certify.py`; pytest; the Baltic config CSVs.

## Global Constraints

- Run tests with `.venv/bin/python -m pytest`; lint `.venv/bin/ruff check`.
- The gate MUST stay inert by default (bit-identical for every existing config unless `reproduction.rv.gate.enabled=true`) — do not change that invariant.
- RV series CSV contract (from `config.py:1124-1132`): columns `year` (contiguous ascending ints) and `spawning_rv` (finite, ≥0); non-empty.
- Gate applies to `n_eggs[sp] *= gate[sp]` for enabled species, **skipping seeded (bootstrap) steps** (`reproduction.py:156-163`) — already handled by the engine.
- Cod is sp0. Config keys: `reproduction.rv.gate.enabled`, `.series.file`, `.species.enabled.sp0`, `.mode` (`mean_preserving`|`raw_cap`), `.floor` [0,1], `.start.year`, `.ref` (raw_cap only).

---

### Task 1: RV time-series data file

**Files:**
- Create: `data/baltic/reference/baltic_cod_reproductive_volume.csv`
- Create: `data/baltic/reference/baltic_cod_reproductive_volume.README.md` (provenance)
- Test: `tests/test_baltic_rv_series.py`

**Interfaces:**
- Produces: a CSV loadable by `_load_rv_gate` — columns `year,spawning_rv`. `spawning_rv` is the eastern-Baltic-cod reproductive volume (km³ of deep-basin water with salinity ≥11 PSU and O₂ ≥2 ml/l), by year.

**Content decision:** for this PoC use a **literature-informed reconstruction** of the eastern Baltic cod RV (Plikshs et al. 1993 and the ICES WGBFAS reproductive-volume indicator): high and variable through the 1970s–80s (large MBIs), a decline through the 1990s, and low/near-collapse post-2000. Absolute units cancel under `mean_preserving`; only the *relative* interannual pattern matters. Document the values' source. A CMEMS-`o2b`+bottom-`so`-derived RV (needs a bottom-oxygen NetCDF, absent today) is the production refinement — note it in the README, do NOT block on it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baltic_rv_series.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("osmose").resolve().parent))
from osmose.engine.config import _load_rv_gate  # noqa: E402


def _cfg(tmp_series):
    return {
        "reproduction.rv.gate.enabled": "true",
        "reproduction.rv.gate.series.file": str(tmp_series),
        "reproduction.rv.gate.species.enabled.sp0": "true",
        "reproduction.rv.gate.mode": "mean_preserving",
    }


def test_baltic_rv_series_loads_and_is_valid():
    p = Path("data/baltic/reference/baltic_cod_reproductive_volume.csv")
    assert p.exists(), "RV series file missing"
    import pandas as pd
    df = pd.read_csv(p)
    assert list(df.columns[:2]) == ["year", "spawning_rv"]
    years = df["year"].to_numpy()
    assert np.array_equal(years, np.arange(years[0], years[0] + len(years)))  # contiguous
    rv = df["spawning_rv"].to_numpy(dtype=float)
    assert np.all(np.isfinite(rv)) and np.all(rv >= 0)
    assert rv[:5].mean() > rv[-5:].mean()  # documents the historical decline (high early, low late)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_rv_series.py -v`
Expected: FAIL (file missing).

- [ ] **Step 3: Create the RV series CSV**

Author `data/baltic/reference/baltic_cod_reproductive_volume.csv` with `year,spawning_rv` rows for a contiguous span (e.g. 1974–2020) encoding the literature pattern (high/variable early → declining → low late), and the `.README.md` citing Plikshs et al. (1993) and the ICES WGBFAS RV indicator, plus the salinity≥11/O₂≥2 definition and the CMEMS-refinement note.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_rv_series.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add data/baltic/reference/baltic_cod_reproductive_volume.csv data/baltic/reference/baltic_cod_reproductive_volume.README.md tests/test_baltic_rv_series.py
git commit -m "feat(baltic): eastern-cod reproductive-volume series for the RV recruitment gate"
```

---

### Task 2: Enable the RV gate for cod in the committed config

**Files:**
- Modify: `data/baltic/baltic_param-reproduction.csv` (append the RV-gate keys)
- Test: `tests/test_baltic_rv_gate_config.py`

**Interfaces:**
- Consumes: the RV series (Task 1), `_load_rv_gate`.
- Produces: the committed Baltic config loads the RV gate enabled for cod; still inert-identical if disabled.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baltic_rv_gate_config.py
from osmose.config import OsmoseConfigReader
from osmose.engine.config import _load_rv_gate


def test_committed_config_enables_rv_gate_for_cod_only():
    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    # signature: _load_rv_gate(cfg, n_species, n_dt_per_year, n_year)
    #   -> (factor_by_index (n_years,) | None, enabled_mask (n_species,) | None, offset)
    factor_by_index, enabled, _offset = _load_rv_gate(cfg, 8, 24, 40)
    assert enabled is not None and enabled[0] and not any(enabled[1:])  # cod (sp0) only
    assert factor_by_index is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_rv_gate_config.py -v`
Expected: FAIL (gate not enabled → `_load_rv_gate` returns the disabled tuple).

- [ ] **Step 3: Append the RV-gate keys**

Append to `data/baltic/baltic_param-reproduction.csv`:
```
# Reproductive-volume recruitment gate — eastern-Baltic-cod recruitment limiter
# (deep-basin salinity>=11 & O2>=2). mean_preserving: adds RV-driven interannual
# recruitment variability without shifting the long-run mean. See
# docs/superpowers/plans/2026-07-24-baltic-rv-recruitment-gate.md.
reproduction.rv.gate.enabled;true
reproduction.rv.gate.series.file;reference/baltic_cod_reproductive_volume.csv
reproduction.rv.gate.mode;mean_preserving
reproduction.rv.gate.floor;0.0
reproduction.rv.gate.species.enabled.sp0;true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_rv_gate_config.py -v` → PASS. Also run `tests/test_salinity_gate.py` and the engine config-validation test to confirm no validation regression.

- [ ] **Step 5: Commit**

```bash
git add data/baltic/baltic_param-reproduction.csv tests/test_baltic_rv_gate_config.py
git commit -m "feat(baltic): enable RV recruitment gate for cod (mean_preserving)"
```

---

### Task 3: Validate on the 5/8 baseline

**Files:**
- Run: `scripts/baltic_stability_certify.py`; a small comparison probe in the scratchpad.

**Interfaces:**
- Consumes: the RV-gated committed config (Tasks 1–2), the reconciled certifier.

- [ ] **Step 1: Effect probe — gated vs ungated, 40-yr, 3 seeds**

Run the committed config with the gate on vs off (override `reproduction.rv.gate.enabled`), compare per-species biomass and cod inter-annual CV. Expected: cod recruitment gains RV-driven interannual variability; the mean is ~preserved (mean_preserving), so cod's decade-mean stays near-in-range while its trajectory tracks the RV series.

- [ ] **Step 2: Re-certify long-horizon stability**

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --years 50`
Compare the persistent-&-in-envelope count to the pre-gate baseline (2/8). Record whether the RV gate moves cod (and the released trophic pressure, other species) toward stability, and note honestly if it does not — a null result here is a real finding that steers Phase 1 (per-substock RV on eastern cod, `raw_cap` mode to drive the eastern collapse).

- [ ] **Step 3: Document + commit**

Write the gated-vs-ungated + re-certification result into `docs/baltic_stability_certification_2026-07-01.md` (or a new dated note) and commit. This result is the go/no-go input for the cod E/W disaggregation Phase 1 (mode choice: `mean_preserving` for variability vs `raw_cap` for the eastern collapse).

---

## Self-Review

- **Spec coverage:** Task 1 = RV data; Task 2 = config wiring (enabled, cod-only, inert-default preserved); Task 3 = validation + the go/no-go for Phase 1. The gate code itself is pre-existing and tested (out of scope, per Architecture).
- **Placeholder scan:** Task 2 Step 1 explicitly says to read `_load_rv_gate`'s return signature before finalizing the unpack (the one genuine unknown) — flagged, not hidden. All config keys are exact (from `config.py:1116-1157`).
- **Type consistency:** `year,spawning_rv` CSV contract used consistently in Task 1 (creation) and Task 2 (consumption); the config-key names match `_load_rv_gate`.
- **Open risk:** `mean_preserving` preserves the mean, so it adds variability but won't by itself lower cod's mean — if Phase 1 needs the eastern collapse, `raw_cap` (recruitment cap) is the mode, decided by Task 3's result.
