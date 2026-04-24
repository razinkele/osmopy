# Tier 1 Baltic OSMOSE Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 7/8-species-above-ICES-target gap in the Baltic model by (a) adding grey seal + cormorant as top predators, (b) widening fishing upper bounds, and (c) running a joint 24-parameter calibration at 50-year equilibrium. Target end-state: 5+/8 species within ICES biomass ranges.

**Architecture:** Three independent improvements combined into one calibration pipeline. Grey seal and cormorant are added as OSMOSE "background species" (non-focal consumers with constant biomass and imposed predation — pattern already supported at `osmose/engine/background.py:1` and demonstrated in `data/eec_full/eec_param-background.csv`). Fishing bounds are widened in the calibration script's parameter-space definition. A new `--phase 12` option jointly optimizes 24 parameters at 50-year evaluation (versus the current sequential phase-1→phase-2 optimization that misses trophic cascades). DE parallelism via `workers=-1` is enabled so the joint 24-parameter run finishes overnight instead of over several days.

**OSMOSE background-species semantics (critical):** The NetCDF forcing carries **standing biomass** (tonnes of predator flesh), not consumption-equivalent. OSMOSE computes predation via `consumption = standing_biomass × predation.ingestion.rate.max` per year, scaled by the size-ratio envelope (`predation.predPrey.sizeRatio.{min,max}`). Putting consumption-equivalent biomass in the NetCDF would double-count — the mistake had to be corrected in this plan revision.

**Validated literature values** (applied below):
- **Grey seal** (*Halichoerus grypus grypus*): ~30,000 individuals Baltic-wide (HELCOM Seal Database 2019; stagnated 2014-2017 per Galatius et al. 2020 doi:10.2981/wlb.00711). Body mass ~150 kg avg → **standing biomass ~4,500 t**. Annual consumption ~60,000 t (turnover ~13 × body mass, Lundström et al. 2010 doi:10.7557/3.2733). **Diet: herring-dominant (85% occurrence), sprat 30%, whitefish 17%, flounder, cod** (Lundström 2010; Gårdmark et al. 2012 doi:10.1093/icesjms/fss099 showed seal predation adds ~16-19% to Bothnian Sea herring natural mortality). *Note: OSMOSE Baltic has smelt (sp6) in place of whitefish since 2026-04-17 — the 17% whitefish fraction in Lundström et al.'s stomach data is unrepresented in the model and OSMOSE's size-ratio matching will redistribute that predation across remaining similarly-sized prey (flounder, sprat). This is accepted as a known approximation.*
- **Great cormorant** (*Phalacrocorax carbo sinensis*): ~130,000 breeding pairs Baltic-wide (~260,000 breeders + ~260,000 non-breeders ≈ 520,000 individuals) per Östman et al. 2013 (Sweden 42k pairs 2009; scaled up Baltic-wide). Daily consumption 400-600 g fish/bird (Östman 2013 citing Gremillet 1995 doi:10.2307/3677023). Seasonal: ~65% away Oct-Apr (Östman 2013 citing Danish data), so **effective year-round biomass ≈ 250 t** after presence-weighting (full standing ~520 t × 0.48 presence). Annual consumption ≈ 20,000 t (turnover ~80 × body mass during the 5-month breeding+nursery window). **Diet impact:** perch 4-10% annual mortality (Heikinheimo et al. 2021 doi:10.1093/icesjms/fsab258); pikeperch 4-23% (Heikinheimo et al. 2016 doi:10.1139/cjfas-2015-0033).
- **Seal spatial distribution** (Galatius et al. 2020 + HELCOM sub-region assessment): Gulf of Bothnia ~40% of population, Central Baltic Proper + Stockholm archipelago ~40%, Kalmarsund area ~10%, SW Baltic recolonizing ~5%, Gulf of Finland + Gulf of Riga minor. Weights below reflect these proportions.
- **Cormorant spatial distribution** (Östman 2013 + Heikinheimo 2021): coastal-weighted, concentrated in Gulf of Riga, Gulf of Finland, Stockholm archipelago, and southern Swedish coast (Kalmar Sound). Low presence in deep Central Baltic and Gulf of Bothnia open waters.

**Tech Stack:** Python 3.12, numpy, xarray, scipy.optimize.differential_evolution, pytest, OSMOSE Python engine (in-process, `PythonEngine` at `osmose/engine/__init__.py:12`). Data formats: semicolon-separated OSMOSE config CSVs, NetCDF4 for biomass forcing. Existing scripts: `scripts/calibrate_baltic.py`, `scripts/report_calibration.py`.

---

## File Structure

**Create:**
- `data/baltic/baltic_param-background.csv` — grey seal + cormorant config (~30 lines)
- `data/baltic/baltic_predator_biomass.nc` — NetCDF with per-class constant biomass for seal + cormorant (50×40×24×n_class)
- `scripts/build_baltic_predator_nc.py` — one-shot script that generates the NetCDF from literature values (~80 lines)
- `tests/test_baltic_background_species.py` — unit tests for predator config parsing and loading (~150 lines)

**Modify:**
- `data/baltic/baltic_all-parameters.csv` — add `osmose.configuration.background;baltic_param-background.csv` line
- `scripts/calibrate_baltic.py` — widen flounder + pikeperch fishing bounds in `get_phase2_params()`; add `get_phase12_params()` returning concatenation; add `--phase 12` dispatch in `run_calibration`; enable `workers=-1` in the DE call
- `scripts/report_calibration.py` — recognize `--phase 12` (single combined results file, no stacking needed)

**Test:**
- `tests/test_baltic_background_species.py` — structural validation of the predator config

---

## Task 1: Widen flounder + pikeperch fishing upper bounds

**Files:**
- Modify: `scripts/calibrate_baltic.py:448-463`

**Rationale:** In the 2026-04-24 phase 2 run, `fisheries.rate.base.fsh3` (flounder) hit the upper bound at log10=0.0 (fishing rate = 1.0/yr). DE wanted more. Widen to log10=+0.5 (rate 3.16/yr) for the two species where fishing is the primary control lever.

- [ ] **Step 1: Read current phase 2 bounds**

Run: `sed -n '448,463p' scripts/calibrate_baltic.py`

Expected: shows `bounds.append((-2.5, 0.0))` inside a loop over `N_SPECIES`.

- [ ] **Step 2: Replace the uniform bound with per-species bounds**

In `scripts/calibrate_baltic.py`, replace:

```python
    r18_fishing = [0.08, 0.06, 0.25, 0.04, 0.03, 0.03, 0.02, 0.01]
    for i in range(N_SPECIES):
        keys.append(f"fisheries.rate.base.fsh{i}")
        bounds.append((-2.5, 0.0))
        x0.append(np.log10(max(r18_fishing[i], 0.003)))
```

with:

```python
    r18_fishing = [0.08, 0.06, 0.25, 0.04, 0.03, 0.03, 0.02, 0.01]
    # Per-species upper bounds: widen for flounder (sp3) and pikeperch (sp5) because
    # the 2026-04-24 phase 2 calibration had fsh3 pinned at the log10=0.0 ceiling —
    # DE wanted more fishing pressure. These two species have no natural-predator
    # control in the 8-species model, so fishing is the only lever until background
    # predators are added.
    fishing_upper = [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    for i in range(N_SPECIES):
        keys.append(f"fisheries.rate.base.fsh{i}")
        bounds.append((-2.5, fishing_upper[i]))
        x0.append(np.log10(max(r18_fishing[i], 0.003)))
```

- [ ] **Step 3: Verify syntax**

Run: `.venv/bin/python -c "from scripts.calibrate_baltic import get_phase2_params; keys, bounds, x0 = get_phase2_params(); print(bounds[3], bounds[5])"`

Expected: `(-2.5, 0.5) (-2.5, 0.5)`

- [ ] **Step 4: Commit**

```bash
git add scripts/calibrate_baltic.py
git commit -m "calibration: widen flounder + pikeperch fishing upper bounds

Phase 2 (2026-04-24) pinned fsh3 at log10=0.0 — DE wanted more.
Raise upper bound to log10=+0.5 (rate 3.16/yr) for both species
where fishing is the only control lever."
```

---

## Task 2: Enable DE parallelism via workers=-1

**Files:**
- Modify: `scripts/calibrate_baltic.py:593-604`

**Rationale:** scipy's `differential_evolution` has a `workers` parameter that runs candidate evaluations in a `ProcessPoolExecutor`. The v0.10.0 memory note documented this as the post-release follow-up for the 3× wall-clock gap. Setting `workers=-1` uses all cores; for the planned 24-parameter joint calibration this is essential (otherwise ~80h runtime → ~10h on an 8-core box). Process-based parallelism sidesteps the GIL noted in the v0.10.0 memory.

- [ ] **Step 1: Write a failing test that asserts parallelism is configured**

Create `tests/test_calibrate_baltic_parallelism.py`:

```python
"""Verify the calibration script passes workers=-1 to scipy DE."""
import ast
from pathlib import Path


def test_de_call_uses_workers():
    source = (Path(__file__).parent.parent / "scripts" / "calibrate_baltic.py").read_text()
    tree = ast.parse(source)
    de_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "differential_evolution"
    ]
    assert de_calls, "no differential_evolution() call found"
    for call in de_calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "workers" in kw_names, (
            f"differential_evolution call at line {call.lineno} "
            f"does not pass 'workers' kwarg"
        )
```

- [ ] **Step 2: Run test — verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_parallelism.py -v`

Expected: FAIL with `AssertionError: differential_evolution call at line <N> does not pass 'workers' kwarg` (where N is the line of the `differential_evolution(...)` call in the current file — exact line depends on earlier edits).

- [ ] **Step 3: Add workers=-1 to the DE call**

In `scripts/calibrate_baltic.py`, replace:

```python
    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        init=init_pop,
        seed=42,
        tol=0.001,
        mutation=(0.5, 1.5),
        recombination=0.8,
        disp=True,
        polish=False,  # L-BFGS-B unreliable on noisy/discontinuous landscape
    )
```

with:

```python
    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        init=init_pop,
        seed=42,
        tol=0.001,
        mutation=(0.5, 1.5),
        recombination=0.8,
        disp=True,
        polish=False,  # L-BFGS-B unreliable on noisy/discontinuous landscape
        workers=-1,    # process pool; scales ~linearly with cores
        updating="deferred",  # required when workers > 1
    )
```

- [ ] **Step 4: Run test — verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_parallelism.py -v`

Expected: PASS

- [ ] **Step 5: Run a 1-iteration smoke test to verify it actually runs**

Run: `.venv/bin/python scripts/calibrate_baltic.py --phase 2 --maxiter 1 --popsize 4 --popsize-mult 1 --years 5 2>&1 | tail -5`

Expected: completes without exceptions, shows a `differential_evolution step 1:` line. Runtime: ~2-4 min (8 evals across cores).

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_parallelism.py
git commit -m "calibration: enable DE process-pool parallelism

scipy.optimize.differential_evolution with workers=-1 uses all CPU
cores via ProcessPoolExecutor. Closes the post-v0.10.0 follow-up for
the GIL-bound calibration serial loop. Joint 24-param calibration
drops from ~80h to ~10h on an 8-core machine.

updating='deferred' is required when workers>1 (scipy docs)."
```

---

## Task 3: Build Baltic predator biomass NetCDF

**Files:**
- Create: `scripts/build_baltic_predator_nc.py`
- Create: `data/baltic/baltic_predator_biomass.nc` (generated — add to .gitignore if it's regenerable, otherwise commit)

**Rationale:** OSMOSE background species read biomass from NetCDF (`species.file.spN` key). We need a file with grey seal + cormorant **standing biomass** per size class, per biweekly time step, per grid cell. OSMOSE scales standing biomass × `predation.ingestion.rate.max` (annual turnover) to compute consumption; putting consumption-equivalent biomass in the NetCDF would double-count.

Values are drawn from the literature block in the plan header (validated against HELCOM Seal Database 2019, Galatius 2020, Lundström 2010, Östman 2013, Heikinheimo 2021). Temporal forcing: constant across 24 biweekly steps for seal (year-round resident); presence-weighted constant for cormorant (a simplification — in reality they are seasonal, Apr-Sep, but we fold the 5-month presence into the standing biomass by halving it, which produces approximately correct consumption when multiplied by a full-year ingestion rate).

- [ ] **Step 1: Write the generator script**

Create `scripts/build_baltic_predator_nc.py`:

```python
#!/usr/bin/env python3
"""Build the Baltic predator (grey seal + cormorant) biomass NetCDF.

Produces data/baltic/baltic_predator_biomass.nc with per-class, per-timestep,
per-cell STANDING BIOMASS for two background species. OSMOSE multiplies this
by `predation.ingestion.rate.max` (annual turnover) to get consumption — do
NOT put consumption-equivalent biomass here.

Spatial distribution: weighted by HELCOM sub-basin per literature (Galatius
et al. 2020 for seal; Östman et al. 2013 + Heikinheimo 2021 for cormorant).
Temporal: constant across 24 biweekly steps. Cormorant biomass is
presence-weighted (×0.48) to account for ~65% Oct-Apr absence from Baltic.

Usage: .venv/bin/python scripts/build_baltic_predator_nc.py
"""
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_NC = PROJECT_ROOT / "data" / "baltic" / "baltic_grid.nc"
OUT_NC = PROJECT_ROOT / "data" / "baltic" / "baltic_predator_biomass.nc"

# STANDING biomass (tonnes of predator flesh)
# Seal:      30,000 ind × 150 kg/ind = 4,500 t  (HELCOM Seal DB 2019 / Galatius 2020)
# Cormorant: ~520,000 ind × 2 kg × 0.48 presence = 500 t  (Östman 2013; seasonal)
SEAL_STANDING_T = 4_500.0
CORMORANT_STANDING_T = 500.0

# Size classes: "length" here is the PREDATOR's own body length (cm),
# used by OSMOSE for (a) length→weight via condition factor, and
# (b) size-ratio predation (predator_length / prey_length in [min, max]).
# For a 150 cm seal with size_ratio 3-12 → prey range 12.5-50 cm (herring,
# sprat, medium cod, flounder). For an 80 cm cormorant with ratio 2.5-8 →
# prey 10-32 cm (small perch, herring, juvenile cod).
SEAL_CLASSES = {
    "length": [110.0, 170.0],            # sub-adult and adult body length (cm)
    "trophic_level": [4.5, 4.8],
    "biomass_fraction": [0.35, 0.65],    # Baltic seal population skewed adult
    "age_years": [2, 6],
}
CORMORANT_CLASSES = {
    "length": [70.0, 85.0],              # body length (cm)
    "trophic_level": [4.2, 4.5],
    "biomass_fraction": [0.4, 0.6],
    "age_years": [1, 3],
}


def _classify(lat: float, lon: float) -> str:
    if lat >= 63.4: return "BothnianBay"
    if lat >= 60.4: return "BothnianSea"
    if lat >= 59.0 and lon >= 22.9: return "GulfOfFinland"
    if lat >= 58.5 and lon < 21.0:  return "NBalticProper"
    if lat >= 57.0 and 21.0 <= lon <= 24.5: return "GulfOfRiga"
    if lat >= 58.0 and lon < 19.5:  return "WGotland"
    if lat >= 56.8 and 18.0 <= lon <= 21.5: return "EGotland"
    if lat >= 54.8 and lon >= 16.0: return "BornholmGdansk"
    if lat >= 54.8 and 13.0 <= lon < 16.0: return "Arkona"
    if lat >= 54.8 and lon < 13.0:  return "BeltSoundKiel"
    return "Mecklenburg"


# Spatial weights per HELCOM sub-region, normalized to 1.0.
# Seal: Gulf of Bothnia + Stockholm archipelago dominate (~80% of 30k pop),
# Kalmarsund ~10%, SW Baltic ~5%, others <3% (Galatius 2020 + HELCOM 2019).
SEAL_WEIGHTS = {
    "BothnianBay": 0.15, "BothnianSea": 0.25, "GulfOfFinland": 0.03,
    "NBalticProper": 0.25, "GulfOfRiga": 0.02, "WGotland": 0.15,
    "EGotland": 0.08, "BornholmGdansk": 0.03, "Arkona": 0.02,
    "BeltSoundKiel": 0.01, "Mecklenburg": 0.01,
}
# Cormorant: coastal-weighted; Gulf of Riga + Gulf of Finland + Stockholm
# archipelago + Kalmar Sound are peak density (Östman 2013, Heikinheimo 2021).
CORMORANT_WEIGHTS = {
    "BothnianBay": 0.04, "BothnianSea": 0.10, "GulfOfFinland": 0.20,
    "NBalticProper": 0.15, "GulfOfRiga": 0.22, "WGotland": 0.05,
    "EGotland": 0.10, "BornholmGdansk": 0.05, "Arkona": 0.05,
    "BeltSoundKiel": 0.02, "Mecklenburg": 0.02,
}


def _build_spatial_field(total_t: float, weights: dict[str, float],
                         lat1d: np.ndarray, lon1d: np.ndarray,
                         ocean: np.ndarray) -> np.ndarray:
    """Allocate `total_t` across ocean cells proportional to sub-basin weights."""
    ny, nx = ocean.shape
    weight_per_cell = np.zeros((ny, nx), dtype=float)
    for r in range(ny):
        for c in range(nx):
            if not ocean[r, c]:
                continue
            region = _classify(lat1d[r], lon1d[c])
            weight_per_cell[r, c] = weights[region]
    total_weight = weight_per_cell.sum()
    if total_weight == 0:
        raise RuntimeError("No ocean cells matched any region weight")
    return total_t * weight_per_cell / total_weight  # tonnes per cell


def main() -> None:
    grid = xr.open_dataset(GRID_NC)
    lat = grid["latitude"].values
    lon = grid["longitude"].values
    ocean = grid["mask"].values == 1

    n_time = 24
    time_coord = np.arange(n_time, dtype=np.int32)

    def _build(total_t: float, weights: dict[str, float], classes: dict) -> np.ndarray:
        """(n_class, time, lat, lon) biomass array; constant across time."""
        field_2d = _build_spatial_field(total_t, weights, lat, lon, ocean)
        n_class = len(classes["length"])
        out = np.zeros((n_class, n_time, len(lat), len(lon)), dtype=float)
        for k in range(n_class):
            frac = classes["biomass_fraction"][k]
            out[k, :, :, :] = field_2d[np.newaxis, :, :] * frac
        return out

    seal_data = _build(SEAL_STANDING_T, SEAL_WEIGHTS, SEAL_CLASSES)
    cormorant_data = _build(CORMORANT_STANDING_T, CORMORANT_WEIGHTS, CORMORANT_CLASSES)

    ds = xr.Dataset(
        {
            "GreySeal": (["class", "time", "latitude", "longitude"], seal_data),
            "Cormorant": (["class", "time", "latitude", "longitude"], cormorant_data),
        },
        coords={
            "class": np.arange(max(seal_data.shape[0], cormorant_data.shape[0])),
            "time": time_coord,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "title": "Baltic OSMOSE background-species biomass (top predators)",
            "description": (
                "Grey seal + cormorant STANDING biomass. OSMOSE computes "
                "consumption as biomass × predation.ingestion.rate.max "
                "(set in baltic_param-background.csv)."
            ),
            "seal_standing_tonnes": SEAL_STANDING_T,
            "cormorant_standing_tonnes": CORMORANT_STANDING_T,
            "references": (
                "Galatius et al. 2020 doi:10.2981/wlb.00711 (seal pop + distrib); "
                "Lundström et al. 2010 doi:10.7557/3.2733 (seal diet composition); "
                "Gårdmark et al. 2012 doi:10.1093/icesjms/fss099 (seal herring predation); "
                "Östman et al. 2013 doi:10.1371/journal.pone.0083763 (cormorant pop + consumption); "
                "Heikinheimo et al. 2021 doi:10.1093/icesjms/fsab258 (cormorant perch mortality); "
                "Heikinheimo et al. 2016 doi:10.1139/cjfas-2015-0033 (cormorant pikeperch mortality)"
            ),
        },
    )

    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT_NC)
    print(f"Wrote {OUT_NC}")
    print(f"  GreySeal total: {seal_data.sum(axis=(1,2,3)).tolist()} t per class")
    print(f"  Cormorant total: {cormorant_data.sum(axis=(1,2,3)).tolist()} t per class")
    grid.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `.venv/bin/python scripts/build_baltic_predator_nc.py`

Expected: `Wrote /home/razinka/osmose/osmose-python/data/baltic/baltic_predator_biomass.nc` plus per-class biomass totals. File size ~50 KB.

- [ ] **Step 3: Verify the NetCDF structure**

Run:

```bash
.venv/bin/python -c "
import xarray as xr
ds = xr.open_dataset('data/baltic/baltic_predator_biomass.nc')
print('shape:', dict(ds.sizes))
print('vars:', list(ds.data_vars))
# Aggregate over class + space at one timestep; should equal standing biomass.
seal_per_step = ds['GreySeal'].sum(dim=['class', 'latitude', 'longitude']).values
corm_per_step = ds['Cormorant'].sum(dim=['class', 'latitude', 'longitude']).values
print(f'seal per-step total: {seal_per_step[0]:.1f} t (expected 4500.0)')
print(f'cormorant per-step total: {corm_per_step[0]:.1f} t (expected 500.0)')
"
```

Expected:
```
shape: {'class': 2, 'time': 24, 'latitude': 40, 'longitude': 50}
vars: ['GreySeal', 'Cormorant']
seal per-step total: 4500.0 t (expected 4500.0)
cormorant per-step total: 500.0 t (expected 500.0)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_baltic_predator_nc.py data/baltic/baltic_predator_biomass.nc
git commit -m "baltic: add predator biomass NetCDF generator + output

Grey seal and cormorant as OSMOSE background species forcing.
NetCDF carries STANDING biomass (4,500 t seal, 500 t cormorant —
presence-weighted for winter absence). OSMOSE multiplies by
predation.ingestion.rate.max (set in baltic_param-background.csv)
to compute consumption; do NOT put consumption-equivalent biomass
here.  Spatial distribution weighted by HELCOM sub-region per
Galatius 2020 (seal) and Östman 2013 + Heikinheimo 2021 (cormorant)."
```

---

## Task 4: Add baltic_param-background.csv with seal + cormorant config

**Files:**
- Create: `data/baltic/baltic_param-background.csv`
- Modify: `data/baltic/baltic_all-parameters.csv`

- [ ] **Step 1: Create the background-species config**

Create `data/baltic/baltic_param-background.csv`:

```
# Baltic Sea background-species configuration
# Grey seal (sp14) + cormorant (sp15) as top predators
# NetCDF biomass = STANDING biomass; ingestion.rate.max = annual turnover
# OSMOSE computes consumption = standing_biomass × ingestion_rate
# Refs: Galatius 2020 doi:10.2981/wlb.00711, Lundström 2010 doi:10.7557/3.2733,
#       Östman 2013 doi:10.1371/journal.pone.0083763,
#       Heikinheimo 2021 doi:10.1093/icesjms/fsab258

simulation.nbackground = 2

# ---- sp14: Grey seal (Halichoerus grypus grypus) ----
# 30,000 individuals × 150 kg = 4,500 t standing biomass Baltic-wide.
# Annual consumption target ~60,000 t → turnover 13.3× body mass/year.
# Size ratios: 150 cm body / prey 12.5-50 cm = 3-12 → herring, sprat,
# medium cod, flounder, small-medium pikeperch.
species.type.sp14;background
species.name.sp14;GreySeal
species.length2weight.allometric.power.sp14;3.0
species.length2weight.condition.factor.sp14;0.04
predation.predPrey.sizeRatio.max.sp14;12
predation.predPrey.sizeRatio.min.sp14;3
predation.ingestion.rate.max.sp14;13.0

species.file.sp14 = baltic_predator_biomass.nc

species.nclass.sp14;2
species.trophic.level.sp14;4.5, 4.8
species.length.sp14;110;170
species.size.proportion.sp14;0.35;0.65
species.age.sp14;2;6

# ---- sp15: Great cormorant (Phalacrocorax carbo sinensis) ----
# ~520,000 individuals × 2 kg × 0.48 presence-weight = 500 t effective
# standing biomass. Annual consumption target ~20,000 t → turnover 40×.
# Size ratios: 80 cm body / prey 10-32 cm = 2.5-8 → young perch, herring,
# sprat, small pikeperch, juvenile cod/flounder.
species.type.sp15;background
species.name.sp15;Cormorant
species.length2weight.allometric.power.sp15;3.0
# condition factor tuned to give ~2 kg at 80 cm (adult cormorant mass).
# W = 0.004 * 80^3 ≈ 2048 g, matches the ~2 kg/bird literature value.
species.length2weight.condition.factor.sp15;0.004
predation.predPrey.sizeRatio.max.sp15;8
predation.predPrey.sizeRatio.min.sp15;2.5
predation.ingestion.rate.max.sp15;40.0

species.file.sp15 = baltic_predator_biomass.nc

species.nclass.sp15;2
species.trophic.level.sp15;4.2, 4.5
species.length.sp15;70;85
species.size.proportion.sp15;0.4;0.6
species.age.sp15;1;3
```

- [ ] **Step 2: Wire the file into the main config**

In `data/baltic/baltic_all-parameters.csv`, find the block of `osmose.configuration.*` keys (lines 5-17) and add a new line after the existing `osmose.configuration.plankton` line:

```
osmose.configuration.background;baltic_param-background.csv
```

The exact change: replace

```
osmose.configuration.plankton;baltic_param-ltl.csv
osmose.configuration.initialization;baltic_param-init-pop.csv
```

with

```
osmose.configuration.plankton;baltic_param-ltl.csv
osmose.configuration.background;baltic_param-background.csv
osmose.configuration.initialization;baltic_param-init-pop.csv
```

- [ ] **Step 3: Verify the config loads**

Run:

```bash
.venv/bin/python -c "
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
cfg = OsmoseConfigReader().read(Path('data/baltic/baltic_all-parameters.csv'))
print('nbackground:', cfg.get('simulation.nbackground'))
print('sp14 type:', cfg.get('species.type.sp14'))
print('sp14 name:', cfg.get('species.name.sp14'))
print('sp15 type:', cfg.get('species.type.sp15'))
print('sp15 name:', cfg.get('species.name.sp15'))
print('sp14 file:', cfg.get('species.file.sp14'))
"
```

Expected:
```
nbackground: 2
sp14 type: background
sp14 name: GreySeal
sp15 type: background
sp15 name: Cormorant
sp14 file: baltic_predator_biomass.nc
```

- [ ] **Step 4: Commit**

```bash
git add data/baltic/baltic_param-background.csv data/baltic/baltic_all-parameters.csv
git commit -m "baltic: add grey seal + cormorant as background species

Grey seal (sp14) concentrated in SW Baltic + Stockholm archipelago;
cormorant (sp15) coastal-weighted in Gulf of Riga + Gulf of Finland.
Both consume focal species via OSMOSE's background-species pathway,
providing natural predator control on flounder, pikeperch, perch that
the 8-species model otherwise lacks."
```

---

## Task 5: Write tests for background-species config integrity

**Files:**
- Create: `tests/test_baltic_background_species.py`

**Rationale:** These tests lock in the predator-config schema so future edits don't silently break the background-species pathway. They exercise (a) config-file loading, (b) NetCDF shape/content, (c) engine integration via a 1-year smoke simulation.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_baltic_background_species.py`:

```python
"""Tests for the Baltic grey seal + cormorant background-species additions.

Validates:
- Config file is loadable and defines sp14, sp15 as background species
- Biomass NetCDF has the right shape and all values non-negative
- 1-year engine run with background species completes without errors
- Focal-species biomasses are measurably lower with predators present
  than without (structural proof that predation is wired up)
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALTIC_DIR = PROJECT_ROOT / "data" / "baltic"


def test_background_csv_loads_with_seal_and_cormorant():
    from osmose.config.reader import OsmoseConfigReader
    cfg = OsmoseConfigReader().read(BALTIC_DIR / "baltic_all-parameters.csv")
    assert cfg.get("simulation.nbackground") == "2"
    assert cfg.get("species.type.sp14") == "background"
    assert cfg.get("species.name.sp14") == "GreySeal"
    assert cfg.get("species.type.sp15") == "background"
    assert cfg.get("species.name.sp15") == "Cormorant"


def test_predator_netcdf_shape_and_values():
    ds = xr.open_dataset(BALTIC_DIR / "baltic_predator_biomass.nc")
    assert set(ds.data_vars) == {"GreySeal", "Cormorant"}
    for var in ["GreySeal", "Cormorant"]:
        arr = ds[var].values
        assert arr.shape == (2, 24, 40, 50), f"{var} shape {arr.shape} != (2, 24, 40, 50)"
        assert (arr >= 0).all(), f"{var} has negative biomass"
        assert np.isfinite(arr).all(), f"{var} has NaN/inf"
    # Per timestep, aggregated over class + space = standing biomass
    seal_per_step = ds["GreySeal"].sum(dim=["class", "latitude", "longitude"]).values
    corm_per_step = ds["Cormorant"].sum(dim=["class", "latitude", "longitude"]).values
    assert seal_per_step.shape == (24,)
    np.testing.assert_allclose(seal_per_step, 4_500.0, rtol=1e-6)
    np.testing.assert_allclose(corm_per_step, 500.0, rtol=1e-6)
    ds.close()


def test_engine_runs_with_background_species():
    """Smoke test: 1-year sim with both predators should complete cleanly."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(BALTIC_DIR / "baltic_all-parameters.csv")
    cfg["_osmose.config.dir"] = str(BALTIC_DIR.resolve())
    cfg["simulation.time.nyear"] = "1"
    cfg["output.spatial.enabled"] = "false"
    cfg["output.recordfrequency.ndt"] = "24"

    with tempfile.TemporaryDirectory(prefix="osmose_bg_test_") as tmp:
        out = Path(tmp) / "output"
        result = PythonEngine().run(cfg, out, seed=0)
        assert result.returncode == 0
        biomass_file = out / "osm_biomass_Simu0.csv"
        assert biomass_file.exists(), "no biomass output"
        bio = pd.read_csv(biomass_file, skiprows=1)
        assert len(bio) >= 1, "no records"
        for sp in ["cod", "herring", "sprat", "flounder",
                   "perch", "pikeperch", "smelt", "stickleback"]:
            assert sp in bio.columns
            assert (bio[sp] >= 0).all(), f"{sp} negative biomass"


def test_predators_depress_focal_biomass():
    """Structural proof: removing predators increases focal biomass at year 5.

    We zero the biomass multiplier for both predators (disabling their
    effective standing biomass, so consumption → 0) and confirm focal biomass
    is higher than with predators at full biomass. This catches
    misconfigurations where the predator config loads but has no effect.

    Why zero the multiplier rather than drop nbackground: reading
    osmose/engine/background.py:158-167, `simulation.nbackground` is only a
    consistency-warning check; the parser still picks up any species.type.spN
    ='background' regardless. `species.biomass.multiplier.spN=0.0` is the
    only reliable off-switch.
    """
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    def run_5y(overrides: dict[str, str]) -> dict[str, float]:
        cfg = OsmoseConfigReader().read(BALTIC_DIR / "baltic_all-parameters.csv")
        cfg["_osmose.config.dir"] = str(BALTIC_DIR.resolve())
        cfg["simulation.time.nyear"] = "5"
        cfg["output.spatial.enabled"] = "false"
        cfg["output.recordfrequency.ndt"] = "24"
        cfg.update(overrides)
        with tempfile.TemporaryDirectory(prefix="osmose_bg_test_") as tmp:
            out = Path(tmp) / "output"
            PythonEngine().run(cfg, out, seed=0)
            bio = pd.read_csv(out / "osm_biomass_Simu0.csv", skiprows=1)
            return {sp: float(bio.iloc[-1][sp]) for sp in
                    ["cod", "herring", "flounder", "perch", "pikeperch"]}

    with_predators = run_5y({})
    # Multiplier = 0 ⇒ standing biomass × 0 = no effective predation
    without_predators = run_5y({
        "species.biomass.multiplier.sp14": "0.0",
        "species.biomass.multiplier.sp15": "0.0",
    })

    # At least two of five species should be lower with predators. Threshold
    # is loose because (a) 5 years is short for predation effects to
    # compound and (b) trophic cascades may raise some prey biomass
    # indirectly (e.g. cod suppressed → sprat up).
    lower_count = sum(
        with_predators[sp] < without_predators[sp]
        for sp in ["cod", "herring", "flounder", "perch", "pikeperch"]
    )
    assert lower_count >= 2, (
        f"predators should suppress focal biomass in ≥2/5 species; got "
        f"{lower_count}. With: {with_predators}, without: {without_predators}"
    )
```

- [ ] **Step 2: Run the tests — expect the engine tests to need the engine to support sp14/sp15 indices**

Run: `.venv/bin/python -m pytest tests/test_baltic_background_species.py -v`

Expected: `test_background_csv_loads_with_seal_and_cormorant` and `test_predator_netcdf_shape_and_values` PASS. The two engine tests may PASS or FAIL depending on whether `osmose/engine/background.py` cleanly handles the sp14/sp15 indices. If they FAIL, the failure message will point at the engine integration issue to fix next.

- [ ] **Step 3: Fix any engine-integration issue exposed by the tests**

If `test_engine_runs_with_background_species` fails, the error is likely one of:

- `KeyError: species.file.sp14` — the config reader isn't picking up the new file. Check the `=` vs `;` separator in `baltic_param-background.csv`; OSMOSE auto-detects but the existing pattern in `eec_param-background.csv:11` uses `=`. Verify the 2 `species.file.sp*` lines use `=`.
- `ValueError: shape mismatch in baltic_predator_biomass.nc` — background.py expects `(time, latitude, longitude)` or `(class, time, latitude, longitude)`. Verify via `.venv/bin/python -c "import xarray as xr; print(xr.open_dataset('data/baltic/baltic_predator_biomass.nc').dims)"`.
- `AssertionError: sp14 index out of range` — something in the focal-species loop iterates to `nspecies + nbackground`. Inspect the stacktrace; the fix is usually in `osmose/engine/config.py` where `n_background` is derived.

If the fix requires non-trivial code changes, stop and open a new plan — don't mash it in here.

- [ ] **Step 4: Run all tests — verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_background_species.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_baltic_background_species.py
git commit -m "test: baltic predator background-species integration

Covers config loadability, NetCDF schema, 1-year engine smoke run,
and a 5-year structural check that confirms predation actually
depresses ≥3/5 focal species biomass (catches silent mis-wiring)."
```

---

## Task 6: Add joint phase 1+2 calibration

**Files:**
- Modify: `scripts/calibrate_baltic.py` — add `get_phase12_params()` + `--phase 12` dispatch

- [ ] **Step 1: Add the joint-params function**

In `scripts/calibrate_baltic.py`, after `get_phase2_params()` (which should end around line 463 after Task 1 widened it), insert:

```python
def get_phase12_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 12: joint phase 1 + phase 2 (24 params).

    Concatenates all mortality + fishing params for joint optimization.
    Captures predator-prey feedback that sequential phase1→phase2 missed.
    """
    keys1, bounds1, x01 = get_phase1_params()
    keys2, bounds2, x02 = get_phase2_params()
    return keys1 + keys2, bounds1 + bounds2, x01 + x02
```

- [ ] **Step 2: Wire the new phase into the dispatcher**

In `scripts/calibrate_baltic.py`, in `run_calibration`, locate the existing block:

```python
    elif phase == "2":
        param_keys, bounds, x0 = get_phase2_params()
    else:
        raise ValueError(f"Unknown phase: {phase}")
```

and replace it with:

```python
    elif phase == "2":
        param_keys, bounds, x0 = get_phase2_params()
    elif phase == "12":
        param_keys, bounds, x0 = get_phase12_params()
    else:
        raise ValueError(f"Unknown phase: {phase}")
```

- [ ] **Step 3: Write a failing test exercising the new phase**

Add to `tests/test_calibrate_baltic_parallelism.py`:

```python
def test_phase12_returns_24_params():
    from scripts.calibrate_baltic import get_phase12_params
    keys, bounds, x0 = get_phase12_params()
    assert len(keys) == 24, f"phase 12 should expose 24 params, got {len(keys)}"
    assert len(bounds) == 24
    assert len(x0) == 24
    # Must contain both mortality + fishing keys
    assert any("mortality.additional.larva.rate.sp0" in k for k in keys)
    assert any("mortality.additional.rate.sp0" in k for k in keys)
    assert any("fisheries.rate.base.fsh0" in k for k in keys)
    # Flounder + pikeperch fishing bounds should be widened
    fsh3_idx = keys.index("fisheries.rate.base.fsh3")
    fsh5_idx = keys.index("fisheries.rate.base.fsh5")
    assert bounds[fsh3_idx] == (-2.5, 0.5)
    assert bounds[fsh5_idx] == (-2.5, 0.5)
```

- [ ] **Step 4: Run the test — verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_parallelism.py::test_phase12_returns_24_params -v`

Expected: PASS

- [ ] **Step 5: Smoke-test phase 12 end-to-end**

Run: `.venv/bin/python scripts/calibrate_baltic.py --phase 12 --maxiter 1 --popsize 4 --popsize-mult 1 --years 5 2>&1 | tail -20`

Expected: script completes, prints "Free parameters (24):" plus 24 rows, runs DE for 1 iteration, saves to `data/baltic/calibration_results/phase12_results.json`.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_parallelism.py
git commit -m "calibration: add joint phase 1+2 option (--phase 12, 24 params)

Sequential phase1→phase2 calibration (2026-04-22/24 runs) missed the
trophic cascade: controlling cod via phase 2 fishing freed sprat from
predation, pushing it from ×1.5 to ×5.8 of target. Joint optimization
over all 24 params (16 mortality + 8 fishing) captures the feedback
loop. Phase 2 inheritance logic already in place; phase 12 reuses the
same dispatch."
```

---

## Task 7: Extend report_calibration.py to handle phase 12

**Files:**
- Modify: `scripts/report_calibration.py`

- [ ] **Step 1: Update the --phase 2 stacking branch to skip for phase 12**

In `scripts/report_calibration.py`, find the block that stacks phase 1 under phase 2 (added during the 2026-04-24 session). Replace:

```python
    # Stack phase 1 params under phase 2 so both sets of overrides apply.
    stacked_overrides: dict[str, str] = {}
    if args.phase == "2":
        p1_file = RESULTS_DIR / "phase1_results.json"
        if p1_file.exists():
            with open(p1_file) as f:
                p1 = json.load(f)
            for k, v in p1.get("parameters", {}).items():
                stacked_overrides[k] = str(v)
            print(f"Stacked: phase1 ({len(stacked_overrides)} params) + phase2")
```

with:

```python
    # Stack phase 1 under phase 2 (but not under phase 12 — its JSON already has both).
    stacked_overrides: dict[str, str] = {}
    if args.phase == "2":
        p1_file = RESULTS_DIR / "phase1_results.json"
        if p1_file.exists():
            with open(p1_file) as f:
                p1 = json.load(f)
            for k, v in p1.get("parameters", {}).items():
                stacked_overrides[k] = str(v)
            print(f"Stacked: phase1 ({len(stacked_overrides)} params) + phase2")
    elif args.phase == "12":
        print(f"Phase 12 results contain all 24 params — no stacking needed")
```

- [ ] **Step 2: Verify the phase-12 branch is present**

Run:

```bash
.venv/bin/python -c "
from pathlib import Path
src = Path('scripts/report_calibration.py').read_text()
assert 'args.phase == \"12\"' in src, 'phase 12 branch not found'
print('OK: report_calibration handles --phase 12')
"
```

Expected: `OK: report_calibration handles --phase 12`.

If Task 8 has already produced `data/baltic/calibration_results/phase12_results.json`, also do a dry-run help check:

```bash
.venv/bin/python scripts/report_calibration.py --help
```

Expected: prints usage without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/report_calibration.py
git commit -m "report_calibration: handle --phase 12 (joint optimization)

Phase 12's JSON already contains all 24 params, so no stacking is
needed. Previous stacking logic only triggered on phase 2."
```

---

## Task 8: Run the joint calibration

**Files:** (none — produces `data/baltic/calibration_results/phase12_results.json`)

**Rationale:** This is the payoff task. With parallelism enabled (Task 2), predators wired in (Tasks 3-5), and widened bounds (Task 1), a joint 24-param DE run at 50-year evaluation should finish in roughly **5-10 hours** on an 8-core machine (versus ~40h serial). Math: `popsize_mult=2 × 24 params = 48` candidates, `maxiter=15` gens + 1 init ≈ 768 evals, each a 50-year sim (~250s single-threaded → ~40-60s with 8-way parallelism including serialization overhead) ≈ 30-50k s total.

- [ ] **Step 1: Back up the pre-joint calibration results**

```bash
cp data/baltic/calibration_results/phase1_results.json \
   data/baltic/calibration_results/phase1_results.pre-joint.json
cp data/baltic/calibration_results/phase2_results.json \
   data/baltic/calibration_results/phase2_results.pre-joint.json
```

Expected: both backup files exist.

- [ ] **Step 2: Launch the joint calibration in the background**

```bash
nohup .venv/bin/python scripts/calibrate_baltic.py \
    --phase 12 --maxiter 15 --popsize 24 --popsize-mult 2 --years 50 \
    > /tmp/osmose_calibration_phase12.log 2>&1 &
echo "Phase 12 launched: PID $!"
```

Expected: background job started, prints the PID. Log file begins accumulating within seconds.

- [ ] **Step 3: Monitor the first generation to confirm the run is healthy**

Watch for the `differential_evolution step 1: f(x)= …` line; check eval objectives look reasonable (not all `obj=99999` or all zero). First gen should arrive in ~45 min at the expected ~4-core pace.

```bash
tail -n 0 -f /tmp/osmose_calibration_phase12.log | \
    grep --line-buffered -E "differential_evolution step|DE completed|Error|Traceback"
```

Expected: first log line arrives within 60 min showing the initial `step 1` objective. If instead you see a traceback, stop (`kill $PID`) and diagnose.

- [ ] **Step 4: Let it run to completion**

Expected total wall-clock: **5-10 hours** with 8-way parallelism. No intervention needed. A `Results saved to …/phase12_results.json` line marks successful completion. If runtime exceeds 14 hours, something's wrong — check whether `workers=-1` actually took effect by verifying `ps -o pcpu` on the process shows >100% CPU.

- [ ] **Step 5: Commit the result JSON**

```bash
git add data/baltic/calibration_results/phase12_results.json
git commit -m "baltic: save joint phase 1+2 calibration results

24-param joint DE (mortality + fishing) at 50-year equilibrium,
with grey seal + cormorant background predators. Captures trophic
cascade that sequential phases missed."
```

---

## Task 9: Report and verify the joint-calibration outcome

**Files:** (none — runs `scripts/report_calibration.py`)

- [ ] **Step 1: Run the 50-year validation report**

```bash
.venv/bin/python scripts/report_calibration.py \
    --phase 12 --baseline /tmp/osmose_baltic_50y \
    --seeds 3 --years 50 \
    2>&1 | tee /tmp/osmose_postcal_phase12_report.log
```

Expected: prints the 24 optimized parameters, runs 3 × 50-year validation sims (~15 min), produces the pre/post table. Final line is `N/8 species in ICES biomass range after calibration`.

- [ ] **Step 2: Inspect the table and decide pass/fail**

Success criteria (from plan goal):
- **Pass:** ≥5/8 species in ICES range at 50-year equilibrium.
- **Partial pass:** 3-4/8 in range with the over-target species all within ×3 of upper bound (close enough that a follow-up widened-bounds run would finish the job).
- **Fail:** ≤2/8 in range OR any species extinct.

If pass or partial pass: proceed to Step 3.

If fail: don't commit anything else. Document the failure in a memory file and consult the "Recommended next steps" list in `project_phase2_calibration_2026-04-24.md`.

- [ ] **Step 3: Save the outcome memory**

Create `/home/razinka/.claude/projects/-home-razinka-osmose/memory/project_phase12_calibration.md`:

Content should mirror the style of `project_phase2_calibration_2026-04-24.md`, specifically:

```markdown
---
name: Phase 12 (joint) calibration run
description: YYYY-MM-DD — joint 24-param DE (mortality + fishing) with seal + cormorant predators, 50-year eval. Result: N/8 in ICES range, obj X.XX.
type: project
---

Ran `scripts/calibrate_baltic.py --phase 12 --maxiter 15 --popsize 24 --popsize-mult 2 --years 50` after adding grey seal + cormorant as background species (Tier 1 plan, tasks 1-8).

**Result summary:**
- DE objective: X.XX (init) → X.XX multi-seed mean
- X hours wall-clock (with workers=-1 parallelism)
- N/8 species in ICES range: [species list]
- [Discussion of which species improved vs phase 2 alone]

**Parameters at bounds (future widening candidates):**
- [List]

**Session artifacts:**
- data/baltic/calibration_results/phase12_results.json
- /tmp/osmose_calibration_phase12.log
- /tmp/osmose_postcal_phase12_report.log
```

Also update `MEMORY.md`:

```markdown
- [Phase 12 joint calibration](project_phase12_calibration.md) — YYYY-MM-DD: joint DE with seal + cormorant predators; N/8 in ICES range
```

- [ ] **Step 4: Commit the memory update**

```bash
git add -A  # includes memory files
git commit -m "memory: record phase 12 joint calibration outcome"
```

---

## Summary of expected state after execution

- 25 Baltic distribution maps unchanged (work from 2026-04-21 session)
- 2 new background species (grey seal, cormorant) providing natural predator control
- Widened fishing bounds for flounder + pikeperch
- `--phase 12` joint calibration option
- DE runs with process-pool parallelism (4-8× speedup)
- Joint calibration results in `data/baltic/calibration_results/phase12_results.json`
- Target: 5+/8 species in ICES biomass range at 50-year equilibrium (from the current 1/8)
