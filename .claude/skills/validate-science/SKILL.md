---
name: validate-science
description: Run ecological plausibility checks on OSMOSE simulation outputs (biomass stability, trophic structure, size spectrum, diversity, extinction, mass conservation)
disable-model-invocation: true
---

Validate that OSMOSE simulation outputs are ecologically plausible. This goes beyond engine parity (Python vs Java) to check whether the modeled ecosystem behaves like a real marine ecosystem.

## Arguments

- `output_dir` (required): path to the simulation output directory
- `config` (optional): "bob" (Bay of Biscay), "eec" (EEC), or path to config. Used for context in the report
- `years` (optional): number of simulation years to analyze (default: all available)
- `checks` (optional): comma-separated list of checks to run (default: all). Options: `stability`, `trophic`, `spectrum`, `diversity`, `extinction`, `mortality`, `growth`, `diet`

## Ecological Checks

Run checks in order. For each, load data via `OsmoseResults`, compute the diagnostic, and compare against the ecological threshold. Report PASS/WARN/FAIL for each.

### 1. Biomass Stability (`stability`)

**Goal**: Biomass trajectories should stabilize, not crash to zero or explode.

```python
from osmose.results import OsmoseResults
results = OsmoseResults(Path(output_dir))
bio = results.biomass()
```

For each species, compute the coefficient of variation (CV) of biomass over the **last 50%** of the time series (skip spin-up):

- **PASS**: CV < 0.5 — biomass fluctuates within realistic bounds
- **WARN**: 0.5 <= CV < 1.0 — high variability, may indicate instability
- **FAIL**: CV >= 1.0 — chaotic or divergent dynamics

Also flag if any species has **final biomass < 1% of initial biomass** (population collapse).

### 2. Trophic Structure (`trophic`)

**Goal**: Mean trophic levels should follow expected marine ecosystem patterns.

```python
tl = results.mean_trophic_level()
```

Check per species:
- **PASS**: 2.0 <= mean TL <= 5.5 — realistic range for marine organisms
- **WARN**: TL outside [2.0, 5.5] — unusual but not impossible
- **FAIL**: TL < 1.0 or TL > 6.0 — biologically implausible

Check community-level:
- Mean TL of catch (if yield data exists) should be 2.5–4.5 for a balanced fishery. Use `analysis.mean_tl_catch()` if yield data is available.

### 3. Size Spectrum (`spectrum`)

**Goal**: Community size spectrum should approximate a power law with negative slope.

```python
from osmose.analysis import size_spectrum_slope
spectrum = results.size_spectrum()
slope, intercept, r2 = size_spectrum_slope(spectrum)
```

- **PASS**: -2.5 < slope < -1.0 and R² > 0.7 — consistent with marine theory (Sheldon spectrum)
- **WARN**: slope outside [-2.5, -1.0] or R² in [0.5, 0.7] — possibly unrealistic structure
- **FAIL**: slope > 0 (inverted) or R² < 0.5 — spectrum does not follow expected pattern

### 4. Diversity (`diversity`)

**Goal**: Species diversity should remain stable and non-zero.

```python
from osmose.analysis import shannon_diversity
bio = results.biomass()
shannon = shannon_diversity(bio)
```

- **PASS**: Final Shannon H' > 1.0 and H' decline < 30% from initial — healthy community
- **WARN**: H' in [0.5, 1.0] or decline 30–60% — reduced diversity
- **FAIL**: H' < 0.5 or decline > 60% — community collapse / monodominance

### 5. Extinction Check (`extinction`)

**Goal**: No focal species should go extinct unless expected.

For each species, check if biomass drops to zero at any point after spin-up:

- **PASS**: All focal species persist throughout simulation
- **WARN**: 1 species goes extinct — may be realistic or a model issue
- **FAIL**: 2+ species go extinct — likely a parameter or model problem

Report which species go extinct and at what timestep.

### 6. Mortality Rates (`mortality`)

**Goal**: Mortality rates should be within ecologically realistic bounds.

```python
mort = results.mortality_rate()
```

For each mortality source per species:
- Total annual mortality rate: **PASS** if 0.01 < Z < 5.0 (per year)
- Predation mortality: should be > 0 for non-apex species
- Starvation mortality: **WARN** if > 50% of total mortality (food web imbalance)
- Fishing mortality: **WARN** if F > natural mortality M for unfished species

### 7. Growth Validation (`growth`)

**Goal**: Mean size at age should follow Von Bertalanffy-like curves (non-decreasing).

```python
size_by_age = results.mean_size_by_age()
```

For each species:
- **PASS**: Mean size is non-decreasing with age
- **WARN**: Mean size decreases between 1 age-class pair (sampling noise)
- **FAIL**: Mean size decreases across multiple age classes (growth model error)

### 8. Diet Composition (`diet`)

**Goal**: Diet matrices should be non-trivial and reflect size-based feeding.

```python
diet = results.diet_matrix()
```

- **PASS**: Each predator eats >= 2 prey types, no single prey > 95% of diet
- **WARN**: A predator has > 80% of diet from one prey (specialist or model artifact)
- **FAIL**: A predator has 100% empty diet (starvation) or eats only 1 prey species

## Output Format

Print a structured report:

```
══════════════════════════════════════════════════════════════
  OSMOSE Scientific Validation Report
  Config: {config}  |  Output: {output_dir}
══════════════════════════════════════════════════════════════

── 1. Biomass Stability ──────────────────────────────────────
  Species             Final (t)    CV (last 50%)    Status
  Anchovy              1,234.5        0.21           PASS
  Hake                   892.1        0.73           WARN
  ...

── 2. Trophic Structure ──────────────────────────────────────
  Species             Mean TL      Status
  Anchovy               3.1        PASS
  ...
  Community mean TL of catch: 3.4  PASS

── 3. Size Spectrum ──────────────────────────────────────────
  Slope: -1.82  Intercept: 12.3  R²: 0.89  PASS

── 4. Diversity ──────────────────────────────────────────────
  Initial Shannon H': 1.92
  Final Shannon H':   1.78 (-7.3%)  PASS

── 5. Extinction Check ───────────────────────────────────────
  All 8 focal species persist  PASS

── 6. Mortality Rates ────────────────────────────────────────
  Species      Z (total)  Pred%  Starv%  Fish%  Status
  Anchovy        1.23     62%     5%     33%    PASS
  ...

── 7. Growth ─────────────────────────────────────────────────
  Species      Ages monotonic?   Status
  Anchovy      Yes               PASS
  ...

── 8. Diet Composition ───────────────────────────────────────
  Predator     N prey   Max prey %   Status
  Hake           5        42%        PASS
  ...

══════════════════════════════════════════════════════════════
  SUMMARY: 7 PASS  |  1 WARN  |  0 FAIL
══════════════════════════════════════════════════════════════
```

## Rules

- Always use `.venv/bin/python`, never system python
- Run from `/home/razinka/osmose/osmose-python/`
- Use `OsmoseResults` and `osmose.analysis` — do NOT parse CSV files manually
- If an output type is missing (e.g., no size spectrum data), skip that check with a NOTE, don't FAIL
- Spin-up period: skip the first 50% of timesteps for stability and diversity checks
- Resource species (background/LTL) are excluded from focal species checks
- Report all findings even if some checks FAIL — don't stop at first failure
- This skill is diagnostic only — do NOT modify engine code or parameters based on results
