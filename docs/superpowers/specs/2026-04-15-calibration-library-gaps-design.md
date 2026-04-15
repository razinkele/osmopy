# Calibration Library Gaps — Phase 1 Design

> Phase 1 of 2. This phase covers the `osmose/calibration/` library module.
> Phase 2 (UI gaps) will be designed separately after Phase 1 ships.

## Problem

The calibration infrastructure has 6 library-level gaps identified during review:

1. The banded loss function (log-ratio with stability penalties) lives only in `scripts/calibrate_baltic.py` — not reusable
2. `SurrogateCalibrator` fits a GP but never validates it (no cross-validation or fit score)
3. No multi-seed validation utility — the Baltic script has one but it's not reusable
4. No evaluation caching — identical parameter sets re-run the full engine
5. `SensitivityAnalyzer` only handles a single objective (1D output)
6. Config overrides are validated by regex only, not against the schema registry
7. `BiomassTarget` dataclass and `load_targets()` live in the Baltic script, not the library
8. ICES reference point types (SSB, Fmsy, Blim, Bpa) have no structured representation

## Scope

Six files in `osmose/calibration/`:

| File | Status | Purpose |
|------|--------|---------|
| `targets.py` | **new** | `BiomassTarget` dataclass, `load_targets()`, ICES reference point types |
| `losses.py` | **new** | Composable banded loss objectives |
| `multiseed.py` | **new** | Multi-seed validation and candidate re-ranking |
| `problem.py` | **modify** | Evaluation cache, schema validation of overrides |
| `surrogate.py` | **modify** | Cross-validation, fit score |
| `sensitivity.py` | **modify** | Multi-objective support (2D Y array) |
| `__init__.py` | **modify** | Export new public API |

Plus updates to `data/baltic/reference/biomass_targets.csv` (add `reference_point_type` column).

## Design

### 1. `targets.py` — ICES Target Data Model

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BiomassTarget:
    species: str
    target: float              # Target tonnes
    lower: float               # Lower acceptable bound
    upper: float               # Upper acceptable bound
    weight: float = 1.0        # Importance weight (1.0=well-assessed, 0.2=uncertain)
    reference_point_type: str = "biomass"  # "biomass", "ssb", "fmsy", "blim", "bpa"
    source: str = ""           # ICES assessment reference
    notes: str = ""

def load_targets(path: Path) -> tuple[list[BiomassTarget], dict]:
    """Load targets from CSV. Skips # comment lines.
    Optional structured metadata lines prefixed with #! are parsed as YAML key-value pairs.
    
    Returns:
        (targets, metadata) where targets is list[BiomassTarget] and metadata is a dict
        of parsed #! lines (empty dict if none present).
        The reference_point_type column is optional (defaults to "biomass").
    """
```

**CSV format** — backward-compatible. Existing `biomass_targets.csv` works unchanged since `reference_point_type` defaults to `"biomass"`. The new column is optional. Structured metadata uses `#!` prefix lines (machine-readable YAML) vs plain `#` (human comments):

```csv
#! version: "1.0"
#! last_updated: "2026-04-15"
#! ices_advice_year: "2022-2023"
# Human-readable comment here
species,target_tonnes,lower_tonnes,upper_tonnes,weight,reference_point_type,source,notes
cod,120000,60000,250000,1.0,ssb,ICES SD24-32...,Post-2015 collapse...
```

### 2. `losses.py` — Composable Banded Loss Objectives

Three independent primitives plus a convenience factory:

```python
def banded_log_ratio_loss(sim_biomass: float, lower: float, upper: float) -> float:
    """Per-species loss: 0 inside [lower, upper], squared log-distance outside.
    
    Returns:
        0.0 if lower <= sim_biomass <= upper
        log10(lower / sim_biomass) ** 2 if sim_biomass < lower
        log10(sim_biomass / upper) ** 2 if sim_biomass > upper
        100.0 if sim_biomass <= 0
    """

def stability_penalty(
    cv: float, trend: float,
    cv_threshold: float = 0.2, trend_threshold: float = 0.05,
) -> float:
    """Penalty for oscillations (CV) and non-equilibrium (trend).
    Returns sum of (cv - threshold)^2 + (trend - threshold)^2
    for values exceeding their threshold; zero otherwise.
    """

def worst_species_penalty(species_errors: list[float]) -> float:
    """Max of weighted per-species errors. Returns max(species_errors)."""

def make_banded_objective(
    targets: list[BiomassTarget],
    species_names: list[str],
    w_stability: float = 5.0,
    w_worst: float = 0.5,
) -> Callable[[dict[str, float]], float]:
    """Factory: returns callable(species_stats) -> scalar objective.
    
    species_stats is a dict with keys '{species}_mean', '{species}_cv', '{species}_trend'.
    Composes: sum of weighted banded losses + stability penalties + worst-species term.
    
    This matches the objective structure in calibrate_baltic.py (lines 141-235)
    but is parameterized and reusable for any region/target set.
    """
```

### 3. `problem.py` — Evaluation Cache + Schema Validation

**Evaluation cache** — transparent, opt-in via constructor flag:

```python
class OsmoseCalibrationProblem(Problem):
    def __init__(
        self,
        ...,
        enable_cache: bool = False,
        cache_dir: Path | None = None,  # defaults to work_dir / ".cache"
        registry: ParameterRegistry | None = None,  # for schema validation
    ):
        # Cache key = sha256(sorted overrides items + jar_path.stat().st_mtime + base_config_hash)
        # base_config_hash computed once in __init__
```

New methods:

```python
    def _cache_key(self, overrides: dict[str, str]) -> str:
        """Deterministic hash of overrides + JAR mtime + base config hash."""

    def cache_stats(self) -> dict:
        """Returns {'hits': int, 'misses': int, 'size_mb': float}."""

    def clear_cache(self) -> None:
        """Remove all cached evaluations."""
```

Cache logic in `_run_single`:
1. Compute cache key from overrides
2. If hit: read cached JSON, return objective values, increment hits counter
3. If miss: run OSMOSE subprocess, write result JSON to cache dir, increment misses counter

JAR `st_mtime` in the hash ensures a recompiled JAR invalidates the cache automatically.

**Schema validation** — optional, gated on `registry` parameter:

```python
    def _validate_overrides(
        self, overrides: dict[str, str], registry: ParameterRegistry | None,
    ) -> None:
        """Validate override keys exist in registry and values within [min_val, max_val].
        Collects all violations and raises a single ValueError listing them all.
        Called in _run_single before subprocess when registry is not None.
        """
```

### 4. `surrogate.py` — Cross-Validation

```python
class SurrogateCalibrator:
    fit_score_: float | None = None  # LOO R², set during fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP models. Computes leave-one-out R² stored in fit_score_.
        Uses sklearn GP's built-in alpha for efficient LOO computation.
        """

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, k_folds: int = 5, seed: int = 42,
    ) -> dict:
        """K-fold cross-validation of the surrogate.
        
        Returns:
            {'fold_rmse': list[float], 'fold_r2': list[float],
             'mean_rmse': float, 'mean_r2': float,
             'std_rmse': float, 'std_r2': float}
        """
```

`fit_score_` follows scikit-learn convention (trailing underscore = attribute set during `fit()`).

### 5. `sensitivity.py` — Multi-Objective Support

```python
class SensitivityAnalyzer:
    def analyze(
        self, Y: np.ndarray, objective_names: list[str] | None = None,
    ) -> dict:
        """Compute Sobol indices for one or more objectives.
        
        Args:
            Y: 1D array (single objective, backward-compatible) or
               2D array of shape (n_samples, n_objectives).
            objective_names: Labels for each objective column.
                Defaults to ["obj_0", "obj_1", ...] if not provided.
        
        Returns:
            1D input (backward-compatible, same shape as today):
                {'S1': array(n_params,), 'ST': ..., 'S1_conf': ..., 'ST_conf': ...,
                 'param_names': list[str]}
            2D input:
                {'S1': array(n_obj, n_params), 'ST': array(n_obj, n_params),
                 'S1_conf': ..., 'ST_conf': ...,
                 'param_names': list[str], 'objective_names': list[str]}
        """
```

Implementation: 1D path unchanged. 2D path loops `sobol_analyze.analyze()` per column and stacks.

### 6. `multiseed.py` — Multi-Seed Validation

```python
def validate_multiseed(
    make_objective: Callable[[int], Callable[[np.ndarray], float]],
    x: np.ndarray,
    seeds: list[int] = (42, 123, 7, 999, 2024),
) -> dict:
    """Re-evaluate a candidate solution across multiple random seeds.
    
    Args:
        make_objective: Factory(seed) -> objective(x) -> float.
            Matches calibrate_baltic.py pattern where make_objective() accepts seed.
        x: Parameter vector to evaluate.
        seeds: Random seeds to test against.
    
    Returns:
        {'per_seed': list[float], 'mean': float, 'std': float,
         'cv': float, 'worst_seed': int, 'worst_value': float}
    """

def rank_candidates_multiseed(
    make_objective: Callable[[int], Callable[[np.ndarray], float]],
    candidates: np.ndarray,
    seeds: list[int] = (42, 123, 7, 999, 2024),
) -> dict:
    """Re-rank multiple candidates by mean objective across seeds.
    
    Args:
        candidates: Array of shape (n_candidates, n_params).
    
    Returns:
        {'rankings': list[int], 'scores': list[dict]}
        where each score dict has the same fields as validate_multiseed output.
    """
```

### 7. `__init__.py` — Updated Exports

New public API additions:

```python
from osmose.calibration.targets import BiomassTarget, load_targets
from osmose.calibration.losses import (
    banded_log_ratio_loss,
    stability_penalty,
    worst_species_penalty,
    make_banded_objective,
)
from osmose.calibration.multiseed import validate_multiseed, rank_candidates_multiseed
```

### 8. `biomass_targets.csv` — Updated Format

Add `reference_point_type` column and `#!` metadata lines:

```csv
#! version: "1.0"
#! last_updated: "2026-04-15"
#! ices_advice_year: "2022-2023"
# Baltic Sea equilibrium biomass targets for calibration
# Sources: ICES stock assessments (2018-2022 averages), FishBase, published literature
species,target_tonnes,lower_tonnes,upper_tonnes,weight,reference_point_type,source,notes
cod,120000,60000,250000,1.0,ssb,ICES SD24-32...,Post-2015 collapse...
herring,1500000,800000,3000000,1.0,biomass,ICES aggregate Baltic...,Central Baltic...
```

## Testing

Each new/modified module gets a corresponding test file:

| Test file | Covers |
|-----------|--------|
| `tests/test_calibration_targets.py` | `load_targets()`, BiomassTarget validation, backward-compat CSV, #! metadata |
| `tests/test_calibration_losses.py` | All 3 primitives + `make_banded_objective` with known species_stats |
| `tests/test_calibration_multiseed.py` | `validate_multiseed`, `rank_candidates_multiseed` with mock objectives |
| `tests/test_calibration_problem.py` | Extended: cache hit/miss/invalidation, schema validation errors |
| `tests/test_calibration_surrogate.py` | New: `cross_validate()` returns, `fit_score_` set after fit |
| `tests/test_calibration_sensitivity.py` | New: 2D Y array, backward-compat 1D path |

## Out of Scope (Phase 2 — UI)

- Banded loss / stability penalties exposed in calibration UI
- Multi-seed validation checkbox in UI
- Calibration history browser page
- Parameter correlation visualization on Pareto front
- Sensitivity chart updates for multi-objective display
