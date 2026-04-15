# Calibration UI Phase 2 — Design

> Covers all 5 UI items deferred from Phase 1 (library gaps).
> Phase 1 spec: `docs/superpowers/specs/2026-04-15-calibration-library-gaps-design.md`

## Problem

The calibration library now has banded loss objectives, multi-seed validation, multi-objective
sensitivity analysis, and structured ICES targets — but none of these are exposed in the UI.
Additionally, calibration runs are ephemeral (lost on page reload) with no way to compare
past results.

## Scope

Five features, all within the existing Calibration page:

| Feature | Summary |
|---------|---------|
| Banded loss objective | Third objective option using `BiomassTarget` ranges |
| Multi-seed validation | Post-hoc button to re-evaluate top candidates across seeds |
| Calibration history | JSON persistence + browsable History tab |
| Parameter correlations | Parallel coordinates plot of Pareto candidates |
| Multi-objective sensitivity | Upgrade Sobol chart for 2D Y arrays |

## Architecture

### Tab Restructure

The calibration results panel moves from a flat 4-tab layout to a two-level grouped
navigation:

```
Calibration Page
├── Left Panel: Configuration (unchanged except banded loss addition)
└── Right Panel: Results
    ├── Run (group)
    │   └── Progress (convergence chart + status text)
    ├── Results (group)
    │   ├── Pareto Front (existing 2D scatter)
    │   ├── Correlations (NEW — parallel coordinates)
    │   ├── Best Parameters (existing table + NEW validate button)
    │   └── Sensitivity (UPGRADED — multi-objective)
    └── History (group)
        └── History Browser (NEW — list + detail view)
```

Implementation: Shiny `navset_tab` inside `navset_pill` (outer = groups, inner = sub-tabs).
The "Run" group has only one tab but maintains the grouping for consistency.

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ui/pages/calibration.py` | **modify** | Restructure tabs into 3 groups; add banded loss objective section; add validate button; add history browser UI |
| `ui/pages/calibration_handlers.py` | **modify** | Add banded loss objective builder; multi-seed validation handler; history save on completion; history load/delete handlers |
| `ui/pages/calibration_charts.py` | **modify** | Add `make_correlation_chart()`; upgrade `make_sensitivity_chart()` for 2D; add `make_history_list()` |
| `osmose/calibration/history.py` | **create** | `save_run()`, `load_run()`, `list_runs()`, `delete_run()` |
| `tests/test_calibration_history.py` | **create** | Tests for history module |

## Design

### 1. Banded Loss Objective

**Configuration panel change:** Add a third objective section below the existing biomass
and diet uploads.

```
Objectives
├── Biomass RMSE:      [Upload observed biomass CSV]
├── Diet Distance:     [Upload diet matrix CSV]
└── Banded Loss:       [Upload targets CSV] or [Use Baltic defaults ▼]
                       Stability weight: [5.0]
                       Worst-species weight: [0.5]
```

The banded loss section has:
- A file upload for a targets CSV (parsed by `load_targets()`)
- A dropdown with "Upload custom..." and "Baltic defaults" options. "Baltic defaults"
  loads `data/baltic/reference/biomass_targets.csv` automatically.
- Two numeric inputs for `w_stability` and `w_worst` (defaults: 5.0, 0.5)
- A checkbox to enable/disable this objective (unchecked by default)

**Handler change:** When banded loss is enabled, `handle_start_cal()` builds an additional
objective function using `make_banded_objective()`:

```python
from osmose.calibration.losses import make_banded_objective
from osmose.calibration.targets import load_targets

targets, _ = load_targets(targets_path)
banded_obj = make_banded_objective(
    targets, species_names,
    w_stability=w_stability, w_worst=w_worst,
)

def banded_objective_fn(results):
    stats = _extract_species_stats(results)  # helper: last-10yr mean/cv/trend
    return banded_obj(stats)
```

The `_extract_species_stats(results)` helper extracts `{species}_mean`, `{species}_cv`,
`{species}_trend` from `OsmoseResults.biomass()` — same logic as the Baltic script's
`run_simulation()` return value.

**Interaction with existing objectives:** All enabled objectives are passed to
`OsmoseCalibrationProblem` together. If both biomass RMSE and banded loss are enabled,
the problem has 2+ objectives (multi-objective optimization). The Pareto front chart
handles arbitrary objective counts already (it plots the first two dimensions).

### 2. Multi-Seed Validation

**UI:** A "Validate Top N" button appears in the Best Parameters tab after calibration
completes. Next to it: a numeric input for N (default 5) and a seed count input
(default 5 seeds).

```
Best Parameters
┌─────────────────────────────────────────────────────┐
│ [Top 10 candidates table]                            │
│                                                      │
│ ── Validation ──────────────────────────────────────│
│ Top N: [5]   Seeds: [5]   [▶ Validate]              │
│                                                      │
│ [Validation results table — appears after running]   │
│  Rank | Original Obj | Mean (5 seeds) | CV | Worst  │
│  1    | 0.342        | 0.358          | 4% | 0.401  │
│  ...                                                 │
└─────────────────────────────────────────────────────┘
```

**Handler:** When clicked, extracts the top N candidates from the Pareto front, calls
`rank_candidates_multiseed()` with a factory that rebuilds the objective for each seed.
Results displayed in a second table below the existing one.

The validation reuses the same engine path as the main calibration (Java subprocess via
`OsmoseCalibrationProblem._run_single`). The factory wraps the problem's evaluate method
with a different seed override:

```python
from osmose.calibration.multiseed import rank_candidates_multiseed

def make_objective_factory(seed):
    # Override the simulation seed, reuse the existing problem's _run_single
    def objective(x):
        overrides = _params_to_overrides(x, param_keys)
        overrides["simulation.random.seed"] = str(seed)
        obj_values = problem._run_single(overrides, run_id=0)
        return sum(obj_values)  # scalar for ranking
    return objective

result = rank_candidates_multiseed(
    make_objective_factory, top_n_candidates, seeds=seeds
)
```

The validation runs in a background thread (same pattern as calibration) with progress
messages. The existing `CalibrationMessageQueue` carries validation status.

### 3. Calibration History

**Storage format:** One JSON file per run in `data/calibration_history/`.

Filename: `{timestamp}_{algorithm}.json` (e.g., `2026-04-15T14-30-00_nsga2.json`).

```json
{
  "version": 1,
  "timestamp": "2026-04-15T14:30:00",
  "algorithm": "nsga2",
  "settings": {
    "population_size": 50,
    "generations": 100,
    "n_parallel": 4
  },
  "parameters": [
    {"key": "species.k.sp0", "lower": 0.01, "upper": 1.0}
  ],
  "objectives": {
    "biomass_rmse": true,
    "diet_distance": false,
    "banded_loss": {"enabled": true, "w_stability": 5.0, "w_worst": 0.5}
  },
  "results": {
    "best_objective": 0.342,
    "n_evaluations": 5000,
    "duration_seconds": 847,
    "convergence": [[0, 12.5], [1, 8.3], ...],
    "pareto_X": [[0.3, 100.0, ...], ...],
    "pareto_F": [[0.34, 0.12], ...]
  }
}
```

**Library module:** `osmose/calibration/history.py`

```python
HISTORY_DIR = Path("data/calibration_history")

def save_run(run_data: dict, history_dir: Path = HISTORY_DIR) -> Path:
    """Save a calibration run to JSON. Returns the path written."""

def load_run(path: Path) -> dict:
    """Load a single run from JSON."""

def list_runs(history_dir: Path = HISTORY_DIR) -> list[dict]:
    """List all runs, sorted by timestamp descending.
    Returns list of dicts with 'path', 'timestamp', 'algorithm',
    'best_objective', 'n_params', 'duration_seconds'."""

def delete_run(path: Path) -> None:
    """Delete a run file."""
```

**UI — History tab:**

```
History
┌─────────────────────────────────────────────────────┐
│ Calibration History (3 runs)                         │
│                                                      │
│ ┌─ 2026-04-15 14:30 ── NSGA-II ──────────────────┐ │
│ │ Best: 0.342 | 8 params | 14m 7s   [Load] [Delete]│ │
│ └────────────────────────────────────────────────────┘ │
│ ┌─ 2026-04-14 09:15 ── GP Surrogate ─────────────┐ │
│ │ Best: 0.518 | 4 params | 3m 22s   [Load] [Delete]│ │
│ └────────────────────────────────────────────────────┘ │
│                                                      │
│ [Load] restores results into the Results tabs        │
│ [Delete] removes the run file (with confirmation)    │
└─────────────────────────────────────────────────────┘
```

**Load behavior:** Clicking "Load" populates the Results tabs (Pareto front, Best Parameters,
Correlations) with the historical run's data. The configuration panel is NOT restored
(user may have changed settings since). A banner shows "Viewing historical run from
2026-04-15 14:30" in the Results group header.

**Auto-save:** Every completed calibration is automatically saved. No manual save button.

### 4. Parameter Correlations — Parallel Coordinates

**Chart:** `plotly.express.parallel_coordinates` with:
- One axis per calibrated parameter
- Lines colored by total objective value (continuous color scale, lower = better)
- Interactive range filtering on each axis (Plotly built-in)

```python
import plotly.express as px

def make_correlation_chart(X, F, param_names):
    """Parallel coordinates plot of Pareto candidates.

    Args:
        X: Parameter array (n_candidates, n_params)
        F: Objective array (n_candidates, n_objectives)
        param_names: Parameter labels
    """
    df = pd.DataFrame(X, columns=param_names)
    df["objective"] = F[:, 0] if F.shape[1] == 1 else np.sum(F, axis=1)
    fig = px.parallel_coordinates(
        df, color="objective",
        dimensions=param_names,
        color_continuous_scale="Viridis_r",  # lower = better = yellow
    )
    return fig
```

**Location:** Correlations sub-tab under the Results group. Only shows data after
calibration completes (empty state: "Run calibration to see parameter correlations").

### 5. Multi-Objective Sensitivity

**Upgrade path:** The existing `make_sensitivity_chart()` currently renders a single grouped
bar chart (S1 + ST for each parameter). With 2D Sobol output, it needs to show indices
per objective.

**Design:** When `analyze()` returns a 2D result (has `"objective_names"` key):
- Add a dropdown above the chart to select which objective to display
- Default: first objective
- Chart title updates to show selected objective name
- S1/ST bars update reactively on dropdown change

```python
def make_sensitivity_chart(result, selected_objective=0):
    """Sobol sensitivity bar chart.

    Handles both 1D (backward-compat) and 2D (multi-objective) results.
    For 2D, selected_objective indexes into the objectives axis.
    """
    if "objective_names" in result:
        s1 = result["S1"][selected_objective]
        st = result["ST"][selected_objective]
        title = f"Sensitivity — {result['objective_names'][selected_objective]}"
    else:
        s1 = result["S1"]
        st = result["ST"]
        title = "Sensitivity Analysis"
    # ... existing bar chart logic with s1, st, param_names
```

**Handler change:** The sensitivity handler already calls `analyzer.analyze(Y)`. When
the user has multiple objectives enabled, `Y` is a 2D array. The result is stored in a
reactive value. The dropdown selection triggers a re-render of the chart.

## Testing

| Test file | Covers |
|-----------|--------|
| `tests/test_calibration_history.py` | `save_run()`, `load_run()`, `list_runs()`, `delete_run()`, version field, sort order, missing dir creation |

UI-level tests (handler logic, chart rendering) are covered by manual testing on the
deployed Shiny server, consistent with the project's existing UI testing approach.

## Out of Scope

- Calibration run comparison (overlay two historical runs' Pareto fronts)
- Export calibration results to CSV/Excel
- Calibration parameter presets / templates
- Real-time Pareto front animation during calibration
- Parallel coordinates for historical runs comparison
