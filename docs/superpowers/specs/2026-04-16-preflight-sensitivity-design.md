# Pre-flight Sensitivity Analysis — Design Spec

> Calibration pre-start procedure: two-stage sensitivity analysis (Morris screening + targeted Sobol) that validates parameter influence and problem well-posedness before committing to a long optimization run.

## Problem

Users can start a calibration with 20+ free parameters, poorly chosen bounds, or objectives that don't respond to the selected parameters. This wastes hours of compute before the user discovers the problem is mis-specified. There is no automated way to check whether:

- Selected parameters actually influence the objectives
- Bounds are reasonable (no blow-ups, not too tight)
- Objectives respond to the parameter space at all

The existing `SensitivityAnalyzer` (Sobol) is a post-calibration diagnostic only.

## Solution

A two-stage pre-flight that runs automatically when the user clicks "Start Calibration":

1. **Morris screening** — cheap qualitative ranking to eliminate negligible parameters
2. **Targeted Sobol** — quantitative diagnostics on surviving parameters to validate bounds and objective responsiveness

Results are presented in a blocking modal if issues are found. Happy path (no issues) proceeds directly to calibration.

## Scope

| Component | Files | Action |
|-----------|-------|--------|
| Library | `osmose/calibration/preflight.py` | **new** |
| Library | `osmose/calibration/__init__.py` | **modify** — export new API |
| UI | `ui/pages/calibration.py` | **modify** — checkbox, modal, reactive value |
| UI | `ui/pages/calibration_handlers.py` | **modify** — pre-flight thread, modal handlers, fix application |
| Tests | `tests/test_calibration_preflight.py` | **new** — 6 unit tests |
| Tests | `tests/test_ui_calibration_preflight.py` | **new** — 4 UI tests |

No changes to: `sensitivity.py` (reused as-is for stage 2), `problem.py`, `configure.py`, `multiphase.py`, engine code.

## Design

### 1. Data Model

```python
@dataclass
class ParameterScreening:
    key: str                    # OSMOSE parameter key
    mu_star: float              # Morris mean absolute elementary effect
    sigma: float                # Morris std of effects (interaction indicator)
    influential: bool           # mu_star above threshold

@dataclass
class PreflightIssue:
    category: str               # "negligible" | "blowup" | "flat_objective" | "bound_tight"
    severity: str               # "warning" | "error"
    param_key: str | None       # Which parameter (None for objective-level issues)
    message: str                # Human-readable description
    suggestion: str             # Recommended fix
    auto_fixable: bool          # Can the modal offer a one-click fix?

@dataclass
class PreflightResult:
    screening: list[ParameterScreening]     # Morris results per parameter
    sobol: dict | None                      # SensitivityAnalyzer output (survivors only)
    issues: list[PreflightIssue]            # Diagnosed problems
    survivors: list[str]                    # Parameter keys that passed screening
    elapsed_seconds: float
```

### 2. Issue Detection Rules

| Category | Condition | Severity | Suggestion | Auto-fixable |
|----------|-----------|----------|------------|--------------|
| `negligible` | mu_star < 0.01 * max(mu_star) | warning | Remove parameter from calibration | yes |
| `blowup` | Any Sobol evaluation returned inf/NaN | error | Tighten bounds on flagged parameter | yes (bound adjustment) |
| `flat_objective` | Total S1 across all params < 0.05 for an objective | warning | Check objective function or target data | no |
| `bound_tight` | ST > 0.3 for a parameter AND >50% of its high-effect Morris trajectories land in the outer 10% of the bound range | warning | Widen bound by 20% | yes |

Additional error-level abort: if >30% of Morris evaluations fail (crash/inf/NaN), screening aborts early with an error issue: "Too many evaluations failed — check configuration."

### 3. `run_preflight()` Orchestrator

```python
def run_preflight(
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    eval_fn: Callable[[np.ndarray], np.ndarray],
    objective_names: list[str] | None = None,
    morris_n: int = 10,
    sobol_n_base: int = 64,
    negligible_threshold: float = 0.01,
    on_progress: Callable[[str], None] | None = None,
) -> PreflightResult:
```

**Stage 1 — Morris screening** (~10*(k+1) evaluations):
1. Generate Morris samples via `SALib.sample.morris`
2. Evaluate all samples through `eval_fn`
3. Analyze with `SALib.analyze.morris` to get mu_star, sigma per parameter
4. Classify each parameter as influential (`mu_star >= threshold * max(mu_star)`) or negligible
5. Record any inf/NaN evaluations as blowup issues
6. If >30% of evaluations failed, abort with error issue — skip Sobol stage

**Stage 2 — Filter and targeted Sobol** (~sobol_n_base*(2*k_survivors+2) evaluations):
1. Build survivor list (influential parameters only)
2. If no survivors remain (all negligible), report error issue and skip Sobol
3. Use existing `SensitivityAnalyzer` on survivors only
4. Compute S1, ST indices
5. Detect flat objectives: any objective with total S1 < 0.05
6. Detect bound tightness: ST > 0.3 AND >50% of high-effect Morris trajectories in outer 10% of bound range

**Cancellation:** The handler thread checks a cancel flag between stages and between evaluation batches (every ~50 evaluations).

### 4. Evaluation Function Factory

```python
def make_preflight_eval_fn(
    config: dict[str, str],
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    sim_years: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
```

Key behaviors:

- **Shortened simulations:** Runs `min(5, configured_years)` to keep screening fast. Parameter influence rankings are stable across simulation lengths — a parameter that doesn't matter in 5 years won't dominate at 30.
- **Single seed:** No replicate averaging. Morris trajectory design handles noise.
- **Transform handling:** Respects `Transform.LOG` / `Transform.LINEAR` from `FreeParameter`, same as `OsmoseCalibrationProblem._evaluate_candidate`.
- **Python engine only:** Uses `PythonEngine` directly (proven parity: 14/14 EEC, 8/8 BoB). No Java subprocess overhead. This is an internal implementation detail — the user doesn't see or choose the engine for screening.
- **Failure handling:** Crashed or inf/NaN evaluations are recorded (for blowup detection) but don't abort individual evaluation. The >30% failure threshold triggers early abort at the stage level.

### 5. UI Integration

**Calibration setup panel changes** (`calibration.py`):

Checkbox below the Start/Stop button row:
```
[✓] Pre-flight screening (recommended)
[ Start Calibration ]  [ Stop ]
```

Default: checked. When unchecked, "Start Calibration" goes directly to the optimizer.

**Flow when screening is enabled:**

1. User clicks "Start Calibration"
2. Button disables, status shows "Pre-flight: Screening parameters (1/2)..."
3. Background thread runs `run_preflight()` using constructed eval function
4. Progress updates: "Pre-flight: Screening parameters (1/2)..." then "Pre-flight: Validating bounds (2/2)..."
5. On completion:
   - **No issues** → status flashes "Pre-flight passed", calibration starts immediately
   - **Issues found** → modal appears

**Issue modal:**

```
┌─────────────────────────────────────────────────┐
│  Pre-flight Screening Results                   │
│                                                 │
│  Screened 20 parameters in 18 min.              │
│  12 influential, 8 negligible.                  │
│                                                 │
│  Issues (3):                                    │
│                                                 │
│  [x] Remove mortality.additional.rate.sp3       │
│      (negligible, mu* = 0.002)                  │
│                                                 │
│  [x] Remove mortality.additional.rate.sp5       │
│      (negligible, mu* = 0.001)                  │
│                                                 │
│  [x] Adjust species.k.sp0 upper → 0.8          │
│      (objective blow-up near current bound 1.0) │
│                                                 │
│  [ Apply Selected & Start ]  [ Cancel ]         │
└─────────────────────────────────────────────────┘
```

- Auto-fixable issues have pre-checked checkboxes
- Non-auto-fixable issues (e.g., flat_objective) show as informational text with no checkbox
- "Apply Selected & Start" removes checked negligible params from `free_params`, adjusts checked bounds, then starts the optimizer
- "Cancel" returns to setup — nothing changed
- `PreflightResult` stored in a reactive value so the Sensitivity tab can display pre-flight Sobol results alongside post-calibration SA

### 6. Estimation of Run Times

For a typical 20-parameter problem:

| Stage | Formula | Evaluations | Time @ 3s/eval |
|-------|---------|-------------|----------------|
| Morris | 10 * (20 + 1) | 210 | ~10 min |
| Sobol (12 survivors) | 64 * (2*12 + 2) | 1,664 | ~83 min |
| **Total** | | **1,874** | **~93 min** |

For a smaller 10-parameter problem:

| Stage | Formula | Evaluations | Time @ 3s/eval |
|-------|---------|-------------|----------------|
| Morris | 10 * (10 + 1) | 110 | ~5 min |
| Sobol (7 survivors) | 64 * (2*7 + 2) | 1,024 | ~51 min |
| **Total** | | **1,134** | **~57 min** |

The shortened simulation (5 years vs full run) keeps per-evaluation cost at ~2-5 seconds. Users can reduce `sobol_n_base` for faster but less precise diagnostics.

## Testing

### Unit tests (`tests/test_calibration_preflight.py`)

1. **Morris ranking** — Synthetic function `y = 3*a + 0*b + 0.5*c` → `a` ranked highest, `b` flagged negligible, `c` influential
2. **Issue detection: negligible** — Feed Morris output with one param below threshold → verify `negligible` issue created
3. **Issue detection: blowup** — Eval function returns inf for certain samples → verify `blowup` issue
4. **Issue detection: flat objective** — All S1 values near zero → verify `flat_objective` issue
5. **`run_preflight()` end-to-end** — Synthetic eval function, verify `PreflightResult` has correct survivors, issues, and Sobol dict populated
6. **Edge cases** — 1 parameter (Morris still works); all negligible (error issue, empty survivors); >30% failure (early abort)

### UI tests (`tests/test_ui_calibration_preflight.py`)

7. **Checkbox state** — Screening enabled by default; unchecking skips pre-flight
8. **Modal rendering** — Mock `PreflightResult` with issues → verify modal contains issue rows with checkboxes
9. **Happy path** — `PreflightResult` with no issues → verify calibration starts without modal
10. **Apply fixes** — Check negligible params → verify they are removed from `free_params`; check bound adjustment → verify param bounds updated

## Dependencies

- `SALib` (already installed — used by `SensitivityAnalyzer`)
  - `SALib.sample.morris` and `SALib.analyze.morris` — new imports
  - `SALib.sample.sobol` and `SALib.analyze.sobol` — existing usage via `SensitivityAnalyzer`
- No new pip dependencies required
