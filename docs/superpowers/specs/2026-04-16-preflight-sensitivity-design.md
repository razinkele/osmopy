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
class IssueCategory(enum.Enum):
    NEGLIGIBLE = "negligible"
    BLOWUP = "blowup"
    FLAT_OBJECTIVE = "flat_objective"
    BOUND_TIGHT = "bound_tight"

class IssueSeverity(enum.Enum):
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ParameterScreening:
    key: str                    # OSMOSE parameter key (matches FreeParameter.key)
    mu_star: float              # Morris mean absolute elementary effect (aggregated across objectives)
    sigma: float                # Morris std of effects (interaction indicator)
    mu_star_conf: float         # 95% confidence interval on mu_star
    influential: bool           # mu_star above threshold

@dataclass
class PreflightIssue:
    category: IssueCategory
    severity: IssueSeverity
    param_key: str | None       # Which parameter (None for objective-level issues)
    message: str                # Human-readable description
    suggestion: str             # Recommended fix
    auto_fixable: bool          # Can the modal offer a one-click fix?

@dataclass
class PreflightResult:
    screening: list[ParameterScreening]     # Morris results per parameter
    sobol: dict | None                      # SensitivityAnalyzer output (survivors only)
    issues: list[PreflightIssue]            # Diagnosed problems
    survivors: list[str]                    # FreeParameter.key values that passed screening
    elapsed_seconds: float
```

Dataclasses follow the project convention of enum-typed constrained fields (see `Transform` in `problem.py`) and validate in `__post_init__` (e.g., `mu_star >= 0`).

### 2. Issue Detection Rules

| Category | Condition | Severity | Suggestion | Auto-fixable |
|----------|-----------|----------|------------|--------------|
| `negligible` | `mu_star + mu_star_conf < 0.01 * max(mu_star)` — conservative: only flag when confident the parameter is truly negligible even accounting for Morris sampling noise | warning | Remove parameter from calibration | yes |
| `blowup` | Any evaluation (Morris OR Sobol stage) returned inf/NaN in the objective, OR engine crashed (returncode != 0). Tracked per-parameter: which parameter was perturbed in the trajectory step that caused the blowup | error | Tighten bounds on flagged parameter | yes (bound adjustment) |
| `flat_objective` | `sum(max(0, S1_i))` < 0.05 for an objective — negative S1 values (common with small n_base) clamped to 0 before summing. A responsive additive model has sum(S1) ≈ 1.0 | warning | Check objective function or target data | no |
| `bound_tight` | Sobol ST > 0.3 for a parameter AND Morris sigma/mu_star ratio > 1.5 — high sigma relative to mu_star indicates the parameter's effect varies dramatically across the space, correlating with boundary sensitivity. Avoids custom trajectory parsing (SALib only returns aggregated statistics) | warning | Widen bound by 20% | yes |
| `all_negligible` | All parameters classified as negligible after Morris — no survivors for Sobol stage | error | Review parameter selection and bounds; current parameters do not influence objectives | no |

**Failure abort thresholds:**
- Morris stage: if >30% of evaluations fail (crash/inf/NaN), screening aborts early. The higher threshold than post-calibration SA (which uses 10% in the existing handler) is justified because Morris uses fewer samples and a single bad trajectory disproportionately affects the failure rate.
- Sobol stage: if >10% of evaluations fail, Sobol aborts (consistent with existing `handle_sensitivity` handler threshold).

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
1. Generate Morris samples via `SALib.sample.morris` → store as `X_morris`
2. Evaluate all samples through `eval_fn` → `Y` array (n_samples, n_objectives)
3. **Multi-objective handling:** Run `SALib.analyze.morris(problem, X_morris, Y[:, col])` per objective (Morris is single-objective). Aggregate mu_star across objectives via `max(mu_star_per_objective)` — a parameter is influential if it matters to *any* objective.
4. Classify each parameter as influential (`aggregated_mu_star >= threshold * max(aggregated_mu_star)`) or negligible
5. Record any inf/NaN evaluations as blowup issues
6. If >30% of evaluations failed, abort with error issue — skip Sobol stage

**Note:** `SALib.analyze.morris` requires both `X` (sample matrix) and `Y` (output vector), unlike Sobol which only needs `Y`. The orchestrator must retain `X_morris` from step 1 for use in step 3.

**Stage 2 — Filter and targeted Sobol** (~sobol_n_base*(2*k_survivors+2) evaluations):
1. Build survivor list (influential parameters only)
2. If no survivors remain (all negligible), report error issue and skip Sobol
3. Use existing `SensitivityAnalyzer` on survivors only
4. Compute S1, ST indices
5. Detect flat objectives: any objective with total S1 < 0.05
6. Detect bound tightness: ST > 0.3 AND Morris sigma/mu_star ratio > 1.5

**Cancellation:** The handler thread checks a cancel flag between stages and between evaluation batches (every ~50 evaluations).

### 4. Evaluation Function Factory

```python
def make_preflight_eval_fn(
    config: dict[str, str],
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    work_dir: Path,
    sim_years: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
```

**How it works internally:**

1. Clones `config` dict, overrides `simulation.time.nyear` to `min(5, configured_years)`
2. For each call with parameter vector `x`:
   a. Maps `x[j]` → config overrides using `FreeParameter.key` and `Transform` (LOG/LINEAR)
   b. Applies overrides to the config dict clone
   c. Runs `PythonEngine().run(cfg, output_dir=work_dir / "preflight_eval", seed=0)`
   d. Opens `OsmoseResults(output_dir, strict=False)` and calls each `objective_fn(results)` — same signature as `OsmoseCalibrationProblem` objective functions (they receive an `OsmoseResults` object)
   e. Returns `np.array([obj_fn(results) for obj_fn in objective_fns])`
3. Reuses a single output directory per evaluation (overwritten each call) to avoid temp dir proliferation across ~1000+ evaluations

**Caller responsibility:** The `config` dict must be ready-to-use with `_osmose.config.dir` set and all referenced data files accessible (the UI handler already does this via `OsmoseConfigWriter` + `copy_data_files` before starting calibration). The factory does not write config files or copy data.

Key behaviors:

- **Shortened simulations:** Runs `min(5, configured_years)` to keep screening fast. Parameter influence rankings are stable across simulation lengths — a parameter that doesn't matter in 5 years won't dominate at 30.
- **Single seed:** No replicate averaging. Morris trajectory design handles noise. Fixed `seed=0`.
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
2. Handler captures reactive values (`state.config.get()`, `state.config_dir.get()`, `state.key_case_map.get()`) before spawning thread — same thread-safety pattern as existing calibration/sensitivity handlers
3. Spawns background thread, stores in `cal_thread.set(thread)` — Stop button works during pre-flight via existing `cancel_event`
4. Thread uses existing `CalibrationMessageQueue`:
   - `msg_queue.post_status("Pre-flight: Screening parameters (1/2)...")` for progress
   - New `msg_queue.post_preflight(result)` message type for the `PreflightResult`
5. The existing `_poll_cal_messages` reactive poll (0.5s interval) picks up messages:
   - `"status"` → updates status text as usual
   - `"preflight"` → stores result in `preflight_result` reactive value, then:
     - **No issues** → status flashes "Pre-flight passed", immediately starts calibration (calls the same optimization logic as the current `handle_start_cal`)
     - **Issues found** → calls `ui.modal_show()` from the reactive poll context (modals must be shown from main reactive context, not background threads)

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

Tests follow existing calibration test patterns: `pytest` with `@pytest.fixture`, class grouping, synthetic functions. No real engine runs — all eval functions are synthetic.

### Unit tests (`tests/test_calibration_preflight.py`)

**`class TestMorrisScreening`** — Morris stage in isolation:
1. **Ranking correctness** — Synthetic `y = 3*a + 0*b + 0.5*c` → `a` ranked highest, `b` flagged negligible, `c` influential
2. **Multi-objective aggregation** — 2 objectives: obj1 depends on `a` only, obj2 depends on `b` only → both `a` and `b` classified influential (max aggregation across objectives)

**`class TestIssueDetection`** — Issue diagnosis logic:
3. **Negligible** — Feed Morris output with one param below threshold (including confidence interval) → verify `IssueCategory.NEGLIGIBLE` issue created
4. **Blowup** — Eval function returns inf for certain samples → verify `IssueCategory.BLOWUP` issue, param_key identifies the perturbed parameter
5. **Flat objective** — All Sobol S1 values near zero for one objective → verify `IssueCategory.FLAT_OBJECTIVE` issue
6. **Bound tight** — Construct a case with ST > 0.3 and sigma/mu_star > 1.5 → verify `IssueCategory.BOUND_TIGHT` issue
7. **All negligible** — All params below threshold → verify `IssueCategory.ALL_NEGLIGIBLE` error, empty survivors, Sobol skipped

**`class TestRunPreflight`** — Orchestrator end-to-end:
8. **Happy path** — Synthetic eval function with clear signal, verify `PreflightResult` has correct survivors, issues list, and Sobol dict populated
9. **Failure abort** — >30% of Morris evals return inf → early abort, no Sobol stage, error issue present
10. **Single parameter** — Morris with 1 param still works (edge case: k=1)
11. **Cancellation** — Set cancel flag mid-Morris → verify partial result or clean abort

**`class TestMakePreflightEvalFn`** — Factory function:
12. **Sim years clamping** — Configured 30yr → factory uses 5yr; configured 3yr → factory uses 3yr
13. **Transform handling** — LOG transform applied correctly (same as `OsmoseCalibrationProblem._evaluate_candidate`)

### UI tests (`tests/test_ui_calibration_preflight.py`)

Following existing `test_ui_calibration_handlers.py` patterns (mock input, `reactive.isolate`):

14. **Checkbox state** — Screening enabled by default; unchecking skips pre-flight
15. **Modal rendering** — Mock `PreflightResult` with issues → verify modal HTML contains issue rows with checkboxes for auto-fixable issues, plain text for non-auto-fixable
16. **Happy path** — `PreflightResult` with no issues → verify calibration proceeds without modal
17. **Apply fixes: negligible removal** — Check negligible param issues → verify they are removed from `free_params` list
18. **Apply fixes: bound adjustment** — Check bound_tight issue → verify param bounds updated (widened by 20%)

## Dependencies

- `SALib` (already installed — used by `SensitivityAnalyzer`)
  - `SALib.sample.morris` and `SALib.analyze.morris` — new imports
  - `SALib.sample.sobol` and `SALib.analyze.sobol` — existing usage via `SensitivityAnalyzer`
- No new pip dependencies required
