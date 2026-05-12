# Calibration Progress Dashboard — Design

> Date: 2026-05-12
> Status: Design (revised after round-1 in-loop review)
> Backlog item: feature-improvements top-5 #3
> Author: brainstorming session, model claude-opus-4-7[1m]

## 1. Context

OSMOSE calibration runs (DE, CMA-ES, surrogate-DE, NSGA-II) currently surface progress in three disjoint places:

- **CLI runs** (`scripts/calibrate_baltic.py` → DE, plus the optimizer dispatchers for CMA-ES and surrogate-DE) write a `phase{N}_checkpoint.json` to `data/baltic/calibration_results/` every N generations *(DE only)* and emit progress lines to `logs/phase{N}_TIMESTAMP.log`. The Shiny UI does not watch either; the operator's workflow is `tail -f` the log.
- **In-UI NSGA-II runs** launched from the calibration page's `Start Calibration` button feed a live convergence chart via `_make_progress_callback` in `ui/pages/calibration_handlers.py`, but write nothing to disk and leave no history.
- **Authoritative ICES verdicts** are produced only post-run by `compare_outputs_to_ices()` in `osmose/validation/ices.py` (shipped on origin/master via PR #46) and surface in the Results tab.

The backlog memory `project_feature_improvements_backlog.md` describes this work as "live NSGA-II generation count + per-species ICES-range status. Replaces tail-the-log workflow."

## 2. Goals

1. Unified live view of the most-recently-active calibration run, regardless of which optimizer launched it or whether it was started from the UI or the CLI.
2. Per-species in-band status displayed mid-run, computed from data already produced by the optimizer (no extra simulation).
3. Completed runs persist into `data/calibration_history/` so the existing History tab on the calibration page starts populating automatically.

## 3. Non-goals

1. **No full multi-run view.** The dashboard binds to one active run at a time. Concurrent runs are *disclosed* via a small "N other live runs · [switch]" badge (see §7) but the main surface still focuses on a single run.
2. **No ETA computation.** `gens N / M` plus elapsed only — generations have non-uniform cost across optimizers, and a misleading ETA is worse than no ETA.
3. **No per-species mid-run trend sparklines.** Aggregate convergence chart only.
4. **No mid-run call to the authoritative ICES validator.** Proxy only; the post-run validator remains the source of truth and surfaces in the Results tab.
5. **No watchdog/inotify push.** 1 s polling is sufficient relative to per-generation cadence.
6. **No new Shiny page or sidebar entry.** Everything fits inside the existing Run tab of the calibration page.
7. **No collision handling for two same-phase same-optimizer concurrent runs.** Such a pair shares the same checkpoint filename; one overwrites the other. Documented as a known limitation in this case only.
8. **No new Python dependency.**
9. **No replacement of the existing `phase{N}_results.json` write.** `save_run()` is additive.
10. **No retroactive backfill of `data/calibration_history/` from old result files.**

## 4. Architecture

```
runners (DE / CMA-ES / surrogate-DE / NSGA-II)
   │
   │ per-generation: atomic-write checkpoint JSON
   ▼
<results_dir>/phase{N}_checkpoint.json          (results_dir is passed in;
   │                                            Baltic default lives in
   │ 1 s reactive.poll (mtime-derived scalar)   osmose/calibration/checkpoint.py)
   ▼
ui/pages/calibration_handlers.py
   │
   ▼
Run-tab widgets:
  • run header (optimizer · phase · gens N/M · elapsed · patience · live/stalled/idle)
  • "N other live runs · [switch]" disclosure badge (when applicable)
  • per-species ICES proxy table (✓ / ✗ / n/a — or banner if all-n/a)
  • existing convergence chart (unchanged)
  • collapsed current-best-parameters block

run completion (all 4 runners)
   │
   ▼
osmose.calibration.history.save_run({…})
   │
   ▼
data/calibration_history/{ts}_{algo}.json  →  History tab list
```

Two write paths, one read path. In-UI runs also write the checkpoint to disk rather than bypassing — so behaviour is uniform across both run sources and the History tab works without a second code path.

**In-scope dependency changes:** *two* objective modules must be extended so per-species residuals are accessible: `scripts/calibrate_baltic.py:_ObjectiveWrapper` (CLI path, used by DE/CMA-ES/surrogate-DE) and `osmose/calibration/losses.py:make_banded_objective` (UI path, used by NSGA-II). See §6.5 — without these changes the proxy table has no data source.

## 5. Data contract — `CalibrationCheckpoint`

New module `osmose/calibration/checkpoint.py`:

```python
@dataclass(frozen=True)
class CalibrationCheckpoint:
    optimizer: Literal["de", "cmaes", "surrogate-de", "nsga2"]
    phase: str                              # e.g. "12", "1g_pilot"
    generation: int                         # >= 0
    generation_budget: int | None           # None when the runner doesn't expose one
    best_fun: float                         # finite (not NaN/Inf)
    per_species_residuals: tuple[float, ...] | None   # None when banded-loss is disabled
    per_species_sim_biomass: tuple[float, ...] | None  # parallel: model-window
        # mean biomass for each species, captured at the same evaluation that
        # produced per_species_residuals. Enables the proxy table's magnitude
        # factor (UX1) — sim_biomass[i] / target_mean[i] = "X.YY× overshoot
        # / undershoot". None iff per_species_residuals is None.
    species_labels: tuple[str, ...] | None  # parallel to per_species_residuals
                                            # and per_species_sim_biomass
    best_x_log10: tuple[float, ...]         # parallel to param_keys
    best_parameters: dict[str, float]       # keys == set(param_keys); finite floats
    param_keys: tuple[str, ...]
    bounds_log10: dict[str, tuple[float, float]]   # keys == set(param_keys), [lo, hi] with lo <= hi
    gens_since_improvement: int             # 0 <= n <= generation
    elapsed_seconds: float                  # >= 0
    timestamp_iso: str                      # ISO-8601
    banded_targets: dict[str, tuple[float, float]] | None  # {species: (lo, hi)} or None
    proxy_source: Literal["banded_loss", "objective_disabled", "not_implemented"]
        # banded_loss     → per_species_residuals is populated, proxy renders ✓/✗
        # objective_disabled → banded-loss not used; per_species_residuals is None
        # not_implemented → losses.py returned no vector despite banded-loss configured (bug surface)

    def __post_init__(self) -> None:
        # length/range invariants — see "Invariants enforced" below for the full list
        ...

@dataclass(frozen=True)
class CheckpointReadResult:
    """Discriminated read result so the UI can distinguish transient partial
    writes from persistent corruption from "no active run"."""
    kind: Literal["ok", "no_run", "partial", "corrupt"]
    checkpoint: CalibrationCheckpoint | None
        # Populated only when kind == "ok". UI code MUST null-check before
        # dereferencing — `kind == "ok"` does not by itself imply non-None.
    error_summary: str | None  # populated for partial/corrupt; rendered in UI banner

    def __post_init__(self) -> None:
        # Two-sided invariant. The first branch rules out "ok with None
        # checkpoint" (the sentinel must be kind='no_run'); the second
        # branch rules out the symmetric error of attaching a checkpoint
        # to a non-ok result.
        if self.kind == "ok" and self.checkpoint is None:
            raise ValueError(
                "CheckpointReadResult(kind='ok') requires non-None checkpoint; "
                "use kind='no_run' for the empty sentinel"
            )
        if self.kind != "ok" and self.checkpoint is not None:
            raise ValueError(
                f"CheckpointReadResult(kind={self.kind!r}) must have "
                "checkpoint=None"
            )

def default_results_dir() -> Path:
    """Baltic default — Path(__file__).resolve().parent.parent.parent / 'data' /
    'baltic' / 'calibration_results'. Callers may pass a different directory to
    write_checkpoint / read_checkpoint to support non-Baltic configurations."""

def write_checkpoint(path: Path, ckpt: CalibrationCheckpoint) -> None:
    """Atomic write (tmp + rename). Raises (OSError, TypeError, ValueError) on
    failure; caller decides handling. Coerces numpy scalars/arrays to plain
    floats / tuples before serialising."""

MAX_CHECKPOINT_BYTES: Final = 1_048_576  # 1 MiB — a real checkpoint is ~10 KB

def read_checkpoint(path: Path) -> CheckpointReadResult:
    """Reads and validates a checkpoint file. Returns:
      kind='ok'      — file present, JSON valid, all invariants pass, checkpoint set
      kind='no_run'  — file does NOT exist (vanished between glob and read, or
                       cleanup deleted it); checkpoint is None, no banner shown
      kind='partial' — JSONDecodeError AND mtime within last 3 s (concurrent write
                       race against the writer); UI shows previous frame with
                       "(updating…)" badge — transient by construction
      kind='corrupt' — JSONDecodeError + older, OR file size > MAX_CHECKPOINT_BYTES,
                       OR invariant failure on a parsed dict, OR field-type
                       mismatch; UI shows red banner with error_summary.

    Size guard: read_checkpoint checks file size via stat() first and returns
    kind='corrupt' with error_summary='file exceeds MAX_CHECKPOINT_BYTES (size=N)'
    if size > MAX_CHECKPOINT_BYTES, BEFORE reading the file content. Prevents a
    runaway writer from OOM'ing the Shiny server.

    File-vanish handling: if path.stat() raises FileNotFoundError (race with a
    deletion / cleanup between glob and read) OR PermissionError, returns
    kind='no_run' — never raises into the caller. If path.stat() raises any
    other OSError, returns kind='corrupt' with error_summary='stat failed: %s'.

    Invariant-violation handling: when JSON parses but CalibrationCheckpoint
    __post_init__ raises ValueError on any of the 14 invariants in §5,
    read_checkpoint converts it to kind='corrupt' with
    error_summary=str(value_error). The function NEVER propagates the
    ValueError to the caller.

    Decode-error scope: the partial/corrupt branches catch
    (json.JSONDecodeError, UnicodeDecodeError, ValueError) — UnicodeDecodeError
    is raised by json.load on non-UTF-8 bytes (it is a ValueError subclass that
    is NOT a JSONDecodeError, so a narrower except-clause would miss it)."""

def is_live(path: Path, max_age_s: float = 60.0, now: float | None = None) -> bool:
    """True iff (now or time.time()) − max_age_s < mtime <= (now or time.time()).
    Strict on the lower bound, inclusive on the upper:
      - mtime exactly at (now − max_age_s) is NOT live (just rolled out)
      - mtime exactly at now IS live
    Future-mtime files (clock jump / NTP rewind) return False — a real run
    cannot have a future mtime. The §7 'liveness states' table renders this
    as '≤ 60 s = live' inclusively for display; the strict-less-than on the
    lower boundary makes the live → stalled transition deterministic at
    exactly 60 s instead of flickering."""

def probe_writable(results_dir: Path) -> None:
    """At runtime startup, attempt a temporary write to results_dir via
    tempfile.NamedTemporaryFile(dir=results_dir, delete=True). Raises OSError
    on failure with a message naming the resolved path. The temp file is
    auto-deleted on close — no sentinel leaks across Shiny restarts."""
```

### Invariants enforced in `__post_init__`

1. `generation >= 0`
2. `gens_since_improvement >= 0` and `<= generation`
3. `elapsed_seconds >= 0`
4. `best_fun` is a finite float (`math.isfinite(best_fun)`); reject NaN/Inf.
5. `set(best_parameters.keys()) == set(param_keys)`.
6. `set(bounds_log10.keys()) == set(param_keys)`.
7. `len(best_x_log10) == len(param_keys)`.
8. For every `(lo, hi)` in `bounds_log10.values()`: length is 2 and `lo <= hi`.
9. If `banded_targets is not None`: same `(lo, hi)` check on its values, AND `lo > 0`. The lower-bound positivity is required because the §8 magnitude factor uses `target_mean = sqrt(lo * hi)` as the band centre — `lo = 0` would make `target_mean = 0` and `sim_biomass / target_mean = inf`, which the proxy table cannot render. Real banded targets used today are all strictly positive (biomass in tonnes); a calibrator wanting to model "near-extinction acceptable" should encode it with a small positive `lo` (e.g. 0.01 t) rather than zero.
10. If `per_species_residuals is not None`: `species_labels is not None` AND `len(per_species_residuals) == len(species_labels)` AND every value is a finite float `>= 0`.
11. If `species_labels is not None` and `banded_targets is not None`: every species in `species_labels` has a corresponding entry in `banded_targets` (otherwise the proxy table cannot render the band).
12. `proxy_source == "banded_loss"` iff `per_species_residuals is not None`.
13. `per_species_sim_biomass is not None` iff `per_species_residuals is not None`. When both non-None: `len(per_species_sim_biomass) == len(species_labels)` AND every value is a finite float `>= 0`. A value of `0.0` means the model simulated species extinction at the evaluated parameters; the §8 proxy table renders these rows with an `extinct` badge (loss is 100.0 by the fast-path penalty at `scripts/calibrate_baltic.py:207`, so the row already classifies as `out_of_range`). Negative or non-finite values are rejected.
14. `phase` matches the regex `^[A-Za-z0-9][A-Za-z0-9_\-\.]*$` AND `1 <= len(phase) <= 64` AND does NOT contain `..` (the `..` substring is rejected explicitly, since the regex allows a literal `.`). The leading-char-must-be-alphanumeric rule blocks `.hidden`, `.`, and `.bashrc`-style names; the `..` check still blocks parent-traversal even with embedded literal dots. Null bytes are implicitly rejected because `\x00` is not in the character class. Real phase strings observed in `data/baltic/calibration_results/` all pass: `"12"`, `"1g_pilot"`, `"12.no-predators"`, `"12.predators_inert_bug"`, `"1g_final_diagnostic"`. The filename pattern in §6 generalises to `phase{S}_checkpoint.json` where `S` is the validated phase string — `{N}` was a notation convenience, not a constraint to integers.

`__post_init__` violations raise `ValueError`, so a corrupt file that decodes as valid JSON but fails an invariant is caught by `read_checkpoint` and surfaced as `kind='corrupt'`.

### Numeric coercion in `write_checkpoint`

DE and CMA-ES produce `numpy.float64` for `best_fun` and `numpy.ndarray` for parameter vectors. `write_checkpoint` coerces both at the boundary:
- scalars via `float(x)` (raises `TypeError` on non-numeric); rejects non-finite values
- arrays/tuples via `tuple(float(v) for v in arr)`
- dicts of arrays via the same map
- `json.dump(..., allow_nan=False)` so any residual NaN is loud, not silent

### Checkpoint-directory resolution

`scripts/calibrate_baltic.py:34-37` already defines `PROJECT_ROOT = Path(__file__).resolve().parent.parent` and `RESULTS_DIR = PROJECT_ROOT / "data" / "baltic" / "calibration_results"`. The new module exposes `default_results_dir()` returning the same Path so both runner and UI code share one source of truth. Callers may pass a different directory.

The UI **must not** use a cwd-relative `Path("data/...")` — Shiny's working directory at runtime depends on how the server was launched.

## 6. Runner integration

| Runner | File | Change |
|---|---|---|
| DE | `scripts/calibrate_baltic.py`, inside `_make_checkpoint_callback` | Replace the inline `snapshot = {...}` dict with `CalibrationCheckpoint(...)` plus `write_checkpoint(path, ckpt)`. **At checkpoint cadence, re-evaluate `best_x` in the main process** by calling `evaluator(_decode(best_x))` once, then read `evaluator.last_per_species_residuals` (see §6.5.1). Stamp `banded_targets` from `evaluator.target_dict` once at run-start into a closure-captured snapshot the callback reuses. |
| CMA-ES | `osmose/calibration/cmaes_runner.py` | New per-generation callback hook (the existing `tell()` boundary) added at the optimizer wrapper level. The wrapper is objective-agnostic; the callback receives the evaluator (Path-A `_ObjectiveWrapper` for CLI invocations) and follows the same main-thread re-evaluation pattern as DE. |
| surrogate-DE | `osmose/calibration/surrogate_de.py` | Same per-generation callback hook in the outer DE loop, same main-thread re-evaluation pattern. Inner GP-surrogate iterations do NOT trigger checkpoint writes — only the outer real-evaluation generations do. |
| NSGA-II | `ui/pages/calibration_handlers.py`, in the `"results"` branch of `_poll_cal_messages` at lines 451-454 (after both `cal_X.set(X)` and `cal_F.set(F)` calls) | Additionally call `write_checkpoint(...)`. **No re-evaluation needed** — pymoo runs in the main thread, so the Path-B `banded_residuals()` accessor (per §6.5.2) returns fresh data from the most-recent objective call directly. The existing in-memory chart-feed path stays untouched — the new write is additive. |

**Checkpoint-file collision (same phase, same optimizer).** One file per phase per optimizer means concurrent same-phase same-optimizer runs would overwrite. Concurrent runs of *different* phase OR different optimizer are file-disjoint and handled correctly. Documented limitation; see §3 non-goal #7.

**Exception handling at write time.** Wrap the write in `try/except (OSError, TypeError, ValueError) as e`:
- Log a warning at WARNING level with `e.__class__.__name__`, the resolved path, and the offending field if discoverable.
- On the *first* failure within a process, emit a single Shiny `ui.notification_show(..., type="warning", duration=None)` so the user sees "Dashboard persistence failed — check logs" once, not repeated noise. (Runner-side code uses a tiny module-level flag.)
- Subsequent failures only log.
- Calibration continues — persistence failure must not fail the run.

**Startup probe-write.** Each runner calls `probe_writable(results_dir)` once at start. If it raises, log an explicit error naming the path, emit a Shiny notification, and continue without checkpoint writes for the run (the convergence chart still works for in-UI runs via the in-memory path; see §7 "in-UI fallback"). This is a startup check only; mid-session FS failures (mount unmount, EROFS) surface via the `try/except` around `write_checkpoint` (above). A mid-session re-probe is explicitly out of scope; documented in §11.

**Logging conventions.** Every new module and runner uses `logging.getLogger("osmose.calibration.<module>")` (e.g. `osmose.calibration.checkpoint`, `osmose.calibration.history`, `osmose.ui.calibration_dashboard`). Failure paths log at WARNING (transient, expected — disk full, vanished file) or ERROR (structural — `RESULTS_DIR` unreachable, invariant violation). No `print()` — the calibration runners already use the logging module via `osmose/logging.py:setup_logging`, this PR follows that pattern. Each log line includes the offending field name (`field=path`, `gen=42`) so log searches can pinpoint the failing tick.

### 6.5 Objective-function changes (in scope) — two parallel paths

There are **two banded-loss implementations** in the repo today (confirmed by grep). The proxy table needs per-species residuals from both:

**Path A — CLI runs (`scripts/calibrate_baltic.py:_ObjectiveWrapper`, class starts around line 143).** Used by DE and, via dispatch, by CMA-ES and surrogate-DE. The constructor at `__init__` (around line 150) accepts `targets`, `species_names`, …, `seed: int = 42`, and stores them on `self`. The callable is picklable so scipy/joblib can spawn workers. Per-species `weighted_error` is computed inside `__call__` (around line 216) and discarded.

**Path B — UI NSGA-II (`osmose/calibration/losses.py:44-87` `make_banded_objective`).** Single call site at `ui/pages/calibration_handlers.py:818, 839`. Signature today is `make_banded_objective(targets: list[BiomassTarget], species_names: list[str], w_stability: float = 5.0, w_worst: float = 0.5) -> Callable[[dict[str, float]], float]`. Per-species `weighted_errors` is built at `losses.py:61` (empty list initialised inside the nested `objective`) and the value is `appended` at `losses.py:78`; the local list is discarded when `objective` returns the scalar `total_error`.

#### 6.5.1 Path A change — `_ObjectiveWrapper`

Add an attribute `last_per_species_residuals: list[tuple[str, float, float]] | None` to the evaluator. Each entry is `(species_name, weighted_error, sim_biomass)`. Inside `__call__`, after the existing `weighted_error` is computed per species (line 216), append `(sp, weighted_error, stats.get(mean_key, 0.0))` to a **local** list and assign to `self.last_per_species_residuals` **as the last statement before return**. Pre-existing fast paths (mean_key not in stats / target_dict, sim_biomass <= 0) record a residual of 100.0 to match the existing scalar contribution and `sim_biomass = 0.0` (so the proxy table flags the row as extinct per §5 invariant 14). The sim_biomass value enables the §8 magnitude factor `sim_biomass / target_mean` — the most actionable signal per operator UX research (per the project memory, Baltic phase 12 perch/pikeperch overshoots reach ×100-450× and the raw loss alone cannot distinguish a 1.1× overshoot from a catastrophic one).

**Load-bearing invariant: assign-at-end.** Building the residual list locally and assigning to `self.last_per_species_residuals` *only* on the success path of `__call__` is what makes the §6.5.1 "clear-before, fail-safe" pattern work. A future refactor that initialises `self.last_per_species_residuals = []` up-front and mutates it in place would break the contract — a mid-call raise would leave the attribute holding a partially-populated list paired with whatever `best_fun` the runner just received from the worker. Test `test_residuals_attribute_unset_when_call_raises_midway` pins this.

**Multiprocessing caveat.** scipy DE with `workers > 1` evaluates the objective in worker processes via joblib/multiprocessing; `self.last_per_species_residuals` set in a worker is invisible to the main process. The DE/CMA-ES/surrogate-DE checkpoint callbacks therefore **re-evaluate `best_x` in the main process** at every checkpoint write.

**Determinism mechanism — engine-driven, not legacy-numpy.** `osmose/engine/rng.py:build_rng(seed, n_species, fixed=True)` constructs per-species PCG64 generators from a `SeedSequence(seed)` on every engine run. `_ObjectiveWrapper.__call__` passes `self.seed` (constant for the life of the wrapper) to `run_simulation`, which feeds it to `build_rng`. As long as the main-process re-evaluation calls the wrapper without mutating `self.seed`, the engine deterministically rebuilds the same per-species PCG64 state the worker built and produces bit-identical biomass means. No `np.random.seed/get_state/set_state` is needed — those touch only the legacy MT19937 module-level state, which `osmose/engine/rng.py` documents the engine does not use (per its module docstring at lines 4-16 and CLAUDE.md's RNG note).

```python
# Inside _make_checkpoint_callback at checkpoint write time
if every_n > 0 and state["gen"] % every_n == 0:
    # Clear the residuals slot BEFORE the re-eval. If the eval raises, we
    # never write a checkpoint pairing fresh best_fun with stale residuals.
    evaluator.last_per_species_residuals = None
    proxy_source: Literal["banded_loss", "objective_disabled", "not_implemented"]
    residuals: list[tuple[str, float]] | None
    try:
        _ = evaluator(np.power(10.0, best_x_log10))   # main-thread eval
        residuals = evaluator.last_per_species_residuals
        if residuals is None:
            # Re-eval succeeded but residuals weren't populated. This is a
            # bug surface in _ObjectiveWrapper.__call__ — log + surface.
            proxy_source = "not_implemented"
        else:
            proxy_source = "banded_loss"
    except Exception as e:  # noqa: BLE001 — bounded log+continue, see below
        logger.warning(
            "checkpoint re-eval failed at gen %d: %s (%s)",
            state["gen"], e.__class__.__name__, e,
        )
        residuals = None
        proxy_source = "not_implemented"
        # Continue: calibration must not die because the dashboard re-eval
        # failed. Checkpoint is written without per-species residuals; the
        # proxy table shows the not_implemented banner for this tick.
    ...build CalibrationCheckpoint(..., proxy_source=proxy_source,
        per_species_residuals=(tuple(r for _, r, _ in residuals) if residuals
                               else None),
        per_species_sim_biomass=(tuple(b for _, _, b in residuals) if residuals
                                 else None),
        species_labels=(tuple(s for s, _, _ in residuals) if residuals
                        else None))
```

**Reconciliation property (downgraded).** Because the engine rebuilds the same PCG64 state, the worker's `best_fun` and the main-thread re-eval's residuals are derived from bit-identical simulations. `sum(residuals)` matches the worker's `best_fun` exactly **modulo `w_worst` and stability composition** — the proxy table at `eps = 1e-9` is well within tolerance. The earlier "1e-12" claim is dropped; the relevant guarantee for the user is that the proxy's in-band / out-of-band classification matches what the worker saw, not arbitrary numeric reconciliation.

The re-evaluation cost is one extra simulation per checkpoint. With `checkpoint_every=5` and default `popsize=15`, that's `1 / (5*15) = ~1.3 %` overhead — acceptable. For longer `checkpoint_every`, overhead drops further. surrogate-DE and CMA-ES use the same re-eval pattern in their own per-generation callbacks. The mechanism depends on `simulation.rng.fixed=true` (the calibration default per CLAUDE.md's RNG note); a test asserts this is set in the calibrator's base config.

**When banded-loss is enabled but `_ObjectiveWrapper` has not yet been called once** (very first generation): `last_per_species_residuals is None`, runner emits `proxy_source="not_implemented"` (this is genuinely a startup edge, treat as bug to investigate if it persists past gen 1).

#### 6.5.2 Path B change — `make_banded_objective`

Extend the factory to return a `(callable, residuals_accessor)` tuple:

```python
def make_banded_objective(
    targets: list[BiomassTarget],
    species_names: list[str],
    w_stability: float = 5.0,
    w_worst: float = 0.5,
) -> tuple[ObjectiveFn, ResidualsAccessor]:
    """Returns (objective_callable, residuals_accessor).

    objective_callable(species_stats) -> float    # unchanged contract & defaults
    residuals_accessor() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]] | None
        Returns (species_labels, residuals, sim_biomass) from the most-recent
        objective call, or None if no call has been made yet. Labels are
        species_names order (stable, deterministic). The reconciliation
        guarantee is the same as Path A's (§6.5.1): the per-species verdict
        (in-band / out-of-band) matches what the optimizer's scalar reflected,
        and the magnitude factor sim_biomass[i] / target_mean[i] is
        recoverable for the proxy table's UX1 column. Exact 1e-12
        numeric reconciliation does NOT hold once w_worst and w_stability
        compose into the scalar; the proxy table at eps=1e-9 is well within
        the tolerance the UI actually needs.
    """
```

The closure stores residuals in a non-local variable updated on each call. The accessor reads it. Single-threaded by construction — pymoo's NSGA-II in the in-UI path doesn't fork workers via this callable, so no multiprocessing caveat applies on Path B.

**Failure-mode parity with Path A.** The closure also clears its non-local residual variable to `None` at the START of each `objective_callable(...)` invocation and only re-assigns it as the last statement before returning the scalar. The same load-bearing assign-at-end invariant applies — a mid-call raise leaves the accessor returning `None`. The NSGA-II checkpoint hook (§6, runner table) sees `banded_residuals()` is `None`, emits `proxy_source="not_implemented"`, writes the checkpoint without residuals, and logs a WARNING. Symmetric with Path A so the dashboard renders the same banner regardless of which path failed.

**Backward compatibility.** Two known consumers must be updated:

1. **`ui/pages/calibration_handlers.py:839`** — change from `banded_obj = make_banded_objective(...)` to `banded_obj, banded_residuals = make_banded_objective(...)`. The `banded_obj` callable's signature and behaviour are unchanged.

2. **`osmose/calibration/__init__.py:30, 61`** — the package re-exports `make_banded_objective` in both the `from osmose.calibration.losses import …` line and `__all__`. External callers that do `from osmose.calibration import make_banded_objective` and treat the return value as a bare callable will silently receive a tuple after this change, breaking everything that follows. Mitigation: keep `make_banded_objective` re-exported with the *new* tuple-returning signature (don't try to preserve the old shape — there are no known third-party consumers; this is internal). Add a CHANGELOG entry documenting the signature change. Tests in `tests/test_calibration_losses.py` import from `osmose.calibration.losses` directly; they are unaffected by the package re-export but must be updated for the new return shape.

**Why two parallel changes rather than unifying.** `_ObjectiveWrapper` is picklable to support DE workers; `make_banded_objective` is a closure factory (more idiomatic Python but not trivially picklable). Unifying them is a real refactor that breaks the multiprocessing contract DE depends on. Out of scope for this PR; flagged in §11.

#### 6.5.3 `proxy_source` field semantics (recap)

| Value | When |
|---|---|
| `"banded_loss"` | banded-loss enabled, residuals captured successfully |
| `"objective_disabled"` | banded-loss not in use for this run; no proxy is meaningful |
| `"not_implemented"` | banded-loss enabled but residuals were not captured (bug surface) |

## 7. UI surface

Inserted into the existing Run tab between `cal_status` (line 264) and `convergence_chart` (line 265) in `ui/pages/calibration.py`:

```
┌─ Run tab ──────────────────────────────────────────────────┐
│  ┌─ Run header ──────────────────────────────────────────┐ │
│  │  DE · phase 12  |  gen 42 / 200  |  elapsed 1h 23m    │ │
│  │  ⏱ patience 3/20   ●  live (last update 2 s ago)      │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ 1 other live run · CMA-ES phase 1 [switch] ──────────┐ │  (only when applicable)
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Per-species ICES proxy ─────────── [proxy] ───────────┐│
│  │  sprat        loss 0.42  band [0.6, 1.6]   ✗            ││
│  │  cod          loss 0.00  band [0.9, 1.8]   ✓            ││
│  │  herring      loss 0.00  band [0.7, 1.4]   ✓            ││
│  │  flounder     —                            n/a          ││  (or banner — see §8)
│  │  …                                                       ││
│  │  5/8 in-band (proxy) · 2 out · 1 n/a · authoritative    ││
│  │  ICES verdict appears in Results tab after completion. ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─ Convergence (unchanged) ─────────────────────────────┐ │
│  │   <existing convergence_chart>                         │ │
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ▾  Current best parameters  (collapsed by default)        │
└────────────────────────────────────────────────────────────┘
```

**Reactive plumbing — sketch:**

The key correctness point: `_read_active_checkpoint` and `_other_live_runs` **must derive from a single shared snapshot per tick**, otherwise the header and badge can disagree when the writer races between two independent globs. The sketch below uses a single `@reactive.poll` that builds a frozen `LiveSnapshot` consumed by both downstream renderers.

```python
import dataclasses
import html
import logging
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from shiny import reactive, ui

from osmose.calibration.checkpoint import (
    CheckpointReadResult,
    default_results_dir,
    is_live,
    probe_writable,
    read_checkpoint,
)

logger = logging.getLogger("osmose.ui.calibration_dashboard")
# Logger NAME, not module name. The dashboard code in this sketch lives
# in the existing module ui/pages/calibration_handlers.py — that file is
# the integration target named throughout §6 and §9. See §10.4 fixture
# notes for the monkeypatch target.

RESULTS_DIR: Path = default_results_dir()  # module-level constant, also the
                                            # symbol §10.4 fixture patches

# Called once at server startup
try:
    probe_writable(RESULTS_DIR)
except OSError as e:
    ui.notification_show(
        f"Calibration results directory unreachable: {html.escape(str(e))} — "
        "dashboard will not show CLI runs",
        type="error", duration=None,
    )

@dataclass(frozen=True)
class LiveSnapshot:
    """One atomic view of the results directory per tick. Shared by all
    rendering reactives so they cannot disagree about which run is active."""
    active: CheckpointReadResult                # newest live, or kind='no_run' sentinel
    other_live_paths: tuple[Path, ...]          # all other live files
    snapshot_monotonic: float                   # time.monotonic() of capture

_EMPTY_SNAPSHOT = LiveSnapshot(
    active=CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None),
    other_live_paths=(),
    snapshot_monotonic=0.0,
)

def _scan_results_dir() -> LiveSnapshot:
    """Atomic scan: glob once, stat each path with FileNotFoundError tolerance,
    sort, partition into (newest_live, other_live, idle). Single source of
    truth per tick. Top-level try/except guards against RESULTS_DIR being
    deleted/unmounted mid-session — never raises into the reactive runtime.

    Symlink policy: skips paths that are symlinks. A hostile or accidental
    symlink to /etc/shadow (or any non-checkpoint file) would otherwise be
    opened by read_checkpoint, and bytes from the target file could leak
    through UnicodeDecodeError.__str__ into error_summary -> UI banner. By
    refusing symlinks entirely the dashboard cannot be tricked into reading
    files outside the results directory."""
    try:
        paths_with_mtime: list[tuple[Path, float]] = []
        for p in RESULTS_DIR.glob("phase*_checkpoint.json"):
            try:
                # follow_symlinks=False makes is_symlink() detectable; we
                # skip symlinks rather than reading their targets.
                st = p.lstat()
                if stat.S_ISLNK(st.st_mode):
                    continue
                paths_with_mtime.append((p, st.st_mtime))
            except (FileNotFoundError, PermissionError):
                continue
        paths_with_mtime.sort(key=lambda pm: pm[1], reverse=True)
        live: list[Path] = []
        for p, _mt in paths_with_mtime:
            try:
                if is_live(p):
                    live.append(p)
            except (FileNotFoundError, PermissionError):
                continue
        if live:
            active = read_checkpoint(live[0])
            others = tuple(live[1:])
        else:
            active = CheckpointReadResult(kind="no_run", checkpoint=None,
                                          error_summary=None)
            others = ()
        return LiveSnapshot(active=active, other_live_paths=others,
                            snapshot_monotonic=time.monotonic())
    except OSError as e:
        # RESULTS_DIR deleted, unmounted, EACCES, etc. Surface once per
        # error class via module-level seen-set; return empty snapshot so
        # downstream renderers see "no_run" gracefully.
        _notify_scan_failure_once(e)
        return dataclasses.replace(_EMPTY_SNAPSHOT, snapshot_monotonic=time.monotonic())

_signature_tick: int = 0  # monotonically incremented on any OSError so the
                          # poll dependency invalidates and _scan_results_dir
                          # re-runs (and _notify_scan_failure_once fires)
                          # even on persistent failures.

def _scan_signature() -> tuple[float, int, int]:
    """Cheap poll dependency — invalidates when any mtime OR file count
    changes. Keeps the 1 s tick non-busy when nothing changes. On any
    OSError, returns a strictly-increasing third element so the dependency
    invalidates regardless of underlying state — this guarantees
    _scan_results_dir re-runs (and surfaces the error via
    _notify_scan_failure_once) on every tick while the directory is
    broken, instead of silently latching.

    Uses lstat() to match _scan_results_dir's symlink-skip policy. Without
    this, a symlink in RESULTS_DIR would (a) be skipped by the scan but
    (b) contribute its target's mtime here, undermining the symlink-refusal
    security rationale. lstat also avoids a permission error on a symlink
    whose target is outside the directory and unreadable."""
    global _signature_tick
    try:
        pairs = []
        for p in RESULTS_DIR.glob("phase*_checkpoint.json"):
            try:
                st = p.lstat()
                if stat.S_ISLNK(st.st_mode):
                    continue
                pairs.append(st.st_mtime)
            except (FileNotFoundError, PermissionError):
                continue
        return (max(pairs, default=0.0), len(pairs), 0)
    except OSError:
        _signature_tick += 1
        return (0.0, 0, _signature_tick)

_seen_scan_errors: set[type] = set()
_seen_scan_errors_lock = threading.Lock()  # _notify_scan_failure_once may be
                                            # called from the Shiny async loop
                                            # AND from CLI runner threads in
                                            # the same process; lock keeps the
                                            # set update atomic.

def _notify_scan_failure_once(e: OSError) -> None:
    cls = type(e)
    with _seen_scan_errors_lock:
        if cls in _seen_scan_errors:
            return
        _seen_scan_errors.add(cls)
    logger.error("calibration results scan failed: %s: %s", cls.__name__, e)
    # HTML-escape the exception's str() because Shiny's notification_show
    # does NOT auto-escape its `message` argument — a hostile filename
    # could otherwise inject HTML/JS via the OSError path (e.g.
    # "phase<img src=x onerror=fetch(...)>_checkpoint.json"). Apply the
    # same html.escape() to every notification_show in this module.
    ui.notification_show(
        f"Calibration directory scan failed "
        f"({html.escape(cls.__name__)}: {html.escape(str(e))}) — "
        "dashboard will retry. Check the results directory's mount/perms.",
        type="warning", duration=None,
    )

@reactive.poll(_scan_signature, interval_secs=1.0)
def _live_snapshot() -> LiveSnapshot:
    return _scan_results_dir()
```

All UI renderers (`run_header`, `ices_proxy_table`, `other_live_badge`) consume `_live_snapshot()` — they share one snapshot per tick by construction.

**Previous-frame cache invalidation.** The "previous frame" referenced under `kind='partial'` handling lives in a small `reactive.Value[LiveSnapshot | None]` updated only when `kind == 'ok'`. It is reset to `None` whenever (a) `snapshot.active.checkpoint.phase` differs from the previous-frame's `phase`, or (b) `snapshot.active.checkpoint.optimizer` differs, or (c) the previous frame's age (monotonic delta) exceeds 5 min. Prevents a stale frame from run-1 rendering with run-2's banner.

**Liveness states.** A checkpoint file's age (now − mtime) determines the indicator:

| Age | State | Render |
|---|---|---|
| ≤ 60 s | live | green dot, "live (last update Xs ago)" |
| 60 s – 5 min | stalled | amber dot, "stalled — no checkpoint update for Xs (process may be paused, on a long generation, or the clock jumped)" |
| > 5 min | idle | grey dot, "idle (last X min ago)" |

Stalled distinguishes "a long-generation legitimate run" from "SIGSTOP'd or crashed". Without it, both look the same; with it the user is prompted to investigate. The amber state's message is intentionally informative rather than alarming.

**Clock-jump resilience.** `is_live` requires `mtime > now − max_age_s` **and** `mtime <= now`. A future-mtime file (NTP rewind) returns False — a real running process cannot have a future mtime. The UI also captures a `monotonic()` snapshot at each successful read; if monotonic deltas diverge sharply from wall-clock deltas, the header shows "(clock jump detected)".

**Stale-data guard.** When transitioning live→stalled→idle, the dashboard continues showing the most-recent checkpoint's values, but the header text changes and stale rows are dimmed. The user always knows whether the displayed values are current.

**"N other live runs" badge.** When `len(_other_live_runs()) >= 1`, render the strip with a one-click switch. Spec uses single-action `[switch]` for one other; for `>=2` the strip lists `2 other live runs · [switch to <newest_other>] · […]`. The badge is not a list view (§3 non-goal #1); it's disclosure that other runs exist plus a one-step jump to the next-most-recent.

**Patience badge.** Uses `gens_since_improvement`. `⏱ patience 3/20` means three generations since improvement, will early-stop at 20. Hidden when patience is not configured (DE has it; CMA-ES doesn't expose it as such).

**Current best parameters.** Collapsed by default. Renders `best_parameters` sorted by key.

**Bound-distance hints (UX3).** Each param row shows a small badge when `best_x_log10` is within 5% of its bound's range from either end:
- `[at upper bound]` when `(hi - best_x_log10[i]) / (hi - lo) < 0.05`
- `[at lower bound]` when `(best_x_log10[i] - lo) / (hi - lo) < 0.05`
- no badge otherwise

Operator workflow rationale: when DE is "chasing a corner" — i.e., the optimum is pinned at a parameter bound — the run will not improve no matter how long it runs. The badge surfaces this immediately and supports the "kill and re-tune the bounds" decision. Param values within 5% of either bound are the most-actionable mid-run signal for tuning the search space; the alternative is reading bound values from the config and the current params from the table and computing the comparison by eye.

**Best-ever reference line on convergence chart (UX2).** The existing Plotly convergence chart adds a horizontal dashed line at the lowest `best_objective` across all runs in `data/calibration_history/` matching the current `(optimizer, phase)` pair (queried via `osmose.calibration.history.list_runs()` filtered in the reactive). Hover label: `"best ever: f=<X.XX> (run <ts>)"`. If no prior matching run exists, no line is drawn. Cheap (~10 LOC: one filter pass over `list_runs()`'s output, one `fig.add_hline(y=best_ever, line_dash="dash", annotation_text=f"best ever: f={best_ever:.3f}")`). The data wiring already exists because §9 populates `data/calibration_history/`.

**In-UI fallback.** If `probe_writable` failed at startup or any `write_checkpoint` raises, the in-UI NSGA-II path keeps feeding the convergence chart via the existing in-memory `cal_history` reactive. The run header and proxy table show "Dashboard persistence degraded — convergence chart only (History tab will not record this run)." Single source of truth: when disk works, the header reads disk; when it doesn't, the header explicitly degrades with a banner rather than silently mixing in-memory and on-disk state.

**Error rendering.** A `CheckpointReadResult` with `kind='partial'` renders the previous frame's checkpoint values (cached in the `reactive.Value` described above, with invalidation rules) and a small "(updating…)" badge — partial writes are transient (≤ 3 s) and rendering the previous frame is correct UX. A `kind='corrupt'` result renders an inline red banner: "Checkpoint `<filename>` is unreadable (`<error_summary>`). Calibration may still be running — check `logs/`." A `kind='no_run'` result is silent — no banner, just the "No active run" header text.

**Proxy-table rendering cost.** The proxy table re-renders every tick the underlying `LiveSnapshot` changes. To keep 1 Hz cost negligible, the table is rendered as plain HTML (`ui.tags.table` with rows) via `@render.ui`, NOT as a Plotly figure or shiny `DataGrid`. A 5-10 row HTML table at 1 Hz is essentially free; a Plotly redraw cycle at the same cadence would be perceptible. The convergence chart remains Plotly (it already exists and updates only when `cal_history` grows — not every tick). The proxy table memoizes on the tuple `(generation, proxy_source, tuple(per_species_residuals or ()))` so identical successive frames skip the HTML rebuild.

**Accessibility.** The run header and proxy table update once per second when a run is active. Without ARIA attributes, screen readers would announce every change. The implementation:

- Run header is wrapped in `aria-live="polite"` with `aria-atomic="false"`; only changed parts (gen counter, elapsed) trigger an announcement, batched at SR-default cadence. State words ("live", "stalled", "idle") are inside the SR-visible span; colour dots have an additional `aria-label` so the state is not colour-only.
- Proxy-table rows have `aria-label` on the verdict column (`"in band"`, `"out of band — loss 0.42 against band 0.6 to 1.6"`, `"proxy unavailable"`) so SR users get the verdict without parsing the ✓/✗/— glyphs.
- The `[switch]` link in the "N other live runs" badge is a real `<button>` with `aria-label="Switch to <optimizer> phase <phase>"`, keyboard-focusable.
- The previous-frame "(updating…)" badge has `aria-live="off"` so the transient flicker doesn't dominate SR output.

These are minimum-bar a11y choices — fuller WCAG compliance (heading hierarchy audit, keyboard-only navigation across the whole dashboard) is documented as out of scope in §11.

## 8. Per-species ICES proxy semantics

The proxy approximates the post-run validator from the banded-loss components the optimizer is already minimizing. Three states:

Let `eps = 1e-9` (small but well above float64 round-off in residual sums; banded-loss returns exactly 0.0 when in-band, so any value above `eps` is unambiguously a real penalty).

The "magnitude factor" per row is `sim_biomass[i] / target_mean[i]` where `target_mean[i] = sqrt(lo * hi)` (geometric mean of the band — biomass is log-scaled; the geometric mean is the band's centre on that scale). Display modes per state:

| State | Condition | Display |
|---|---|---|
| `in_range` | `banded_targets[species]` exists AND `per_species_residuals[i] <= eps` | `✓` green, magnitude column shows `≈1.0×` |
| `out_of_range` | banded target exists AND `per_species_residuals[i] > eps` AND `per_species_sim_biomass[i] > 0` | `✗` red + raw loss + band + magnitude e.g. `2.4× overshoot` (factor > 1) or `0.4× undershoot` (factor < 1) |
| `extinct` | `per_species_sim_biomass[i] == 0` (model simulated extinction) | `☠` red + `loss 100.0` + `extinct` label — magnitude is undefined |
| `n/a` | `banded_targets[species] is None` for this species | `—` muted, with visible inline reason column (not tooltip-only) |

The magnitude factor is the operator's most-actionable signal. Per project memory, Baltic phase 12 perch/pikeperch overshoots reach ×100-450×; the raw loss value alone (e.g. `loss 5.4`) cannot distinguish a structural ×450 catastrophic overshoot from a 1.5× minor exceedance. Showing `442× overshoot` makes the failure mode unambiguous and supports the operator's mid-run "kill and re-tune" decision (§7 "mid-run pivots" rationale).

**All-n/a banner.** If `proxy_source != "banded_loss"` (i.e. banded-loss is not enabled or the residual accessor returned no data), the table is *replaced* by a single banner — not rendered as 8 rows of dashes:
- `proxy_source == "objective_disabled"`: blue banner — "ICES proxy unavailable: this run does not use banded-loss objectives. Authoritative verdict will appear in Results tab on completion."
- `proxy_source == "not_implemented"`: red banner — "ICES proxy: per-species residuals were not exposed by `losses.py` despite banded-loss being configured. This is a bug — please file an issue and include checkpoint `<filename>`."

The distinction prevents the user from reading "feature unimplemented" as "this run is broken".

**Magnitude column.** Now in scope as UX1. Computed mid-run from `per_species_sim_biomass / target_mean`. The relationship to the post-run validator's `magnitude_factor` is: the validator computes `model_window_mean / ices_envelope_mean`; the proxy computes `per_species_sim_biomass / sqrt(lo * hi)`. These agree when the calibrator's banded targets are derived from ICES envelopes (the Baltic case). For non-ICES-derived bands, the proxy is still a directionally-accurate "how far from the band's centre" measure, just not the validator's exact metric.

**Default sort.** `out_of_range` first (red rows on top), then `in_range`, then `n/a`.

**Footer.** `5/8 in-band (proxy) · 2 out · 1 n/a · authoritative ICES verdict appears in Results tab after completion.` Reinforces the proxy framing.

## 9. History wiring

Each runner, on successful completion, calls `osmose.calibration.history.save_run()` with:

```python
save_run({
    "timestamp": datetime.now(timezone.utc).isoformat(),
        # UTC with explicit "+00:00" suffix. datetime.now().isoformat() (no tz)
        # would sort wrong across machines in different tz, and the dashboard's
        # History tab sorts by this string. The same change must apply to
        # CalibrationCheckpoint.timestamp_iso (set in write_checkpoint).
    "algorithm": "de" | "cmaes" | "surrogate-de" | "nsga2",
    "phase": "12",
    "parameters": [...],  # list of param keys
    "results": {
        "best_objective": ckpt.best_fun,
        "best_parameters": ckpt.best_parameters,
        "duration_seconds": ckpt.elapsed_seconds,
        "n_evaluations": <runner-specific>,
        "per_species_residuals_final": list(ckpt.per_species_residuals)
            if ckpt.per_species_residuals is not None else None,
        "per_species_sim_biomass_final": list(ckpt.per_species_sim_biomass)
            if ckpt.per_species_sim_biomass is not None else None,
        "species_labels": list(ckpt.species_labels)
            if ckpt.species_labels is not None else None,
    },
})
```

This payload is a strict superset of what `list_runs()` already reads (`osmose/calibration/history.py:43-49`). No History-tab UI changes needed; runs start appearing automatically.

Call sites:

| Runner | Hook |
|---|---|
| DE | After `differential_evolution()` returns, before the existing `phase{N}_results.json` write |
| CMA-ES | End of `run()` |
| surrogate-DE | End of `run()` |
| NSGA-II | In the `"results"` branch of `_poll_cal_messages` at `ui/pages/calibration_handlers.py:451-454`, after both `cal_X.set(X)` and `cal_F.set(F)` calls |

The existing `phase{N}_results.json` write is preserved unchanged — `save_run()` is additive, so `report_calibration.py` and the Results tab keep working.

**Failure handling.** Wrap `save_run()` in a layered exception scope:

```python
try:
    save_run(payload)
except (OSError, TypeError, ValueError, OverflowError,
        UnicodeError, RecursionError, MemoryError) as e:
    _save_run_fallback(payload, e)  # log + write /tmp fallback + UI banner
except Exception as e:  # noqa: BLE001 — defensive catch-all
    # Any other exception means a future change introduced an unexpected
    # error class. Treat as a fallback case rather than crash the runner.
    logger.exception("unexpected save_run failure: %s", e)
    _save_run_fallback(payload, e)
```

`_save_run_fallback`:
- Calls `logger.exception(...)` (NOT `logger.error`) so the stack trace is captured unconditionally — covers both the explicit-exception-list branch and the catch-all `Exception` branch. The log message includes `e.__class__.__name__` and the payload's top-level keys (not values — they may contain large vectors). Without `logger.exception`, a `ValueError`/`MemoryError` from a code bug would lose the traceback and become hard to postmortem.
- Writes a fallback payload to `tempfile.gettempdir() / "calibration_history_fallback_{ts}_{algo_sanitized}.json"` with mode 0o600 via `os.open(path, O_WRONLY|O_CREAT|O_EXCL, 0o600)` — NOT the default 0o644. The fallback contains the operator's calibrated parameters; world-readable is the wrong default. The `O_EXCL` is **intentional**: if the file already exists (e.g., a clock skew produced the same `ts` for two concurrent fallback writes), do NOT overwrite the prior fallback — that prior file is still the most-likely-recoverable copy of the operator's run data. The second `save_run` failure logs to stderr only (via the outer catch-all `except Exception`) and surfaces a banner including the existing-fallback path; the operator can rename and retry. Implementers must not "fix" `O_EXCL` to `O_TRUNC`. `tempfile.gettempdir()` picks `/tmp` on Linux/macOS, `%TEMP%` on Windows. `algo_sanitized` is `re.sub(r"[^A-Za-z0-9_-]", "_", str(payload.get("algorithm","unknown")))[:32]` — `payload["algorithm"]` is loaded from JSON and is not statically typed at this point, so the f-string-into-filename path must defensively sanitise rather than trust the literal-union annotation.
- Sets a Shiny `reactive.Value[str | None]` `history_persistence_failure_banner` to the message: `"Last run ({html.escape(algo)} {html.escape(ts)}) failed to persist ({html.escape(error_class)}) — fallback at {html.escape(str(path))}."` (every interpolation HTML-escaped — same XSS class as the scan-failure notification). The History tab renders the banner whenever this is non-None and clears it on the next successful `save_run`.

## 10. Testing plan

### 10.1 Unit — checkpoint module (`tests/test_calibration_checkpoint.py`)

Reuses a new `tmp_results_dir` fixture in `tests/conftest.py` that monkeypatches `osmose.calibration.checkpoint.default_results_dir` to `tmp_path`.

**Happy path:**
- `test_write_read_roundtrip_preserves_all_fields` — every field of `CalibrationCheckpoint`.
- `test_is_live_within_window` and `test_is_live_outside_window` across mtime boundaries.

**Negative path — write:**
- `test_write_is_atomic_no_partial_file` — patch `os.replace` to raise; assert `phase*_checkpoint.json` does not exist (only the `.tmp`).
- `test_write_coerces_numpy_scalars` — pass `np.float64` for `best_fun`; assert JSON contains a plain float.
- `test_write_coerces_numpy_arrays` — pass `np.ndarray` for `best_x_log10`; assert tuple in JSON.
- `test_write_rejects_nan` — pass `float('nan')` for `best_fun`; assert `ValueError`.
- `test_write_rejects_inf` — same with `float('inf')`.
- `test_write_to_readonly_dir_raises_oserror` — chmod 0o555 on `tmp_path`; assert `OSError`. Skip on Windows.

**Negative path — read:**
- `test_read_returns_partial_on_truncated_json_recent_mtime` — write `{"optimizer": "de"`, assert `kind='partial'`.
- `test_read_returns_corrupt_on_truncated_json_old_mtime` — same payload, set mtime to 1 hour ago; assert `kind='corrupt'`.
- `test_read_returns_no_run_on_file_vanishing` — file deleted between glob and read (patch `Path.stat` to raise `FileNotFoundError`); assert `kind='no_run'`, no exception. Matches the §5 docstring contract (file-vanish is `no_run`, not `partial`).
- `test_read_returns_no_run_on_stat_permission_error` — patch `Path.stat` to raise `PermissionError`; assert `kind='no_run'`.
- `test_read_returns_corrupt_on_invariant_violation` — write valid JSON with `generation=-1`; assert `kind='corrupt'` with `error_summary` mentioning `generation`.
- `test_read_returns_corrupt_on_mismatched_parallel_arrays` — `species_labels` of length 2 with `per_species_residuals` of length 3; assert `kind='corrupt'`.
- `test_read_returns_corrupt_on_invalid_utf8` — write `b'\xff\xfe\x00not-json'` to the checkpoint file with old mtime; `json.load` raises `UnicodeDecodeError` (a `ValueError` subclass that is NOT `JSONDecodeError`); assert `kind='corrupt'` with `error_summary` mentioning `UnicodeDecodeError`. Spec contract for `read_checkpoint` must explicitly handle this — extend the partial/corrupt branches to "(JSONDecodeError, UnicodeDecodeError, ValueError)".

**Liveness state machine:**
- `test_is_live_false_when_mtime_in_future` — mtime = now + 60 s; assert False.
- `test_is_live_uses_injected_clock` — pass `now=…`; assert deterministic.
- `test_liveness_state_live_boundary` — file with mtime = now − 30 s ⇒ state = "live". Drives whatever helper resolves the §7 three-state classification (live ≤ 60 s, stalled 60-300 s, idle > 300 s).
- `test_liveness_state_stalled_boundary_60s` — mtime = now − 90 s ⇒ "stalled" (transition just inside the stalled band).
- `test_liveness_state_stalled_to_idle_boundary_300s` — mtime = now − 4 min ⇒ "stalled"; mtime = now − 6 min ⇒ "idle". Pins the 300 s boundary.

**`__post_init__` invariants:** one test per invariant in §5 (14 cases).

### 10.2 Unit — proxy renderer (`tests/test_ices_proxy.py`)

Pure helper `_render_ices_proxy_table(ckpt)`:
- `test_in_range_zero_loss` → `✓`.
- `test_out_of_range_positive_loss` → `✗` with raw loss and band.
- `test_eps_boundary` — `loss = 1e-12` (below eps) → `✓`; `loss = 1e-8` → `✗`.
- `test_proxy_source_objective_disabled_renders_blue_banner` (not 8 dashes).
- `test_proxy_source_not_implemented_renders_red_banner`.
- `test_mixed_species_some_with_band_some_without` — bands present for 5 of 8 species → 5 valid rows + 3 n/a rows with inline reason.
- `test_default_sort_out_first_then_in_then_na`.
- `test_magnitude_column_overshoot` — `sim_biomass=2.4`, `band=[0.5, 1.0]`; assert magnitude displays as `≈3.4× overshoot` (where `3.4 = 2.4 / sqrt(0.5 * 1.0)`).
- `test_magnitude_column_undershoot` — `sim_biomass=0.2`, `band=[0.5, 1.0]`; assert displays as `≈0.28× undershoot`.
- `test_magnitude_column_extinct` — `sim_biomass=0.0`; assert state is `extinct`, magnitude column shows `—`, badge says "extinct".
- `test_magnitude_column_in_band` — `sim_biomass=0.7`, `band=[0.5, 1.0]`; assert magnitude shows `≈0.99×` (close to 1).
- `test_bound_distance_badge_at_upper` — param at 95% of bound range; assert `[at upper bound]` badge.
- `test_bound_distance_badge_at_lower` — symmetric.
- `test_bound_distance_no_badge_in_middle` — param at 50% of range; assert no badge.
- `test_best_ever_reference_line_drawn_when_prior_runs_exist` — seed `list_runs()` with two prior matching `(optimizer, phase)` runs of `best_objective=4.2` and `5.1`; assert chart contains a horizontal line at `y=4.2` with annotation "best ever: f=4.200".
- `test_best_ever_reference_line_omitted_when_no_prior_runs` — `list_runs()` returns no matches; assert no horizontal line.

### 10.3 Unit — per-species residual exposure (Paths A and B)

The proxy table depends on **two** new accessor contracts. Tests cover both.

**Path A — `_ObjectiveWrapper` (`tests/test_objective_evaluator_residuals.py`, new file):**
- `test_last_per_species_residuals_none_before_first_call` — fresh evaluator; assert `evaluator.last_per_species_residuals is None`.
- `test_last_per_species_residuals_populated_after_call` — call evaluator once with the synthetic-2-species toy fixture (§10.4); assert tuple of `(species_name, weighted_error, sim_biomass)` — the 3-element shape introduced by UX1 — with correct length and stable ordering matching `species_names` constructor argument.
- `test_residuals_sum_equals_total_minus_worst_term` — for the same input, `sum(weighted_error for _, weighted_error, _ in evaluator.last_per_species_residuals)` equals the scalar returned minus `w_worst * worst_error` and minus stability penalties. The 3-element unpacking discards the `sim_biomass` field (covered by separate tests below).
- `test_sim_biomass_captured_per_species` — call evaluator with `stats={"sp_a_mean": 1.7, "sp_b_mean": 0.3, ...}`; assert each entry's third element matches the corresponding `mean_key` value (`1.7` and `0.3`).
- `test_sim_biomass_zero_on_extinction_fast_path` — call evaluator with `stats={"sp_a_mean": 0.0, ...}`; assert the residual record is `("sp_a", 100.0, 0.0)` (matching the fast-path penalty AND the extinction sim_biomass).
- `test_residuals_zero_when_in_band` — input biomass inside [lower, upper]; assert that species' residual is 0.0 exactly.
- `test_residuals_record_fast_path_penalties` — mean_key not in stats / target_dict / sim_biomass <= 0; assert residual recorded as 100.0 to match the existing scalar contribution.
- `test_evaluator_round_trips_through_multiprocessing` — verify the evaluator can be serialized and reconstructed by the same protocol scipy/joblib use to ship objectives to DE workers. Adding `last_per_species_residuals` must not break this contract; spawn a real `multiprocessing.Pool(1)` and call the evaluator through it to ground-truth.

**Path B — `make_banded_objective` accessor (`tests/test_calibration_losses.py`, additions):**
- `test_residuals_accessor_returns_none_before_first_call`.
- `test_residuals_accessor_returns_most_recent` — call objective twice with different inputs; accessor reflects the second call.
- `test_residuals_sum_matches_scalar_within_eps` — `sum(residuals) == scalar` within `1e-12` (before `w_worst`/`w_stability` composition).
- `test_residuals_ordering_matches_species_names` — call with scrambled inputs; assert `residuals[i]` corresponds to `species_names[i]` (constructor-order, not alphabetical — match the actual current code, not a re-imagined ordering).
- `test_make_banded_objective_callable_unchanged` — the scalar `Callable[[dict], float]` returned by the tuple's first element produces the same number as the previous-signature version for the same input. Locks the backward-compat guarantee.

### 10.4 New test fixtures (built as part of this work)

The existing optimizer tests (`tests/test_cmaes_runner.py`, `tests/test_surrogate_de.py`) use Rosenbrock/Sphere analytical fixtures with no concept of species — unsuitable for testing the per-species residual contract. The fixtures listed below are new to this PR and become reusable for future calibration work.

**`tests/conftest.py` additions:**
- `tmp_results_dir` — redirects all checkpoint writes to `tmp_path`. Because `read_checkpoint`/`write_checkpoint` take explicit `path` arguments (not derived from `default_results_dir()` internally), the fixture must monkeypatch TWO existing constants plus ONE constant this PR adds: (a) `osmose.calibration.checkpoint.default_results_dir` to return `tmp_path`, (b) `scripts.calibrate_baltic.RESULTS_DIR` to `tmp_path`, (c) the **new** `RESULTS_DIR` constant introduced at the top of `ui/pages/calibration_handlers.py` by the dashboard wiring in §7. (The §7 sketch's `logger = logging.getLogger("osmose.ui.calibration_dashboard")` chose a logger *name*, not a module name; the actual file the new code lives in is the existing `ui/pages/calibration_handlers.py`, which is the integration target named throughout §6 and §9.) Note: (c) does NOT exist in that file today — this PR adds it at module top. Tests use `tmp_results_dir` as a function argument and the fixture yields the `tmp_path` for assertions.
- `synthetic_two_species_targets` — returns `(targets, species_names)` for a 2-species toy with `(sp_a target=1.0, band=[0.5, 1.5])` and `(sp_b target=2.0, band=[1.5, 2.5])`. Used by every Path-A test and every checkpoint-write integration test.
- `synthetic_stats_in_band` / `synthetic_stats_out_of_band` — pre-built `species_stats` dicts that drive the evaluator into each branch.
- `mock_pymoo_algorithm` — minimal stand-in exposing `.opt.get("F")` and `.callback`, used by the NSGA-II callback tests (§10.6). Drives `_ProgressCallback.notify` without a real pymoo run.

### 10.5 Integration — runner writes checkpoint (`tests/test_runner_checkpoints.py`)

Parametrised across optimizers, using the fixtures in §10.4:

- `test_runner_writes_checkpoint_each_generation[de|cmaes|surrogate-de|nsga2]` — 3-generation run on the synthetic 2-species toy; assert one checkpoint file exists with the expected schema after each generation.
- `test_checkpoint_per_species_none_when_banded_disabled` — same toy without banded-loss; assert `per_species_residuals is None` and `proxy_source == "objective_disabled"`.
- `test_checkpoint_per_species_populated_when_banded_enabled` — toy with banded-loss; assert `per_species_residuals` non-empty and `proxy_source == "banded_loss"`.
- `test_de_checkpoint_main_thread_reevaluates_best_x` — patch `_ObjectiveWrapper.__call__` to count invocations; run DE with `workers=2`, `popsize=4`, `maxiter=3`, `checkpoint_every=1`; assert the main-process evaluator was called exactly once per checkpoint **in addition** to whatever the workers did. Pins the §6.5.1 re-evaluation contract.
- `test_de_checkpoint_reeval_raises_does_not_crash_run` — patch `_ObjectiveWrapper.__call__` to raise on the main-thread call only; assert DE completes, the checkpoint for that gen is written with `per_species_residuals=None` and `proxy_source="not_implemented"`, and a WARNING is logged with the exception class name. Verifies §6.5.1's bounded-failure contract.
- `test_de_checkpoint_reeval_clears_stale_residuals_on_failure` — patch the wrapper so the first re-eval succeeds (writes residuals), the second raises; assert the second checkpoint does NOT carry the first call's residuals (i.e. `last_per_species_residuals` is `None` before the failed call, so no stale data leaks).
- `test_calibrator_config_has_rng_fixed_true` — read the calibrator's base config; assert `simulation.rng.fixed == "true"`. Pins the §6.5.1 determinism precondition.

### 10.6 Integration — history persistence (`tests/test_history_wiring.py`)

- `test_de_end_to_end_writes_history` — synthetic 2-parameter 1-species quadratic objective `f(x) = (x[0]-1)**2 + (x[1]-2)**2`, `maxiter=1`, `popsize=4`, `workers=1`. Total runtime < 1 s. Real `differential_evolution` → real `_make_checkpoint_callback` → real `write_checkpoint` → real `save_run`. Asserts (a) checkpoint file exists, (b) `list_runs()` returns the run with `algorithm == "de"`, (c) `per_species_residuals_final is None`.
- `test_de_end_to_end_with_banded_loss_2_species` — same fixture with banded-loss; assert saved record has both elements and labels parallel.
- `test_save_run_payload_is_superset_of_list_runs_schema` — load saved record and pass through `list_runs()`'s field parser without `KeyError`.

### 10.7 Regression — in-UI NSGA-II path (`tests/test_ui_calibration_handlers.py`, additions)

The existing in-UI run already works for users; this is the highest-risk regression surface. The current file at `tests/test_ui_calibration_handlers.py` does NOT yet test `_make_progress_callback` or `_ProgressCallback.notify` — these tests are new additions, not extensions of an existing pattern. They use the `mock_pymoo_algorithm` fixture from §10.4.

- `test_results_branch_sets_cal_X_cal_F_when_checkpoint_write_raises_typeerror` — patch `write_checkpoint` to raise `TypeError`; drive the `"results"` branch of `_poll_cal_messages`; assert `cal_X.set` and `cal_F.set` were both called, and the message was consumed (not re-queued). Protects the user-facing chart from a future numpy-typing regression in `write_checkpoint`.
- `test_results_branch_sets_cal_X_cal_F_when_checkpoint_write_raises_oserror` — same with `OSError`.
- `test_results_branch_order_set_calls_first_then_write` — pin ordering: if `write_checkpoint` raises uncaught, the chart still updates. Future refactors that flip the order break this test.
- `test_progress_callback_writes_checkpoint_per_generation` — drive `_ProgressCallback.notify` with the mock pymoo algorithm three times; assert a checkpoint file exists in `tmp_results_dir` after each call with monotonically-increasing `generation`.

### 10.8 Reactive poll behaviour (`tests/test_calibration_dashboard_reactive.py`)

- `test_poll_returns_no_run_sentinel_when_no_files` — empty `tmp_results_dir`; assert sentinel `CheckpointReadResult(kind='no_run', checkpoint=None, error_summary=None)`. The earlier `kind='ok'` form is now an invariant violation per §5.
- `test_poll_returns_newest_live_file` — write three files at different mtimes; assert reader picks newest, skips stale.
- `test_poll_skips_files_that_disappear_between_glob_and_stat` — patch one path to raise `FileNotFoundError` on `stat()`; assert no exception escapes, other live files still picked.
- `test_other_live_runs_returns_all_but_newest` — three live files; assert two are in the "other" list.
- `test_other_live_runs_empty_when_one_or_zero` — assert returns `[]` for both cases.
- `test_scan_returns_empty_snapshot_when_results_dir_unmounted` — monkeypatch `RESULTS_DIR.glob` to raise `OSError("ENOENT")`; assert `_scan_results_dir` returns `_EMPTY_SNAPSHOT` (kind='no_run', other_live_paths=()), assert `_notify_scan_failure_once` fires exactly once, and assert `_scan_signature` returns a strictly-increasing tick on each subsequent failing call (so the poll keeps invalidating).
- `test_scan_signature_invalidates_on_persistent_oserror` — call `_scan_signature()` three times in a row while glob raises; assert the third element of the returned tuple is strictly greater each time.

### 10.9 Manual UI smoke (PR description checklist)

- Launch a DE run via CLI; open the Shiny Run tab; confirm header populates within 2 s, proxy table renders, convergence chart still works, header transitions live → stalled at ~60 s if the CLI process is SIGSTOP'd, returns to live on SIGCONT, transitions live → idle 5 min after the run completes.
- Launch an NSGA-II run via the UI's `Start Calibration` button; same checks.
- Disable persistence (chmod 0o555 the results dir); confirm the in-UI run still updates the convergence chart and the "Dashboard persistence degraded" banner appears.
- Concurrent: DE phase 1 + CMA-ES phase 12 in two shells; confirm the active view shows phase 12 (most recent) and the "1 other live run · DE phase 1 [switch]" badge appears.
- Run with banded-loss disabled; confirm proxy table is replaced by the blue banner.
- Trigger the "not_implemented" path by patching `make_banded_objective` to return a no-op accessor; confirm the red banner.
- Kill -9 the CLI runner mid-generation; confirm the header transitions live → stalled within 60 s.

**Conventions.** Ruff format, 100-char lines, run with `.venv/bin/python -m pytest`. Per repo CLAUDE.md. Every new test module starts with `from __future__ import annotations` to match the existing test-file convention (`tests/test_calibration_losses.py`, `tests/test_ui_calibration_handlers.py` both do this).

## 11. Open follow-ups (out of scope for this PR)

- **Validator-resim path.** Promote the proxy to an opt-in `--validator-resim` flag that runs a short simulation per checkpoint and surfaces the authoritative per-species verdict mid-run. Adds engine state-management complexity.
- **Warning ticker for extinctions / NaN sims (UX4).** A scrolling sidebar that shows the last N WARNING-or-above log lines from the calibrator. Currently invisible — tailing operators see these in stderr; the dashboard doesn't surface them. Cheap to wire if the calibrator writes them to a sidecar JSONL the dashboard can tail.
- **Per-species trajectory sparklines (UX5).** §3 non-goal #3 currently excludes these. Reconsider when the checkpoint write rate justifies storing a rolling residual history (e.g., last 50 generations) per species and rendering a 50-point sparkline per row. The direction-of-travel signal answers the operator's actual question ("are we converging?") better than the current snapshot in/out boolean.
- **Side-by-side concurrent-run comparison (UX6).** §3 non-goal #1 currently restricts the dashboard to one active run + a disclosure badge. The actual operator workflow (per project memory) is comparative — running phase-1 + phase-12 to evaluate whether changes help. A two-column layout (with the proxy table shared / convergence chart per-column) matches that workflow.
- **`RunRecord` dataclass for `history.py`.** The current dict-based contract has worked through PR #46; tightening it is a natural moment when the next consumer needs richer fields.
- **Watchdog/inotify push.** 1 s polling is sufficient at current generation cadence; if cadence shortens (e.g., GPU-accelerated calibration), push-based monitoring removes the latency floor.
- **Full multi-run view.** The disclosure badge in §7 makes concurrent runs visible; a real multi-run page is a follow-up if users routinely run more than two simultaneously.
- **Phase as `Literal[...]` or normalised `PhaseId`.** Type-design-analyzer flagged that `phase: str` admits typos. Keep as `str` for v1; tighten when the phase set stabilises.
- **Nested `SpeciesObjective` dataclass** to encode the `(species_labels, per_species_residuals)` parallel-array invariant as a single optional field rather than two correlated optionals. Real refactor; defer.
- **Mid-session `probe_writable` re-check.** Today's spec probes once at startup. A FS mount/unmount or disk-fill during a long Shiny session is swallowed by the runner-side `try/except` but not surfaced to the dashboard's own banner. A periodic re-probe (e.g., once per minute) would close the gap; trade-off is an additional cross-process signal channel.
- **Proper algebraic sum type for `CheckpointReadResult`.** The current shape requires runtime `__post_init__` invariants (`kind='ok' iff checkpoint is not None`) that a real tagged-union (Rust-style `Ok(ckpt) | NoRun | Partial(err) | Corrupt(err)`) would eliminate. Same applies to `proxy_source="not_implemented"` being a normal value. Defer until Python's `typing.TypeAliasType` / a small typed-union helper (or a stable third-party lib) is judged worth the dependency.
- **Full WCAG audit.** §7 specifies minimum-bar a11y for the new widgets. A pass that audits heading hierarchy, keyboard-only navigation of the whole calibration page, focus-trap on modals, and SR voicing of all chart interactions is a separate workstream.
- **Unifying Path A and Path B objective implementations.** `_ObjectiveWrapper` (CLI) and `make_banded_objective` (UI) both implement banded-log-ratio loss and now both need a per-species residual hook. A single shared implementation would eliminate the duplication but requires breaking the multiprocessing-picklability constraint DE depends on; real refactor, defer.
