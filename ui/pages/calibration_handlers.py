# ui/pages/calibration_handlers.py
"""Event handlers and helper functions for the calibration page."""

from __future__ import annotations

import dataclasses
import html
import logging
import math
import os
import queue as _queue_mod
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from shiny import reactive, render, ui
from shiny.types import SilentException

from osmose.calibration.checkpoint import (
    RESULTS_DIR,
    CalibrationCheckpoint,
    CheckpointReadResult,
    LiveSnapshot,
    is_live,
    probe_writable,
    read_checkpoint,
)
from osmose.logging import setup_logging
from osmose.schema.base import ParamType
from osmose.schema.registry import ParameterRegistry

_log = setup_logging("osmose.calibration.ui")

logger = logging.getLogger("osmose.ui.calibration_dashboard")

# Startup probe (call once at module import; failure logs but does not raise).
try:
    probe_writable(RESULTS_DIR)
except OSError as e:
    logger.error("RESULTS_DIR probe failed: %s", e)

_signature_tick: int = 0
_seen_scan_errors: set[type] = set()
_seen_scan_errors_lock = threading.Lock()
_EMPTY_SNAPSHOT = LiveSnapshot(
    active=CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None),
    other_live_paths=(),
    snapshot_monotonic=0.0,
)


def _notify_scan_failure_once(e: OSError) -> None:
    cls = type(e)
    with _seen_scan_errors_lock:
        if cls in _seen_scan_errors:
            return
        _seen_scan_errors.add(cls)
    logger.error("calibration results scan failed: %s: %s", cls.__name__, e)
    try:
        ui.notification_show(
            f"Calibration directory scan failed "
            f"({html.escape(cls.__name__)}: {html.escape(str(e))}) — "
            "dashboard will retry. Check the results directory's mount/perms.",
            type="warning", duration=None,
        )
    except Exception:
        pass  # outside a Shiny session (tests) — log-only


def _scan_signature() -> tuple[float, int, int]:
    """Cheap poll dependency. Uses lstat() to match the symlink-skip policy.

    Persistent failures advance _signature_tick so the poll keeps invalidating
    (otherwise it would latch on (0.0, 0) and never re-fire _scan_results_dir,
    which would silence _notify_scan_failure_once after first call).
    """
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


def _scan_results_dir() -> LiveSnapshot:
    """Atomic scan; never raises into the reactive runtime.

    Symlinks are skipped (security: a symlink in RESULTS_DIR cannot trick
    read_checkpoint into reading /etc/shadow, and bytes from a target file
    cannot leak through UnicodeDecodeError.__str__ into the UI banner).
    """
    try:
        paths_with_mtime: list[tuple[Path, float]] = []
        for p in RESULTS_DIR.glob("phase*_checkpoint.json"):
            try:
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
            active = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
            others = ()
        return LiveSnapshot(active=active, other_live_paths=others,
                            snapshot_monotonic=time.monotonic())
    except OSError as e:
        _notify_scan_failure_once(e)
        return dataclasses.replace(_EMPTY_SNAPSHOT, snapshot_monotonic=time.monotonic())


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def _ckpt_mtime_for(snap: LiveSnapshot) -> float:
    """Wall-clock time the active checkpoint was written."""
    if snap.active.checkpoint is None:
        return time.time()
    try:
        from datetime import datetime
        return datetime.fromisoformat(snap.active.checkpoint.timestamp_iso).timestamp()
    except ValueError:
        return time.time()


_PROXY_EPS = 1e-9
_STATE_ORDER = {"out_of_range": 0, "in_range": 1, "extinct": 2}


def _build_proxy_rows(ckpt: CalibrationCheckpoint) -> list[dict]:
    """Compute proxy-table rows. When proxy_source != 'banded_loss',
    returns a single sentinel row signalling the renderer to display a banner."""
    if ckpt.proxy_source != "banded_loss":
        return [{"state": ckpt.proxy_source, "species": "", "loss": 0.0,
                 "band": (0.0, 0.0), "magnitude": 0.0, "direction": ""}]

    assert ckpt.species_labels is not None
    assert ckpt.per_species_residuals is not None
    assert ckpt.per_species_sim_biomass is not None
    assert ckpt.banded_targets is not None

    rows: list[dict] = []
    for i, sp in enumerate(ckpt.species_labels):
        residual = ckpt.per_species_residuals[i]
        sim_biomass = ckpt.per_species_sim_biomass[i]
        lo, hi = ckpt.banded_targets[sp]
        target_mean = math.sqrt(lo * hi)  # Inv 9 ensures lo > 0
        if sim_biomass == 0.0:
            state, direction, magnitude = "extinct", "", 0.0
        elif residual <= _PROXY_EPS:
            state = "in_range"
            magnitude = sim_biomass / target_mean
            direction = ""
        else:
            state = "out_of_range"
            magnitude = sim_biomass / target_mean
            direction = "overshoot" if sim_biomass > hi else "undershoot"
        rows.append({
            "species": sp, "state": state, "loss": residual,
            "band": (lo, hi), "magnitude": magnitude, "direction": direction,
        })
    rows.sort(key=lambda r: _STATE_ORDER[r["state"]])
    return rows


def _aria_for_state(state: str, magnitude: float, direction: str) -> str:
    if state == "in_range":
        return "in band"
    if state == "out_of_range":
        return f"out of band — {magnitude:.2f} times {direction}"
    if state == "extinct":
        return "extinct"
    return "proxy unavailable"


def _require_preflight(
    base_config: Path | None,
    jar_path: Path | None,
    work_dir: Path | None,
) -> tuple[Path, Path | None, Path]:
    """Return the base_config and work_dir (required) plus an optional jar_path.

    The Python engine is the default and needs no JAR, so ``jar_path`` may be
    ``None``. Only ``base_config`` and ``work_dir`` are mandatory — both are
    produced by the preflight stage itself, so their absence really does mean
    preflight hasn't run yet.
    """
    if base_config is None or work_dir is None:
        raise RuntimeError(
            "Calibration preflight must run before optimization. "
            "Click 'Preflight' first in the Calibration tab."
        )
    return base_config, jar_path, work_dir


class CalibrationMessageQueue:
    """Thread-safe message queue for calibration thread -> UI communication."""

    def __init__(self):
        self._q: _queue_mod.Queue = _queue_mod.Queue()

    def post_status(self, msg: str) -> None:
        self._q.put(("status", msg))

    def post_history_append(self, value: float) -> None:
        self._q.put(("history_append", value))

    def post_results(self, X, F) -> None:
        self._q.put(("results", (X, F)))

    def post_error(self, msg: str) -> None:
        self._q.put(("error", msg))

    def post_sensitivity(self, result) -> None:
        self._q.put(("sensitivity", result))

    def post_validation(self, result) -> None:
        self._q.put(("validation", result))

    def post_history_saved(self) -> None:
        self._q.put(("history_saved", None))

    def post_preflight(self, result) -> None:
        self._q.put(("preflight", result))

    def post_preflight_error(self, exc) -> None:
        self._q.put(("preflight_error", exc))

    def post_surrogate_optimum(self, optimum) -> None:
        self._q.put(("surrogate_optimum", optimum))

    def drain(self) -> list[tuple]:
        msgs = []
        while True:
            try:
                msgs.append(self._q.get_nowait())
            except _queue_mod.Empty:
                break
        return msgs


def _clamp_int(value: int, lo: int, hi: int, name: str) -> int:
    """Validate integer is within [lo, hi]; raises ValueError if not."""
    if value < lo or value > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _resolve_optimum_weights(input, n_obj: int) -> list[float] | None:
    """Read the weights inputs if mode=='weighted', else return None.

    Returns None for the Pareto path. Returns a list of non-negative
    floats for the Weighted path. If any weight input is not yet
    rendered (SilentException, e.g. the @render.ui for weights hasn't
    flushed yet, or n_obj shrank mid-run), falls back to None rather
    than halting silently — matches the preflight_fix_N pattern at
    calibration_handlers.py:815-819.
    """
    from shiny.types import SilentException

    mode = getattr(input, "cal_optimum_mode", lambda: "pareto")()
    if mode != "weighted":
        return None
    weights: list[float] = []
    for i in range(n_obj):
        try:
            raw = getattr(input, f"cal_weight_{i}")()
        except SilentException:
            return None
        try:
            weights.append(float(raw or 0.0))
        except (TypeError, ValueError):
            return None
    return weights


def _clamp_n_workers(requested: int | None, cpu: int | None) -> int:
    """Clamp a user-supplied worker count into [1, max(1, cpu)].

    None or non-positive input → 1 (sequential, the library default).
    ``cpu`` is ``os.cpu_count()`` which returns None on some platforms;
    fall back to 1 in that case.
    """
    ceiling = max(1, cpu) if cpu else 1
    if requested is None:
        return 1
    try:
        n = int(requested)
    except (TypeError, ValueError):
        return 1
    if n < 1:
        return 1
    return min(n, ceiling)


def _make_progress_callback(
    cal_history_append,
    cancel_check,
    *,
    checkpoint_path: Path | None = None,
    phase: str = "unknown",
    param_keys: list[str] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    banded_residuals_accessor=None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
):
    """Create a pymoo callback that feeds cal_history AND writes a checkpoint.

    The existing in-memory chart feed is preserved (cal_history_append); the
    new write is additive and wrapped in try/except so a disk failure cannot
    regress the convergence chart. See spec §6 runner table (NSGA-II row).
    """
    from datetime import datetime, timezone
    import logging
    import time

    from pymoo.core.callback import Callback  # type: ignore[import-untyped]

    logger = logging.getLogger("osmose.ui.calibration_dashboard")
    state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": time.time(),
    }

    class _ProgressCallback(Callback):
        def __init__(self):
            super().__init__()

        def notify(self, algorithm):
            if cancel_check():
                algorithm.termination.force_termination = True
                return
            F = algorithm.opt.get("F")
            best = float(np.min(F.sum(axis=1)))
            cal_history_append(best)  # existing — MUST run first

            if checkpoint_path is None or param_keys is None or bounds is None:
                return

            state["gen"] += 1
            prior_best = state["best_fun_seen"]
            if best < prior_best:
                state["best_fun_seen"] = best
                state["gens_since_improvement"] = 0
            else:
                state["gens_since_improvement"] += 1

            # Path B: in-process single-threaded → no re-eval needed.
            if banded_residuals_accessor is None or banded_targets is None:
                residuals_tuple = None
            else:
                residuals_tuple = banded_residuals_accessor()

            if residuals_tuple is None:
                if banded_targets is None:
                    proxy_source = "objective_disabled"
                else:
                    proxy_source = "not_implemented"
                    logger.warning(
                        "NSGA-II checkpoint at gen %d: banded_residuals_accessor "
                        "returned None despite banded-loss being configured "
                        "(proxy_source=not_implemented; cause=accessor_returned_none).",
                        state["gen"],
                    )
                per_species_residuals = None
                per_species_sim_biomass = None
                species_labels = None
            else:
                species_labels, per_species_residuals, per_species_sim_biomass = residuals_tuple
                proxy_source = "banded_loss"

            best_x = algorithm.opt.get("X")[0]
            best_x_log10 = tuple(float(v) for v in best_x)

            try:
                from osmose.calibration.checkpoint import (
                    CalibrationCheckpoint,
                    write_checkpoint,
                )

                ckpt = CalibrationCheckpoint(
                    optimizer="nsga2",
                    phase=phase,
                    generation=state["gen"],
                    generation_budget=None,
                    best_fun=best,
                    per_species_residuals=per_species_residuals,
                    per_species_sim_biomass=per_species_sim_biomass,
                    species_labels=species_labels,
                    best_x_log10=best_x_log10,
                    best_parameters={
                        k: float(10.0 ** v) for k, v in zip(param_keys, best_x_log10)
                    },
                    param_keys=tuple(param_keys),
                    bounds_log10={
                        k: (float(lo), float(hi))
                        for k, (lo, hi) in zip(param_keys, bounds)
                    },
                    gens_since_improvement=state["gens_since_improvement"],
                    elapsed_seconds=time.time() - state["start_time"],
                    timestamp_iso=datetime.now(timezone.utc).isoformat(),
                    banded_targets=banded_targets,
                    proxy_source=proxy_source,
                )
                write_checkpoint(checkpoint_path, ckpt)
            except (OSError, TypeError, ValueError) as e:
                logger.warning(
                    "NSGA-II checkpoint write failed at gen %d: %s", state["gen"], e,
                )

    return _ProgressCallback()


def _save_run_for_nsga2(payload, X, F, phase: str, param_keys: list[str]) -> None:
    """Thin NSGA-II-side wrapper around _save_run_safe in history.py.

    Mirrors _save_run_for_de in scripts/calibrate_baltic.py.
    """
    from datetime import datetime, timezone
    import logging

    from osmose.calibration.history import _save_run_safe

    logger = logging.getLogger("osmose.ui.calibration_dashboard")
    best_idx = int(np.argmin(F.sum(axis=1)))
    best_F = float(F.sum(axis=1)[best_idx])
    best_x = X[best_idx]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "nsga2",
        "phase": phase,
        "parameters": list(param_keys),
        "results": {
            "best_objective": best_F,
            "best_parameters": {k: float(v) for k, v in zip(param_keys, best_x)},
            "duration_seconds": 0.0,
            "n_evaluations": int(getattr(X, "shape", [0])[0]),
            "per_species_residuals_final": None,
            "per_species_sim_biomass_final": None,
            "species_labels": None,
        },
    }
    _save_run_safe(record, logger=logger, with_fallback=True)


def get_calibratable_params(registry: ParameterRegistry, n_species: int) -> list[dict]:
    """Get all numeric species-indexed params suitable for calibration."""
    params = []
    for field in registry.all_fields():
        if not field.indexed:
            continue
        if field.param_type not in (ParamType.FLOAT, ParamType.INT):
            continue
        if field.min_val is None or field.max_val is None:
            continue
        for i in range(n_species):
            key = field.resolve_key(i)
            params.append(
                {
                    "key": key,
                    "label": f"{field.description} (sp{i})",
                    "category": field.category,
                    "lower": field.min_val,
                    "upper": field.max_val,
                }
            )
    return params


def collect_selected_params(
    input: object, state, config: dict[str, str] | None = None
) -> list[dict]:
    """Return calibratable param dicts where the checkbox is checked.

    Args:
        config: If provided, use this instead of reading state.config (for thread safety).
    """
    cfg = config if config is not None else state.config.get()
    try:
        n_species = int(float(cfg.get("simulation.nspecies", "3") or "3"))
    except (ValueError, TypeError):
        n_species = 3
    all_params = get_calibratable_params(state.registry, n_species)
    selected = []
    for p in all_params:
        input_id = f"cal_param_{p['key'].replace('.', '_')}"
        try:
            val = getattr(input, input_id)
        except AttributeError:
            continue
        try:
            if val():
                selected.append(p)
        except (SilentException, AttributeError):
            continue
    return selected


def build_free_params(selected: list[dict]) -> list:
    """Convert selected param dicts to FreeParameter objects."""
    from osmose.calibration.problem import FreeParameter

    return [
        FreeParameter(key=p["key"], lower_bound=p["lower"], upper_bound=p["upper"])
        for p in selected
    ]


def _extract_species_stats(results, species_names: list[str], n_eval_years: int = 10) -> dict:
    """Extract mean/cv/trend per species from simulation results."""
    bio = results.biomass()
    total_years = len(bio)
    eval_data = bio.iloc[-n_eval_years:] if total_years > n_eval_years else bio

    stats: dict[str, float] = {}
    for sp in species_names:
        if sp not in eval_data.columns:
            continue
        vals = eval_data[sp].values.astype(float)
        mean_val = float(np.mean(vals))
        stats[f"{sp}_mean"] = mean_val

        if mean_val > 0:
            stats[f"{sp}_cv"] = float(np.std(vals) / mean_val)
        else:
            stats[f"{sp}_cv"] = 10.0

        if len(vals) >= 3:
            x = np.arange(len(vals), dtype=float)
            slope = np.polyfit(x, vals, 1)[0]
            stats[f"{sp}_trend"] = float(abs(slope) / (mean_val + 1.0))
        else:
            stats[f"{sp}_trend"] = 0.0

    return stats


def build_preflight_modal(result_or_error):
    """Dispatch on payload: PreflightEvalError → red-banner modal;
    anything else → existing success-shape modal."""
    from osmose.calibration.preflight import PreflightEvalError

    if isinstance(result_or_error, PreflightEvalError):
        return _build_preflight_error_modal(result_or_error)
    return _build_preflight_success_modal(result_or_error)


def _build_preflight_error_modal(exc):
    """Render a red-banner modal for PreflightEvalError. Dismissal-only —
    no retry button in this iteration."""
    stage = getattr(exc, "stage", "unknown")
    body = ui.div(
        ui.div(
            ui.tags.strong(f"Preflight failed — {stage} stage"),
            ui.br(),
            ui.tags.span(str(exc)),
            class_="alert alert-danger",
            role="alert",
        ),
        ui.p(
            "Recovery: try narrowing parameter bounds, reducing the Morris "
            "sample budget, or re-running with n_workers=1 to see the "
            "underlying sample-evaluation exception.",
            class_="mt-3",
        ),
    )
    footer = ui.tags.button("Close", class_="btn btn-secondary", **{"data-bs-dismiss": "modal"})
    return ui.modal(
        body,
        title="Pre-flight Screening Failed",
        footer=footer,
        easy_close=True,
        size="l",
    )


def _build_preflight_success_modal(result):
    """Build a modal dialog showing pre-flight issues with fix checkboxes.

    Returns a ``ui.modal()`` if there are issues, or ``None`` if the result is clean.
    Auto-fixable issues get checkboxes; non-fixable issues are shown as text.
    """
    if not result.issues:
        return None

    body_items = []
    body_items.append(
        ui.p(
            f"Pre-flight screening completed in {result.elapsed_seconds:.1f}s. "
            f"{len(result.survivors)} of {len(result.screening)} parameters survived."
        )
    )

    fixable_idx = 0
    for issue in result.issues:
        severity_class = "text-danger" if issue.severity.value == "error" else "text-warning"
        if issue.auto_fixable:
            body_items.append(
                ui.div(
                    ui.input_checkbox(
                        f"preflight_fix_{fixable_idx}",
                        ui.span(
                            ui.tags.strong(issue.message, class_=severity_class),
                            ui.br(),
                            ui.tags.em(issue.suggestion),
                        ),
                        value=True,
                    ),
                    class_="mb-2",
                )
            )
            fixable_idx += 1
        else:
            body_items.append(
                ui.div(
                    ui.tags.strong(issue.message, class_=severity_class),
                    ui.br(),
                    ui.tags.em(issue.suggestion),
                    class_="mb-2",
                )
            )

    footer = ui.div(
        ui.input_action_button(
            "btn_preflight_apply", "Apply Selected & Start", class_="btn-success me-2"
        ),
        ui.tags.button("Cancel", class_="btn btn-secondary", **{"data-bs-dismiss": "modal"}),
    )

    return ui.modal(
        *body_items,
        title="Pre-flight Screening Results",
        footer=footer,
        easy_close=True,
        size="l",
    )


def apply_preflight_fixes(free_params, issues, checked):
    """Apply selected pre-flight fixes to the free parameter list.

    - NEGLIGIBLE: remove the parameter entirely.
    - BOUND_TIGHT: widen bounds by 10% of span on each side.
    - BLOWUP: tighten bounds by 10% of span on each side.

    Args:
        free_params: List of FreeParameter objects.
        issues: List of auto-fixable PreflightIssue objects.
        checked: List of booleans indicating which fixes are selected.

    Returns:
        Updated list of FreeParameter objects.
    """
    from osmose.calibration.preflight import IssueCategory
    from osmose.calibration.problem import FreeParameter

    # Build a map of param_key -> list of (issue, is_checked)
    removals: set[str] = set()
    bound_adjustments: dict[str, list[tuple]] = {}

    for issue, is_checked in zip(issues, checked):
        if not is_checked:
            continue
        if issue.category is IssueCategory.NEGLIGIBLE and issue.param_key:
            removals.add(issue.param_key)
        elif issue.category is IssueCategory.BOUND_TIGHT and issue.param_key:
            bound_adjustments.setdefault(issue.param_key, []).append(("widen",))
        elif issue.category is IssueCategory.BLOWUP and issue.param_key:
            bound_adjustments.setdefault(issue.param_key, []).append(("tighten",))

    result = []
    for fp in free_params:
        if fp.key in removals:
            continue
        if fp.key in bound_adjustments:
            lo, hi = fp.lower_bound, fp.upper_bound
            span = hi - lo
            for (action,) in bound_adjustments[fp.key]:
                if action == "widen":
                    lo -= 0.1 * span
                    hi += 0.1 * span
                elif action == "tighten":
                    lo += 0.1 * span
                    hi -= 0.1 * span
            if lo >= hi:
                # Safety: don't create invalid bounds, keep original
                result.append(fp)
            else:
                result.append(
                    FreeParameter(
                        key=fp.key, lower_bound=lo, upper_bound=hi, transform=fp.transform
                    )
                )
        else:
            result.append(fp)

    return result


def register_calibration_handlers(
    input,
    output,
    session,
    state,
    cal_history,
    cal_F,
    cal_X,
    sensitivity_result,
    cal_thread,
    surrogate_status,
    copy_data_files,
    validation_result,
    cal_param_names,
    preflight_result,
    history_banner_text,
    history_trigger,
    surrogate_optimum,
):
    """Register all reactive event handlers for the calibration page."""

    msg_queue = CalibrationMessageQueue()
    cancel_event = threading.Event()

    @reactive.poll(_scan_signature, interval_secs=1.0)
    def _live_snapshot() -> LiveSnapshot:
        return _scan_results_dir()


    @output
    @render.ui
    def run_header():
        snap = _live_snapshot()
        if snap.active.kind == "no_run":
            return ui.tags.div("No active calibration run", class_="text-muted small")
        if snap.active.kind == "corrupt":
            return ui.tags.div(
                f"Checkpoint unreadable ({html.escape(snap.active.error_summary or '')})",
                class_="alert alert-danger",
            )
        if snap.active.kind == "partial":
            return ui.tags.div("Checkpoint updating…", class_="text-warning small")
        ckpt = snap.active.checkpoint
        assert ckpt is not None

        elapsed_str = _format_elapsed(ckpt.elapsed_seconds)
        gen_str = (
            f"gen {ckpt.generation} / {ckpt.generation_budget}"
            if ckpt.generation_budget else f"gen {ckpt.generation}"
        )
        from osmose.calibration.checkpoint import liveness_state
        age = time.time() - _ckpt_mtime_for(snap)
        state_text = liveness_state(age)
        state_dot = "●" if state_text in ("live", "stalled") else "○"
        patience = (
            f"⏱ patience {ckpt.gens_since_improvement}"
            if ckpt.gens_since_improvement > 0 else ""
        )
        return ui.tags.div(
            ui.tags.div(
                f"{ckpt.optimizer.upper()} · phase {html.escape(ckpt.phase)}  |  "
                f"{gen_str}  |  elapsed {elapsed_str}",
                class_="fw-bold",
            ),
            ui.tags.div(
                f"{patience}   {state_dot} {state_text} (last update {int(age)}s ago)",
                class_="small text-muted",
                **{"aria-live": "polite", "aria-atomic": "false"},
            ),
            class_="run-header mb-2",
        )

    @output
    @render.ui
    def ices_proxy_table():
        snap = _live_snapshot()
        if snap.active.kind != "ok":
            return ui.tags.div()
        ckpt = snap.active.checkpoint
        rows = _build_proxy_rows(ckpt)

        if rows and rows[0]["state"] == "objective_disabled":
            return ui.tags.div(
                "ICES proxy unavailable: this run does not use banded-loss objectives. "
                "Authoritative verdict will appear in Results tab on completion.",
                class_="alert alert-info small",
            )
        if rows and rows[0]["state"] == "not_implemented":
            return ui.tags.div(
                "ICES proxy: per-species residuals were not exposed by losses.py "
                "despite banded-loss being configured. This is a bug — please file an "
                "issue and include the checkpoint filename.",
                class_="alert alert-danger small",
            )

        table_rows = []
        n_in, n_out, n_na = 0, 0, 0
        for r in rows:
            if r["state"] == "in_range":
                badge, n_in = "✓", n_in + 1
                mag_text = f"≈{r['magnitude']:.2f}×"
            elif r["state"] == "out_of_range":
                badge, n_out = "✗", n_out + 1
                mag_text = f"{r['magnitude']:.2f}× {r['direction']}"
            elif r["state"] == "extinct":
                badge, n_out = "☠", n_out + 1
                mag_text = "extinct"
            else:
                badge, n_na = "—", n_na + 1
                mag_text = ""
            table_rows.append(ui.tags.tr(
                ui.tags.td(html.escape(r["species"])),
                ui.tags.td(f"loss {r['loss']:.2f}"),
                ui.tags.td(f"band [{r['band'][0]:.2f}, {r['band'][1]:.2f}]"),
                ui.tags.td(badge, **{"aria-label": _aria_for_state(r['state'], r['magnitude'], r['direction'])}),
                ui.tags.td(mag_text),
            ))
        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("species"), ui.tags.th("loss"), ui.tags.th("band"),
                    ui.tags.th(""), ui.tags.th("magnitude"),
                )),
                ui.tags.tbody(*table_rows),
                class_="table table-sm",
            ),
            ui.tags.div(
                f"{n_in}/{n_in + n_out + n_na} in-band (proxy) · {n_out} out · {n_na} n/a · "
                "authoritative ICES verdict appears in Results tab after completion.",
                class_="small text-muted",
            ),
        )

    @reactive.poll(lambda: time.time(), interval_secs=0.5)
    def _poll_cal_messages():
        msgs = msg_queue.drain()
        for kind, payload in msgs:
            if kind == "status":
                surrogate_status.set(payload)
            elif kind == "history_append":
                current = cal_history.get()
                cal_history.set(current + [payload])
            elif kind == "results":
                X, F = payload
                cal_X.set(X)
                cal_F.set(F)
                # Scope note: phase and param_keys are NOT in scope at this
                # callsite. NSGA-II UI runs use a fixed phase string. param_keys
                # comes from cal_param_names (register_calibration_handlers closure parameter).
                try:
                    _save_run_for_nsga2(
                        payload, X, F,
                        phase="ui_nsga2",
                        param_keys=cal_param_names.get() or [],
                    )
                except Exception as e:  # noqa: BLE001
                    surrogate_status.set(f"history persist failed: {e}")
            elif kind == "error":
                surrogate_status.set(f"Failed: {payload}")
                ui.notification_show(f"Calibration error: {payload}", type="error", duration=15)
            elif kind == "sensitivity":
                sensitivity_result.set(payload)
            elif kind == "history_saved":
                history_trigger.set(history_trigger.get() + 1)
            elif kind == "validation":
                validation_result.set(payload)
            elif kind == "preflight":
                preflight_result.set(payload)
                modal = build_preflight_modal(payload)
                if modal is None:
                    surrogate_status.set("Pre-flight passed — starting calibration...")
                    selected = collect_selected_params(input, state)
                    fp = build_free_params(selected)
                    _start_optimization_with_params(fp)
                else:
                    ui.modal_show(modal)
            elif kind == "preflight_error":
                # build_preflight_modal dispatches on payload type → error helper
                modal = build_preflight_modal(payload)
                if modal is not None:
                    ui.modal_show(modal)
            elif kind == "surrogate_optimum":
                surrogate_optimum.set(payload)

    @reactive.effect
    def _consume_cal_poll():
        _poll_cal_messages()

    # Shared state for preflight -> optimization handoff (nonlocal across handlers)
    _shared_objective_fns = []
    _shared_obj_names = []
    _shared_obs_bio = None
    _shared_obs_diet = None
    _shared_banded_enabled = False
    _shared_base_config: Path | None = None
    _shared_jar_path: Path | None = None
    _shared_work_dir: Path | None = None
    _shared_current_config = None
    _shared_n_parallel = 4
    _shared_algorithm_choice = "nsga2"
    _shared_pop_size = 50
    _shared_generations = 100
    _shared_banded_residuals_accessor = None
    _shared_banded_targets_dict: dict[str, tuple[float, float]] | None = None

    def _start_optimization_with_params(free_params):
        """Launch the optimization thread (surrogate or NSGA-II) with given params.

        All shared state (_shared_*) must be set before calling this.
        """
        nonlocal _shared_objective_fns, _shared_obj_names, _shared_obs_bio
        nonlocal _shared_obs_diet, _shared_banded_enabled, _shared_base_config
        nonlocal _shared_jar_path, _shared_work_dir, _shared_n_parallel
        nonlocal _shared_algorithm_choice, _shared_pop_size, _shared_generations
        nonlocal _shared_banded_residuals_accessor, _shared_banded_targets_dict

        from osmose.calibration.problem import OsmoseCalibrationProblem

        objective_fns = _shared_objective_fns
        obj_names = list(_shared_obj_names)
        obs_bio = _shared_obs_bio
        obs_diet = _shared_obs_diet
        banded_enabled = _shared_banded_enabled
        base_config, jar_path, work_dir = _require_preflight(
            _shared_base_config, _shared_jar_path, _shared_work_dir
        )
        n_parallel = _shared_n_parallel
        algorithm_choice = _shared_algorithm_choice
        pop_size = _shared_pop_size
        generations = _shared_generations

        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=objective_fns,
            base_config_path=base_config,
            jar_path=jar_path,
            work_dir=work_dir,
            n_parallel=n_parallel,
        )

        cancel_event.clear()
        cal_history.set([])
        cal_F.set(None)
        cal_X.set(None)
        surrogate_status.set("")

        if algorithm_choice == "surrogate":

            def run_surrogate():
                try:
                    import time as _time

                    from osmose.calibration.surrogate import SurrogateCalibrator

                    _t0 = _time.time()
                    bounds = [(fp.lower_bound, fp.upper_bound) for fp in free_params]
                    n_obj = len(objective_fns)
                    calibrator = SurrogateCalibrator(param_bounds=bounds, n_objectives=n_obj)

                    n_samples = pop_size
                    msg_queue.post_status(f"Generating {n_samples} Latin hypercube samples...")
                    samples = calibrator.generate_samples(n_samples=n_samples)

                    Y = np.zeros((n_samples, n_obj))
                    for idx in range(n_samples):
                        if cancel_event.is_set():
                            msg_queue.post_status("Cancelled.")
                            return

                        msg_queue.post_status(f"Evaluating sample {idx + 1}/{n_samples}...")
                        overrides = {
                            fp.key: str(samples[idx, j]) for j, fp in enumerate(free_params)
                        }
                        try:
                            result = problem._run_single(overrides, run_id=idx)
                            for k in range(n_obj):
                                Y[idx, k] = result[k]
                        except (
                            subprocess.TimeoutExpired,
                            subprocess.CalledProcessError,
                            FileNotFoundError,
                            OSError,
                        ) as exc:
                            _log.error("Surrogate sample %d/%d failed: %s", idx + 1, n_samples, exc)
                            Y[idx, :] = float("inf")
                            msg_queue.post_status(f"Sample {idx + 1}/{n_samples} failed: {exc}")

                    if cancel_event.is_set():
                        msg_queue.post_status("Cancelled.")
                        return

                    msg_queue.post_status("Fitting GP model...")
                    calibrator.fit(samples, Y)

                    msg_queue.post_status("Finding optimum on surrogate...")
                    n_obj = calibrator.surrogate.n_objectives
                    weights = _resolve_optimum_weights(input, n_obj)
                    if weights is not None:
                        optimum = calibrator.find_optimum(weights=weights)
                    else:
                        optimum = calibrator.find_optimum()  # weights=None → Pareto
                    msg_queue.post_surrogate_optimum(optimum)

                    msg_queue.post_results(X=samples, F=Y)
                    history = [float(np.min(Y[: i + 1].sum(axis=1))) for i in range(n_samples)]
                    for val in history:
                        msg_queue.post_history_append(val)
                    msg_queue.post_status(
                        f"Done. Best predicted objective: {optimum['predicted_objectives']}"
                    )

                    from osmose.calibration.history import save_run

                    save_run(
                        {
                            "version": 1,
                            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "algorithm": "surrogate",
                            "settings": {
                                "population_size": pop_size,
                                "generations": 0,
                                "n_parallel": n_parallel,
                            },
                            "parameters": [
                                {
                                    "key": fp.key,
                                    "lower": fp.lower_bound,
                                    "upper": fp.upper_bound,
                                }
                                for fp in free_params
                            ],
                            "objectives": {
                                "biomass_rmse": bool(obs_bio),
                                "diet_distance": bool(obs_diet),
                                "banded_loss": {"enabled": banded_enabled},
                            },
                            "results": {
                                "best_objective": float(np.min(Y.sum(axis=1))),
                                "n_evaluations": n_samples,
                                "duration_seconds": int(_time.time() - _t0),
                                "objective_names": obj_names,
                                "convergence": [[i, v] for i, v in enumerate(history)],
                                "pareto_X": samples.tolist(),
                                "pareto_F": Y.tolist(),
                            },
                        }
                    )
                    msg_queue.post_history_saved()
                except Exception as exc:
                    _log.error("Surrogate calibration failed: %s", exc, exc_info=True)
                    msg_queue.post_error(f"Surrogate calibration failed: {exc}")

            thread = threading.Thread(target=run_surrogate, daemon=True)
            thread.start()
            cal_thread.set(thread)

        else:
            # NSGA-II (default)
            def run_optimization():
                try:
                    import time as _time

                    from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore[import-untyped]

                    _t0 = _time.time()
                    from pymoo.optimize import minimize  # type: ignore[import-untyped]
                    from pymoo.termination import get_termination  # type: ignore[import-untyped]

                    algorithm = NSGA2(pop_size=pop_size)
                    termination = get_termination("n_gen", generations)

                    _local_convergence: list[float] = []

                    def _tracked_append(val: float) -> None:
                        _local_convergence.append(val)
                        msg_queue.post_history_append(val)

                    callback = _make_progress_callback(
                        cal_history_append=_tracked_append,
                        cancel_check=cancel_event.is_set,
                        # NEW dashboard wiring:
                        checkpoint_path=RESULTS_DIR / "phase_ui_nsga2_checkpoint.json",
                        phase="ui_nsga2",
                        param_keys=[fp.key for fp in free_params],
                        bounds=[(fp.lower_bound, fp.upper_bound) for fp in free_params],
                        banded_residuals_accessor=_shared_banded_residuals_accessor,
                        banded_targets=_shared_banded_targets_dict,
                    )

                    res = minimize(
                        problem,
                        algorithm,
                        termination,
                        seed=42,
                        verbose=False,
                        callback=callback,
                    )

                    if cancel_event.is_set():
                        msg_queue.post_status("Cancelled.")
                    elif res.F is not None:
                        msg_queue.post_results(X=res.X, F=res.F)

                        import time as _time
                        from osmose.calibration.history import save_run

                        save_run(
                            {
                                "version": 1,
                                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "algorithm": "nsga2",
                                "settings": {
                                    "population_size": pop_size,
                                    "generations": generations,
                                    "n_parallel": n_parallel,
                                },
                                "parameters": [
                                    {
                                        "key": fp.key,
                                        "lower": fp.lower_bound,
                                        "upper": fp.upper_bound,
                                    }
                                    for fp in free_params
                                ],
                                "objectives": {
                                    "biomass_rmse": bool(obs_bio),
                                    "diet_distance": bool(obs_diet),
                                    "banded_loss": {"enabled": banded_enabled},
                                },
                                "results": {
                                    "best_objective": float(np.min(res.F.sum(axis=1))),
                                    "n_evaluations": pop_size * generations,
                                    "duration_seconds": int(_time.time() - _t0),
                                    "objective_names": obj_names,
                                    "convergence": [
                                        [i, v] for i, v in enumerate(_local_convergence)
                                    ],
                                    "pareto_X": res.X.tolist(),
                                    "pareto_F": res.F.tolist(),
                                },
                            }
                        )
                        msg_queue.post_history_saved()
                except Exception as exc:
                    _log.error("Calibration failed: %s", exc, exc_info=True)
                    msg_queue.post_error(str(exc))

            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            cal_thread.set(thread)

    @reactive.effect
    @reactive.event(input.btn_start_cal)
    def handle_start_cal():
        nonlocal _shared_objective_fns, _shared_obj_names, _shared_obs_bio
        nonlocal _shared_obs_diet, _shared_banded_enabled, _shared_base_config
        nonlocal _shared_jar_path, _shared_work_dir, _shared_current_config
        nonlocal _shared_n_parallel, _shared_algorithm_choice
        nonlocal _shared_pop_size, _shared_generations
        nonlocal _shared_banded_residuals_accessor, _shared_banded_targets_dict

        selected = collect_selected_params(input, state)
        if not selected:
            cal_history.set([])
            ui.notification_show(
                "Select at least one parameter to calibrate.", type="warning", duration=5
            )
            return

        # JAR is only needed for the opt-in Java engine backend; the Python
        # engine (default) runs entirely in-process. Resolve the state value
        # to an existing Path or None so downstream callers don't assume a
        # JAR exists.
        _raw_jar = Path(state.jar_path.get())
        jar_path = _raw_jar if _raw_jar.exists() else None

        obs_bio = input.observed_biomass()
        obs_diet = input.observed_diet()
        banded_check = False
        try:
            banded_check = bool(input.cal_banded_loss_enabled())
        except (SilentException, AttributeError):
            pass
        if not obs_bio and not obs_diet and not banded_check:
            ui.notification_show(
                "Upload observed data or enable banded loss objective.",
                type="warning",
                duration=5,
            )
            return

        import pandas as pd

        from osmose.calibration.objectives import biomass_rmse, diet_distance
        from osmose.config.writer import OsmoseConfigWriter

        # Reset banded-loss shared state — stale values from a prior run would
        # cause the NSGA-II checkpoint to write stale residuals/targets.
        _shared_banded_residuals_accessor = None
        _shared_banded_targets_dict = None

        free_params = build_free_params(selected)
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_cal_"))

        writer = OsmoseConfigWriter()
        config_dir = work_dir / "config"
        with reactive.isolate():
            current_config = state.config.get()
            source_dir = state.config_dir.get()
            case_map = state.key_case_map.get()
        writer.write(current_config, config_dir, key_case_map=case_map)
        if source_dir and source_dir.is_dir():
            copy_data_files(current_config, source_dir, config_dir)
        base_config = config_dir / "osm_all-parameters.csv"

        objective_fns = []
        if obs_bio:
            obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])
            objective_fns.append(lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df))
        if obs_diet:
            obs_diet_df = pd.read_csv(obs_diet[0]["datapath"])
            objective_fns.append(lambda r, df=obs_diet_df: diet_distance(r.diet_matrix(), df))

        obj_names = []
        if obs_bio:
            obj_names.append("Biomass RMSE")
        if obs_diet:
            obj_names.append("Diet Distance")

        banded_enabled = False
        try:
            banded_enabled = bool(input.cal_banded_loss_enabled())
        except (SilentException, AttributeError):
            pass

        if banded_enabled:
            from osmose.calibration.losses import make_banded_objective
            from osmose.calibration.targets import load_targets

            banded_source = input.cal_banded_source()
            if banded_source == "baltic":
                targets_path = Path("data/baltic/reference/biomass_targets.csv")
            else:
                banded_file = input.cal_banded_targets_file()
                if not banded_file:
                    ui.notification_show(
                        "Upload a targets CSV or select Baltic defaults.",
                        type="warning",
                        duration=5,
                    )
                    return
                targets_path = Path(banded_file[0]["datapath"])

            targets, _ = load_targets(targets_path)
            species_names = [t.species for t in targets]
            w_stability = float(input.cal_w_stability())
            w_worst = float(input.cal_w_worst())
            banded_obj, banded_residuals_accessor = make_banded_objective(
                targets, species_names, w_stability=w_stability, w_worst=w_worst
            )
            _shared_banded_residuals_accessor = banded_residuals_accessor
            _shared_banded_targets_dict = {
                t.species: (t.lower, t.upper) for t in targets
            }

            def banded_objective_fn(results, _banded=banded_obj, _sp=species_names):
                stats = _extract_species_stats(results, _sp)
                return _banded(stats)

            objective_fns.append(banded_objective_fn)
            obj_names.append("Banded Loss")

        if not objective_fns:
            ui.notification_show(
                "Enable at least one objective (upload data or enable banded loss).",
                type="warning",
                duration=5,
            )
            return

        # Store param names for charts
        cal_param_names.set([p["key"].split(".")[-1] for p in selected])

        n_parallel = _clamp_int(int(input.cal_n_parallel()), 1, 32, "n_parallel")

        # Store shared state for preflight -> optimization handoff
        _shared_objective_fns = objective_fns
        _shared_obj_names = obj_names
        _shared_obs_bio = obs_bio
        _shared_obs_diet = obs_diet
        _shared_banded_enabled = banded_enabled
        _shared_base_config = base_config
        _shared_jar_path = jar_path
        _shared_work_dir = work_dir
        _shared_current_config = current_config
        _shared_n_parallel = n_parallel
        _shared_algorithm_choice = input.cal_algorithm()
        _shared_pop_size = _clamp_int(int(input.cal_pop_size()), 10, 500, "pop_size")
        _shared_generations = _clamp_int(int(input.cal_generations()), 10, 1000, "generations")

        # Check if preflight is enabled
        preflight_enabled = False
        try:
            preflight_enabled = bool(input.cal_preflight_enabled())
        except (SilentException, AttributeError):
            pass

        if not preflight_enabled:
            _start_optimization_with_params(free_params)
            return

        # Run preflight in background thread
        surrogate_status.set("Running pre-flight screening...")
        param_names = [fp.key for fp in free_params]
        param_bounds = [(fp.lower_bound, fp.upper_bound) for fp in free_params]

        from osmose.calibration.preflight import (
            PreflightEvalError,
            make_preflight_eval_fn,
            run_preflight,
        )

        raw_workers = getattr(input, "cal_preflight_workers", lambda: 1)()
        n_workers = _clamp_n_workers(raw_workers, os.cpu_count())

        eval_fn = make_preflight_eval_fn(
            free_params=free_params,
            base_config=dict(current_config),
            output_dir=work_dir / "preflight_output",
            objective_fns=objective_fns,
            n_workers=n_workers,
        )

        def _run_preflight_thread():
            try:
                result = run_preflight(
                    param_names=param_names,
                    param_bounds=param_bounds,
                    evaluation_fn=eval_fn,
                    cancel_event=cancel_event,
                )
                msg_queue.post_preflight(result)
            except PreflightEvalError as exc:
                msg_queue.post_preflight_error(exc)
            except Exception as exc:
                _log.error("Pre-flight screening failed: %s", exc, exc_info=True)
                msg_queue.post_error(f"Pre-flight screening failed: {exc}")

        cancel_event.clear()
        thread = threading.Thread(target=_run_preflight_thread, daemon=True)
        thread.start()
        cal_thread.set(thread)

    @reactive.effect
    @reactive.event(input.btn_preflight_apply)
    def handle_preflight_apply():
        result = preflight_result.get()
        if result is None:
            return
        ui.modal_remove()
        selected = collect_selected_params(input, state)
        free_params = build_free_params(selected)
        fixable_issues = [i for i in result.issues if i.auto_fixable]
        checked = []
        for idx in range(len(fixable_issues)):
            try:
                checked.append(bool(getattr(input, f"preflight_fix_{idx}")()))
            except (SilentException, AttributeError):
                checked.append(False)
        updated_params = apply_preflight_fixes(free_params, fixable_issues, checked)
        if not updated_params:
            ui.notification_show("All parameters removed.", type="warning", duration=5)
            return
        cal_param_names.set([fp.key.split(".")[-1] for fp in updated_params])
        _start_optimization_with_params(updated_params)

    @reactive.effect
    @reactive.event(input.btn_stop_cal)
    def handle_stop_cal():
        cancel_event.set()

    @reactive.effect
    @reactive.event(input.btn_validate)
    def handle_validate():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            ui.notification_show("Run calibration first.", type="warning", duration=5)
            return

        top_n = int(input.cal_val_top_n())
        n_seeds = int(input.cal_val_seeds())
        seeds = list(range(n_seeds))

        order = np.argsort(F.sum(axis=1))[:top_n]
        candidates = X[order]

        selected = collect_selected_params(input, state)
        param_keys = [p["key"] for p in selected]

        # JAR is only required by the opt-in Java engine; fall back to None
        # when the configured path is missing so validation works on the
        # default Python engine.
        _raw_jar = Path(state.jar_path.get())
        jar_path = _raw_jar if _raw_jar.exists() else None
        with reactive.isolate():
            current_config = state.config.get()
            source_dir = state.config_dir.get()
            case_map = state.key_case_map.get()

        from osmose.calibration.problem import OsmoseCalibrationProblem
        from osmose.config.writer import OsmoseConfigWriter

        work_dir = Path(tempfile.mkdtemp(prefix="osmose_val_"))
        writer = OsmoseConfigWriter()
        config_dir = work_dir / "config"
        writer.write(current_config, config_dir, key_case_map=case_map)
        if source_dir and source_dir.is_dir():
            copy_data_files(current_config, source_dir, config_dir)
        base_config = config_dir / "osm_all-parameters.csv"

        free_params = build_free_params(selected)
        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=[lambda r: 0.0],
            base_config_path=base_config,
            jar_path=jar_path,
            work_dir=work_dir,
        )

        validation_result.set(None)
        surrogate_status.set("Validating candidates...")

        def run_validation():
            try:
                from osmose.calibration.multiseed import rank_candidates_multiseed

                _counter = [0]

                def make_factory(seed):
                    def objective(x):
                        overrides = {
                            param_keys[j]: str(float(x[j])) for j in range(len(param_keys))
                        }
                        overrides["simulation.random.seed"] = str(seed)
                        run_id = _counter[0]
                        _counter[0] += 1
                        obj_values = problem._run_single(overrides, run_id=run_id)
                        problem.cleanup_run(run_id)
                        return sum(obj_values)

                    return objective

                result = rank_candidates_multiseed(make_factory, candidates, seeds=seeds)
                msg_queue.post_status("")
                msg_queue.post_validation(result)
            except Exception as exc:
                _log.error("Validation failed: %s", exc, exc_info=True)
                msg_queue.post_error(f"Validation: {exc}")

        thread = threading.Thread(target=run_validation, daemon=True)
        thread.start()

    @reactive.effect
    def _handle_history_buttons():
        """Watch for dynamically created history load/delete buttons.

        Gated on: (1) History tab must be active, (2) history_trigger must have
        changed (save or delete). This avoids disk I/O on every 0.5s poll cycle.
        """
        # Only run when History tab is selected
        try:
            if input.cal_groups() != "History":
                return
        except (SilentException, AttributeError):
            return

        # Depend on the trigger so we re-read after save/delete
        history_trigger.get()

        from osmose.calibration.history import delete_run, list_runs, load_run

        runs = list_runs()
        for i, run in enumerate(runs):
            load_id = f"btn_load_run_{i}"
            delete_id = f"btn_delete_run_{i}"
            try:
                if getattr(input, load_id)():
                    data = load_run(Path(run["path"]))
                    cal_X.set(np.array(data["results"]["pareto_X"]))
                    cal_F.set(np.array(data["results"]["pareto_F"]))
                    cal_param_names.set([p["key"].split(".")[-1] for p in data["parameters"]])
                    conv = data["results"].get("convergence", [])
                    cal_history.set([v for _, v in conv])
                    history_banner_text.set(f"Viewing historical run from {data['timestamp']}")
                    validation_result.set(None)
            except (SilentException, AttributeError):
                pass
            try:
                if getattr(input, delete_id)():
                    delete_run(Path(run["path"]))
                    history_trigger.set(history_trigger.get() + 1)
                    ui.notification_show("Run deleted.", type="message", duration=3)
            except (SilentException, AttributeError):
                pass

    @reactive.effect
    @reactive.event(input.btn_sensitivity)
    def handle_sensitivity():
        import pandas as pd

        from osmose.calibration.objectives import biomass_rmse
        from osmose.calibration.problem import OsmoseCalibrationProblem
        from osmose.calibration.sensitivity import SensitivityAnalyzer
        from osmose.config.writer import OsmoseConfigWriter

        selected = collect_selected_params(input, state)
        if not selected:
            ui.notification_show(
                "Select at least one parameter for sensitivity.", type="warning", duration=5
            )
            return

        # Same rationale as handle_start_cal: Python engine is the default,
        # JAR is optional.
        _raw_jar = Path(state.jar_path.get())
        jar_path = _raw_jar if _raw_jar.exists() else None

        param_names = [p["key"] for p in selected]
        param_bounds = [(p["lower"], p["upper"]) for p in selected]
        analyzer = SensitivityAnalyzer(param_names, param_bounds)

        obs_bio = input.observed_biomass()
        if not obs_bio:
            ui.notification_show("Upload observed biomass CSV.", type="warning", duration=5)
            return
        obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])

        # Capture reactive values before spawning thread (thread safety)
        sens_config = state.config.get()
        sens_source_dir = state.config_dir.get()
        sens_case_map = state.key_case_map.get()

        def run_sensitivity():
            try:
                samples = analyzer.generate_samples(n_base=64)
                sens_work_dir = Path(tempfile.mkdtemp(prefix="osmose_sens_"))
                sens_writer = OsmoseConfigWriter()
                config_dir = sens_work_dir / "config"
                sens_writer.write(sens_config, config_dir, key_case_map=sens_case_map)
                if sens_source_dir and sens_source_dir.is_dir():
                    copy_data_files(sens_config, sens_source_dir, config_dir)
                base_config = config_dir / "osm_all-parameters.csv"

                objective_fns_sens = []
                obj_names_sens = []
                if obs_bio:
                    objective_fns_sens.append(
                        lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df)
                    )
                    obj_names_sens.append("Biomass RMSE")

                n_obj_sens = max(1, len(objective_fns_sens))
                Y = (
                    np.zeros((samples.shape[0], n_obj_sens))
                    if n_obj_sens > 1
                    else np.zeros(samples.shape[0])
                )

                prob = OsmoseCalibrationProblem(
                    free_params=build_free_params(selected),
                    objective_fns=objective_fns_sens or [lambda r: 0.0],
                    base_config_path=base_config,
                    jar_path=jar_path,
                    work_dir=sens_work_dir,
                )
                for idx, row in enumerate(samples):
                    overrides = {selected[j]["key"]: str(row[j]) for j in range(len(selected))}
                    try:
                        result = prob._run_single(overrides, run_id=idx)
                        if n_obj_sens > 1:
                            for k in range(n_obj_sens):
                                Y[idx, k] = result[k]
                        else:
                            Y[idx] = result[0]
                    except (
                        subprocess.TimeoutExpired,
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                        OSError,
                    ) as exc:
                        _log.warning("Sensitivity sample %d failed: %s", idx, exc)
                        if n_obj_sens > 1:
                            Y[idx, :] = float("inf")
                        else:
                            Y[idx] = float("inf")
                    finally:
                        prob.cleanup_run(idx)

                inf_mask = np.isinf(Y).any(axis=1) if Y.ndim > 1 else np.isinf(Y)
                n_inf = int(inf_mask.sum())
                if n_inf > len(Y) * 0.1:
                    msg_queue.post_error(
                        f"Sensitivity aborted: {n_inf}/{len(Y)} samples failed (>10% threshold)"
                    )
                    return

                Y_1d = Y.sum(axis=1) if Y.ndim > 1 else Y
                sens_result = analyzer.analyze(Y_1d)
                msg_queue.post_sensitivity(sens_result)
            except Exception as exc:
                _log.error("Sensitivity analysis failed: %s", exc, exc_info=True)
                msg_queue.post_error(f"Sensitivity: {exc}")

        thread = threading.Thread(target=run_sensitivity, daemon=True)
        thread.start()
