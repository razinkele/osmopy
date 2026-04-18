# ui/pages/calibration_handlers.py
"""Event handlers and helper functions for the calibration page."""

from __future__ import annotations

import os
import queue as _queue_mod
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from shiny import reactive, ui
from shiny.types import SilentException

from osmose.logging import setup_logging
from osmose.schema.base import ParamType
from osmose.schema.registry import ParameterRegistry

_log = setup_logging("osmose.calibration.ui")


def _require_preflight(
    base_config: Path | None,
    jar_path: Path | None,
    work_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """Return non-Optional paths or raise if preflight never ran."""
    if base_config is None or jar_path is None or work_dir is None:
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


def _make_progress_callback(cal_history_append, cancel_check):
    """Create a pymoo callback (lazy import to avoid loading pymoo at startup)."""
    from pymoo.core.callback import Callback  # type: ignore[import-untyped]

    class _ProgressCallback(Callback):
        def __init__(self):
            super().__init__()

        def notify(self, algorithm):
            if cancel_check():
                algorithm.termination.force_termination = True
                return
            F = algorithm.opt.get("F")
            best = float(np.min(F.sum(axis=1)))
            cal_history_append(best)

    return _ProgressCallback()


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

    def _start_optimization_with_params(free_params):
        """Launch the optimization thread (surrogate or NSGA-II) with given params.

        All shared state (_shared_*) must be set before calling this.
        """
        nonlocal _shared_objective_fns, _shared_obj_names, _shared_obs_bio
        nonlocal _shared_obs_diet, _shared_banded_enabled, _shared_base_config
        nonlocal _shared_jar_path, _shared_work_dir, _shared_n_parallel
        nonlocal _shared_algorithm_choice, _shared_pop_size, _shared_generations

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

        selected = collect_selected_params(input, state)
        if not selected:
            cal_history.set([])
            ui.notification_show(
                "Select at least one parameter to calibrate.", type="warning", duration=5
            )
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            ui.notification_show(f"JAR not found: {jar_path}", type="error", duration=15)
            return

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
            banded_obj = make_banded_objective(
                targets, species_names, w_stability=w_stability, w_worst=w_worst
            )

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

        jar_path = Path(state.jar_path.get())
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

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            ui.notification_show(f"JAR not found: {jar_path}", type="error", duration=15)
            return

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
