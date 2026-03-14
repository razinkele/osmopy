# ui/pages/calibration_handlers.py
"""Event handlers and helper functions for the calibration page."""

from __future__ import annotations

import queue as _queue_mod
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from shiny import reactive, ui

from osmose.logging import setup_logging
from osmose.schema.base import ParamType
from osmose.schema.registry import ParameterRegistry

_log = setup_logging("osmose.calibration.ui")


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

    def drain(self) -> list[tuple]:
        msgs = []
        while True:
            try:
                msgs.append(self._q.get_nowait())
            except _queue_mod.Empty:
                break
        return msgs


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
        except (TypeError, RuntimeError):
            continue
    return selected


def build_free_params(selected: list[dict]) -> list:
    """Convert selected param dicts to FreeParameter objects."""
    from osmose.calibration.problem import FreeParameter

    return [
        FreeParameter(key=p["key"], lower_bound=p["lower"], upper_bound=p["upper"])
        for p in selected
    ]


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
                ui.notification_show(
                    f"Calibration error: {payload}", type="error", duration=10
                )
            elif kind == "sensitivity":
                sensitivity_result.set(payload)

    @reactive.effect
    @reactive.event(input.btn_start_cal)
    def handle_start_cal():
        selected = collect_selected_params(input, state)
        if not selected:
            cal_history.set([])
            ui.notification_show("Select at least one parameter to calibrate.", type="warning")
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            ui.notification_show(f"JAR not found: {jar_path}", type="error")
            return

        obs_bio = input.observed_biomass()
        obs_diet = input.observed_diet()
        if not obs_bio and not obs_diet:
            ui.notification_show("Upload observed data (biomass or diet CSV).", type="warning")
            return

        import pandas as pd

        from osmose.calibration.objectives import biomass_rmse, diet_distance
        from osmose.calibration.problem import OsmoseCalibrationProblem
        from osmose.config.writer import OsmoseConfigWriter

        free_params = build_free_params(selected)
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_cal_"))

        writer = OsmoseConfigWriter()
        config_dir = work_dir / "config"
        writer.write(state.config.get(), config_dir)
        source_dir = state.config_dir.get()
        if source_dir and source_dir.is_dir():
            copy_data_files(state.config.get(), source_dir, config_dir)
        base_config = config_dir / "osm_all-parameters.csv"

        objective_fns = []
        if obs_bio:
            obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])
            objective_fns.append(lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df))
        if obs_diet:
            obs_diet_df = pd.read_csv(obs_diet[0]["datapath"])
            objective_fns.append(lambda r, df=obs_diet_df: diet_distance(r.diet_matrix(), df))

        if not objective_fns:
            return

        n_parallel = int(input.cal_n_parallel())

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

        algorithm_choice = input.cal_algorithm()
        pop_size = int(input.cal_pop_size())
        generations = int(input.cal_generations())

        if algorithm_choice == "surrogate":

            def run_surrogate():
                try:
                    from osmose.calibration.surrogate import SurrogateCalibrator

                    bounds = [(fp.lower_bound, fp.upper_bound) for fp in free_params]
                    n_obj = len(objective_fns)
                    calibrator = SurrogateCalibrator(param_bounds=bounds, n_objectives=n_obj)

                    n_samples = pop_size
                    msg_queue.post_status(f"Generating {n_samples} Latin hypercube samples...")
                    samples = calibrator.generate_samples(n_samples=n_samples)

                    # Evaluate OSMOSE for each sample
                    Y = np.zeros((n_samples, n_obj))
                    for idx in range(n_samples):
                        if cancel_event.is_set():
                            msg_queue.post_status("Cancelled.")
                            return

                        msg_queue.post_status(f"Evaluating sample {idx + 1}/{n_samples}...")
                        overrides = {fp.key: str(samples[idx, j]) for j, fp in enumerate(free_params)}
                        try:
                            result = problem._run_single(overrides, run_id=idx)
                            for k in range(n_obj):
                                Y[idx, k] = result[k]
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
                            _log.error("Surrogate sample %d/%d failed: %s", idx + 1, n_samples, exc)
                            Y[idx, :] = float("inf")
                            msg_queue.post_status(f"Sample {idx + 1}/{n_samples} failed: {exc}")

                    if cancel_event.is_set():
                        msg_queue.post_status("Cancelled.")
                        return

                    msg_queue.post_status("Fitting GP model...")
                    calibrator.fit(samples, Y)

                    msg_queue.post_status("Finding optimum on surrogate...")
                    optimum = calibrator.find_optimum()

                    # Set results for the UI
                    msg_queue.post_results(X=samples, F=Y)
                    history = [float(np.min(Y[: i + 1].sum(axis=1))) for i in range(n_samples)]
                    for val in history:
                        msg_queue.post_history_append(val)
                    msg_queue.post_status(
                        f"Done. Best predicted objective: {optimum['predicted_objectives']}"
                    )
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
                    from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore[import-untyped]
                    from pymoo.optimize import minimize  # type: ignore[import-untyped]
                    from pymoo.termination import get_termination  # type: ignore[import-untyped]

                    algorithm = NSGA2(pop_size=pop_size)
                    termination = get_termination("n_gen", generations)

                    callback = _make_progress_callback(
                        cal_history_append=msg_queue.post_history_append,
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
                except Exception as exc:
                    _log.error("Calibration failed: %s", exc, exc_info=True)
                    msg_queue.post_error(str(exc))

            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            cal_thread.set(thread)

    @reactive.effect
    @reactive.event(input.btn_stop_cal)
    def handle_stop_cal():
        cancel_event.set()

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
            ui.notification_show("Select at least one parameter for sensitivity.", type="warning")
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            ui.notification_show(f"JAR not found: {jar_path}", type="error")
            return

        param_names = [p["key"] for p in selected]
        param_bounds = [(p["lower"], p["upper"]) for p in selected]
        analyzer = SensitivityAnalyzer(param_names, param_bounds)

        obs_bio = input.observed_biomass()
        if not obs_bio:
            ui.notification_show("Upload observed biomass CSV.", type="warning")
            return
        obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])

        # Capture reactive values before spawning thread (thread safety)
        sens_config = state.config.get()
        sens_source_dir = state.config_dir.get()

        def run_sensitivity():
            try:
                samples = analyzer.generate_samples(n_base=64)
                sens_work_dir = Path(tempfile.mkdtemp(prefix="osmose_sens_"))
                sens_writer = OsmoseConfigWriter()
                config_dir = sens_work_dir / "config"
                sens_writer.write(sens_config, config_dir)
                if sens_source_dir and sens_source_dir.is_dir():
                    copy_data_files(sens_config, sens_source_dir, config_dir)
                base_config = config_dir / "osm_all-parameters.csv"

                Y = np.zeros(samples.shape[0])
                for idx, row in enumerate(samples):
                    overrides = {selected[j]["key"]: str(row[j]) for j in range(len(selected))}
                    try:
                        prob = OsmoseCalibrationProblem(
                            free_params=build_free_params(selected),
                            objective_fns=[lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df)],
                            base_config_path=base_config,
                            jar_path=jar_path,
                            work_dir=sens_work_dir / f"sens_{idx}",
                        )
                        result = prob._run_single(overrides, run_id=idx)
                        Y[idx] = result[0]
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
                        _log.warning("Sensitivity sample %d failed: %s", idx, exc)
                        Y[idx] = float("inf")

                n_inf = int(np.isinf(Y).sum())
                if n_inf > len(Y) * 0.1:
                    msg_queue.post_error(
                        f"Sensitivity aborted: {n_inf}/{len(Y)} samples failed "
                        f"(>10% threshold)"
                    )
                    return

                sens_result = analyzer.analyze(Y)
                msg_queue.post_sensitivity(sens_result)
            except Exception as exc:
                _log.error("Sensitivity analysis failed: %s", exc, exc_info=True)
                msg_queue.post_error(f"Sensitivity: {exc}")

        thread = threading.Thread(target=run_sensitivity, daemon=True)
        thread.start()
