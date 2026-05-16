"""Tests for calibration page handler helper functions."""

import threading

from shiny import reactive

from pathlib import Path

import pytest

from osmose.calibration.preflight import PreflightEvalError
from tests.helpers import make_catch_all_input, make_multi_input
from ui.pages.calibration import build_free_params, collect_selected_params
from ui.pages.calibration_handlers import (
    _clamp_n_workers,
    _require_preflight,
    build_preflight_modal,
)


def test_collect_selected_params():
    """Should return keys where the corresponding checkbox is True."""
    from ui.state import AppState

    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()

        # Simulate checkboxes: only species.k.sp0 is checked
        params = collect_selected_params(
            make_multi_input(cal_param_species_k_sp0=True, default=False), state
        )
        keys = [p["key"] for p in params]
        assert "species.k.sp0" in keys


def test_collect_selected_params_empty():
    """Should return empty list when nothing is checked."""
    from ui.state import AppState

    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()

        params = collect_selected_params(make_catch_all_input(False), state)
        assert params == []


def test_build_free_params():
    """Should create FreeParameter objects from selected param dicts."""
    from osmose.calibration.problem import FreeParameter

    selected = [
        {"key": "species.k.sp0", "lower": 0.1, "upper": 1.0},
        {"key": "species.k.sp1", "lower": 0.1, "upper": 1.0},
    ]
    free_params = build_free_params(selected)
    assert len(free_params) == 2
    assert isinstance(free_params[0], FreeParameter)
    assert free_params[0].key == "species.k.sp0"
    assert free_params[0].lower_bound == 0.1
    assert free_params[0].upper_bound == 1.0


def test_run_surrogate_workflow():
    """Test SurrogateCalibrator end-to-end: generate samples, fit, predict, find_optimum."""
    import numpy as np

    from osmose.calibration.surrogate import SurrogateCalibrator

    # Two free params, 1 objective
    bounds = [(0.1, 1.0), (10.0, 100.0)]
    cal = SurrogateCalibrator(param_bounds=bounds, n_objectives=1)

    # Step 1: generate samples
    n_samples = 20
    samples = cal.generate_samples(n_samples=n_samples)
    assert samples.shape == (n_samples, 2)
    # All samples within bounds
    assert np.all(samples[:, 0] >= 0.1) and np.all(samples[:, 0] <= 1.0)
    assert np.all(samples[:, 1] >= 10.0) and np.all(samples[:, 1] <= 100.0)

    # Step 2: simulate OSMOSE evaluations (use a known function)
    Y = np.sum(samples**2, axis=1)  # simple quadratic

    # Step 3: fit GP model
    cal.fit(samples, Y)
    assert cal._is_fitted

    # Step 4: predict on new points
    test_X = cal.generate_samples(n_samples=5, seed=99)
    means, stds = cal.predict(test_X)
    assert means.shape == (5, 1)
    assert stds.shape == (5, 1)
    assert np.all(stds >= 0)

    # Step 5: find optimum
    result = cal.find_optimum(n_candidates=500)
    assert "params" in result
    assert "predicted_objectives" in result
    assert "predicted_uncertainty" in result
    assert result["params"].shape == (2,)


def test_run_surrogate_workflow_multi_objective():
    """Test SurrogateCalibrator with multiple objectives."""
    import numpy as np

    from osmose.calibration.surrogate import SurrogateCalibrator

    bounds = [(0.0, 5.0), (0.0, 5.0)]
    cal = SurrogateCalibrator(param_bounds=bounds, n_objectives=2)

    samples = cal.generate_samples(n_samples=30)
    Y = np.column_stack(
        [
            np.sum(samples**2, axis=1),
            np.sum((samples - 3) ** 2, axis=1),
        ]
    )

    cal.fit(samples, Y)
    assert cal.n_objectives == 2

    means, stds = cal.predict(samples[:5])
    assert means.shape == (5, 2)

    result = cal.find_optimum(n_candidates=500)
    assert result["predicted_objectives"].shape == (2,)


def test_calibration_message_queue():
    """Thread-safe message queue relays updates without reactive writes."""
    from ui.pages.calibration_handlers import CalibrationMessageQueue

    q = CalibrationMessageQueue()

    def worker():
        q.post_status("Fitting GP model...")
        q.post_history_append(0.5)
        q.post_history_append(0.3)
        q.post_results(X=[[1, 2]], F=[[0.5]])
        q.post_error("Something broke")
        q.post_sensitivity({"S1": [0.5]})

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    msgs = q.drain()
    assert len(msgs) == 6
    assert msgs[0] == ("status", "Fitting GP model...")
    assert msgs[1] == ("history_append", 0.5)
    assert msgs[2] == ("history_append", 0.3)
    assert msgs[3][0] == "results"
    assert msgs[4] == ("error", "Something broke")
    assert msgs[5][0] == "sensitivity"


def test_calibration_message_queue_drain_empties():
    """Second drain returns empty list after first drain consumes all messages."""
    from ui.pages.calibration_handlers import CalibrationMessageQueue

    q = CalibrationMessageQueue()
    q.post_status("hello")
    assert len(q.drain()) == 1
    assert len(q.drain()) == 0  # second drain is empty


def test_surrogate_error_relayed_to_queue():
    """If surrogate calibration raises, error must be posted to queue."""
    from ui.pages.calibration_handlers import CalibrationMessageQueue

    q = CalibrationMessageQueue()

    def fake_surrogate_with_handler():
        try:
            raise RuntimeError("GP fit singular matrix")
        except Exception as exc:
            q.post_error(f"Surrogate calibration failed: {exc}")

    t = threading.Thread(target=fake_surrogate_with_handler)
    t.start()
    t.join()

    msgs = q.drain()
    assert len(msgs) == 1
    assert msgs[0][0] == "error"
    assert "singular matrix" in msgs[0][1]


def test_preflight_modal_renders_error_banner_for_PreflightEvalError():
    """A PreflightEvalError payload must produce a red-banner modal with
    the stage name and the error message — not the generic success-shape
    modal."""
    exc = PreflightEvalError("half the Morris samples failed", stage="morris")
    modal = build_preflight_modal(exc)
    assert modal is not None, "Error modal should render, not return None"
    html = str(modal)
    assert "alert-danger" in html, "Expected red alert banner"
    assert "morris" in html, "Expected stage label in banner"
    assert "half the Morris samples failed" in html, "Expected error message in banner"
    # Sanity: ensure it's NOT the success-shape modal
    assert "Apply Selected & Start" not in html


def test_require_preflight_allows_missing_jar_path():
    """Regression for final-review Important issue: the Python engine is
    the UI default and doesn't need a JAR. _require_preflight must permit
    jar_path=None so users without a local OSMOSE JAR can still launch
    calibration."""
    base = Path("/tmp/base.csv")
    work = Path("/tmp/work")
    out_base, out_jar, out_work = _require_preflight(base, None, work)
    assert out_base is base
    assert out_jar is None
    assert out_work is work


def test_require_preflight_still_blocks_missing_base_config():
    """base_config and work_dir remain mandatory — they're produced by the
    preflight stage itself, so their absence means preflight hasn't run."""
    with pytest.raises(RuntimeError, match="preflight"):
        _require_preflight(None, None, Path("/tmp/work"))
    with pytest.raises(RuntimeError, match="preflight"):
        _require_preflight(Path("/tmp/base.csv"), None, None)


def test_clamp_n_workers_honors_valid_input():
    assert _clamp_n_workers(4, 8) == 4


def test_clamp_n_workers_clamps_to_cpu_count():
    assert _clamp_n_workers(16, 4) == 4


def test_clamp_n_workers_defaults_on_invalid():
    assert _clamp_n_workers(None, 8) == 1
    assert _clamp_n_workers(0, 8) == 1
    assert _clamp_n_workers(-1, 8) == 1


def test_clamp_n_workers_survives_null_cpu_count():
    # Platforms where os.cpu_count() returns None: fall back to 1
    assert _clamp_n_workers(4, None) == 1


def test_resolve_optimum_weights_pareto_mode_returns_none():
    from ui.pages.calibration_handlers import _resolve_optimum_weights

    inp = make_multi_input(cal_optimum_mode="pareto", default=None)
    assert _resolve_optimum_weights(inp, n_obj=3) is None


def test_resolve_optimum_weights_weighted_mode_reads_N_inputs():
    from ui.pages.calibration_handlers import _resolve_optimum_weights

    inp = make_multi_input(
        cal_optimum_mode="weighted",
        cal_weight_0=1.0,
        cal_weight_1=2.0,
        default=None,
    )
    assert _resolve_optimum_weights(inp, n_obj=2) == [1.0, 2.0]


def test_resolve_optimum_weights_falls_back_on_silent_exception():
    """Simulate the @render.ui not having flushed — cal_weight_1 raises
    SilentException. Helper must return None (fallback to Pareto) rather
    than propagate the exception and halt the click handler."""
    from shiny.types import SilentException

    from ui.pages.calibration_handlers import _resolve_optimum_weights

    class _PartialInput:
        cal_optimum_mode = staticmethod(lambda: "weighted")
        cal_weight_0 = staticmethod(lambda: 1.0)

        def __getattr__(self, name):
            if name == "cal_weight_1":
                raise SilentException()
            raise AttributeError(name)

    assert _resolve_optimum_weights(_PartialInput(), n_obj=2) is None


def test_weights_inputs_renders_zero_inputs_in_pareto_mode():
    """In Pareto mode, _render_weights_inputs emits no input_numeric tags."""
    import numpy as np  # noqa: F401  (kept for consistency with sibling tests)
    from ui.pages.calibration import _render_weights_inputs

    out = _render_weights_inputs(mode="pareto", optimum=None)
    html = str(out)
    assert "input_numeric" not in html and "cal_weight_" not in html


def test_weights_inputs_renders_N_inputs_in_weighted_mode():
    import numpy as np
    from ui.pages.calibration import _render_weights_inputs

    optimum = {
        "predicted_objectives": np.array([1.0, 2.0, 3.0]),
        "objective_labels": ["a", "b", "c"],
    }
    out = _render_weights_inputs(mode="weighted", optimum=optimum)
    html = str(out)
    for i in range(3):
        assert f"cal_weight_{i}" in html, f"Missing input cal_weight_{i}"


def test_surrogate_pareto_scatter_empty_for_high_dim():
    """n_obj >= 3: scatter returns empty figure; table still produces M rows."""
    import numpy as np
    import pandas as pd
    from ui.pages.calibration import _render_pareto_scatter, _render_pareto_table

    optimum = {
        "pareto": {
            "objectives": np.arange(12).reshape(4, 3).astype(float),
            "uncertainty": np.full((4, 3), 0.1),
            "params": np.arange(8).reshape(4, 2).astype(float),
        },
        "predicted_objectives": np.array([0.0, 0.0, 0.0]),
        "predicted_uncertainty": np.array([0.1, 0.1, 0.1]),
        "params": np.array([0.0, 0.0]),
    }
    fig = _render_pareto_scatter(optimum)
    assert len(fig.data) == 0, "Expected empty scatter for n_obj >= 3"

    df_out = _render_pareto_table(optimum)
    df = df_out.data if hasattr(df_out, "data") else df_out
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"


def test_progress_callback_writes_checkpoint_per_generation(tmp_results_dir):
    """Driving the NSGA-II callback three times produces a checkpoint with
    monotonically-increasing generation."""
    from unittest.mock import MagicMock

    import numpy as np

    from osmose.calibration.checkpoint import read_checkpoint
    from ui.pages.calibration_handlers import _make_progress_callback

    history_appended = []
    cb = _make_progress_callback(
        cal_history_append=history_appended.append,
        cancel_check=lambda: False,
        checkpoint_path=tmp_results_dir / "phase_test_checkpoint.json",
        phase="test",
        param_keys=["k_a", "k_b"],
        bounds=[(-1.0, 1.0), (0.0, 3.0)],
        banded_residuals_accessor=lambda: None,
        banded_targets=None,
    )

    for gen in range(1, 4):
        mock_alg = MagicMock()
        mock_alg.opt = MagicMock()
        mock_alg.opt.get.side_effect = lambda key, _g=gen: {
            "F": np.array([[float(_g) * 0.5]]),
            "X": np.array([[0.1, 0.2]]),
        }[key]
        cb.notify(mock_alg)

    result = read_checkpoint(tmp_results_dir / "phase_test_checkpoint.json")
    assert result.kind == "ok"
    assert result.checkpoint.optimizer == "nsga2"
    assert result.checkpoint.generation == 3


def test_results_branch_handles_write_failure_gracefully(monkeypatch):
    """write_checkpoint raising in the NSGA-II callback MUST NOT prevent
    cal_history_append from updating the convergence chart."""
    from pathlib import Path
    from unittest.mock import MagicMock

    import numpy as np

    from ui.pages.calibration_handlers import _make_progress_callback

    def failing_write(*args, **kwargs):
        raise OSError("simulated disk-full")

    monkeypatch.setattr(
        "osmose.calibration.checkpoint.write_checkpoint", failing_write,
    )
    appended = []
    cb = _make_progress_callback(
        cal_history_append=appended.append,
        cancel_check=lambda: False,
        checkpoint_path=Path("/tmp/should_not_be_used.json"),
        phase="test",
        param_keys=["k_a"],
        bounds=[(-1.0, 1.0)],
        banded_residuals_accessor=lambda: None,
        banded_targets=None,
    )
    mock_alg = MagicMock()
    mock_alg.opt.get.side_effect = lambda key: {
        "F": np.array([[3.14]]),
        "X": np.array([[0.0]]),
    }[key]
    cb.notify(mock_alg)
    assert appended == [3.14]  # chart updated even though disk write raised

