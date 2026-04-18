"""Tests for calibration page handler helper functions."""

import threading

from shiny import reactive

from osmose.calibration.preflight import PreflightEvalError
from tests.helpers import make_catch_all_input, make_multi_input
from ui.pages.calibration import build_free_params, collect_selected_params
from ui.pages.calibration_handlers import build_preflight_modal


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
