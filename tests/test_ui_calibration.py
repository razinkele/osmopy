"""Tests for calibration page helper functions."""

import numpy as np
import plotly.graph_objects as go

from ui.pages.calibration import (
    get_calibratable_params,
    make_convergence_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)


def test_get_calibratable_params():
    from ui.state import REGISTRY

    params = get_calibratable_params(REGISTRY, n_species=3)
    assert len(params) > 0
    # Should include growth K for all 3 species
    keys = [p["key"] for p in params]
    assert "species.k.sp0" in keys
    assert "species.k.sp1" in keys
    assert "species.k.sp2" in keys


def test_make_convergence_chart():
    history = [10.0, 5.0, 3.0, 2.0, 1.5]
    fig = make_convergence_chart(history)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_make_convergence_chart_empty():
    fig = make_convergence_chart([])
    assert isinstance(fig, go.Figure)


def test_make_pareto_chart():
    F = np.array([[1.0, 2.0], [0.5, 3.0], [0.8, 1.5]])
    fig = make_pareto_chart(F, ["Biomass RMSE", "Diet Distance"])
    assert isinstance(fig, go.Figure)


def test_make_sensitivity_chart():
    result = {
        "param_names": ["k", "linf", "mort"],
        "S1": np.array([0.5, 0.3, 0.2]),
        "ST": np.array([0.6, 0.4, 0.3]),
    }
    fig = make_sensitivity_chart(result)
    assert isinstance(fig, go.Figure)


def test_make_convergence_chart_incremental():
    """Verify convergence chart updates correctly as history grows (simulates live updates)."""
    history = []

    # Initially empty — chart should render with no data traces
    fig0 = make_convergence_chart(history)
    assert isinstance(fig0, go.Figure)
    assert len(fig0.data) == 0  # empty chart has no traces

    # After generation 1
    history = [10.0]
    fig1 = make_convergence_chart(history)
    assert isinstance(fig1, go.Figure)
    assert len(fig1.data) == 1
    assert list(fig1.data[0].y) == [10.0]
    assert list(fig1.data[0].x) == [0]

    # After generation 3 — values should decrease (convergence)
    history = [10.0, 7.5, 5.0]
    fig3 = make_convergence_chart(history)
    assert isinstance(fig3, go.Figure)
    assert len(fig3.data) == 1
    assert list(fig3.data[0].y) == [10.0, 7.5, 5.0]
    assert list(fig3.data[0].x) == [0, 1, 2]
    assert fig3.layout.xaxis.title.text == "Generation"
    assert fig3.layout.yaxis.title.text == "Best Objective"


def test_nsga2_progress_callback():
    """Test that ProgressCallback correctly appends history and handles cancellation."""
    from unittest.mock import MagicMock

    from ui.pages.calibration import _make_progress_callback

    # Simulate cal_history as a list-accumulating object
    collected = []

    def append_to_history(val):
        collected.append(val)

    cancel_getter = MagicMock(return_value=False)
    cb = _make_progress_callback(
        cal_history_append=append_to_history,
        cancel_check=cancel_getter,
    )

    # Mock algorithm with opt containing F values
    algorithm = MagicMock()
    F_values = np.array([[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]])
    algorithm.opt.get.return_value = F_values

    # Notify should append the min sum-of-objectives
    cb.notify(algorithm)
    assert len(collected) == 1
    assert collected[0] == 3.0  # min of [5.0, 5.0, 3.0]

    # Second generation with better values
    F_values2 = np.array([[1.0, 1.0], [2.0, 3.0]])
    algorithm.opt.get.return_value = F_values2
    cb.notify(algorithm)
    assert len(collected) == 2
    assert collected[1] == 2.0  # min of [2.0, 5.0]

    # Test cancellation
    cancel_getter.return_value = True
    cb.notify(algorithm)
    assert algorithm.termination.force_termination is True


def test_cal_preflight_workers_input_registered():
    """The n_workers input must appear in the rendered calibration UI.
    Guards against silent removal in a future refactor."""
    from ui.pages.calibration import calibration_ui

    page_ui = calibration_ui()
    html = str(page_ui)
    assert 'id="cal_preflight_workers"' in html or "'cal_preflight_workers'" in html, (
        "cal_preflight_workers input missing from rendered UI"
    )
