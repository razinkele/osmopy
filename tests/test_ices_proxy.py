from __future__ import annotations

import math

import pytest

from osmose.calibration.checkpoint import CalibrationCheckpoint


def _make_ckpt_with_proxy(per_species_residuals, per_species_sim_biomass, banded_targets, proxy_source):
    species_labels = tuple(banded_targets.keys()) if banded_targets else None
    return CalibrationCheckpoint(
        optimizer="de", phase="test", generation=1, generation_budget=10,
        best_fun=1.0,
        per_species_residuals=per_species_residuals,
        per_species_sim_biomass=per_species_sim_biomass,
        species_labels=species_labels,
        best_x_log10=(0.0,), best_parameters={"k": 1.0}, param_keys=("k",),
        bounds_log10={"k": (-1.0, 1.0)},
        gens_since_improvement=0, elapsed_seconds=1.0,
        timestamp_iso="2026-05-12T10:00:00+00:00",
        banded_targets=banded_targets,
        proxy_source=proxy_source,
    )


def test_proxy_in_range_zero_loss():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.0,),
        per_species_sim_biomass=(0.87,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert len(rows) == 1
    assert rows[0]["state"] == "in_range"
    assert abs(rows[0]["magnitude"] - 1.0) < 0.01


def test_proxy_out_of_range_overshoot():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.5,),
        per_species_sim_biomass=(3.0,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert rows[0]["state"] == "out_of_range"
    assert abs(rows[0]["magnitude"] - 3.0 / math.sqrt(0.75)) < 0.01
    assert rows[0]["direction"] == "overshoot"


def test_proxy_out_of_range_undershoot():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.5,),
        per_species_sim_biomass=(0.1,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert rows[0]["state"] == "out_of_range"
    assert rows[0]["direction"] == "undershoot"


def test_proxy_extinct():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(100.0,),
        per_species_sim_biomass=(0.0,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert rows[0]["state"] == "extinct"


def test_proxy_objective_disabled_renders_banner():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=None,
        per_species_sim_biomass=None,
        banded_targets=None,
        proxy_source="objective_disabled",
    )
    rows = _build_proxy_rows(ckpt)
    assert len(rows) == 1
    assert rows[0]["state"] == "objective_disabled"


def test_proxy_default_sort_out_first_then_in_then_extinct():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.0, 0.5, 100.0),
        per_species_sim_biomass=(1.0, 5.0, 0.0),
        banded_targets={"in_band": (0.5, 1.5), "out_band": (0.5, 1.5), "extinct_sp": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    states = [r["state"] for r in rows]
    assert states == ["out_of_range", "in_range", "extinct"]


def test_convergence_chart_adds_best_ever_line(monkeypatch):
    """list_runs() with matching prior runs → chart gets a horizontal line."""
    from osmose.calibration import history as hist_mod
    from ui.pages.calibration_charts import make_convergence_chart

    monkeypatch.setattr(
        hist_mod, "list_runs",
        lambda history_dir=None: [
            {"algorithm": "de", "phase": "test", "best_objective": 4.2,
             "timestamp": "2026-05-01T10:00:00+00:00", "n_params": 2, "duration_seconds": 1.0,
             "path": "x"},
            {"algorithm": "de", "phase": "test", "best_objective": 5.1,
             "timestamp": "2026-05-02T10:00:00+00:00", "n_params": 2, "duration_seconds": 1.0,
             "path": "x"},
        ],
    )
    fig = make_convergence_chart(history=[10.0, 8.0, 7.0], optimizer="de", phase="test")
    shapes = fig.layout.shapes or ()
    assert any(s.y0 == 4.2 and s.y1 == 4.2 for s in shapes), "expected hline at 4.2"


def test_convergence_chart_no_best_ever_line_when_no_prior_runs(monkeypatch):
    from osmose.calibration import history as hist_mod
    from ui.pages.calibration_charts import make_convergence_chart

    monkeypatch.setattr(hist_mod, "list_runs", lambda history_dir=None: [])
    fig = make_convergence_chart(history=[10.0], optimizer="de", phase="test")
    assert not (fig.layout.shapes or ())
