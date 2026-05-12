from __future__ import annotations

from datetime import datetime, timezone


def test_save_run_called_on_de_completion(tmp_path, monkeypatch):
    """End-to-end smoke: _save_run_for_de writes to the history dir and
    list_runs() reads it back.

    save_run reads HISTORY_DIR from hist_mod at call time (not via the default
    arg, which Python binds at function-definition time). Patching the module
    attribute is sufficient.
    """
    from osmose.calibration import history as hist_mod
    from osmose.calibration.history import list_runs
    from scripts.calibrate_baltic import _save_run_for_de

    monkeypatch.setattr(hist_mod, "HISTORY_DIR", tmp_path / "calibration_history")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "de",
        "phase": "test",
        "parameters": ["k_a", "k_b"],
        "results": {
            "best_objective": 0.5,
            "best_parameters": {"k_a": 1.0, "k_b": 2.0},
            "duration_seconds": 1.2,
            "n_evaluations": 12,
            "per_species_residuals_final": None,
            "per_species_sim_biomass_final": None,
            "species_labels": None,
        },
    }
    _save_run_for_de(payload)
    runs = list_runs(tmp_path / "calibration_history")
    assert len(runs) == 1
    assert runs[0]["algorithm"] == "de"
    assert runs[0]["best_objective"] == 0.5


def test_save_run_fallback_writes_to_tempfile_with_restrictive_mode(tmp_path, monkeypatch):
    """When save_run raises, _save_run_fallback writes a 0o600 JSON to tempfile.gettempdir()."""
    import stat
    import tempfile

    from osmose.calibration.history import _save_run_fallback

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    import logging
    logger = logging.getLogger("test")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "de",
        "phase": "test",
        "parameters": ["k_a"],
        "results": {},
    }
    _save_run_fallback(payload, OSError("simulated"), logger)
    files = list(tmp_path.glob("calibration_history_fallback_*.json"))
    assert len(files) == 1
    mode_bits = stat.S_IMODE(files[0].stat().st_mode)
    assert mode_bits == 0o600
