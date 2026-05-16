"""Shared pytest fixtures for the OSMOSE test suite.

Fixtures here are available to all test modules without explicit imports.
Only fixtures that are (or will be) used across multiple test files are placed
here; file-specific fixtures stay in their own test modules.
"""

from pathlib import Path

import pandas as pd
import pytest

from osmose.plotly_theme import ensure_templates
from osmose.schema import build_registry

# Register the "osmose" Plotly template once before any test runs. Without this,
# tests that render charts (tests/test_ui_results.py, test_ui_charts.py, etc.)
# fail in isolation because the template is normally registered only when app.py
# imports ui.charts at server startup. Running conftest.py triggers the
# registration during pytest collection, before any test module is imported.
ensure_templates()


# ---------------------------------------------------------------------------
# Schema / registry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_registry():
    """Build the complete OSMOSE schema registry once per test session.

    Use this fixture when you need the full 150+ parameter registry.
    (Unlike the ``registry`` fixture in test_registry.py, which is a
    minimal hand-built registry used to test the ParameterRegistry API
    itself.)
    """
    return build_registry()


# ---------------------------------------------------------------------------
# Minimal valid config dict
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Minimal valid OSMOSE config dict with 3 species."""
    return {
        "simulation.nspecies": "3",
        "simulation.time.nyear": "10",
        "simulation.time.ndtperyear": "24",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.name.sp2": "Hake",
        "species.linf.sp0": "19.5",
        "species.linf.sp1": "23.0",
        "species.linf.sp2": "130.0",
    }


# ---------------------------------------------------------------------------
# DataFrame fixtures for analysis / plotting tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_biomass_df():
    """Long-format biomass DataFrame with 2 species over 3 timesteps."""
    return pd.DataFrame(
        {
            "time": [1, 1, 2, 2, 3, 3],
            "species": ["A", "B", "A", "B", "A", "B"],
            "biomass": [100.0, 200.0, 110.0, 190.0, 120.0, 180.0],
        }
    )


@pytest.fixture
def sample_yield_df():
    """Long-format yield DataFrame with 2 species over 2 timesteps."""
    return pd.DataFrame(
        {
            "time": [1, 1, 2, 2],
            "species": ["A", "B", "A", "B"],
            "yield": [50.0, 100.0, 55.0, 95.0],
        }
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Pre-created temporary output directory for test runs."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Calibration dashboard fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch) -> Path:
    """Redirect all checkpoint writes to tmp_path."""
    monkeypatch.setattr("osmose.calibration.checkpoint.RESULTS_DIR", tmp_path)
    try:
        import scripts.calibrate_baltic as cb_mod
        monkeypatch.setattr(cb_mod, "RESULTS_DIR", tmp_path, raising=False)
    except ImportError:
        pass
    try:
        import ui.pages.calibration_handlers as ch_mod
        monkeypatch.setattr(ch_mod, "RESULTS_DIR", tmp_path, raising=False)
    except ImportError:
        pass
    return tmp_path


@pytest.fixture
def synthetic_two_species_targets():
    """A 2-species banded-loss target list."""
    from scripts.calibrate_baltic import BiomassTarget

    targets = [
        BiomassTarget(species="sp_a", target=1.0, lower=0.5, upper=1.5, weight=1.0),
        BiomassTarget(species="sp_b", target=2.0, lower=1.5, upper=2.5, weight=1.0),
    ]
    species_names = ["sp_a", "sp_b"]
    return targets, species_names


@pytest.fixture
def synthetic_stats_in_band():
    return {
        "sp_a_mean": 1.0, "sp_a_cv": 0.1, "sp_a_trend": 0.01,
        "sp_b_mean": 2.0, "sp_b_cv": 0.1, "sp_b_trend": 0.01,
    }


@pytest.fixture
def synthetic_stats_sp_b_out_of_band():
    return {
        "sp_a_mean": 1.0, "sp_a_cv": 0.1, "sp_a_trend": 0.01,
        "sp_b_mean": 5.0, "sp_b_cv": 0.1, "sp_b_trend": 0.01,
    }


# ---------------------------------------------------------------------------
# Numba warmup stub (T5 will replace with real Numba JIT compilation fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def numba_warmup() -> None:  # type: ignore[return]
    """Placeholder fixture for Numba JIT compilation warmup.

    T5 will replace this stub with a session-scoped fixture that compiles all
    Numba-decorated functions once before the tutorial tests run.  Until then
    this stub ensures the tutorial tests can be collected and run without
    fixture-not-found errors.
    """
    yield
