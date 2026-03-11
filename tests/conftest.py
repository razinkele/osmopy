"""Shared pytest fixtures for the OSMOSE test suite.

Fixtures here are available to all test modules without explicit imports.
Only fixtures that are (or will be) used across multiple test files are placed
here; file-specific fixtures stay in their own test modules.
"""

from pathlib import Path

import pandas as pd
import pytest

from osmose.schema import build_registry


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
