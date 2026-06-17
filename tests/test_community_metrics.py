from __future__ import annotations

import pandas as pd
import pytest

from osmose.community_metrics import (
    _per_species_window_mean,
    _species_columns,
    _species_lw_coeffs,
    _to_float,
)


def test_to_float_handles_bad_values():
    assert _to_float("1.5") == 1.5
    assert _to_float(None) is None
    assert _to_float("abc") is None


def test_species_columns_excludes_meta():
    df = pd.DataFrame(columns=["Time", "Size", "species", "cod", "herring"])
    assert _species_columns(df) == ["cod", "herring"]


def test_per_species_window_mean_windows_and_means():
    # Time 1..12; window_years=3 keeps Time > 12-3 = 9 -> {10,11,12}. cod mean of 10,20,30 = 20.
    df = pd.DataFrame(
        {
            "Time": [float(t) for t in range(1, 13)],
            "cod": [0.0] * 9 + [10.0, 20.0, 30.0],
            "species": ["all"] * 12,
        }
    )
    out = _per_species_window_mean(df, window_years=3)
    assert out == {"cod": pytest.approx(20.0)}


def test_per_species_window_mean_empty_df():
    assert _per_species_window_mean(pd.DataFrame(), window_years=10) == {}


def test_species_lw_coeffs_reads_config_and_skips_bad():
    config = {
        "simulation.nspecies": "2",
        "species.name.sp0": "cod",
        "species.length2weight.condition.factor.sp0": "0.01",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.name.sp1": "herring",
        "species.length2weight.condition.factor.sp1": "0",  # non-positive -> skip
        "species.length2weight.allometric.power.sp1": "3.0",
    }
    out = _species_lw_coeffs(config)
    assert out == {"cod": (0.01, 3.0)}


def test_species_lw_coeffs_empty_without_nspecies():
    assert _species_lw_coeffs({}) == {}
