from __future__ import annotations

import pandas as pd
import pytest

from osmose.community_metrics import (
    SheldonSpectrum,
    _per_species_window_mean,
    _species_columns,
    _species_lw_coeffs,
    _to_float,
    compute_sheldon_spectrum,
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


def _write_csv(path, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _sheldon_fixture(out_dir):
    # One species "cod", a=1, b=1 so mass == length-midpoint.
    # Size lower-edges 0 and 10 -> inferred width 10 -> midpoints 5 and 15 -> masses 5, 15.
    # w_ref=5: octave k(5)=floor(log2(1))=0, k(15)=floor(log2(3))=1.
    # biomass-at-size: 20 at Size 0, 10 at Size 10 -> bin biomass {k0:20, k1:10}.
    # widths {k0: 5*2^0=5, k1: 5*2^1=10} -> NBSS {4.0, 1.0}; midpoints {7.071, 14.142}.
    # slope of log10(NBSS) vs log10(midpoint) = (0 - 0.602)/(1.150 - 0.849) = -2.0.
    _write_csv(
        out_dir / "osm_biomassDistribBySize_Simu0.csv",
        [(1.0, 0.0, 20.0), (1.0, 10.0, 10.0)],
        ["Time", "Size", "cod"],
    )
    # Totals come from the 1D biomass/abundance files: total B=30, total A=6 -> mean mass=5.
    _write_csv(out_dir / "osm_biomass_Simu0.csv", [(1.0, 30.0)], ["Time", "cod"])
    _write_csv(out_dir / "osm_abundance_Simu0.csv", [(1.0, 6.0)], ["Time", "cod"])


_CONFIG = {
    "simulation.nspecies": "1",
    "species.name.sp0": "cod",
    "species.length2weight.condition.factor.sp0": "1.0",
    "species.length2weight.allometric.power.sp0": "1.0",
}


def test_sheldon_spectrum_bins_and_slope(tmp_path):
    _sheldon_fixture(tmp_path)
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)
    assert isinstance(spec, SheldonSpectrum)
    assert spec.mass_bin_midpoints == pytest.approx([5 * 2**0.5, 5 * 2**1.5], rel=1e-6)
    assert spec.nbss_values == pytest.approx([4.0, 1.0], rel=1e-6)
    assert spec.slope == pytest.approx(-2.0, abs=1e-6)
    assert spec.n_bins_fit == 2
    assert spec.dropped_species == []


def test_sheldon_slope_recovery_powerlaw(tmp_path):
    # Genuine slope-RECOVERY test (the 2-bin test above only checks arithmetic).
    # a=1,b=1 so mass == length midpoint. Size edges 0,10,30,70 -> inferred width
    # median([10,20,40])=20 -> midpoints 10,20,40,80 -> octaves k=0,1,2,3 (distinct).
    # EQUAL biomass per octave => NBSS = biomass/(w_ref*2^k) halves each octave while the
    # midpoint doubles => log-log slope = -1 (the canonical normalized-biomass NBSS slope).
    _write_csv(
        tmp_path / "osm_biomassDistribBySize_Simu0.csv",
        [(1.0, 0.0, 100.0), (1.0, 10.0, 100.0), (1.0, 30.0, 100.0), (1.0, 70.0, 100.0)],
        ["Time", "Size", "cod"],
    )
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)
    assert spec.n_bins_fit == 4
    assert spec.slope == pytest.approx(-1.0, abs=1e-6)


def test_sheldon_totals_and_diversity(tmp_path):
    _sheldon_fixture(tmp_path)
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)
    assert spec.total_biomass == pytest.approx(30.0)
    assert spec.total_abundance == pytest.approx(6.0)
    assert spec.mean_body_mass == pytest.approx(5.0)
    # biomass shares [20/30, 10/30]; H = 0.6365; evenness = H/ln(2) = 0.9183.
    assert spec.size_diversity == pytest.approx(0.9183, abs=1e-3)


def test_sheldon_drops_species_without_coeffs(tmp_path):
    _write_csv(
        tmp_path / "osm_biomassDistribBySize_Simu0.csv",
        [(1.0, 0.0, 20.0, 5.0)],
        ["Time", "Size", "cod", "herring"],
    )
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)  # config has cod only
    assert spec.dropped_species == ["herring"]
    assert "herring" in spec.note


def test_sheldon_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)


def test_sheldon_bad_metric_raises(tmp_path):
    with pytest.raises(ValueError):
        compute_sheldon_spectrum(tmp_path, _CONFIG, metric="nonsense")
