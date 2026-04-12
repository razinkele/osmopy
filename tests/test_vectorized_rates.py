"""Exact-match test: vectorized _precompute_effective_rates vs scalar original."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).parent.parent
EEC_CONFIG = PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv"


@pytest.mark.skipif(not EEC_CONFIG.exists(), reason="No EEC config")
def test_vectorized_rates_eec_smoke():
    """EEC 1yr runs without error with vectorized rates."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    reader = OsmoseConfigReader()
    raw = reader.read(EEC_CONFIG)
    raw["simulation.time.nyear"] = "1"
    cfg = EngineConfig.from_dict(raw)

    grid = Grid.from_netcdf(
        EEC_CONFIG.parent / raw["grid.netcdf.file"],
        mask_var=raw.get("grid.var.mask", "mask"),
    )

    rng = np.random.default_rng(42)
    outputs = simulate(cfg, grid, rng)

    assert len(outputs) > 0
    assert np.all(outputs[-1].biomass >= 0)


def test_vectorized_rates_correctness():
    """Vectorized rates produce correct values for known inputs."""
    from osmose.engine.processes.mortality import _precompute_effective_rates
    from types import SimpleNamespace

    n = 5
    n_species = 2
    n_subdt = 4
    n_dt_per_year = 24
    denom = n_dt_per_year * n_subdt

    # Build a minimal work_state
    class MockState:
        def __init__(self):
            self.species_id = np.array([0, 0, 1, 1, 0], dtype=np.int32)
            self.is_background = np.array([False, False, False, False, True])
            self.age_dt = np.array([5, 0, 3, 10, 5], dtype=np.int32)
            self.first_feeding_age_dt = np.ones(5, dtype=np.int32)
            self.starvation_rate = np.array([0.5, 0.0, 0.3, 0.0, 0.1])
            self.cell_y = np.array([0, 1, 0, 1, 0], dtype=np.int32)
            self.cell_x = np.array([0, 0, 1, 1, 0], dtype=np.int32)
            self.length = np.array([15.0, 5.0, 20.0, 30.0, 12.0])

        def __len__(self):
            return n

    work_state = MockState()

    # Build minimal config
    config = SimpleNamespace(
        n_species=n_species,
        n_dt_per_year=n_dt_per_year,
        additional_mortality_rate=np.array([0.1, 0.2]),
        additional_mortality_by_dt=None,
        additional_mortality_spatial=None,
        fishing_enabled=True,
        fishing_rate=np.array([0.5, 0.3]),
        fishing_rate_by_year=None,
        fishing_selectivity_type=np.array([0, 2], dtype=np.int32),
        fishing_selectivity_a50=np.array([0.1, 0.0]),
        fishing_selectivity_l50=np.array([0.0, 10.0]),
        fishing_selectivity_slope=np.array([0.0, 0.0]),
        fishing_spatial_maps=[None, None],
        mpa_zones=None,
        fishing_seasonality=None,
        fishing_discard_rate=np.array([0.1, 0.0]),
    )

    eff_s, eff_a, eff_f, f_disc = _precompute_effective_rates(work_state, config, n_subdt, step=5)

    # School 0: sp=0, age=5, not bg → starvation=0.5/96, additional=0.1/96, fishing=0.5/96
    assert eff_s[0] == pytest.approx(0.5 / denom)
    assert eff_a[0] == pytest.approx(0.1 / denom)
    assert eff_f[0] == pytest.approx(0.5 / denom)  # age selectivity: age_years=5/24 > a50=0.1
    assert f_disc[0] == pytest.approx(0.1)  # discard rate for sp0

    # School 1: age=0 → all rates zero
    assert eff_s[1] == 0.0
    assert eff_a[1] == 0.0
    assert eff_f[1] == 0.0

    # School 2: sp=1, age=3, length=20 > l50=10 → selectivity=1
    assert eff_a[2] == pytest.approx(0.2 / denom)
    assert eff_f[2] == pytest.approx(0.3 / denom)

    # School 3: sp=1, age=10, length=30 > l50=10 → selectivity=1
    assert eff_f[3] == pytest.approx(0.3 / denom)

    # School 4: is_background → all rates zero
    assert eff_s[4] == 0.0
    assert eff_a[4] == 0.0
    assert eff_f[4] == 0.0

    # No NaN in any output
    assert not np.any(np.isnan(eff_s))
    assert not np.any(np.isnan(eff_a))
    assert not np.any(np.isnan(eff_f))
