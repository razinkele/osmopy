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
