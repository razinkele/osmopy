"""`population.seeding.mode` selects how seeded biomass becomes eggs (GitHub #143).

Java 4.4.1 converts linearly (`SeedingInterface`: 1e6 * seedingBiomass); this engine passes that same
linear quantity through the stock-recruitment relationship. The two are not scalings of each other — a
linear conversion is proportional, a stock-recruitment curve saturates — so they diverge most where
seeded biomass sits far from the curve's linear region, which is where the Baltic clupeids are.

Both are kept selectable so Java parity stays reachable and the two can be scored against ICES.
"""

from __future__ import annotations

import pytest

from osmose.engine.config import EngineConfig


def _cfg(extra: dict[str, str] | None = None) -> dict[str, str]:
    from tests.test_engine_output import _make_output_config

    c = dict(_make_output_config())
    if extra:
        c.update(extra)
    return c


def test_default_is_stock_recruitment_when_key_absent():
    """No existing config may change behaviour."""
    cfg = EngineConfig.from_dict(_cfg())
    assert cfg.seeding_mode == "stock_recruitment"


def test_linear_mode_selected_by_config():
    cfg = EngineConfig.from_dict(_cfg({"population.seeding.mode": "linear"}))
    assert cfg.seeding_mode == "linear"


def test_stock_recruitment_mode_selected_explicitly():
    cfg = EngineConfig.from_dict(_cfg({"population.seeding.mode": "stock_recruitment"}))
    assert cfg.seeding_mode == "stock_recruitment"


def test_unknown_mode_is_rejected_not_silently_defaulted():
    """A typo must fail loudly — silently falling back would make an A/B comparison meaningless."""
    with pytest.raises(ValueError, match="population.seeding.mode"):
        EngineConfig.from_dict(_cfg({"population.seeding.mode": "linaer"}))


def test_linear_seeds_more_than_stock_recruitment_on_the_baltic(tmp_path):
    """Behavioural check: the modes must actually differ, and in the direction the curve implies.

    A stock-recruitment curve saturates, so passing the linear quantity through it can only reduce
    (or leave) it where the seeded biomass sits above the curve's linear region. Baltic clupeids sit
    well above it, so `linear` must seed strictly more.
    """
    import numpy as np

    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    res = osmose_demo("baltic", tmp_path)
    base = dict(OsmoseConfigReader().read(str(res["config_file"])))
    base["simulation.time.nyear"] = "1"

    out = {}
    for mode in ("stock_recruitment", "linear"):
        cfg = dict(base)
        cfg["population.seeding.mode"] = mode
        d = tmp_path / mode
        d.mkdir()
        PythonEngine().run(cfg, d, seed=42)
        f = next(x for x in d.rglob("*_abundance_Simu0.csv") if "Distrib" not in x.name)
        import pandas as pd

        df = pd.read_csv(f, skiprows=1)
        out[mode] = float(np.nanmax(df["sprat"].to_numpy(dtype=float)))

    assert out["linear"] > out["stock_recruitment"], f"modes did not diverge as expected: {out}"
