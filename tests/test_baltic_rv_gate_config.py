"""The committed Baltic config enables the RV recruitment gate for cod (sp0) only."""

from osmose.config import OsmoseConfigReader
from osmose.engine.config import _load_rv_gate


def test_committed_config_enables_rv_gate_for_cod_only():
    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    # signature: _load_rv_gate(cfg, n_species, n_dt_per_year, n_year)
    #   -> (factor_by_index (n_years,) | None, enabled_mask (n_species,) | None, offset)
    factor_by_index, enabled, _offset = _load_rv_gate(cfg, 8, 24, 40)
    assert enabled is not None and enabled[0] and not any(enabled[1:])  # cod (sp0) only
    assert factor_by_index is not None and len(factor_by_index) > 0
