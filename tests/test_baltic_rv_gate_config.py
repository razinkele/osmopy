"""The committed Baltic config enables the RV recruitment gate for cod_east
(sp8) only — after the cod disaggregation, western cod (sp0) uses standard
Shepherd recruitment and eastern cod (sp8) is RV-gated (raw_cap)."""

from osmose.config import OsmoseConfigReader
from osmose.engine.config import _load_rv_gate


def test_committed_config_enables_rv_gate_for_cod_east_only():
    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    # signature: _load_rv_gate(cfg, n_species, n_dt_per_year, n_year)
    #   -> (factor_by_index (n_years,) | None, enabled_mask (n_species,) | None, offset)
    factor_by_index, enabled, _offset = _load_rv_gate(cfg, 9, 24, 40)
    assert enabled is not None and enabled[8]  # cod_east (sp8)
    assert not any(enabled[:8])  # cod_west (sp0) and all others off
    assert factor_by_index is not None and len(factor_by_index) > 0
