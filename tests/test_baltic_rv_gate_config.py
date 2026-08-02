"""The committed Baltic config enables the RV recruitment gate for cod_east
(sp8) only — after the cod disaggregation, western cod (sp0) uses standard
Shepherd recruitment and eastern cod (sp8) is RV-gated (raw_cap)."""

from types import SimpleNamespace

import numpy as np
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine.config import _load_rv_gate
from osmose.engine.processes.recruitment_gate import rv_gate_factor


def test_committed_config_enables_rv_gate_for_cod_east_only():
    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    # signature: _load_rv_gate(cfg, n_species, n_dt_per_year, n_year)
    #   -> (factor_by_index (n_years,) | None, enabled_mask (n_species,) | None, offset)
    factor_by_index, enabled, _offset = _load_rv_gate(cfg, 9, 24, 40)
    assert enabled is not None and enabled[8]  # cod_east (sp8)
    assert not any(enabled[:8])  # cod_west (sp0) and all others off
    assert factor_by_index is not None and len(factor_by_index) > 0


def _committed_factor_profile(n_year=50):
    """Per-model-year gate factor for cod_east on the committed config, via the real code path."""
    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    factor_by_index, enabled, offset = _load_rv_gate(cfg, 9, 24, n_year)
    shim = SimpleNamespace(
        n_species=9,
        n_dt_per_year=24,
        rv_gate_factor_by_index=factor_by_index,
        rv_gate_enabled=enabled,
        rv_gate_offset=offset,
    )
    return np.array([rv_gate_factor(shim, year * 24)[8] for year in range(n_year)])


def test_committed_gate_bites_hardest_in_the_scored_final_decade():
    """Pin the shipped factor profile: it sets what cod_east's certification actually means.

    Measured 2026-08-02 (docs/baltic_rv_gate_mechanism_ab_2026-08-02.md): gate-off puts cod_east at
    167 kt, 1.97x over its 85 kt ICES ceiling, while gate-on lands at 83 kt — IN envelope with only
    ~2.2% headroom. So this profile is load-bearing for the certified verdict, and an edit to the RV
    series or to `ref` would silently move it. Characterisation test, not TDD: it pins behaviour that
    already exists precisely because that behaviour is now depended upon.
    """
    f = _committed_factor_profile()

    # Inert through the seeding bootstrap — and reproduction.py skips the gate on seeded steps anyway.
    assert np.all(f[:12] == 1.0)

    # Strongest bite lands inside the final decade, the window certification scores.
    assert f[40:].mean() == pytest.approx(0.438, abs=5e-3)
    assert f[12:40].mean() == pytest.approx(0.695, abs=5e-3)
    assert f[40:].mean() < f[12:40].mean() < f[:12].mean()

    # Past the 47-row series the clamp holds the 2020 minimum. Intentional (see recruitment_gate.py:
    # post-series years stay low, and it keeps the scored tail consistent across run horizons).
    assert f[47:] == pytest.approx(0.320, abs=5e-3)
    assert f[-1] == pytest.approx(f[47], abs=1e-12)
