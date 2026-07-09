import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from osmose.engine.initialization import (  # noqa: E402
    age_structured_population,
    build_initial_population,
)


# ---------------------------------------------------------------- age-structure math
def test_age_structure_conserves_biomass_and_declines():
    ages_dt, lengths, weights, abund = age_structured_population(
        target_biomass=1000.0,
        linf=100.0,
        k=0.2,
        t0=-0.2,
        cf=0.01,
        ap=3.0,
        mortality=0.3,
        lifespan_years=10,
        n_dt_per_year=24,
    )
    assert len(ages_dt) == 10
    assert abs(float((abund * weights).sum()) - 1000.0) < 1e-6  # biomass conserved
    assert np.all(abund[:-1] >= abund[1:])  # numbers decline with age
    assert np.all(lengths[:-1] <= lengths[1:])  # length grows with age
    assert np.all(ages_dt >= 0)
    assert int(ages_dt.max()) < 10 * 24  # no cohort at/beyond lifespan_dt


def test_age_structure_floor_and_bounds():
    # fractional lifespan floors to 7 classes; oldest strictly below the aging threshold
    ages_dt, _len, w, ab = age_structured_population(
        500.0,
        90.0,
        0.18,
        -0.3,
        0.006,
        3.09,
        0.2,
        7.5,
        24,
    )
    assert len(ages_dt) == 7
    assert int(ages_dt.max()) < int(7.5 * 24)
    assert abs(float((ab * w).sum()) - 500.0) < 1e-6


def test_age_structure_empty_when_no_biomass():
    assert len(age_structured_population(0.0, 100, 0.2, -0.2, 0.01, 3.0, 0.3, 10, 24)[0]) == 0
    assert len(age_structured_population(1000.0, 100, 0.2, -0.2, 0.01, 3.0, 0.3, 0, 24)[0]) == 0


# ---------------------------------------------------------------- builder (fake config)
def _fake_config(enabled, seeding_biomass, key="module.population.initialisation.enabled"):
    n = len(seeding_biomass)
    arr = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    return SimpleNamespace(
        n_species=n,
        raw_config={key: "true"} if enabled else {},
        seeding_biomass=np.array(seeding_biomass, dtype=np.float64),
        linf=arr(100.0),
        k=arr(0.2),
        t0=arr(-0.2),
        condition_factor=arr(0.01),
        allometric_power=arr(3.0),
        additional_mortality_rate=arr(0.3),
        lifespan_dt=arr(10 * 24),
        n_dt_per_year=24,
        n_schools=np.full(n, 10, dtype=np.int32),
    )


def _fake_grid():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True  # 6 ocean cells
    return SimpleNamespace(ocean_mask=mask)


def test_disabled_returns_empty():
    st = build_initial_population(
        _fake_config(False, [1000.0, 2000.0]), _fake_grid(), np.random.default_rng(0)
    )
    assert len(st) == 0


def test_enabled_seeds_conserved_standing_stock():
    cfg = _fake_config(True, [1000.0, 0.0, 2000.0])  # sp1 zero biomass -> no schools
    grid = _fake_grid()
    st = build_initial_population(cfg, grid, np.random.default_rng(0))
    assert len(st) > 0
    assert not st.is_egg.any()  # standing adults, not eggs
    assert (st.abundance > 0).all()
    for sp, target in ((0, 1000.0), (2, 2000.0)):
        m = st.species_id == sp
        assert abs(float((st.abundance[m] * st.weight[m]).sum()) - target) < 1.0
    assert not (st.species_id == 1).any()  # zero-biomass species seeds nothing
    for cx, cy in zip(st.cell_x, st.cell_y):
        assert grid.ocean_mask[cy, cx]  # placed in ocean cells


def test_legacy_key_still_activates():
    cfg = _fake_config(True, [1000.0], key="population.initialization.relativebiomass.enabled")
    st = build_initial_population(cfg, _fake_grid(), np.random.default_rng(0))
    assert len(st) > 0


def test_initialize_parity_off_and_populated_on():
    from osmose.engine.simulate import initialize

    grid = _fake_grid()
    off = initialize(_fake_config(False, [1000.0]), grid, np.random.default_rng(0))
    assert len(off) == 0  # parity: empty when flag off
    on = initialize(_fake_config(True, [1000.0]), grid, np.random.default_rng(0))
    assert len(on) > 0 and not on.is_egg.any()


# ---------------------------------------------------------------- real-config activation (H1 + H2)
def test_real_config_activation_and_seeding_disabled():
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    cfg = OsmoseConfigReader().read(str(_ROOT / "data" / "baltic" / "baltic_all-parameters.csv"))
    cfg["module.population.initialisation.enabled"] = "true"
    ec = EngineConfig.from_dict(cfg)
    # H2: warm-start disables egg-seeding (seeding_max_step -> 0)
    assert int(ec.seeding_max_step[0]) == 0
    # H1: the canonical key actually activates the builder on the real config path
    grid = PythonEngine()._resolve_grid(cfg)
    st = build_initial_population(ec, grid, np.random.default_rng(0))
    assert len(st) > 0
    m = st.species_id == 0
    assert m.any()
    cod_bm = float((st.abundance[m] * st.weight[m]).sum())
    target = float(ec.seeding_biomass[0])
    assert abs(cod_bm - target) / target < 0.02  # biomass conserved to target
