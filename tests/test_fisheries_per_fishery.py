"""Per-fishery fishing (spec docs/superpowers/specs/2026-11-24-per-fishery-fishing-design.md).

Java fishes each school with the SUM over fisheries of q * rate * selectivity * map(step). The
per-species arrays are exact only for species-reducible configs (all shipped configs), which keep
that path; the rest get a FisheryModel. Covered here:
  G1 a species in two fisheries -> rates add        G2 catchability 0.5 -> half the rate
  G3 maps by fishery name                          G4 a map scheduled on some steps -> off on others
  equivalence: a reducible config forced through the per-fishery path == the per-species rates
  C#: per-fishery configs are refused; shipped configs are not
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig, _cfg_dir, _load_spatial_csv, _require_file
from osmose.engine.fisheries import (
    NO_MAP,
    OFF,
    FisheryModel,
    build_fishery_model,
    per_fishery_rates,
)
from osmose.engine.processes.mortality import _lookup_spatial_factors, _precompute_effective_rates
from osmose.engine.state import SchoolState

ROOT = Path(__file__).resolve().parent.parent
NDT, NSUB = 12, 10


def _model(**over) -> FisheryModel:
    base = {
        "names": ["a", "b"],
        "rate": np.array([1.2, 0.6]),
        "sel_type": np.array([-1, -1], dtype=np.int32),  # length knife-edge with l50 0 = all fished
        "sel_a50": np.zeros(2),
        "sel_l50": np.zeros(2),
        "sel_slope": np.ones(2),
        "catchability": np.array([[1.0, 0.0]]),
        "discard": np.zeros((1, 2)),
        "season": None,
        "maps": [],
        "map_files": [],
        "map_index": np.full((2, NDT), NO_MAP, dtype=np.int32),
    }
    base.update(over)
    return FisheryModel(**base)


def _rates(model, n=3, cy=None, cx=None, step=0):
    sp = np.zeros(n, dtype=np.int32)
    age = np.full(n, 100, dtype=np.int32)
    length = np.full(n, 30.0)
    cy = np.zeros(n, dtype=np.int32) if cy is None else np.asarray(cy, dtype=np.int32)
    cx = np.zeros(n, dtype=np.int32) if cx is None else np.asarray(cx, dtype=np.int32)
    return per_fishery_rates(
        model, sp, age, length, cy, cx, step, NDT, NSUB, _lookup_spatial_factors
    )


def test_g1_two_fisheries_on_one_species_add_up():
    one, _ = _rates(_model())
    both, _ = _rates(_model(catchability=np.array([[1.0, 1.0]])))
    assert one[0] == pytest.approx(1.2 / (NDT * NSUB))
    assert both[0] == pytest.approx((1.2 + 0.6) / (NDT * NSUB))


def test_g2_catchability_scales_the_rate():
    full, _ = _rates(_model())
    half, _ = _rates(_model(catchability=np.array([[0.5, 0.0]])))
    assert half[0] == pytest.approx(0.5 * full[0])


def test_g3_each_fishery_uses_its_own_map():
    m_a = np.array([[1.0, 0.0]])  # fishery a fishes cell x=0 only
    m_b = np.array([[0.0, 2.0]])  # fishery b fishes cell x=1, doubled
    idx = np.array([[0] * NDT, [1] * NDT], dtype=np.int32)
    model = _model(
        catchability=np.array([[1.0, 1.0]]), maps=[m_a, m_b], map_files=["a", "b"], map_index=idx
    )
    r, _ = _rates(model, n=2, cy=[0, 0], cx=[0, 1])
    assert r[0] == pytest.approx(1.2 / (NDT * NSUB))  # only a, in cell 0
    assert r[1] == pytest.approx(0.6 * 2.0 / (NDT * NSUB))  # only b, in cell 1


def test_g4_unscheduled_steps_switch_the_fishery_off():
    idx = np.array([[0] * 6 + [OFF] * 6, [NO_MAP] * NDT], dtype=np.int32)
    model = _model(maps=[np.ones((1, 1))], map_files=["m"], map_index=idx)
    on, _ = _rates(model, step=5)
    off, _ = _rates(model, step=6)
    assert on[0] > 0 and off[0] == 0.0


def test_discard_fraction_is_rate_weighted():
    model = _model(catchability=np.array([[1.0, 1.0]]), discard=np.array([[0.1, 0.4]]))
    _, frac = _rates(model)
    assert frac[0] == pytest.approx((1.2 * 0.1 + 0.6 * 0.4) / (1.2 + 0.6))


def _load(name: str) -> dict[str, str]:
    return OsmoseConfigReader().read(next((ROOT / "data" / name).glob("*all-parameters*.csv")))


@pytest.mark.parametrize("demo", ["baltic", "eec_full", "baltic_ev"])
def test_shipped_configs_keep_the_per_species_path(demo):
    assert EngineConfig.from_dict(_load(demo)).fishery_model is None


def _forced(cfg: dict[str, str], ec: EngineConfig) -> FisheryModel:
    names = [ec.species_names[i] for i in range(ec.n_species)]
    return build_fishery_model(
        cfg,
        names,
        ec.n_dt_per_year,
        ec.n_dt_per_year * ec.n_year,
        require_file=lambda v, k: _require_file(v, _cfg_dir(cfg), k),
        load_spatial_csv=_load_spatial_csv,
        reduce=False,
    )


@pytest.mark.parametrize("demo", ["eec_full", "baltic"])
def test_reducible_config_forced_through_per_fishery_path_gives_the_same_rates(demo):
    cfg = _load(demo)
    ec = EngineConfig.from_dict(cfg)
    rng = np.random.default_rng(3)
    n = 400
    state = SchoolState.create(n, species_id=rng.integers(0, ec.n_species, n).astype(np.int32))
    ny, nx = ec.fishing_spatial_maps[0].shape if ec.fishing_spatial_maps[0] is not None else (5, 5)
    state = state.replace(
        age_dt=rng.integers(0, 6 * ec.n_dt_per_year, n).astype(np.int32),
        length=rng.uniform(1, 80, n),
        cell_y=rng.integers(0, ny, n).astype(np.int32),
        cell_x=rng.integers(0, nx, n).astype(np.int32),
        first_feeding_age_dt=np.ones(n, dtype=np.int32),
    )
    for step in (0, 5, ec.n_dt_per_year + 3):
        _, _, per_species, d0 = _precompute_effective_rates(state, ec, NSUB, step)
        ec.fishery_model = _forced(cfg, ec)
        try:
            _, _, per_fishery, d1 = _precompute_effective_rates(state, ec, NSUB, step)
        finally:
            ec.fishery_model = None
        np.testing.assert_allclose(per_fishery, per_species, rtol=1e-12, atol=0)
        np.testing.assert_allclose(d1, d0, rtol=1e-12, atol=0)
        assert per_species.max() > 0  # the comparison is not vacuous


def _write_per_fishery_demo(tmp_path: Path) -> dict[str, str]:
    """EEC with one species moved into a SECOND fishery at catchability 0.5 (G1 + G2)."""
    cfg = _load("eec_full")
    import pandas as pd

    src = _require_file(cfg["fisheries.catchability.file"], _cfg_dir(cfg), "x")
    df = pd.read_csv(src, index_col=0).astype(float)
    sp0 = df.index[0]
    other = df.columns[1] if df.loc[sp0, df.columns[0]] > 0 else df.columns[0]
    df.loc[sp0, other] = 0.5
    out = tmp_path / "catchability.csv"
    df.to_csv(out)
    cfg["fisheries.catchability.file"] = str(out)
    return cfg


def test_a_non_reducible_config_gets_a_model_and_runs_on_csharp(tmp_path):
    """The C# engine implements the per-fishery path (feature 46, bundle contract 3), so it no
    longer refuses such configs; its byte-for-byte equality is tests/test_csharp_per_fishery.py."""
    from osmose.engine_cs.capabilities import csharp_engine_block_reason, required_features

    cfg = _write_per_fishery_demo(tmp_path)
    ec = EngineConfig.from_dict(cfg)
    assert isinstance(ec.fishery_model, FisheryModel)
    assert "per_fishery_fishing" in required_features(ec, cfg)
    assert csharp_engine_block_reason(cfg) is None
    shipped = _load("eec_full")
    assert "per_fishery_fishing" not in required_features(EngineConfig.from_dict(shipped), shipped)


def test_schedule_keys_build_the_map_index(tmp_path):
    cfg = _load("eec_full")
    # map0 on steps 0-5 only, from year 1: every other step of that fishery is off
    cfg["fisheries.movement.steps.map0"] = "0;1;2;3;4;5"
    cfg["fisheries.movement.initialyear.map0"] = "1"
    ec = EngineConfig.from_dict(cfg)
    m = ec.fishery_model
    assert isinstance(m, FisheryModel)  # a scheduled gap is not species-reducible
    f = m.names.index(cfg["fisheries.movement.fishery.map0"])
    ndt = ec.n_dt_per_year
    assert (m.map_index[f, :ndt] == OFF).all()  # year 0: before initialyear
    assert (m.map_index[f, ndt : ndt + 6] >= 0).all()
    assert (m.map_index[f, ndt + 6 : 2 * ndt] == OFF).all()
