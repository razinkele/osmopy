"""Opt-in fish-mediated carbon-flux diagnostic (issue #134).

Two pathways per focal species per step, after Silvar-Viladomiu et al. (2026, ICES JMS,
doi:10.1093/icesjms/fsag095): faecal pellets = eaten biomass x unassimilated fraction x pellet
carbon factor; carcasses = non-predation, non-fishing deaths (tonnes) x carcass carbon factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import StepOutput, _average_step_outputs, _collect_carbon
from osmose.engine.state import MortalityCause, SchoolState
from tests.test_engine_bioen_integration import _make_bioen_config


def _base_cfg(n_sp: int = 1) -> dict[str, str]:
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": str(n_sp),
        "mortality.subdt": "1",
    }
    names = ["TestFish", "OtherFish"]
    for i in range(n_sp):
        cfg.update(
            {
                f"simulation.nschool.sp{i}": "5",
                f"species.name.sp{i}": names[i],
                f"species.linf.sp{i}": "20.0",
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "3",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
                f"movement.distribution.method.sp{i}": "random",
                f"movement.randomwalk.range.sp{i}": "1",
                f"species.maturity.size.sp{i}": "12.0",
            }
        )
    return cfg


# ── config ────────────────────────────────────────────────────────────────────


def test_config_parses_carbon_flags_and_paper_defaults():
    cfg = EngineConfig.from_dict(
        {**_base_cfg(), "output.carbon.enabled": "true", "output.carbon.netcdf.enabled": "true"}
    )
    assert cfg.output_carbon is True
    assert cfg.output_carbon_netcdf is True
    # Irish Sea teleost defaults: U = 0.2, carbon factors 0.10 of wet weight.
    assert cfg.carbon_unassimilated.tolist() == [0.2]
    assert cfg.carbon_pellet_cfactor.tolist() == [0.10]
    assert cfg.carbon_carcass_cfactor.tolist() == [0.10]
    off = EngineConfig.from_dict(_base_cfg())
    assert off.output_carbon is False and off.output_carbon_netcdf is False


def test_config_parses_carbon_coefficients_per_species():
    cfg = EngineConfig.from_dict(
        {
            **_base_cfg(2),
            "carbon.unassimilated.fraction.sp0": "0.3",
            "carbon.pellet.cfactor.sp1": "0.08",
            "carbon.carcass.cfactor.sp1": "0.12",
        }
    )
    assert cfg.carbon_unassimilated.tolist() == [0.3, 0.2]
    assert cfg.carbon_pellet_cfactor.tolist() == [0.10, 0.08]
    assert cfg.carbon_carcass_cfactor.tolist() == [0.10, 0.12]


def test_config_warns_when_carbon_and_bioen_both_on():
    # Under bioen the engine rescales eaten biomass to post-survival ingestion before the
    # output sees it, so the faecal basis changes; the config says so once, loudly.
    cfg = _make_bioen_config(_base_cfg(2))
    cfg["output.carbon.enabled"] = "true"
    with pytest.warns(UserWarning, match="carbon.*bioen|bioen.*carbon"):
        EngineConfig.from_dict(cfg)


# ── collector ─────────────────────────────────────────────────────────────────


class _Cfg:
    n_species = 2
    carbon_unassimilated = np.array([0.2, 0.5])
    carbon_pellet_cfactor = np.array([0.1, 0.08])
    carbon_carcass_cfactor = np.array([0.1, 0.12])


def test_collect_carbon_faecal_is_eaten_times_u_times_cfactor():
    s = SchoolState.create(n_schools=3, species_id=np.array([0, 0, 1], dtype=np.int32))
    s = s.replace(preyed_biomass=np.array([10.0, 5.0, 4.0]))
    faecal, carcass = _collect_carbon(s, _Cfg())
    assert faecal.tolist() == pytest.approx([15.0 * 0.2 * 0.1, 4.0 * 0.5 * 0.08])
    assert carcass.tolist() == [0.0, 0.0]


def test_collect_carbon_carcass_counts_only_other_mortality_in_tonnes():
    s = SchoolState.create(n_schools=2, species_id=np.array([0, 1], dtype=np.int32))
    n_dead = np.zeros((2, len(MortalityCause)))
    # sp0: 100 starved + 50 additional + 10 foraging + 40 aging = 200 "other" deaths;
    # 1000 eaten, 500 fished, 30 discarded, 70 left the domain -> all ignored.
    n_dead[0, MortalityCause.STARVATION] = 100
    n_dead[0, MortalityCause.ADDITIONAL] = 50
    n_dead[0, MortalityCause.FORAGING] = 10
    n_dead[0, MortalityCause.AGING] = 40
    n_dead[0, MortalityCause.PREDATION] = 1000
    n_dead[0, MortalityCause.FISHING] = 500
    n_dead[0, MortalityCause.DISCARDS] = 30
    n_dead[0, MortalityCause.OUT] = 70
    n_dead[1, MortalityCause.ADDITIONAL] = 20
    s = s.replace(n_dead=n_dead, weight=np.array([0.01, 0.05]))  # tonnes per fish
    faecal, carcass = _collect_carbon(s, _Cfg())
    assert faecal.tolist() == [0.0, 0.0]
    assert carcass.tolist() == pytest.approx([200 * 0.01 * 0.1, 20 * 0.05 * 0.12])


def test_collect_carbon_ignores_background_species_and_empty_state():
    s = SchoolState.create(n_schools=1, species_id=np.array([2], dtype=np.int32))  # background
    s = s.replace(preyed_biomass=np.array([99.0]))
    faecal, carcass = _collect_carbon(s, _Cfg())
    assert faecal.tolist() == [0.0, 0.0] and carcass.tolist() == [0.0, 0.0]
    empty = SchoolState.create(n_schools=0, species_id=np.zeros(0, dtype=np.int32))
    faecal, carcass = _collect_carbon(empty, _Cfg())
    assert faecal.shape == (2,) and carcass.shape == (2,)


# ── window aggregation ────────────────────────────────────────────────────────


def _step(step, faecal=None, carcass=None, n_sp=1):
    return StepOutput(
        step=step,
        biomass=np.full(n_sp, 100.0),
        abundance=np.full(n_sp, 1000.0),
        mortality_by_cause=np.zeros((n_sp, len(MortalityCause)), dtype=np.float64),
        carbon_faecal=faecal,
        carbon_carcass=carcass,
    )


def test_window_sums_carbon_flux_like_yield():
    a = _step(0, np.array([1.0]), np.array([0.5]))
    b = _step(1, np.array([3.0]), np.array([0.25]))
    out = _average_step_outputs([a, b], 2, 1)
    assert out.carbon_faecal.tolist() == [4.0]  # a FLOW: summed, not averaged
    assert out.carbon_carcass.tolist() == [0.75]
    single = _average_step_outputs([a], 1, 0)
    assert single.carbon_faecal.tolist() == [1.0] and single.carbon_carcass.tolist() == [0.5]
    none = _average_step_outputs([_step(0), _step(1)], 2, 1)
    assert none.carbon_faecal is None and none.carbon_carcass is None


# ── writers + reader ──────────────────────────────────────────────────────────


def test_carbon_csv_netcdf_and_reader_roundtrip(tmp_path):
    from osmose.engine.grid import Grid
    from osmose.engine.output import write_outputs
    from osmose.results import OsmoseResults, _build_dataframes_from_outputs

    cfg = EngineConfig.from_dict(
        {**_base_cfg(), "output.carbon.enabled": "true", "output.carbon.netcdf.enabled": "true"}
    )
    sp = cfg.species_names[0]
    outputs = [
        _step(0, np.array([1.5]), np.array([0.5])),
        _step(1, np.array([2.5]), np.array([0.75])),
    ]
    write_outputs(outputs, tmp_path, cfg, prefix="run")
    assert (tmp_path / "run_carbonFaecal_Simu0.csv").exists()
    assert (tmp_path / "run_carbonCarcass_Simu0.csv").exists()
    res = OsmoseResults(tmp_path, prefix="run")
    assert res.carbon_flux()[sp].tolist() == [1.5, 2.5]
    assert res.carbon_flux(pathway="carcass")[sp].tolist() == [0.5, 0.75]
    assert res.carbon_flux(pathway="faecal", source="netcdf")[sp].tolist() == [1.5, 2.5]
    assert res.carbon_flux(pathway="carcass", source="netcdf")[sp].tolist() == [0.5, 0.75]
    ds = res.read_netcdf("run_Simu0.nc")
    assert ds["carbonFaecal"].attrs["units"].startswith("t C")
    assert res.export_dataframe("carbon_faecal")[sp].tolist() == [1.5, 2.5]
    assert res.export_dataframe("carbon_carcass")[sp].tolist() == [0.5, 0.75]
    mem = _build_dataframes_from_outputs(outputs, cfg, Grid.from_dimensions(ny=1, nx=1))
    assert mem["carbonFaecal"][sp].tolist() == [1.5, 2.5]
    assert mem["carbonCarcass"][sp].tolist() == [0.5, 0.75]
    with pytest.raises(ValueError, match="pathway"):
        res.carbon_flux(pathway="respiration")


def test_carbon_writers_inert_when_flag_off(tmp_path):
    from osmose.engine.grid import Grid
    from osmose.engine.output import write_outputs
    from osmose.results import _build_dataframes_from_outputs

    cfg = EngineConfig.from_dict(_base_cfg())
    outputs = [_step(0, np.array([1.5]), np.array([0.5]))]  # populated but flag OFF
    write_outputs(outputs, tmp_path, cfg, prefix="run")
    assert not list(tmp_path.glob("run_carbon*"))
    mem = _build_dataframes_from_outputs(outputs, cfg, Grid.from_dimensions(ny=1, nx=1))
    assert not [k for k in mem if k.startswith("carbon")]


# ── end to end on the tutorial Baltic config ──────────────────────────────────
# (data/minimal declares no resources and never produces a predation event, so its eaten
# tally is identically zero; the Baltic-derived tutorial config feeds every step.)


def test_engine_run_produces_carbon_only_when_enabled(tmp_path, numba_warmup):
    from osmose.engine import PythonEngine
    from tests._tutorial_config import build_config

    base = build_config(tmp_path, n_year=1)
    off = PythonEngine().run_in_memory(base, seed=0)
    assert not [k for k in off.list_outputs() if k.startswith("carbon")]
    on = PythonEngine().run_in_memory({**base, "output.carbon.enabled": "true"}, seed=0)
    assert {"carbonFaecal", "carbonCarcass"} <= set(on.list_outputs())
    faecal = on.carbon_flux().drop(columns=["Time", "species"], errors="ignore")
    carcass = on.carbon_flux(pathway="carcass").drop(columns=["Time", "species"], errors="ignore")
    assert list(faecal.columns) == list(carcass.columns) and len(faecal.columns) >= 2
    # Something is eaten and something dies of "other" causes within the first year.
    assert faecal.to_numpy().sum() > 0
    assert carcass.to_numpy().sum() > 0
    assert (faecal.to_numpy() >= 0).all() and (carcass.to_numpy() >= 0).all()
    # The diagnostic is read-only: biomass is bit-identical with the flag on or off.
    np.testing.assert_array_equal(on.biomass().to_numpy(), off.biomass().to_numpy())
