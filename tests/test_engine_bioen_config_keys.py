"""C3 Task 2: bioen keys must survive the reader's lowercasing; larval threshold; merged Imax."""

from pathlib import Path

import numpy as np

from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from tests.test_bioen_orchestration import _make_bioen_config_dict  # 2-species synthetic dict


def _lower(d: dict[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in d.items()}


def test_tp_and_ed_are_read_from_lowercase_keys():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    cfg["species.bioen.mobilized.tp.sp0"] = "9.5"
    cfg["species.bioen.mobilized.e.d.sp1"] = "1.25"
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_tp[0] == 9.5 and ec.bioen_e_d[1] == 1.25


def test_reader_roundtrip_delivers_tp(tmp_path: Path):
    cfg = _make_bioen_config_dict(n_species=2)
    cfg["species.bioen.mobilized.Tp.sp0"] = "11.0"  # mixed case, as a user would write it
    p = tmp_path / "osm_all-parameters.csv"
    p.write_text("".join(f"{k} ; {v}\n" for k, v in cfg.items()))
    raw = dict(OsmoseConfigReader().read(str(p)))
    ec = EngineConfig.from_dict(raw)
    assert ec.bioen_tp[0] == 11.0, "Tp lost through the reader -> engine path"


def test_larvae_threshold_default_is_one_dt_and_key_is_years():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    ec = EngineConfig.from_dict(cfg)
    assert list(ec.bioen_larvae_thres_dt) == [1, 1]
    cfg["species.larvae.growth.threshold.age.sp1"] = "0.5"  # years
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_larvae_thres_dt[1] == round(0.5 * ec.n_dt_per_year)


def test_bioen_i_max_all_has_focal_then_background_entries():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    # One background species at sp10 (minimal keys, mirrors tests/test_engine_background.py's
    # _make_bkg_config) so the background half of the concatenation is actually exercised.
    cfg.update(
        {
            "species.type.sp10": "background",
            "species.name.sp10": "bkgspecies",
            "species.nclass.sp10": "2",
            "species.length.sp10": "10;30",
            "species.size.proportion.sp10": "0.3;0.7",
            "species.trophic.level.sp10": "2;3",
            "species.age.sp10": "1;3",
            "species.length2weight.condition.factor.sp10": "0.00308",
            "species.length2weight.allometric.power.sp10": "3.029",
            "predation.predprey.sizeratio.max.sp10": "3",
            "predation.predprey.sizeratio.min.sp10": "50",
            "predation.ingestion.rate.max.sp10": "7.7",
            "species.biomass.total.sp10": "1000.0",
            "simulation.nbackground": "1",
        }
    )
    ec = EngineConfig.from_dict(cfg)
    assert ec.n_background == 1
    assert ec.bioen_i_max_all.shape[0] == ec.n_species + ec.n_background
    np.testing.assert_array_equal(ec.bioen_i_max_all[: ec.n_species], ec.bioen_i_max)
    np.testing.assert_array_equal(ec.bioen_i_max_all[ec.n_species :], [7.7])


def test_bioen_larvae_thres_dt_and_i_max_all_dtypes_are_pinned():
    """Task 2 review Minor 2 (carried into Task 6, item C.3): nothing previously asserted
    `bioen_larvae_thres_dt` is int32 or `bioen_i_max_all` is float64 -- both are consumed as
    array indices / kernel inputs by `per_fish_ingestion_cap` (`ageDt < larvaeThresDt`,
    `i_max_all[species_id]`), where a silent dtype drift (e.g. int64 or float32) would still
    run but could change Numba specialisation or truncate/round unexpectedly. Pin both.
    """
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_larvae_thres_dt.dtype == np.int32
    assert ec.bioen_i_max_all.dtype == np.float64


def test_bioen_fields_none_when_disabled():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    cfg["module.bioenergetics.enabled"] = "false"
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_larvae_thres_dt is None and ec.bioen_i_max_all is None
