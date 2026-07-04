import numpy as np
import pytest

from osmose.engine.config import EngineConfig, _load_recruitment_ceiling
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState
from osmose.schema import build_registry


def test_ceiling_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.recruitment.ceiling.enabled" in keys
    assert "reproduction.recruitment.ceiling.series.file" in keys
    assert "reproduction.recruitment.ceiling.species.enabled.sp{idx}" in keys


def _write_ceiling_csv(path, n_cols, cols):
    # cols: dict {species_index: [value per season_idx]}
    sp_ids = sorted(cols)
    header = "season_idx," + ",".join(f"ceiling_sp{i}" for i in sp_ids)
    lines = [header]
    for s in range(n_cols):
        row = [str(s)] + [f"{cols[i][s]:.6f}" for i in sp_ids]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _cfg(tmp_path, csv_path, enabled_species=(0,)):
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.recruitment.ceiling.enabled": "true",
        "reproduction.recruitment.ceiling.series.file": csv_path.name,
    }
    for sp in enabled_species:
        cfg[f"reproduction.recruitment.ceiling.species.enabled.sp{sp}"] = "true"
    return cfg


def test_ceiling_off_returns_none(tmp_path):
    ceil, mask = _load_recruitment_ceiling({}, 3, 12, None)
    assert ceil is None and mask is None


def test_ceiling_loads_shape_and_mask(tmp_path):
    csv = _write_ceiling_csv(
        tmp_path / "c.csv", 12, {0: [10.0] * 12, 1: [20.0] * 12, 2: [30.0] * 12}
    )
    ceil, mask = _load_recruitment_ceiling(_cfg(tmp_path, csv, (0, 2)), 3, 12, None)
    assert ceil.shape == (12, 3)
    assert list(mask) == [True, False, True]
    assert ceil[5, 1] == 20.0


def test_ceiling_row_count_must_match_ncols(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 6, {0: [10.0] * 6})
    with pytest.raises(ValueError, match="season"):
        _load_recruitment_ceiling(_cfg(tmp_path, csv), 1, 12, None)


def test_ceiling_rejects_negative(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [-1.0] + [10.0] * 11})
    with pytest.raises(ValueError, match="negative|NaN"):
        _load_recruitment_ceiling(_cfg(tmp_path, csv), 1, 12, None)


def test_ceiling_requires_enabled_species(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [10.0] * 12})
    cfg = _cfg(tmp_path, csv, enabled_species=())
    with pytest.raises(ValueError, match="no species enabled"):
        _load_recruitment_ceiling(cfg, 1, 12, None)


def test_ceiling_missing_file_raises(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        _load_recruitment_ceiling(cfg, 1, 12, None)


def test_ceiling_enabled_species_missing_column(tmp_path):
    # Test 1: Enabled species (sp1) but CSV only has ceiling_sp0 column
    _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [10.0] * 12})
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.recruitment.ceiling.enabled": "true",
        "reproduction.recruitment.ceiling.series.file": "c.csv",
        "reproduction.recruitment.ceiling.species.enabled.sp1": "true",
    }
    with pytest.raises(ValueError, match="no ceiling_sp1 column"):
        _load_recruitment_ceiling(cfg, 2, 12, None)


def test_ceiling_empty_file_key(tmp_path):
    # Test 2: Empty series.file with master switch on
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.recruitment.ceiling.enabled": "true",
        "reproduction.recruitment.ceiling.series.file": "",
        "reproduction.recruitment.ceiling.species.enabled.sp0": "true",
    }
    with pytest.raises(ValueError, match="empty"):
        _load_recruitment_ceiling(cfg, 1, 12, None)


def test_ceiling_missing_season_idx_column(tmp_path):
    # Test 3: CSV missing season_idx column
    csv_path = tmp_path / "c.csv"
    # Write CSV without season_idx column, only ceiling_sp0
    csv_path.write_text(
        "ceiling_sp0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n10.0\n"
    )
    cfg = _cfg(tmp_path, csv_path)
    with pytest.raises(ValueError, match="season_idx"):
        _load_recruitment_ceiling(cfg, 1, 12, None)


def _repro_cfg_dict():
    # Single-species config that produces a large per-step n_eggs at step 0.
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "800",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "50000",
    }


def _mature_state():
    s = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    return s.replace(
        abundance=np.array([1000.0]),
        length=np.array([15.0]),  # > maturity_size 12
        weight=np.array([20.25]),
        biomass=np.array([20250.0]),
        age_dt=np.array([24], dtype=np.int32),
    )


def _eggs_produced(new_state, sp=0):
    fresh = (new_state.age_dt == 0) & new_state.is_egg & (new_state.species_id == sp)
    return float(new_state.abundance[fresh].sum())


def _enable_ceiling(cfg, tmp_path, n_cols, sp0_ceiling):
    csv = _write_ceiling_csv(tmp_path / "c.csv", n_cols, {0: [sp0_ceiling] * n_cols})
    cfg = dict(cfg)
    cfg["_osmose.config.dir"] = str(tmp_path)
    cfg["reproduction.recruitment.ceiling.enabled"] = "true"
    cfg["reproduction.recruitment.ceiling.series.file"] = csv.name
    cfg["reproduction.recruitment.ceiling.species.enabled.sp0"] = "true"
    return cfg


def test_reproduction_uncapped_baseline(tmp_path):
    cfg = EngineConfig.from_dict(_repro_cfg_dict())
    eggs = _eggs_produced(reproduction(_mature_state(), cfg, step=0, rng=np.random.default_rng(0)))
    assert eggs > 0  # sanity: this state produces eggs


def test_reproduction_clamps_when_above_ceiling(tmp_path):
    base = EngineConfig.from_dict(_repro_cfg_dict())
    uncapped = _eggs_produced(
        reproduction(_mature_state(), base, step=0, rng=np.random.default_rng(0))
    )
    cap = uncapped / 2.0
    cfg = EngineConfig.from_dict(_enable_ceiling(_repro_cfg_dict(), tmp_path, 12, cap))
    capped = _eggs_produced(
        reproduction(_mature_state(), cfg, step=0, rng=np.random.default_rng(0))
    )
    assert abs(capped - cap) < 1e-3  # clamped to the ceiling


def test_reproduction_unchanged_when_below_ceiling(tmp_path):
    base = EngineConfig.from_dict(_repro_cfg_dict())
    uncapped = _eggs_produced(
        reproduction(_mature_state(), base, step=0, rng=np.random.default_rng(0))
    )
    cfg = EngineConfig.from_dict(_enable_ceiling(_repro_cfg_dict(), tmp_path, 12, uncapped * 2.0))
    result = _eggs_produced(
        reproduction(_mature_state(), cfg, step=0, rng=np.random.default_rng(0))
    )
    assert abs(result - uncapped) < 1e-3  # ceiling above production: identical


def test_reproduction_ceiling_skips_seeded_step(tmp_path):
    # Empty state -> SSB is seeded from population.seeding.biomass; seeded eggs
    # must NOT be clipped even with a tiny ceiling.
    cfg = EngineConfig.from_dict(_enable_ceiling(_repro_cfg_dict(), tmp_path, 12, 1.0))
    empty = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
    eggs = _eggs_produced(reproduction(empty, cfg, step=0, rng=np.random.default_rng(0)))
    assert eggs > 1.0  # seeded bootstrap exceeds the ceiling, proving it was skipped


def test_ceiling_off_is_bit_identical():
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read("data/eec_full/eec_all-parameters.csv")
    cfg["simulation.time.nyear"] = "2"
    cfg["simulation.rng.fixed"] = "true"
    cfg["movement.randomseed.fixed"] = "true"
    cfg["stochastic.mortality.randomseed.fixed"] = "true"

    base = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    cfg["reproduction.recruitment.ceiling.enabled"] = "false"
    off = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    np.testing.assert_array_equal(base.to_numpy(), off.to_numpy())


import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import derive_recruitment_ceiling as derive  # noqa: E402


def test_zero_fishing_disables_both_modes():
    cfg = {
        "module.multispecies.fisheries.enabled": "true",
        "simulation.fishing.mortality.enabled": "true",
    }
    out = derive.zero_fishing(cfg)
    assert out["module.multispecies.fisheries.enabled"] == "false"
    assert out["simulation.fishing.mortality.enabled"] == "false"
    assert cfg["module.multispecies.fisheries.enabled"] == "true"  # original untouched


def test_per_species_recruitment_counts_fresh_natural_eggs():
    from osmose.engine.state import SchoolState

    s = SchoolState.create(n_schools=3, species_id=np.array([0, 0, 1], dtype=np.int32))
    s = s.replace(
        abundance=np.array([100.0, 50.0, 7.0]),
        age_dt=np.array([0, 1, 0], dtype=np.int32),  # 2nd is last-step egg
        is_egg=np.array([True, True, True]),
    )
    r = derive.per_species_recruitment(s, n_species=2)
    assert r[0] == 100.0  # only the age_dt==0 school for sp0
    assert r[1] == 7.0


def test_late_window_ceiling_buckets_by_season():
    # 2 seasons/year, 4 years; recruitment = season_idx*10 + noise-free
    records = []
    for step in range(8):  # 4 years * 2 seasons
        col = step % 2
        records.append((step, np.array([10.0 * col + 100.0])))
    ceil = derive.late_window_ceiling(records, n_cols=2, n_species=1, n_dt=2, frac=0.5)
    assert ceil.shape == (2, 1)
    assert ceil[0, 0] == 100.0  # season 0
    assert ceil[1, 0] == 110.0  # season 1


def test_write_ceiling_csv_roundtrips(tmp_path):
    ceil = np.array([[100.0, 200.0], [110.0, 210.0]])
    out = derive.write_ceiling_csv(ceil, tmp_path / "c.csv")
    text = out.read_text().strip().splitlines()
    assert text[0] == "season_idx,ceiling_sp0,ceiling_sp1"
    assert text[1].startswith("0,")
    loaded, mask = _load_recruitment_ceiling(
        {
            "_osmose.config.dir": str(tmp_path),
            "reproduction.recruitment.ceiling.enabled": "true",
            "reproduction.recruitment.ceiling.series.file": out.name,
            "reproduction.recruitment.ceiling.species.enabled.sp0": "true",
        },
        2,
        2,
        None,
    )
    np.testing.assert_array_equal(loaded, ceil)


def test_seeding_overlap_warns_when_late_window_inside_seeding_window():
    # 360 steps, n_dt 24 -> 15 years; frac 1/3 -> late window starts step 240.
    smax = np.array([480, 100])  # sp0 eligible past 240 -> warn; sp1 (100) -> no warn
    w = derive.seeding_overlap_warnings(smax, 360, 24, 1.0 / 3.0)
    assert len(w) == 1
    assert "sp0" in w[0]


def test_seeding_overlap_no_warning_when_clear():
    smax = np.array([50, 100])  # both well before step 240
    assert derive.seeding_overlap_warnings(smax, 360, 24, 1.0 / 3.0) == []
