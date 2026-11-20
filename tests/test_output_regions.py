"""OSMOSE 4.4.x output regions (#137).

Java (``Configuration``, ``OutputRegion``, ``FisheryMapSet``, ``AbstractOutput``, v4.4.1): region 0
is the whole domain; a region k > 0 (``output.regions.name.rg<k>``) covers, at each step, the cells
where its map (``output.regions.movement.*``, as for a fishery) is above 0, and every
region-aware output is written again over its schools as ``<prefix>_<output>-<region>_Simu0.csv``.
Anchors: two regions splitting the grid add up to the whole domain, and the whole-domain files
are byte-identical with and without regions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osmose.engine.output_regions import parse_output_regions, region_indices
from osmose.engine.state import SchoolState

NY = NX = 4
NDT = 4


def _map(path, cols) -> str:
    grid = np.zeros((NY, NX))
    grid[:, cols] = 1.0
    path.write_text("\n".join(";".join(str(v) for v in row) for row in grid) + "\n")
    return str(path)


def _regions(tmp_path) -> dict[str, str]:
    return {
        "output.regions.name.rg1": "West_side",
        "output.regions.name.rg2": "East",
        "output.regions.movement.region.map0": "Westside",  # Java matches the cleaned name
        "output.regions.movement.file.map0": _map(tmp_path / "west.csv", [0, 1]),
        "output.regions.movement.region.map1": "East",
        "output.regions.movement.file.map1": _map(tmp_path / "east.csv", [2, 3]),
    }


def _config(tmp_path, regions: bool = True, **extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": str(NDT),
        "simulation.time.nyear": "2",
        "simulation.nspecies": "1",
        "mortality.subdt": "10",
        "grid.nlon": str(NX),
        "grid.nlat": str(NY),
        "simulation.nschool.sp0": "12",
        "species.name.sp0": "Fish0",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.4",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.maturity.size.sp0": "12.0",
        "mortality.additional.rate.sp0": "0.3",
        "mortality.fishing.rate.sp0": "0.4",
        "population.seeding.biomass.sp0": "2000",
        "movement.randomwalk.range.sp0": "1",
        "output.cutoff.age.sp0": "0.5",  # new eggs are unlocated until they move
        "_osmose.config.dir": str(tmp_path),
    }
    return cfg | (_regions(tmp_path) if regions else {}) | extra


class TestTheKeys:
    def test_indices_skip_the_whole_domain_and_disabled_regions(self, tmp_path):
        cfg = _regions(tmp_path) | {
            "output.regions.name.rg3": "Off",
            "output.regions.enabled.rg3": "false",
            "output.regions.enabled.rg2": "true",
        }
        assert region_indices(cfg) == [1, 2]

    def test_region_zero_is_never_a_region_even_with_plural_keys(self, tmp_path):
        # Java removes index 0 from the output.regions.*.rg* indices: it is the whole domain.
        cfg = _regions(tmp_path) | {"output.regions.name.rg0": "Whole"}
        assert region_indices(cfg) == [1, 2]

    def test_anything_but_true_disables_as_javas_getboolean(self, tmp_path):
        cfg = _regions(tmp_path) | {"output.regions.enabled.rg1": "0"}
        assert region_indices(cfg) == [2]

    def test_names_lose_underscores_and_dashes(self, tmp_path):
        names = [r.name for r in parse_output_regions(_regions(tmp_path), NDT, 8, NY, NX)]
        assert names == ["Westside", "East"]

    def test_maps_match_the_cleaned_name_not_the_raw_one(self, tmp_path):
        cfg = _regions(tmp_path) | {"output.regions.movement.region.map0": "West_side"}
        west = parse_output_regions(cfg, NDT, 8, NY, NX)[0]
        assert west.name == "Westside" and (west.index_maps == -1).all()

    def test_region_netcdf_maps_are_refused(self, tmp_path):
        cfg = _regions(tmp_path) | {"output.regions.movement.netcdf.enabled": "true"}
        with pytest.raises(ValueError, match="not supported"):
            parse_output_regions(cfg, NDT, 8, NY, NX)

    def test_a_name_that_is_not_alphanumeric_is_refused(self, tmp_path):
        cfg = _regions(tmp_path) | {"output.regions.name.rg1": "West side"}
        with pytest.raises(ValueError, match="alphanumeric"):
            parse_output_regions(cfg, NDT, 8, NY, NX)

    def test_disabling_the_whole_domain_is_refused(self, tmp_path):
        cfg = _regions(tmp_path) | {"output.region.enabled.rg0": "false"}
        with pytest.raises(ValueError, match="rg0"):
            parse_output_regions(cfg, NDT, 8, NY, NX)

    def test_steps_and_years_set_when_a_map_applies(self, tmp_path):
        cfg = _regions(tmp_path) | {
            "output.regions.movement.steps.map0": "1;2",
            "output.regions.movement.years.map0": "1",
        }
        west = parse_output_regions(cfg, NDT, 8, NY, NX)[0]
        assert list(west.index_maps) == [-1, -1, -1, -1, -1, 0, 0, -1]

    def test_initial_and_last_year_are_inclusive(self, tmp_path):
        cfg = _regions(tmp_path) | {
            "output.regions.movement.initialyear.map0": "1",
            "output.regions.movement.lastyear.map0": "1",
        }
        west = parse_output_regions(cfg, NDT, 12, NY, NX)[0]
        assert list(west.index_maps) == [-1] * 4 + [0] * 4 + [-1] * 4


class TestTheMask:
    def test_in_the_domain_and_above_zero(self, tmp_path):
        west = parse_output_regions(_regions(tmp_path), NDT, 8, NY, NX)[0]
        s = SchoolState.create(n_schools=4, species_id=np.zeros(4, dtype=np.int32)).replace(
            cell_x=np.array([0, 1, 3, -1], dtype=np.int32),
            cell_y=np.array([0, 2, 0, -1], dtype=np.int32),
            is_out=np.array([False, True, False, False]),
        )
        assert list(west.mask(s, 0)) == [True, False, False, False]

    def test_a_step_without_a_map_is_outside(self, tmp_path):
        cfg = _regions(tmp_path) | {"output.regions.movement.steps.map0": "0"}
        west = parse_output_regions(cfg, NDT, 8, NY, NX)[0]
        s = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32)).replace(
            cell_x=np.array([0], dtype=np.int32), cell_y=np.array([0], dtype=np.int32)
        )
        assert west.mask(s, 0)[0] and not west.mask(s, 1)[0]


def _run(cfg: dict[str, str], out_dir):
    from osmose.engine import PythonEngine

    PythonEngine().run(cfg, out_dir, seed=9)
    return out_dir


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1)


class TestAWholeRun:
    def test_two_regions_splitting_the_grid_add_up_to_the_whole(self, tmp_path):
        out = _run(_config(tmp_path), tmp_path / "out")
        for name in ("biomass", "abundance", "yield"):
            whole = _read(out / f"osm_{name}_Simu0.csv")
            west = _read(out / f"osm_{name}-Westside_Simu0.csv")
            east = _read(out / f"osm_{name}-East_Simu0.csv")
            assert np.allclose(west["Fish0"] + east["Fish0"], whole["Fish0"], rtol=1e-9), name
            assert (west["Fish0"] > 0).any() and (east["Fish0"] > 0).any(), name

    def test_mortality_files_are_written_per_region(self, tmp_path):
        out = _run(_config(tmp_path), tmp_path / "out")
        # Java's MortalityOutput puts the region before the species.
        assert (out / "Mortality" / "osm_mortalityRate-Westside-Fish0_Simu0.csv").exists()

    def test_the_whole_domain_files_do_not_change(self, tmp_path):
        with_r = _run(_config(tmp_path), tmp_path / "with")
        without = _run(_config(tmp_path, regions=False), tmp_path / "without")
        whole = sorted(p.relative_to(without) for p in without.rglob("*.csv"))
        assert whole
        for rel in whole:
            assert (with_r / rel).read_bytes() == (without / rel).read_bytes(), rel
        extra = {p.relative_to(with_r) for p in with_r.rglob("*.csv")} - set(whole)
        assert extra and all("-Westside" in p.name or "-East" in p.name for p in extra)


_OPTIONAL = {
    k: "true"
    for k in (
        "output.biomass.byage.enabled",
        "output.abundance.byage.enabled",
        "output.biomass.bysize.enabled",
        "output.abundance.bysize.enabled",
        "output.size.enabled",
        "output.meantl.enabled",
        "output.tl.enabled",
        "output.yield.abundance.enabled",
    )
}
_READERS = (
    "biomass", "abundance", "yield_biomass", "mortality", "mortality_rate", "mean_size",
    "mean_trophic_level", "yield_abundance", "biomass_by_age", "abundance_by_age",
    "biomass_by_size", "abundance_by_size", "yield_by_age", "yield_by_size", "yield_n_by_age",
    "yield_n_by_size", "mean_size_by_age", "mean_tl_by_size", "mean_tl_by_age", "ssb",
)  # fmt: skip


def _call(results, name):
    try:
        return getattr(results, name)()
    except Exception as exc:  # compared below: both runs must fail the same way
        return type(exc)


def test_the_results_readers_ignore_region_files(tmp_path):
    from osmose.results import OsmoseResults

    with_dir = _run(_config(tmp_path, **_OPTIONAL), tmp_path / "with")
    without_dir = _run(_config(tmp_path, regions=False, **_OPTIONAL), tmp_path / "without")
    # the optional families were written per region
    for name in ("biomassByAge_Fish0", "abundanceBySize_Fish0", "meanSize", "meanTL", "yieldN"):
        assert (with_dir / f"osm_{name}-Westside_Simu0.csv").exists(), name
    with_r, without = (OsmoseResults(d, strict=False) for d in (with_dir, without_dir))
    compared = 0
    for reader in _READERS:
        a, b = _call(with_r, reader), _call(without, reader)
        if isinstance(a, type) or isinstance(b, type):
            assert a is b, (reader, a, b)
            continue
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
        if len(b):
            compared += 1
    assert compared >= 8


def test_age_classes_add_up_over_two_regions(tmp_path):
    out = _run(_config(tmp_path, **_OPTIONAL), tmp_path / "out")
    whole, west, east = (
        _read(out / f"osm_biomassByAge_Fish0{s}_Simu0.csv").to_numpy()[:, 2:]
        for s in ("", "-Westside", "-East")
    )
    # Age class 0 holds the new eggs, unlocated until they move; from age 1 every school is
    # in one of the two regions.
    assert whole.sum() > 0 and np.allclose(west + east, whole, rtol=1e-9)


def test_a_region_without_a_map_in_some_steps(tmp_path):
    cfg = _config(tmp_path) | {"output.regions.movement.steps.map0": "0;1"}
    out = _run(cfg, tmp_path / "out")
    west = _read(out / "osm_abundance-Westside_Simu0.csv")["Fish0"].to_numpy()
    whole = _read(out / "osm_abundance_Simu0.csv")["Fish0"].to_numpy()
    assert np.isfinite(west).all() and (west <= whole + 1e-9).all()


def test_the_initial_record_is_taken_after_the_first_movement(tmp_path):
    # The warm-start population exists before step 0 but is unlocated until the first movement.
    cfg = _config(
        tmp_path,
        **{"output.step0.include": "true", "module.population.initialisation.enabled": "true"},
    )
    out = _run(cfg, tmp_path / "out")
    west = _read(out / "osm_abundance-Westside_Simu0.csv")["Fish0"].to_numpy()
    east = _read(out / "osm_abundance-East_Simu0.csv")["Fish0"].to_numpy()
    whole = _read(out / "osm_abundance_Simu0.csv")["Fish0"].to_numpy()
    assert whole[0] > 0 and west[0] + east[0] > 0
    assert west[0] + east[0] == pytest.approx(whole[0], rel=1e-9)


def test_the_csharp_engine_refuses_regions_and_only_them(tmp_path):
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    assert "output_regions" in (csharp_engine_block_reason(_config(tmp_path)) or "")
    assert "output_regions" not in (
        csharp_engine_block_reason(_config(tmp_path, regions=False)) or ""
    )
