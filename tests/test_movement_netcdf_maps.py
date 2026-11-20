"""Movement maps from NetCDF files, ``movement.netcdf.enabled`` (#137).

Java (``MapSet.loadMapsNcMaps``, ``ForcingFile.getNcStep``, v4.4.1): map k reads
``movement.variable.mapk`` from the files matching ``movement.file.mapk``; each time step of the
variable is a map, empty (the school goes out) where its ocean cells sum to 0. Simulation step t
uses file step ``(t // (ndtperyear // nsteps.year)) % n_file_steps``, for ages initialAge to
lastAge. The anchor test: two seasonal CSV maps and one NetCDF file holding the same two maps
build the same map set, the same C# bundle and the same run.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from osmose.engine.movement_maps import MovementMapSet

NY, NX, NDT, NYEAR = 5, 6, 4, 2
LIFESPAN_DT = 5 * NDT


def _maps() -> tuple[np.ndarray, np.ndarray]:
    """Two maps in grid rows (row 0 = south)."""
    a = np.zeros((NY, NX))
    a[0:2, 0:3] = 1.0
    b = np.zeros((NY, NX))
    b[3:5, 2:6] = 0.5
    b[4, 5] = 1.0
    return a, b


def _csv(path, grid: np.ndarray) -> None:
    # CSV maps are written north first: their row 0 is grid row NY - 1.
    path.write_text("\n".join(";".join(str(v) for v in row) for row in grid[::-1]) + "\n")


def _nc(path, frames: list[np.ndarray], var: str = "presence") -> None:
    xr.Dataset({var: (("time", "y", "x"), np.stack(frames))}).to_netcdf(path)


def _base(tmp_path) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": str(NDT),
        "simulation.time.nyear": str(NYEAR),
        "simulation.nspecies": "1",
        "mortality.subdt": "10",
        "grid.nlon": str(NX),
        "grid.nlat": str(NY),
        "simulation.nschool.sp0": "10",
        "species.name.sp0": "Fish0",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "1000",
        "movement.distribution.method.sp0": "maps",
        "movement.randomwalk.range.sp0": "1",
        "_osmose.config.dir": str(tmp_path),
    }
    return cfg


def _csv_config(tmp_path) -> dict[str, str]:
    a, b = _maps()
    _csv(tmp_path / "a.csv", a)
    _csv(tmp_path / "b.csv", b)
    return _base(tmp_path) | {
        "movement.species.map0": "Fish0",
        "movement.initialage.map0": "0",
        "movement.lastage.map0": "5",
        "movement.steps.map0": "0;1",
        "movement.file.map0": "a.csv",
        "movement.species.map1": "Fish0",
        "movement.initialage.map1": "0",
        "movement.lastage.map1": "5",
        "movement.steps.map1": "2;3",
        "movement.file.map1": "b.csv",
    }


def _nc_config(tmp_path, **extra: str) -> dict[str, str]:
    a, b = _maps()
    _nc(tmp_path / "maps.nc", [a, b])
    return (
        _base(tmp_path)
        | {
            "movement.netcdf.enabled": "true",
            "movement.species.map0": "Fish0",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "5",
            "movement.file.map0": "maps.nc",
            "movement.variable.map0": "presence",
            "movement.nsteps.year.map0": "2",
        }
        | extra
    )


def _set(cfg: dict[str, str]) -> MovementMapSet:
    return MovementMapSet(
        config=cfg,
        species_name="Fish0",
        n_dt_per_year=NDT,
        n_years=NYEAR,
        lifespan_dt=LIFESPAN_DT,
        ny=NY,
        nx=NX,
        config_dir=cfg["_osmose.config.dir"],
        ocean_mask=np.ones((NY, NX), dtype=bool),
    )


class TestTheMapSet:
    def test_it_equals_the_seasonal_csv_maps(self, tmp_path):
        csv, nc = _set(_csv_config(tmp_path)), _set(_nc_config(tmp_path))
        assert np.array_equal(csv.index_maps, nc.index_maps)
        assert len(csv.maps) == len(nc.maps) == 2
        for x, y in zip(csv.maps, nc.maps, strict=True):
            assert x is not None and y is not None and np.array_equal(x, y)
        assert np.array_equal(csv.max_proba, nc.max_proba)

    def test_the_file_step_of_a_simulation_step(self, tmp_path):
        frames = [np.full((NY, NX), float(k + 1)) for k in range(3)]
        _nc(tmp_path / "three.nc", frames)
        s = _set(_nc_config(tmp_path, **{"movement.file.map0": "three.nc"}))
        # 2 file steps per year, 4 model steps per year: t -> (t // 2) % 3
        assert list(s.index_maps[0, :8]) == [0, 0, 1, 1, 2, 2, 0, 0]
        assert s.get_map(0, 2)[0, 0] == 2.0  # type: ignore[index]

    def test_an_empty_step_is_an_out_map(self, tmp_path):
        a, _ = _maps()
        _nc(tmp_path / "gap.nc", [a, np.zeros((NY, NX))])
        s = _set(_nc_config(tmp_path, **{"movement.file.map0": "gap.nc"}))
        assert s.get_map(0, 0) is not None and s.get_map(0, 2) is None

    def test_the_file_name_is_a_pattern_over_sorted_files(self, tmp_path):
        a, b = _maps()
        _nc(tmp_path / "part_2.nc", [b])
        _nc(tmp_path / "part_1.nc", [a])
        _nc(tmp_path / "other.nc", [np.full((NY, NX), 9.0)])
        _nc(tmp_path / "old_part_3.nc", [np.full((NY, NX), 9.0)])  # Java matches the whole name
        s = _set(_nc_config(tmp_path, **{"movement.file.map0": r"part_\d\.nc"}))
        assert len(s.maps) == 2
        assert np.array_equal(s.maps[0], a) and np.array_equal(s.maps[1], b)  # type: ignore[arg-type]

    def test_ages_follow_initial_and_last_age(self, tmp_path):
        s = _set(
            _nc_config(tmp_path, **{"movement.initialage.map0": "1", "movement.lastage.map0": "2"})
        )
        assert (s.index_maps[:NDT] == -1).all()
        assert (s.index_maps[NDT : 2 * NDT + 1] >= 0).all()
        assert (s.index_maps[2 * NDT + 1 :] == -1).all()

    def test_a_missing_variable_key_is_refused(self, tmp_path):
        cfg = _nc_config(tmp_path)
        del cfg["movement.variable.map0"]
        with pytest.raises(ValueError, match="movement.variable.map0"):
            _set(cfg)

    def test_more_file_steps_per_year_than_model_steps_is_refused(self, tmp_path):
        # Java: step // (4 // 5) divides by zero.
        with pytest.raises(ValueError, match="nsteps.year.map0 = 5"):
            _set(_nc_config(tmp_path, **{"movement.nsteps.year.map0": "5"}))

    def test_a_non_divisor_uses_javas_integer_division(self, tmp_path):
        frames = [np.full((NY, NX), float(k + 1)) for k in range(3)]
        _nc(tmp_path / "three.nc", frames)
        s = _set(
            _nc_config(
                tmp_path, **{"movement.file.map0": "three.nc", "movement.nsteps.year.map0": "3"}
            )
        )
        # 4 // 3 = 1: file step t % 3
        assert list(s.index_maps[0, :6]) == [0, 1, 2, 0, 1, 2]

    def test_identical_frames_share_an_index_as_javas_twin_elimination(self, tmp_path):
        a, b = _maps()
        _nc(tmp_path / "aab.nc", [a, a.copy(), b])
        s = _set(
            _nc_config(
                tmp_path, **{"movement.file.map0": "aab.nc", "movement.nsteps.year.map0": "4"}
            )
        )
        assert list(s.index_maps[0, :4]) == [0, 0, 2, 0]
        assert s.maps[1] is None and s.maps[0] is not None and s.maps[2] is not None

    def test_frames_differing_only_on_land_are_twins(self, tmp_path):
        a, _ = _maps()
        a2 = a.copy()
        a2[0, 0] = 7.0
        _nc(tmp_path / "land.nc", [a, a2])
        cfg = _nc_config(tmp_path, **{"movement.file.map0": "land.nc"})
        ocean = np.ones((NY, NX), dtype=bool)
        ocean[0, 0] = False
        s = MovementMapSet(
            config=cfg,
            species_name="Fish0",
            n_dt_per_year=NDT,
            n_years=NYEAR,
            lifespan_dt=LIFESPAN_DT,
            ny=NY,
            nx=NX,
            config_dir=str(tmp_path),
            ocean_mask=ocean,
        )
        assert s.index_maps[0, 0] == s.index_maps[0, 2]

    def test_a_land_value_is_dropped_as_java_stores_nan_on_land(self, tmp_path):
        # A probability map (ocean max 0.5) with 5.0 on a land cell: Java keeps NaN on land, so
        # the map stays a probability map (max_proba 0.5), not a presence map (max >= 1 -> 0).
        frame = np.zeros((NY, NX))
        frame[1:3, 1:3] = 0.5
        frame[0, 0] = 5.0
        _nc(tmp_path / "land5.nc", [frame])
        ocean = np.ones((NY, NX), dtype=bool)
        ocean[0, 0] = False
        s = MovementMapSet(
            config=_nc_config(tmp_path, **{"movement.file.map0": "land5.nc"}),
            species_name="Fish0",
            n_dt_per_year=NDT,
            n_years=NYEAR,
            lifespan_dt=LIFESPAN_DT,
            ny=NY,
            nx=NX,
            config_dir=str(tmp_path),
            ocean_mask=ocean,
        )
        assert np.isnan(s.maps[0][0, 0])  # type: ignore[index]
        assert s.max_proba[0] == 0.5

    def test_a_path_with_dot_dot_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="'..' is refused"):
            _set(_nc_config(tmp_path, **{"movement.file.map0": "../maps.nc"}))

    def test_a_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="directory of its map files"):
            _set(_nc_config(tmp_path, **{"movement.file.map0": "nowhere/maps.nc"}))

    def test_without_the_switch_the_csv_path_is_untouched(self, tmp_path):
        cfg = _csv_config(tmp_path) | {"movement.netcdf.enabled": "false"}
        assert np.array_equal(_set(cfg).index_maps, _set(_csv_config(tmp_path)).index_maps)


def _run(cfg: dict[str, str]):
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine

    config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, 11)
    return sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)


def test_a_whole_run_equals_the_csv_run(tmp_path):
    csv, nc = _run(_csv_config(tmp_path)), _run(_nc_config(tmp_path))
    assert all(
        np.array_equal(x.biomass, y.biomass) and np.array_equal(x.abundance, y.abundance)
        for x, y in zip(csv, nc, strict=True)
    )


def test_the_csharp_bundle_carries_the_same_maps(tmp_path):
    from osmose.engine import PythonEngine
    from osmose.engine_cs.build import build_bundle

    def stack(cfg: dict[str, str]):
        config, grid, *_ = PythonEngine()._prepare_run(cfg, 11)
        b = build_bundle(config, config.raw_config, grid, seed=11)
        return {k: v for k, (_, v) in b._entries.items() if k.startswith("movement.maps")}

    csv, nc = stack(_csv_config(tmp_path)), stack(_nc_config(tmp_path))
    assert csv.keys() == nc.keys() and csv
    for k in csv:
        assert np.array_equal(np.asarray(csv[k]), np.asarray(nc[k])), k
