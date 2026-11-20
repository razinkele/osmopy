"""Three OSMOSE 4.4.x keys (#137) the Python engine did not read.

- ``simulation.nschool.multiplier`` (``Configuration.java``, v4.4.1 lines 583-591): the number of
  schools of each focal species is ``(int) (multiplier * nschool)``.
- ``species.multiplier.log.sp#`` (``BackgroundSpecies.java``, lines 223-238): a background
  species' biomass multiplier given as its logarithm; Java refuses it together with
  ``species.multiplier.sp#``.
- ``species.biomass.varname.sp#`` (``ResourceForcing.java``, line 219): the NetCDF variable that
  holds a resource's forcing, when it is not named after the resource.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState


def _two_species(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "mortality.subdt": "10",
    }
    for i in range(2):
        cfg |= {
            f"simulation.nschool.sp{i}": ("5", "7")[i],
            f"species.name.sp{i}": f"Fish{i}",
            f"species.linf.sp{i}": "30.0",
            f"species.k.sp{i}": "0.3",
            f"species.t0.sp{i}": "-0.1",
            f"species.egg.size.sp{i}": "0.1",
            f"species.length2weight.condition.factor.sp{i}": "0.006",
            f"species.length2weight.allometric.power.sp{i}": "3.0",
            f"species.lifespan.sp{i}": "5",
            f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
            f"predation.ingestion.rate.max.sp{i}": "3.5",
            f"predation.efficiency.critical.sp{i}": "0.57",
        }
    return cfg | extra


class TestSchoolMultiplier:
    def test_absent_changes_nothing(self):
        assert EngineConfig.from_dict(_two_species()).n_schools[:2].tolist() == [5, 7]

    @pytest.mark.parametrize(
        ("mul", "want"), [("2", [10, 14]), ("1.5", [7, 10]), ("0.4", [2, 2]), ("1.0", [5, 7])]
    )
    def test_the_count_is_the_truncated_product(self, mul, want):
        # Java: (int) (mul * (float) nschool), a truncation: 1.5 x 7 = 10.5 -> 10, 0.4 x 5 = 2.
        cfg = _two_species(**{"simulation.nschool.multiplier": mul})
        assert EngineConfig.from_dict(cfg).n_schools[:2].tolist() == want

    def test_a_count_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="simulation.nschool.multiplier"):
            EngineConfig.from_dict(_two_species(**{"simulation.nschool.multiplier": "0.1"}))


def _background_cfg(**extra: str) -> dict[str, str]:
    from osmose.engine.background import parse_background_species  # noqa: F401  (import check)

    cfg = _two_species() | {
        "simulation.nbackground": "1",
        "species.type.sp2": "background",
        "species.name.sp2": "Seal",
        "species.nclass.sp2": "1",
        "species.length.sp2": "150",
        "species.size.proportion.sp2": "1.0",
        "species.trophic.level.sp2": "4.0",
        "species.age.sp2": "5",
        "species.length2weight.condition.factor.sp2": "0.01",
        "species.length2weight.allometric.power.sp2": "3.0",
        "predation.predprey.sizeratio.max.sp2": "10",
        "predation.predprey.sizeratio.min.sp2": "2",
        "predation.ingestion.rate.max.sp2": "3.5",
        "species.biomass.total.sp2": "100",
    }
    return cfg | extra


class TestBackgroundLogMultiplier:
    def test_the_log_form_is_exponentiated(self):
        from osmose.engine.background import parse_background_species

        log = parse_background_species(
            _background_cfg(**{"species.multiplier.log.sp2": str(math.log(2.5))}), 2, 12
        )
        assert log[0].multiplier == pytest.approx(2.5, rel=1e-12)
        plain = parse_background_species(
            _background_cfg(**{"species.multiplier.sp2": "2.5"}), 2, 12
        )
        assert plain[0].multiplier == 2.5
        assert parse_background_species(_background_cfg(), 2, 12)[0].multiplier == 1.0

    def test_both_forms_together_are_refused(self):
        from osmose.engine.background import parse_background_species

        cfg = _background_cfg(
            **{"species.multiplier.log.sp2": "0.5", "species.multiplier.sp2": "2"}
        )
        with pytest.raises(ValueError, match="species.multiplier.log.sp2"):
            parse_background_species(cfg, 2, 12)


class TestResourceVarname:
    def _write(self, tmp_path, var: str) -> str:
        data = np.full((2, 2, 2), 7.0, dtype=np.float32)
        path = tmp_path / "forcing.nc"
        xr.Dataset({var: (["time", "lat", "lon"], data)}).to_netcdf(path)
        return str(path)

    def _config(self, nc: str, **extra: str) -> dict[str, str]:
        return {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "2",
            "species.type.sp8": "resource",
            "species.name.sp8": "Zoo",
            "species.size.min.sp8": "0.01",
            "species.size.max.sp8": "0.1",
            "species.trophic.level.sp8": "2.0",
            "species.accessibility2fish.sp8": "0.5",
            "species.biomass.file.sp8": nc,
            "species.biomass.nsteps.year": "2",
        } | extra

    def test_the_named_variable_is_read(self, tmp_path):
        nc = self._write(tmp_path, "zooplankton_biomass")
        grid = Grid.from_dimensions(ny=2, nx=2)
        rs = ResourceState(
            config=self._config(nc, **{"species.biomass.varname.sp8": "zooplankton_biomass"}),
            grid=grid,
        )
        rs.update(0)
        assert np.allclose(rs.biomass[0, :4], 7.0 * 0.5)
        rs.close()

    def test_without_varname_the_variable_is_the_resource_name(self, tmp_path):
        nc = self._write(tmp_path, "Zoo")
        grid = Grid.from_dimensions(ny=2, nx=2)
        rs = ResourceState(config=self._config(nc), grid=grid)
        rs.update(0)
        assert np.allclose(rs.biomass[0, :4], 7.0 * 0.5)
        rs.close()

    def test_a_varname_missing_from_the_file_is_refused(self, tmp_path):
        nc = self._write(tmp_path, "Zoo")
        grid = Grid.from_dimensions(ny=2, nx=2)
        with pytest.raises(ValueError, match="species.biomass.varname.sp8"):
            ResourceState(
                config=self._config(nc, **{"species.biomass.varname.sp8": "nope"}), grid=grid
            )


def test_the_csharp_bundle_reads_the_named_variable(tmp_path):
    """The C# engine's resources are built from the same forcing: a resource whose variable is
    renamed in its file, and named by species.biomass.varname, gets the same forcing in the
    bundle as before the rename. (No runtime is loaded: the bundle is built in Python.)"""
    from pathlib import Path

    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine_cs.build import build_bundle

    def forcing_of(raw):
        config, grid, *_ = PythonEngine()._prepare_run(raw, 1)
        b = build_bundle(config, raw, grid, seed=1)
        return np.array(b.get("resources.forcing"), copy=True)

    master = Path(osmose_demo("eec_full", tmp_path / "eec")["config_file"])
    raw = dict(OsmoseConfigReader().read(master))
    before = forcing_of(raw)
    rsc = [k for k, v in raw.items() if k.startswith("species.type.sp") and v.lower() == "resource"]
    fi = min(int(k.rsplit("sp", 1)[1]) for k in rsc)
    name = raw[f"species.name.sp{fi}"]
    src = master.parent / raw[f"species.file.sp{fi}"]
    with xr.open_dataset(src) as ds:
        ds.load().rename({name: "biomass_of_the_first_resource"}).to_netcdf(
            master.parent / "renamed.nc"
        )
    for k in rsc:
        raw[f"species.file.sp{k.rsplit('sp', 1)[1]}"] = "renamed.nc"
    raw[f"species.biomass.varname.sp{fi}"] = "biomass_of_the_first_resource"
    after = forcing_of(raw)
    assert before.shape == after.shape and (np.nan_to_num(before) != 0).any()
    assert np.array_equal(before, after, equal_nan=True)  # NaN over land
