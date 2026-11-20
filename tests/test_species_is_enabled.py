"""OSMOSE 4.4.1's ``species.is.enabled.sp#`` (#137).

Java (``Configuration.java``, v4.4.1 lines 523-552) drops a disabled focal species from its list
and numbers the remaining ones in order, while every key keeps its file index. The Python engine
numbers focal species 0..n-1 straight from the keys, so it renumbers the config instead: the
disabled species' keys go, the later focal species move down, and keys of background and
resource species (file indices from n up) are left as they are.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig, drop_disabled_species


def _three(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "3",
        "mortality.subdt": "10",
    }
    for i, (name, linf) in enumerate((("Anchovy", "20.0"), ("Sardine", "25.0"), ("Hake", "90.0"))):
        cfg |= {
            f"simulation.nschool.sp{i}": "5",
            f"species.name.sp{i}": name,
            f"species.linf.sp{i}": linf,
            f"species.k.sp{i}": "0.3",
            f"species.t0.sp{i}": "-0.1",
            f"species.egg.size.sp{i}": "0.1",
            f"species.length2weight.condition.factor.sp{i}": "0.006",
            f"species.length2weight.allometric.power.sp{i}": "3.0",
            f"species.lifespan.sp{i}": "5",
            f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
            f"predation.ingestion.rate.max.sp{i}": "3.5",
            f"predation.efficiency.critical.sp{i}": "0.57",
            f"population.seeding.biomass.sp{i}": str(1000 * (i + 1)),
        }
    return cfg | extra


class TestTheRenumbering:
    def test_without_the_switch_nothing_changes(self):
        cfg = _three()
        assert drop_disabled_species(cfg) == cfg
        assert drop_disabled_species(cfg | {"species.is.enabled.sp1": "true"}) == cfg | {
            "species.is.enabled.sp1": "true"
        }

    def test_a_disabled_species_goes_and_the_later_ones_move_down(self):
        out = drop_disabled_species(_three(**{"species.is.enabled.sp1": "false"}))
        assert out["simulation.nspecies"] == "2"
        assert [out["species.name.sp0"], out["species.name.sp1"]] == ["Anchovy", "Hake"]
        assert out["species.linf.sp1"] == "90.0" and out["population.seeding.biomass.sp1"] == "3000"
        assert not any(k.endswith(".sp2") for k in out)
        assert "Sardine" not in out.values()

    def test_background_and_resource_keys_keep_their_index(self):
        cfg = _three(**{
            "species.is.enabled.sp0": "false",
            "species.type.sp3": "resource", "species.name.sp3": "Plankton",
            "species.type.sp4": "background", "species.name.sp4": "Seal",
        })  # fmt: skip
        out = drop_disabled_species(cfg)
        assert out["species.name.sp3"] == "Plankton" and out["species.name.sp4"] == "Seal"
        assert [out["species.name.sp0"], out["species.name.sp1"]] == ["Sardine", "Hake"]
        assert "species.name.sp2" not in out

    def test_the_movement_maps_of_a_disabled_species_go(self):
        cfg = _three(**{
            "species.is.enabled.sp1": "false",
            "movement.species.map0": "Anchovy", "movement.file.map0": "a.csv",
            "movement.species.map1": "Sardine", "movement.file.map1": "s.csv",
            "movement.steps.map1": "0;1",
            "fisheries.movement.fishery.map1": "trawl", "fisheries.movement.file.map1": "f.csv",
        })  # fmt: skip
        out = drop_disabled_species(cfg)
        assert out["movement.species.map0"] == "Anchovy"
        assert not any(k.startswith("movement.") and k.endswith(".map1") for k in out)
        # A fishery's maps share the map numbering but are not the species' maps: kept.
        assert out["fisheries.movement.fishery.map1"] == "trawl"

    def test_disabling_every_species_is_refused(self):
        cfg = _three(**{f"species.is.enabled.sp{i}": "false" for i in range(3)})
        with pytest.raises(ValueError, match="species.is.enabled"):
            drop_disabled_species(cfg)

    @pytest.mark.parametrize("value", ["0", "no", "False", "FALSE", "off", "yes"])
    def test_anything_but_true_disables_as_javas_boolean_valueof(self, value):
        out = drop_disabled_species(_three(**{"species.is.enabled.sp1": value}))
        assert out["simulation.nspecies"] == "2"

    @pytest.mark.parametrize("value", ["true", "True", " TRUE "])
    def test_true_in_any_case_enables(self, value):
        cfg = _three(**{"species.is.enabled.sp1": value})
        assert drop_disabled_species(cfg) == cfg

    def test_an_empty_value_is_refused_not_silently_disabled(self):
        with pytest.raises(ValueError, match="species.is.enabled.sp1 is empty"):
            drop_disabled_species(_three(**{"species.is.enabled.sp1": ""}))


class TestTheEngine:
    def test_the_config_has_the_remaining_species(self):
        c = EngineConfig.from_dict(_three(**{"species.is.enabled.sp1": "false"}))
        assert c.n_species == 2
        assert list(c.species_names) == ["Anchovy", "Hake"]
        assert np.asarray(c.linf)[:2].tolist() == [20.0, 90.0]

    def test_a_run_without_a_species_equals_the_config_written_without_it(self):
        """The switch must give the same run as a config that never had the species."""
        from osmose.engine.grid import Grid
        from osmose.engine.simulate import simulate

        grid = Grid.from_dimensions(ny=3, nx=3)
        switched = simulate(
            EngineConfig.from_dict(_three(**{"species.is.enabled.sp1": "false"})),
            grid,
            np.random.default_rng(4),
        )
        by_hand = {k: v for k, v in _three().items() if not k.endswith(".sp1")}
        by_hand = {
            (k[: -len(".sp2")] + ".sp1" if k.endswith(".sp2") else k): v for k, v in by_hand.items()
        }
        by_hand["simulation.nspecies"] = "2"
        written = simulate(EngineConfig.from_dict(by_hand), grid, np.random.default_rng(4))
        for a, b in zip(switched, written, strict=True):
            assert np.array_equal(a.biomass, b.biomass)
        assert np.asarray(switched[-1].biomass).shape[0] >= 2


def test_baltic_runs_with_one_species_disabled(tmp_path):
    """A real config: species names in movement maps, fishery and accessibility matrices."""
    from pathlib import Path

    import osmose.engine.simulate as sim
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    raw = dict(
        OsmoseConfigReader().read(Path(osmose_demo("baltic", tmp_path / "b")["config_file"]))
    )
    names = [raw[f"species.name.sp{i}"] for i in range(int(raw["simulation.nspecies"]))]
    smelt = names.index("smelt")
    raw |= {
        f"species.is.enabled.sp{smelt}": "false",
        "simulation.time.nyear": "1",
        "output.minimal.enabled": "true",
    }
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 2)
    assert config.n_species == len(names) - 1 and "smelt" not in config.species_names
    records = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    assert len(records) >= 1


def test_the_csharp_engine_accepts_the_switch():
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    reason = csharp_engine_block_reason(_three(**{"species.is.enabled.sp1": "false"})) or ""
    assert "species.is.enabled" not in reason


def test_the_csharp_bundle_is_built_from_the_renumbered_config(tmp_path):
    """The C# engine is handed the config and the raw dict; the raw dict still holds the
    disabled species' keys. The bundle must not depend on them: built with the original raw
    dict, it equals, array for array, the bundle built with the renumbered one. (No runtime is
    loaded: the bundle is built in Python.)"""
    from pathlib import Path

    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine_cs.build import build_bundle

    raw = dict(
        OsmoseConfigReader().read(Path(osmose_demo("baltic", tmp_path / "b")["config_file"]))
    )
    names = [raw[f"species.name.sp{i}"] for i in range(int(raw["simulation.nspecies"]))]
    raw |= {f"species.is.enabled.sp{names.index('smelt')}": "false", "simulation.time.nyear": "1"}
    renumbered = drop_disabled_species(raw)
    assert renumbered != raw

    def bundle(raw_for_builder):
        config, grid, *_ = PythonEngine()._prepare_run(raw, 1)
        return build_bundle(config, raw_for_builder, grid, seed=1)

    a, b = bundle(raw), bundle(renumbered)
    assert a.names() == b.names()
    assert int(a.get("species.n_focal")) == len(names) - 1
    for name in a.names():
        x, y = a.get(name), b.get(name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), name
        else:
            assert x == y, name
