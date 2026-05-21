from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.engine.output import write_outputs
from osmose.engine.simulate import StepOutput, TraitStats


EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"

# Example config has 8 focal species and 8 MortalityCause members.
_N_SPECIES = 8
_N_CAUSES = 8


def _bare_config_dict() -> dict[str, str]:
    # Real fixture — hand-rolled dicts fail EngineConfig.from_dict's
    # schema enforcement on linf/K/t0/length-weight params.
    from osmose.config import OsmoseConfigReader
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    return raw


def _build_outputs_with_trait_stats() -> list[StepOutput]:
    return [
        StepOutput(
            step=0,
            biomass=np.zeros(_N_SPECIES),
            abundance=np.zeros(_N_SPECIES),
            mortality_by_cause=np.zeros((_N_SPECIES, _N_CAUSES)),
            trait_stats={"imax": {0: TraitStats(mean=3.5, variance=0.1, n_individuals=50)}},
        ),
        StepOutput(
            step=1,
            biomass=np.zeros(_N_SPECIES),
            abundance=np.zeros(_N_SPECIES),
            mortality_by_cause=np.zeros((_N_SPECIES, _N_CAUSES)),
            trait_stats={"imax": {0: TraitStats(mean=3.2, variance=0.12, n_individuals=55)}},
        ),
    ]


def test_writer_creates_csv_with_expected_columns(tmp_path: Path) -> None:
    from osmose.engine.config import EngineConfig
    config = EngineConfig.from_dict(_bare_config_dict())
    outputs = _build_outputs_with_trait_stats()

    write_outputs(outputs, tmp_path, config, prefix="osm")

    path = tmp_path / "osm_genetic_trait_means_Simu0.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == ["Time", "species_id", "trait_name", "mean", "variance", "n_individuals"]
    assert len(df) == 2  # one row per (step, species, trait)
    assert set(df["trait_name"]) == {"imax"}


def test_writer_skipped_when_no_trait_stats(tmp_path: Path) -> None:
    from osmose.engine.config import EngineConfig
    config = EngineConfig.from_dict(_bare_config_dict())
    outputs = [
        StepOutput(
            step=0,
            biomass=np.zeros(_N_SPECIES),
            abundance=np.zeros(_N_SPECIES),
            mortality_by_cause=np.zeros((_N_SPECIES, _N_CAUSES)),
        ),
    ]
    write_outputs(outputs, tmp_path, config, prefix="osm")
    assert not (tmp_path / "osm_genetic_trait_means_Simu0.csv").exists()


def test_read_genetic_trait_means_round_trip(tmp_path: Path) -> None:
    from osmose.engine.config import EngineConfig
    from osmose.results import read_genetic_trait_means

    config = EngineConfig.from_dict(_bare_config_dict())
    outputs = _build_outputs_with_trait_stats()
    write_outputs(outputs, tmp_path, config, prefix="osm")

    ds = read_genetic_trait_means(tmp_path, prefix="osm")
    assert "mean" in ds.data_vars
    assert "variance" in ds.data_vars
    assert "n_individuals" in ds.data_vars
    assert set(ds.coords) >= {"Time", "species_id", "trait_name"}
    # Two timesteps, 1 species, 1 trait
    assert ds["mean"].sel(species_id=0, trait_name="imax").shape == (2,)
