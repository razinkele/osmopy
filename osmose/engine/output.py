"""Output writer for the OSMOSE Python engine.

Writes CSV files matching Java's naming convention so that the existing
OsmoseResults reader works with either engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import StepOutput


def write_outputs(
    outputs: list[StepOutput],
    output_dir: Path,
    config: EngineConfig,
    prefix: str = "osmose",
) -> None:
    """Write simulation outputs to CSV files matching Java format.

    Creates:
      - {prefix}_biomass_Simu0.csv
      - {prefix}_abundance_Simu0.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    species = config.species_names

    # Collect time series
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    biomass_data = np.array([o.biomass for o in outputs])
    abundance_data = np.array([o.abundance for o in outputs])

    # Write biomass CSV
    _write_species_csv(
        output_dir / f"{prefix}_biomass_Simu0.csv",
        "Mean biomass (tons), including first ages specified in input",
        times,
        species,
        biomass_data,
    )

    # Write abundance CSV
    _write_species_csv(
        output_dir / f"{prefix}_abundance_Simu0.csv",
        "Mean abundance (number of fish), including first ages specified in input",
        times,
        species,
        abundance_data,
    )


def _write_species_csv(
    path: Path,
    description: str,
    times: np.ndarray,
    species: list[str],
    data: np.ndarray,
) -> None:
    """Write a species time-series CSV matching Java format."""
    df = pd.DataFrame(data, columns=species)
    df.insert(0, "Time", times)

    with open(path, "w") as f:
        f.write(f'"{description}"\n')
        df.to_csv(f, index=False)
