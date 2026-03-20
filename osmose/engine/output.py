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

    # Write mortality CSVs per species
    _write_mortality_csvs(output_dir, prefix, outputs, config)

    # Write age/size distribution CSVs
    _write_distribution_csvs(output_dir, prefix, outputs, config)


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


def _write_distribution_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species age/size distribution CSVs matching Java format."""
    times = np.array([o.step / config.n_dt_per_year for o in outputs])

    for label, attr_name in [
        ("biomassByAge", "biomass_by_age"),
        ("abundanceByAge", "abundance_by_age"),
        ("biomassBySize", "biomass_by_size"),
        ("abundanceBySize", "abundance_by_size"),
    ]:
        first_out = next((o for o in outputs if getattr(o, attr_name) is not None), None)
        if first_out is None:
            continue
        dist_data = getattr(first_out, attr_name)
        for sp_idx, sp_name in enumerate(config.species_names):
            if sp_idx not in dist_data:
                continue
            n_bins = len(dist_data[sp_idx])
            data_matrix = np.zeros((len(outputs), n_bins))
            for t_idx, o in enumerate(outputs):
                d = getattr(o, attr_name)
                if d is not None and sp_idx in d:
                    data_matrix[t_idx, : len(d[sp_idx])] = d[sp_idx]

            if "Age" in label:
                columns = [str(i) for i in range(n_bins)]
            else:
                edges = np.arange(
                    config.output_size_min,
                    config.output_size_min + n_bins * config.output_size_incr,
                    config.output_size_incr,
                )
                columns = [f"{e:.1f}" for e in edges]

            df = pd.DataFrame(data_matrix, columns=columns)
            df.insert(0, "Time", times)
            path = output_dir / f"{prefix}_{label}_{sp_name}_Simu0.csv"
            df.to_csv(path, index=False)


def _write_mortality_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species mortality rate CSVs matching Java format."""
    from osmose.engine.state import MortalityCause

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    cause_names = [c.name.capitalize() for c in MortalityCause]

    mort_dir = output_dir / "Mortality"
    mort_dir.mkdir(exist_ok=True)

    for sp_idx, sp_name in enumerate(config.species_names):
        # Extract mortality data for this species across all timesteps
        data = np.array([o.mortality_by_cause[sp_idx] for o in outputs])
        df = pd.DataFrame(data, columns=cause_names)
        df.insert(0, "Time", times)

        path = mort_dir / f"{prefix}_mortalityRate-{sp_name}_Simu0.csv"
        with open(path, "w") as f:
            f.write(f'"Mortality rates per time step for {sp_name}"\n')
            df.to_csv(f, index=False)
