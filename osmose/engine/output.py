"""Output writer for the OSMOSE Python engine.

Writes CSV files matching Java's naming convention so that the existing
OsmoseResults reader works with either engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import StepOutput

if TYPE_CHECKING:
    from osmose.engine.economics.fleet import FleetState



def write_outputs(
    outputs: list[StepOutput],
    output_dir: Path,
    config: EngineConfig,
    prefix: str = "osm",
) -> None:
    """Write simulation outputs to CSV files matching Java format.

    Creates:
      - {prefix}_biomass_Simu0.csv
      - {prefix}_abundance_Simu0.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    species = config.all_species_names  # includes background

    # Collect time series
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    biomass_data = np.array([o.biomass for o in outputs])
    abundance_data = np.array([o.abundance for o in outputs])

    # Write biomass CSV
    _write_species_csv(
        output_dir / f"{prefix}_biomass_Simu0.csv",
        "Mean biomass (tons), excluding first ages specified in input",
        times,
        species,
        biomass_data,
    )

    # Write abundance CSV
    _write_species_csv(
        output_dir / f"{prefix}_abundance_Simu0.csv",
        "Mean abundance (number of fish), excluding first ages specified in input",
        times,
        species,
        abundance_data,
    )

    # Write mortality CSVs per species
    _write_mortality_csvs(output_dir, prefix, outputs, config)

    # Write yield CSV
    _write_yield_csv(output_dir, prefix, outputs, config)

    # Write age/size distribution CSVs
    _write_distribution_csvs(output_dir, prefix, outputs, config)

    # Write bioenergetic CSVs when enabled
    if config.bioen_enabled:
        _write_bioen_csvs(output_dir, prefix, outputs, config)

    # Write diet CSV if diet data is present
    diet_arrays = [o.diet_by_species for o in outputs if o.diet_by_species is not None]
    if diet_arrays:
        # Sum diet across all timesteps, then normalize at write time
        total_diet = np.sum(diet_arrays, axis=0)
        prey_names = config.all_species_names
        predator_names = config.species_names
        write_diet_csv(
            output_dir / f"{prefix}_dietMatrix_Simu0.csv",
            total_diet,
            predator_names,
            prey_names,
        )


def _write_species_csv(
    path: Path,
    description: str,
    times: np.ndarray,
    species: list[str],
    data: np.ndarray,
) -> None:
    """Write a species time-series CSV matching Java format."""
    df = pd.DataFrame(data, columns=species)  # type: ignore[arg-type]
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

            df = pd.DataFrame(data_matrix, columns=columns)  # type: ignore[arg-type]
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
        df = pd.DataFrame(data, columns=cause_names)  # type: ignore[arg-type]
        df.insert(0, "Time", times)

        path = mort_dir / f"{prefix}_mortalityRate-{sp_name}_Simu0.csv"
        with open(path, "w") as f:
            f.write(f'"Mortality rates per time step for {sp_name}"\n')
            df.to_csv(f, index=False)


# ---------------------------------------------------------------------------
# Diet composition output
# ---------------------------------------------------------------------------


def aggregate_diet_by_species(
    diet_matrix: NDArray[np.float64],
    species_id: NDArray[np.int32],
    n_pred_species: int,
) -> NDArray[np.float64]:
    """Aggregate per-school diet matrix into per-species diet.

    Args:
        diet_matrix: shape (n_schools, n_prey_columns) — biomass eaten per prey.
        species_id: species index for each school.
        n_pred_species: number of predator species (focal).

    Returns:
        Array of shape (n_pred_species, n_prey_columns) with summed biomass.
    """
    n_prey_cols = diet_matrix.shape[1]
    result = np.zeros((n_pred_species, n_prey_cols), dtype=np.float64)
    # Filter to focal species only (exclude background species)
    focal_mask = species_id < n_pred_species
    if focal_mask.any():
        np.add.at(result, species_id[focal_mask], diet_matrix[focal_mask])
    return result


def write_diet_csv(
    path: Path,
    diet_by_species: NDArray[np.float64],
    predator_names: list[str],
    prey_names: list[str],
) -> None:
    """Write diet composition as a CSV with percentage values.

    Rows = prey species, columns = predator species.
    Values are the percentage of each predator's total diet from each prey.

    Args:
        path: Output file path.
        diet_by_species: shape (n_predators, n_prey) — biomass eaten.
        predator_names: names for each predator species.
        prey_names: names for each prey column (focal + resource species).
    """
    # Convert to percentages per predator
    totals = diet_by_species.sum(axis=1, keepdims=True)
    # Avoid division by zero
    safe_totals = np.where(totals > 0, totals, 1.0)
    pct = diet_by_species / safe_totals * 100.0

    # Build DataFrame: rows = prey, columns = predator
    df = pd.DataFrame(pct.T, index=prey_names, columns=predator_names)  # type: ignore[arg-type]
    df.index.name = "Prey"
    df.to_csv(path)


# ---------------------------------------------------------------------------
# Yield output
# ---------------------------------------------------------------------------


def _write_yield_csv(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write fishing yield CSV matching Java format."""
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    yield_data = np.array(
        [
            o.yield_by_species if o.yield_by_species is not None else np.zeros(config.n_species)
            for o in outputs
        ]
    )
    species = config.species_names

    _write_species_csv(
        output_dir / f"{prefix}_yield_Simu0.csv",
        "Fishing yield (tons) per time step",
        times,
        species,
        yield_data,
    )


# ---------------------------------------------------------------------------
# Bioenergetic output
# ---------------------------------------------------------------------------


def _write_bioen_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write bioen-specific per-species CSVs into a Bioen/ subdirectory."""
    bioen_dir = output_dir / "Bioen"
    bioen_dir.mkdir(exist_ok=True)

    times = np.array([o.step / config.n_dt_per_year for o in outputs])

    bioen_outputs = [
        ("bioen_e_net_by_species", "meanEnet", True),
        ("bioen_ingestion_by_species", "ingestion", config.output_bioen_ingest),
        ("bioen_maint_by_species", "maintenance", config.output_bioen_maint),
        ("bioen_rho_by_species", "rho", config.output_bioen_rho),
        ("bioen_size_inf_by_species", "sizeInf", config.output_bioen_sizeinf),
    ]

    for attr, label, enabled in bioen_outputs:
        if not enabled:
            continue
        data_list = [getattr(o, attr) for o in outputs]
        if not any(d is not None for d in data_list):
            continue
        data = np.array([d if d is not None else np.zeros(config.n_species) for d in data_list])
        for sp_idx, sp_name in enumerate(config.species_names):
            df = pd.DataFrame({"Time": times, label: data[:, sp_idx]})
            df.to_csv(bioen_dir / f"{prefix}_{label}_{sp_name}_Simu0.csv", index=False)


# TODO(v0.7): Spatial bioen outputs — Java has SpatialEnetOutput, SpatialEnetOutputjuv,
# SpatialEnetOutputlarvae, SpatialdGOutput. Requires: (1) per-cell aggregation framework
# (see docs/parity-roadmap.md §5.4), and (2) Ev-OSMOSE bioenergetics module
# (see docs/parity-roadmap.md §6.1–6.5). Blocked on both; only needed for Ev-OSMOSE configs.


# ---------------------------------------------------------------------------
# NetCDF output
# ---------------------------------------------------------------------------


def write_outputs_netcdf(
    outputs: list[StepOutput],
    path: Path,
    config: EngineConfig,
) -> None:
    """Write simulation outputs to NetCDF format using xarray.

    Creates a dataset with biomass and abundance as variables,
    with time and species dimensions.
    """
    import xarray as xr

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    biomass_data = np.array([o.biomass for o in outputs])
    abundance_data = np.array([o.abundance for o in outputs])

    n_species = biomass_data.shape[1]
    species_names = config.all_species_names[:n_species]

    ds = xr.Dataset(
        {
            "biomass": (["time", "species"], biomass_data),
            "abundance": (["time", "species"], abundance_data),
        },
        coords={
            "time": times,
            "species": species_names,
        },
        attrs={
            "description": "OSMOSE Python engine output",
            "n_dt_per_year": config.n_dt_per_year,
            "n_year": config.n_year,
        },
    )

    # Add yield if available
    yield_data = [o.yield_by_species for o in outputs if o.yield_by_species is not None]
    if yield_data:
        yield_arr = np.array(yield_data)
        focal_names = config.species_names[: yield_arr.shape[1]]
        ds["yield"] = (["time", "focal_species"], yield_arr)
        ds.coords["focal_species"] = focal_names

    ds.to_netcdf(path)

    # TODO(v0.7): Add size/age distribution outputs — see docs/parity-roadmap.md §5.3.
    # Blocked on per-school data being carried through StepOutput (currently only aggregates).
    # TODO(v0.7): Add spatial biomass/abundance maps — see docs/parity-roadmap.md §5.4.
    # Blocked on per-cell aggregation framework; needs cell-indexed data in StepOutput.


# ---------------------------------------------------------------------------
# Economic output
# ---------------------------------------------------------------------------


def write_economic_outputs(
    fleet_state: "FleetState",
    output_dir: Path,
) -> None:
    """Write economic CSV output files.

    Files written per fleet:
    - econ_effort_<fleet>.csv: Effort map (ny × nx snapshot)
    - econ_revenue_<fleet>.csv: Per-vessel revenue
    - econ_costs_<fleet>.csv: Per-vessel costs

    One file across all fleets:
    - econ_profit_summary.csv: Per-fleet total revenue, costs, profit
    """
    import csv as _csv

    if fleet_state is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    profit_rows: list[list] = []

    for fi, fleet in enumerate(fleet_state.fleets):
        name = fleet.name
        vessel_mask = fleet_state.vessel_fleet == fi

        # Effort map: shape (ny, nx)
        effort = fleet_state.effort_map[fi]
        np.savetxt(output_dir / f"econ_effort_{name}.csv", effort, delimiter=";", fmt="%.1f")

        # Revenue per vessel
        revenue_arr = fleet_state.vessel_revenue[vessel_mask]
        np.savetxt(
            output_dir / f"econ_revenue_{name}.csv",
            revenue_arr.reshape(1, -1),
            delimiter=";",
            fmt="%.2f",
        )

        # Costs per vessel
        costs_arr = fleet_state.vessel_costs[vessel_mask]
        np.savetxt(
            output_dir / f"econ_costs_{name}.csv",
            costs_arr.reshape(1, -1),
            delimiter=";",
            fmt="%.2f",
        )

        total_rev = float(revenue_arr.sum())
        total_cost = float(costs_arr.sum())
        profit_rows.append([name, total_rev, total_cost, total_rev - total_cost])

    with open(output_dir / "econ_profit_summary.csv", "w", newline="") as f:
        writer = _csv.writer(f, delimiter=";")
        writer.writerow(["fleet", "revenue", "costs", "profit"])
        writer.writerows(profit_rows)
