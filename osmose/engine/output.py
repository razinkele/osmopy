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
    *,
    grid=None,
) -> None:
    """Write simulation outputs to CSV files matching Java format.

    Build-then-write: each family's CSV is built as a DataFrame via
    _build_*_dataframes, then _write_species_csv writes it to disk
    with the appropriate commentary header. The build helpers are the
    shared source of truth for disk and in-memory (OsmoseResults.from_outputs)
    paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    species_dfs = _build_species_dataframes(outputs, config)
    headers = {
        "biomass": "Mean biomass (tons), excluding first ages specified in input",
        "abundance": "Mean abundance (number of fish), excluding first ages specified in input",
    }
    for key in ("biomass", "abundance"):
        df = species_dfs[key]
        times = df["Time"].values
        species = [c for c in df.columns if c != "Time"]
        data = df[species].values
        _write_species_csv(
            output_dir / f"{prefix}_{key}_Simu0.csv",
            headers[key],
            times,
            species,
            data,
        )

    _write_mortality_csvs(output_dir, prefix, outputs, config)
    _write_yield_csv(output_dir, prefix, outputs, config)
    _write_distribution_csvs(output_dir, prefix, outputs, config)

    if config.bioen_enabled:
        _write_bioen_csvs(output_dir, prefix, outputs, config)

    # Write diet CSV (Java-parity: one file, one row per recording period)
    if config.diet_output_enabled:
        step_matrices: list[NDArray[np.float64]] = []
        step_times: list[float] = []
        for o in outputs:
            if o.diet_by_species is not None:
                step_matrices.append(o.diet_by_species)
                step_times.append(o.step / config.n_dt_per_year)
        write_diet_csv(
            path=output_dir / f"{prefix}_dietMatrix_Simu0.csv",
            step_diet_matrices=step_matrices,
            step_times=step_times,
            predator_names=config.species_names,
            prey_names=config.all_species_names,
        )

    # Write spatial NetCDF outputs when enabled
    if config.output_spatial_enabled:
        write_outputs_netcdf_spatial(
            outputs,
            output_dir,
            prefix=prefix,
            sim_index=0,
            config=config,
            grid=grid,
        )


def _build_species_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for per-species time series (biomass, abundance).

    Returns a dict with keys "biomass" and "abundance", each mapping to a
    wide-form DataFrame with columns ["Time"] + config.all_species_names.
    These are the in-memory equivalents of the CSVs written by
    _write_species_csv (one row per step, one column per species).
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    species = config.all_species_names
    biomass_data = np.array([o.biomass for o in outputs])
    abundance_data = np.array([o.abundance for o in outputs])

    bio_df = pd.DataFrame(biomass_data, columns=list(species))
    bio_df.insert(0, "Time", times)

    abd_df = pd.DataFrame(abundance_data, columns=list(species))
    abd_df.insert(0, "Time", times)

    return {"biomass": bio_df, "abundance": abd_df}


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


def _build_distribution_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for per-species age/size distributions.

    Returns a dict keyed by f"{output_type}_{species_name}" (e.g.,
    "biomassByAge_cod"), each mapping to a wide-form DataFrame with
    columns ["Time"] + bin labels. One entry per (output_type, species)
    pair that has data in the outputs list.
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    result: dict[str, pd.DataFrame] = {}

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
            result[f"{label}_{sp_name}"] = df

    return result


def _write_distribution_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species age/size distribution CSVs matching Java format."""
    dfs = _build_distribution_dataframes(outputs, config)
    for key, df in dfs.items():
        path = output_dir / f"{prefix}_{key}_Simu0.csv"
        df.to_csv(path, index=False)


def _build_mortality_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for per-species mortality-rate outputs.

    Ported verbatim from _write_mortality_csvs; only change is returning
    DataFrames instead of writing them.

    Returns {f"mortalityRate_{species_name}": df} with columns
    ["Time"] + [cause.name.capitalize() for cause in MortalityCause].
    """
    from osmose.engine.state import MortalityCause

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    cause_names = [c.name.capitalize() for c in MortalityCause]

    result: dict[str, pd.DataFrame] = {}
    for sp_idx, sp_name in enumerate(config.species_names):
        data = np.array([o.mortality_by_cause[sp_idx] for o in outputs])
        df = pd.DataFrame(data, columns=cause_names)  # type: ignore[arg-type]
        df.insert(0, "Time", times)
        result[f"mortalityRate_{sp_name}"] = df
    return result


def _write_mortality_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species mortality rate CSVs matching Java format."""
    mort_dir = output_dir / "Mortality"
    mort_dir.mkdir(exist_ok=True)

    dfs = _build_mortality_dataframes(outputs, config)
    for key, df in dfs.items():
        sp_name = key.split("_", 1)[1]
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


def _build_diet_dataframe(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build the diet matrix DataFrame (long-form: one row per recording period).

    Caller must check config.diet_output_enabled before calling.
    Keyed by "dietMatrix" to match the single-file CSV output.
    Ported verbatim from write_diet_csv; only change is returning a
    DataFrame instead of writing it to disk.
    """
    step_matrices: list[NDArray[np.float64]] = []
    step_times: list[float] = []
    for o in outputs:
        if o.diet_by_species is not None:
            step_matrices.append(o.diet_by_species)
            step_times.append(o.step / config.n_dt_per_year)
    if not step_matrices:
        return {}

    predator_names = config.species_names
    prey_names = config.all_species_names
    n_pred, n_prey = len(predator_names), len(prey_names)
    columns = [f"{pred}_{prey}" for pred in predator_names for prey in prey_names]
    rows: list[list[float]] = []
    for mat, t in zip(step_matrices, step_times, strict=True):
        if mat.shape != (n_pred, n_prey):
            raise ValueError(
                f"diet matrix shape {mat.shape} != ({n_pred}, {n_prey}) at time {t}"
            )
        rows.append([t, *mat.reshape(-1).tolist()])
    df = pd.DataFrame(rows, columns=["Time", *columns])
    return {"dietMatrix": df}


def write_diet_csv(
    *,
    path: Path,
    step_diet_matrices: list[NDArray[np.float64]],
    step_times: list[float],
    predator_names: list[str],
    prey_names: list[str],
) -> None:
    """Write diet composition as one CSV per simulation with one row per
    recording period, matching Java ``DietOutput.java:217-222``.

    Schema: first column ``Time``; remaining columns are one per
    ``{predator}_{prey}`` pair in predator-major, prey-minor order.
    Values are BIOMASS EATEN in tonnes. Per-predator percentage
    normalization is available via ``_normalize_diet_matrix_to_percent``
    when callers need Java's percentage layout.

    No-op when ``step_diet_matrices`` is empty.
    """
    if not step_diet_matrices:
        return
    if len(step_diet_matrices) != len(step_times):
        raise ValueError(
            f"step_diet_matrices length {len(step_diet_matrices)} "
            f"!= step_times length {len(step_times)}"
        )

    n_pred, n_prey = len(predator_names), len(prey_names)
    columns = [f"{pred}_{prey}" for pred in predator_names for prey in prey_names]
    rows: list[list[float]] = []
    for mat, t in zip(step_diet_matrices, step_times, strict=True):
        if mat.shape != (n_pred, n_prey):
            raise ValueError(f"diet matrix shape {mat.shape} != ({n_pred}, {n_prey}) at time {t}")
        rows.append([t, *mat.reshape(-1).tolist()])
    df = pd.DataFrame(rows, columns=["Time", *columns])
    df.to_csv(path, index=False)


def _normalize_diet_matrix_to_percent(
    diet_by_species: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize a single (n_pred, n_prey) matrix to per-predator percentages."""
    totals = diet_by_species.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    return diet_by_species / safe_totals * 100.0


# ---------------------------------------------------------------------------
# Yield output
# ---------------------------------------------------------------------------


def _build_yield_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for yield outputs.

    Ported verbatim from _write_yield_csv; returns {"yield": df} with
    columns ["Time"] + config.species_names.
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    yield_data = np.array(
        [
            o.yield_by_species
            if o.yield_by_species is not None
            else np.zeros(config.n_species)
            for o in outputs
        ]
    )
    species = config.species_names
    df = pd.DataFrame(yield_data, columns=list(species))
    df.insert(0, "Time", times)
    return {"yield": df}


def _write_yield_csv(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write fishing yield CSV matching Java format."""
    dfs = _build_yield_dataframes(outputs, config)
    if "yield" not in dfs:
        return
    df = dfs["yield"]
    times = df["Time"].values
    species = [c for c in df.columns if c != "Time"]
    yield_data = df[species].values
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


def _build_bioen_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for bioenergetic outputs.

    Caller must check config.bioen_enabled before calling. Ported verbatim
    from _write_bioen_csvs; only change is returning DataFrames.

    Returns {f"{label}_{species_name}": df} for each enabled bioen output,
    where df has columns ["Time", label].
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])

    bioen_outputs = [
        ("bioen_e_net_by_species", "meanEnet", True),
        ("bioen_ingestion_by_species", "ingestion", config.output_bioen_ingest),
        ("bioen_maint_by_species", "maintenance", config.output_bioen_maint),
        ("bioen_rho_by_species", "rho", config.output_bioen_rho),
        ("bioen_size_inf_by_species", "sizeInf", config.output_bioen_sizeinf),
    ]

    result: dict[str, pd.DataFrame] = {}
    for attr, label, enabled in bioen_outputs:
        if not enabled:
            continue
        data_list = [getattr(o, attr) for o in outputs]
        if not any(d is not None for d in data_list):
            continue
        data = np.array([d if d is not None else np.zeros(config.n_species) for d in data_list])
        for sp_idx, sp_name in enumerate(config.species_names):
            df = pd.DataFrame({"Time": times, label: data[:, sp_idx]})
            result[f"{label}_{sp_name}"] = df
    return result


def _write_bioen_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write bioen-specific per-species CSVs into a Bioen/ subdirectory."""
    bioen_dir = output_dir / "Bioen"
    bioen_dir.mkdir(exist_ok=True)

    dfs = _build_bioen_dataframes(outputs, config)
    for key, df in dfs.items():
        path = bioen_dir / f"{prefix}_{key}_Simu0.csv"
        df.to_csv(path, index=False)


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
    """Write simulation outputs to NetCDF.

    Each of the 8 possible variables is gated by its own
    ``output.{var}.netcdf.enabled`` config key. When every toggle is
    off, no file is written.

    Ragged per-species distribution bins are padded to cross-species
    max with NaN. Declares CF-1.8 conventions; every float DataArray
    carries ``_FillValue = NaN``.
    """
    import xarray as xr

    want = {
        "biomass": config.output_biomass_netcdf,
        "abundance": config.output_abundance_netcdf,
        "yield": config.output_yield_biomass_netcdf
        and any(o.yield_by_species is not None for o in outputs),
        "biomass_by_age": config.output_biomass_byage_netcdf
        and any(o.biomass_by_age is not None for o in outputs),
        "abundance_by_age": config.output_abundance_byage_netcdf
        and any(o.abundance_by_age is not None for o in outputs),
        "biomass_by_size": config.output_biomass_bysize_netcdf
        and any(o.biomass_by_size is not None for o in outputs),
        "abundance_by_size": config.output_abundance_bysize_netcdf
        and any(o.abundance_by_size is not None for o in outputs),
        "mortality_by_cause": config.output_mortality_netcdf,
    }
    if not any(want.values()):
        return

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    n_species = len(outputs[0].biomass)
    species_names = config.all_species_names[:n_species]

    data_vars: dict = {}
    coords: dict = {"time": times, "species": species_names}

    if want["biomass"]:
        data_vars["biomass"] = (
            ["time", "species"],
            np.array([o.biomass for o in outputs]),
        )
    if want["abundance"]:
        data_vars["abundance"] = (
            ["time", "species"],
            np.array([o.abundance for o in outputs]),
        )
    if want["yield"]:
        yield_arr = np.array(
            [
                o.yield_by_species
                if o.yield_by_species is not None
                else np.full(config.n_species, np.nan)
                for o in outputs
            ]
        )
        data_vars["yield"] = (["time", "focal_species"], yield_arr)
        coords["focal_species"] = config.species_names[: yield_arr.shape[1]]

    def _pad(attr: str) -> tuple[np.ndarray, int]:
        max_bins = 0
        for o in outputs:
            d = getattr(o, attr)
            if d is None:
                continue
            for arr in d.values():
                max_bins = max(max_bins, len(arr))
        result = np.full((len(outputs), n_species, max_bins), np.nan)
        for t_idx, o in enumerate(outputs):
            d = getattr(o, attr)
            if d is None:
                continue
            for sp, arr in d.items():
                result[t_idx, sp, : len(arr)] = arr
        return result, max_bins

    if want["biomass_by_age"]:
        arr, n_age = _pad("biomass_by_age")
        data_vars["biomass_by_age"] = (["time", "species", "age_bin"], arr)
        coords["age_bin"] = np.arange(n_age, dtype=np.float64)
    if want["abundance_by_age"]:
        arr, n_age = _pad("abundance_by_age")
        data_vars["abundance_by_age"] = (["time", "species", "age_bin"], arr)
        coords.setdefault("age_bin", np.arange(n_age, dtype=np.float64))
    if want["biomass_by_size"]:
        arr, n_size = _pad("biomass_by_size")
        data_vars["biomass_by_size"] = (["time", "species", "size_bin"], arr)
        coords["size_bin"] = np.arange(n_size, dtype=np.float64)
    if want["abundance_by_size"]:
        arr, n_size = _pad("abundance_by_size")
        data_vars["abundance_by_size"] = (["time", "species", "size_bin"], arr)
        coords.setdefault("size_bin", np.arange(n_size, dtype=np.float64))

    if want["mortality_by_cause"]:
        from osmose.engine.state import MortalityCause

        data_vars["mortality_by_cause"] = (
            ["time", "species", "cause"],
            np.array([o.mortality_by_cause for o in outputs]),
        )
        coords["cause"] = [c.name.capitalize() for c in MortalityCause]
        # Match the existing CSV writer at osmose/engine/output.py:161
        # ("Predation", "Starvation", ..., "Aging"). Users comparing CSV
        # and NetCDF outputs see identical cause labels.

    dataset_attrs = {
        "description": "OSMOSE Python engine output",
        "n_dt_per_year": config.n_dt_per_year,
        "n_year": config.n_year,
        "Conventions": "CF-1.8",
        "distribution_padding": (
            "Ragged per-species bin counts are padded to cross-species "
            "max with NaN. Structural padding is indistinguishable from "
            "missing data — downstream tools treat both identically."
        ),
    }
    if "mortality_by_cause" in data_vars:
        # Glossary for opaque enum members (Foraging = bioen cost-of-foraging;
        # Out = advected-out-of-domain). Attached when mortality var is present.
        dataset_attrs["cause_descriptions"] = (
            "Predation: schools consumed by other schools; "
            "Starvation: failed energy budget; "
            "Additional: residual/M-other; "
            "Fishing: captured by fishing mortality; "
            "Out: advected out of domain; "
            "Foraging: bioenergetic cost-of-foraging (Ev-OSMOSE only); "
            "Discards: discarded catch; "
            "Aging: senescence at lifespan."
        )
    ds = xr.Dataset(data_vars, coords=coords, attrs=dataset_attrs)
    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name].encoding["_FillValue"] = np.float64("nan")
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Spatial NetCDF output
# ---------------------------------------------------------------------------


def write_outputs_netcdf_spatial(
    outputs: list[StepOutput],
    output_dir: Path,
    *,
    prefix: str = "osm",
    sim_index: int = 0,
    config: EngineConfig,
    grid=None,
) -> None:
    """Write per-cell spatial outputs as NetCDF files.

    One file is produced per enabled spatial variant:
      - {prefix}_spatial_biomass_Simu{i}.nc   (output.spatial.biomass.enabled)
      - {prefix}_spatial_abundance_Simu{i}.nc (output.spatial.abundance.enabled)
      - {prefix}_spatial_yield_Simu{i}.nc     (output.spatial.yield.biomass.enabled)

    Dims: (time, species, lat, lon).
    Land cells (grid.ocean_mask == False) are written as NaN; ocean cells with
    no schools this period hold 0.0 (documented in attrs.nan_semantics).
    """
    import xarray as xr

    has_spatial = any(o.spatial_biomass is not None for o in outputs)
    if not has_spatial:
        return

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    focal_names = list(config.species_names[: config.n_species])
    n_sp = config.n_species
    n_t = len(outputs)

    # Determine grid dimensions: use grid object if present, else infer from data
    if grid is not None:
        ny, nx = grid.ny, grid.nx
        lat = (
            grid.lat
            if hasattr(grid, "lat") and grid.lat is not None
            else np.arange(ny, dtype=np.float64)
        )
        lon = (
            grid.lon
            if hasattr(grid, "lon") and grid.lon is not None
            else np.arange(nx, dtype=np.float64)
        )
        ocean_mask = (
            grid.ocean_mask if hasattr(grid, "ocean_mask") and grid.ocean_mask is not None else None
        )
    else:
        # Infer shape from first non-None spatial_biomass
        sample = next((o.spatial_biomass for o in outputs if o.spatial_biomass is not None), None)
        if sample is None:
            return
        sample_arr = next(iter(sample.values()))
        ny, nx = sample_arr.shape
        lat = np.arange(ny, dtype=np.float64)
        lon = np.arange(nx, dtype=np.float64)
        ocean_mask = None

    coords = {
        "time": times,
        "species": focal_names,
        "lat": lat,
        "lon": lon,
    }

    _dataset_attrs_base = {
        "Conventions": "CF-1.8",
        "description": "OSMOSE Python engine spatial output",
        "n_dt_per_year": config.n_dt_per_year,
        "n_year": config.n_year,
        "time_convention": "step / n_dt_per_year (fractional years from simulation start)",
        "nan_semantics": (
            "NaN marks land cells (ocean_mask==False). "
            "Ocean cells with no schools in the averaging window hold 0.0 — "
            "distinguishable from NaN/land."
        ),
        "spatial_coord_source": "grid object" if grid is not None else "inferred (no grid passed)",
    }

    cutoff = getattr(config, "output_cutoff_age", None)
    if cutoff is not None:
        _dataset_attrs_base["cutoff_age_note"] = (
            "Schools younger than output_cutoff_age are excluded from aggregation."
        )
    _dataset_attrs_base["abundance_period_mean_note"] = (
        "spatial_abundance is the period mean of instantaneous abundance; "
        "it is NOT a cumulative count."
    )

    def _build_arr(attr: str, op: str) -> np.ndarray:
        """Stack per-step spatial dicts into (time, species, lat, lon) array."""
        result = np.full((n_t, n_sp, ny, nx), np.nan)
        for t_idx, o in enumerate(outputs):
            d = getattr(o, attr)
            if d is None:
                continue
            for sp in range(n_sp):
                plane = d.get(sp)
                if plane is None:
                    continue
                arr = plane.copy().astype(np.float64)
                if ocean_mask is not None:
                    arr[~ocean_mask] = np.nan
                result[t_idx, sp] = arr
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_Simu{sim_index}.nc"

    if config.output_spatial_biomass:
        arr = _build_arr("spatial_biomass", "mean")
        da = xr.DataArray(
            arr,
            dims=["time", "species", "lat", "lon"],
            attrs={
                "units": "tonnes",
                "long_name": "Spatial mean biomass per species per cell",
                "cell_methods": "time: mean",
            },
        )
        ds = xr.Dataset({"spatial_biomass": da}, coords=coords, attrs=dict(_dataset_attrs_base))
        ds["spatial_biomass"].encoding["_FillValue"] = np.float64("nan")
        ds.to_netcdf(output_dir / f"{prefix}_spatial_biomass{suffix}")

    if config.output_spatial_abundance:
        arr = _build_arr("spatial_abundance", "mean")
        da = xr.DataArray(
            arr,
            dims=["time", "species", "lat", "lon"],
            attrs={
                "units": "individuals",
                "long_name": "Spatial mean abundance per species per cell",
                "cell_methods": "time: mean",
            },
        )
        ds = xr.Dataset({"spatial_abundance": da}, coords=coords, attrs=dict(_dataset_attrs_base))
        ds["spatial_abundance"].encoding["_FillValue"] = np.float64("nan")
        ds.to_netcdf(output_dir / f"{prefix}_spatial_abundance{suffix}")

    if config.output_spatial_yield_biomass:
        arr = _build_arr("spatial_yield", "sum")
        da = xr.DataArray(
            arr,
            dims=["time", "species", "lat", "lon"],
            attrs={
                "units": "tonnes",
                "long_name": "Spatial yield biomass per species per cell (fishing mortality * weight)",
                "cell_methods": "time: sum",
            },
        )
        ds = xr.Dataset({"spatial_yield": da}, coords=coords, attrs=dict(_dataset_attrs_base))
        ds["spatial_yield"].encoding["_FillValue"] = np.float64("nan")
        ds.to_netcdf(output_dir / f"{prefix}_spatial_yield{suffix}")


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
