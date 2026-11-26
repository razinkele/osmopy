"""Restart files in Java 4.4.1's format (``SchoolSetSnapshot`` / ``NetcdfPopulator``).

A snapshot is the school table after a step: ``restart/<prefix>_snapshot_step<i>.nc.<rank>``
with one row per school (``nschool``): ``species`` (int), ``x``/``y`` (cell indices, float),
``abundance`` (double), ``age`` (years = ageDt / ndt), ``length`` (cm), ``weight`` (g),
``trophiclevel``; with bioenergetics ``gonadWeight`` (g) and ``maturity`` (0/1). Global attribute
``step`` = i. Background schools and resources are not in it (the engine re-injects / resets them
every step), and neither are the RNG state, the lagged starvation rate or depletable resource
biomass: as in Java, a restart re-initialises from a population, it is not a bit-exact
continuation.

Writing (``Simulation.run``): after step ``i`` when ``i >= spinup - 1`` and
``(i - spinup + 1) % frequency == 0``, and always after the last step;
``simulation.restart.enabled`` absent means TRUE (owner decision 2026-11-26: mirror Java).

Reading (``Simulation.init`` / ``PopulatingProcess``): ``simulation.restart.file`` resumes at step
``step + 1``; otherwise ``population.initialization.file`` (R's ``initialize_osmose()`` files)
starts at step 0. A file ``<name>`` or ``<name>.<rank>`` is accepted, the ranked one preferred.

Genetics (genotypes) is not restored or written yet: a genetic config with a restart file is
refused, and a genetic run writes no snapshot (with a warning).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from osmose.engine.state import SchoolState
from osmose.netcdf_io import open_dataset

_log = logging.getLogger("osmose")
RANK = 0


def snapshot_due(step: int, spinup: int, frequency: int) -> bool:
    """Java ``Simulation.run``: a snapshot after ``step`` (the final one is added by the caller)."""
    if frequency <= 0:
        return False
    return step >= spinup - 1 and (step - spinup + 1) % frequency == 0


def snapshot_path(output_dir: Path, prefix: str, step: int) -> Path:
    return Path(output_dir) / "restart" / f"{prefix}_snapshot_step{step}.nc.{RANK}"


def resolve_restart_file(name: str) -> Path:
    """Java's lookup: ``<name>.<rank>`` preferred over ``<name>``; error when neither exists."""
    plain = Path(name)
    ranked = Path(f"{name}.{RANK}")
    if ranked.exists():
        if plain.exists():
            _log.warning("Found %s and %s: using %s (as Java)", plain, ranked, ranked)
        return ranked
    if plain.exists():
        return plain
    raise FileNotFoundError(f"restart file not found: neither {plain} nor {ranked} exists")


def _validate(path, n, sp, x, y, abundance, length, weight, tl, age_dt, config, grid) -> None:
    """Reject a file whose values the engines cannot take safely: the cells index the kernels'
    arrays, so they must be inside the grid (or -1 for an unlocated school); the numbers must
    be finite and non-negative; ages no older than the species' lifespan."""
    problems = []

    def bad(name, mask):
        if mask.any():
            problems.append(f"{name}: {int(mask.sum())} of {n} rows")

    unlocated = (x == -1) & (y == -1)
    bad(
        "x/y not whole numbers",
        ~np.isfinite(x) | ~np.isfinite(y) | (x != np.floor(x)) | (y != np.floor(y)),
    )
    if grid is not None:
        inside = (x >= 0) & (x < grid.nx) & (y >= 0) & (y < grid.ny)
        bad("x/y outside the grid (and not -1, -1)", ~(inside | unlocated))
    for name, v in (("abundance", abundance), ("length", length), ("weight", weight)):
        bad(f"{name} negative or not finite", ~np.isfinite(v) | (v < 0))
    bad("trophiclevel not finite", ~np.isfinite(tl))
    lifespan = np.asarray(config.lifespan_dt)[sp] if n else np.zeros(0)
    bad("age negative or beyond the lifespan", (age_dt < 0) | (age_dt > lifespan))
    if problems:
        raise ValueError(f"{path}: invalid restart file: " + "; ".join(problems))


def read_snapshot(path: Path, config, grid=None) -> tuple[SchoolState, int]:
    """The school table of a snapshot as a ``SchoolState``, and the step it was taken after.

    Java ``NetcdfPopulator``: each row becomes a new school of that species at cell (x, y), with
    ``ageDt = round(age * ndt)``, weight g -> t; everything else as for a new school (egg while
    ``ageDt < firstFeedingAgeDt``, which is 1 in both engines)."""
    with open_dataset(path) as ds:
        ds = ds.load()
    if "step" not in ds.attrs:
        raise ValueError(f"{path}: the restart file is missing a 'step' global attribute")
    step = int(ds.attrs["step"])
    sp = ds["species"].values.astype(np.int32)
    n = len(sp)
    if n and (sp.min() < 0 or sp.max() >= config.n_species):
        raise ValueError(f"{path}: species index outside 0..{config.n_species - 1}")
    ndt = config.n_dt_per_year
    age_dt = np.rint(ds["age"].values.astype(np.float32) * ndt).astype(np.int32)
    length = ds["length"].values.astype(np.float32).astype(np.float64)
    weight = ds["weight"].values.astype(np.float32).astype(np.float64) * 1e-6  # g -> t
    abundance = ds["abundance"].values.astype(np.float64)
    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)
    tl = ds["trophiclevel"].values.astype(np.float32).astype(np.float64)
    _validate(path, n, sp, x, y, abundance, length, weight, tl, age_dt, config, grid)
    unlocated = (x == -1) & (y == -1)
    first_feeding = np.ones(n, dtype=np.int32)
    state = SchoolState.create(n_schools=n, species_id=sp).replace(
        abundance=abundance,
        biomass=abundance * weight,
        length=length,
        length_start=length.copy(),
        weight=weight,
        age_dt=age_dt,
        trophic_level=tl,
        cell_x=x.astype(np.int32),
        cell_y=y.astype(np.int32),
        is_out=unlocated,
        first_feeding_age_dt=first_feeding,
        is_egg=age_dt < first_feeding,
    )
    if getattr(config, "bioen_enabled", False):
        if "gonadWeight" in ds:
            state = state.replace(
                gonad_weight=ds["gonadWeight"].values.astype(np.float32).astype(np.float64) * 1e-6
            )
        if "maturity" in ds:
            state = state.replace(is_mature=ds["maturity"].values.astype(np.int32) == 1)
    return state, step


def write_snapshot(state: SchoolState, step: int, path: Path, config) -> None:
    """Java ``SchoolSetSnapshot.makeSnapshot``: the focal schools, Java's names, types and units."""
    import xarray as xr

    focal = np.flatnonzero(np.asarray(state.species_id) < config.n_species)
    ndt = config.n_dt_per_year
    data = {
        "species": ("nschool", np.asarray(state.species_id)[focal].astype(np.int32),
                    {"description": "index of the species"}),
        "x": ("nschool", np.asarray(state.cell_x)[focal].astype(np.float32),
              {"units": "scalar", "description": "x-grid index of the school"}),
        "y": ("nschool", np.asarray(state.cell_y)[focal].astype(np.float32),
              {"units": "scalar", "description": "y-grid index of the school"}),
        "abundance": ("nschool", np.asarray(state.abundance)[focal].astype(np.float64),
                      {"units": "scalar", "description": "number of fish in the school"}),
        "age": ("nschool", (np.asarray(state.age_dt)[focal] / ndt).astype(np.float32),
                {"units": "year", "description": "age of the school in year"}),
        "length": ("nschool", np.asarray(state.length)[focal].astype(np.float32),
                   {"units": "cm", "description": "length of the fish in the school in centimeter"}),
        "weight": ("nschool", (np.asarray(state.weight)[focal] * 1e6).astype(np.float32),
                   {"units": "g", "description": "weight of the fish in the school in gram"}),
        "trophiclevel": ("nschool", np.asarray(state.trophic_level)[focal].astype(np.float32),
                         {"units": "scalar", "description": "trophiclevel of the fish in the school"}),
    }  # fmt: skip
    if getattr(config, "bioen_enabled", False):
        gw = getattr(state, "gonad_weight", None)
        mature = getattr(state, "is_mature", None)
        n = len(focal)
        data["gonadWeight"] = (
            "nschool",
            ((np.asarray(gw)[focal] if gw is not None else np.zeros(n)) * 1e6).astype(np.float32),
            {"units": "g", "description": "gonad weight of the fish in the school in gram"},
        )
        data["maturity"] = (
            "nschool",
            (np.asarray(mature)[focal] if mature is not None else np.zeros(n, bool)).astype(
                np.int32
            ),
            {"units": "", "description": "True if the school is mature"},
        )
    names = " ".join(f"{i}={n}" for i, n in enumerate(config.species_names[: config.n_species]))
    ds = xr.Dataset(
        {k: (dims, v, attrs) for k, (dims, v, attrs) in data.items()},
        attrs={"step": str(step), "species": names},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def write_snapshots(snapshots: list[tuple[int, SchoolState]], output_dir: Path, config) -> None:
    """Write every captured snapshot (``simulate``'s ``restart_snapshots``)."""
    for step, state in snapshots:
        write_snapshot(
            state, step, snapshot_path(output_dir, config.output_file_prefix, step), config
        )
