"""Incoming flux: external biomass injection from CSV time-series.

Simulates immigration/migration from outside the modeled domain by creating
new schools each timestep based on CSV-defined biomass inputs per age or size class.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.state import SchoolState


class _SpeciesFlux:
    """Pre-computed flux data for a single species."""

    def __init__(
        self,
        species_id: int,
        biomass_table: NDArray[np.float64],  # (n_timesteps, n_classes)
        lengths: NDArray[np.float64],  # per class
        ages_dt: NDArray[np.int32],  # per class (in simulation dt)
        weights: NDArray[np.float64],  # per class
    ) -> None:
        self.species_id = species_id
        self.biomass_table = biomass_table
        self.lengths = lengths
        self.ages_dt = ages_dt
        self.weights = weights
        self.n_rows = biomass_table.shape[0]


def _parse_flux_csv(path: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse a semicolon-separated flux CSV file.

    Returns:
        boundaries: 1D array of class boundary values from the header.
        data: 2D array of shape (n_timesteps, n_classes) with biomass values.
    """
    path = Path(path)
    text = path.read_text().strip()
    lines = text.split("\n")
    if len(lines) < 2:
        raise ValueError(f"Flux CSV must have header + at least 1 data row: {path}")

    boundaries = np.array([float(x.strip()) for x in lines[0].split(";") if x.strip()])
    n_classes = len(boundaries)
    data_lines = lines[1:]
    if n_classes == 1:
        # Single column: no delimiter needed
        data = np.array([[float(line.strip())] for line in data_lines])
    else:
        data = np.loadtxt(StringIO("\n".join(data_lines)), delimiter=";", ndmin=2)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return boundaries, data


def _age_to_length(age: float, linf: float, k: float, t0: float) -> float:
    """Von Bertalanffy: age (years) -> length (cm)."""
    return linf * (1.0 - np.exp(-k * (age - t0)))


def _length_to_age(length: float, linf: float, k: float, t0: float) -> float:
    """Inverse Von Bertalanffy: length (cm) -> age (years)."""
    ratio = length / linf
    if ratio >= 1.0:
        ratio = 0.999
    return t0 - np.log(1.0 - ratio) / k


class IncomingFluxState:
    """Manages incoming flux (external biomass injection) for the simulation.

    Loads CSV time-series for each species and generates new schools each timestep.
    """

    def __init__(
        self,
        config: dict[str, str],
        engine_config: EngineConfig,
        grid: Grid,
    ) -> None:
        self.enabled = config.get("simulation.incoming.flux.enabled", "false").lower() == "true"
        self._fluxes: list[_SpeciesFlux] = []
        self._engine_config = engine_config
        self._grid = grid

        if not self.enabled:
            return

        # Pre-compute ocean cell coordinates
        ys, xs = np.where(grid.ocean_mask)
        self._ocean_ys = ys.astype(np.int32)
        self._ocean_xs = xs.astype(np.int32)

        n_sp = engine_config.n_species
        for sp in range(n_sp):
            flux = self._load_species_flux(config, sp)
            if flux is not None:
                self._fluxes.append(flux)

    def _load_species_flux(self, config: dict[str, str], sp: int) -> _SpeciesFlux | None:
        """Load flux CSV for a species (byAge or bySize). Returns None if no file."""
        ec = self._engine_config

        age_file = config.get(f"flux.incoming.bydt.byage.file.sp{sp}", "")
        size_file = config.get(f"flux.incoming.bydt.bysize.file.sp{sp}", "")

        if age_file and size_file:
            raise ValueError(
                f"Species {sp} has both byAge and bySize flux files; only one is allowed."
            )

        if not age_file and not size_file:
            return None

        is_by_age = bool(age_file)
        file_path = age_file if is_by_age else size_file

        # Try relative paths
        resolved = None
        config_dir = config.get("_osmose.config.dir", "")
        search_bases = []
        if config_dir:
            search_bases.append(Path(config_dir))
        search_bases.extend([Path("."), Path("data/examples")])
        for base in search_bases:
            candidate = base / file_path
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = Path(file_path)

        boundaries, data = _parse_flux_csv(resolved)
        n_classes = data.shape[1]

        linf = float(ec.linf[sp])
        k_val = float(ec.k[sp])
        t0_val = float(ec.t0[sp])
        cf = float(ec.condition_factor[sp])
        ap = float(ec.allometric_power[sp])
        lifespan_years = float(ec.lifespan_dt[sp]) / ec.n_dt_per_year
        n_dt_per_year = ec.n_dt_per_year

        lengths = np.zeros(n_classes, dtype=np.float64)
        ages_dt = np.zeros(n_classes, dtype=np.int32)

        for c in range(n_classes):
            lo = boundaries[c]
            if c + 1 < len(boundaries):
                hi = boundaries[c + 1]
            else:
                # Last class: upper bound is lifespan (age) or Linf (size)
                hi = lifespan_years if is_by_age else linf
            midpoint = (lo + hi) / 2.0

            if is_by_age:
                age_years = midpoint
                length = _age_to_length(age_years, linf, k_val, t0_val)
            else:
                length = midpoint
                age_years = _length_to_age(length, linf, k_val, t0_val)

            lengths[c] = max(length, 0.001)  # avoid zero length
            ages_dt[c] = max(0, round(age_years * n_dt_per_year))

        # Convert grams to tonnes (Java convention)
        weights = cf * lengths**ap * 1e-6

        return _SpeciesFlux(
            species_id=sp,
            biomass_table=data,
            lengths=lengths,
            ages_dt=ages_dt,
            weights=weights,
        )

    def get_incoming_schools(self, step: int, rng: np.random.Generator) -> SchoolState | None:
        """Generate new schools from incoming flux for the given timestep.

        Returns SchoolState to append, or None if no flux this step.
        """
        if not self.enabled or not self._fluxes:
            return None

        all_new: list[SchoolState] = []
        ec = self._engine_config

        for flux in self._fluxes:
            row_idx = step % flux.n_rows
            biomass_row = flux.biomass_table[row_idx]

            for c in range(len(biomass_row)):
                bm = biomass_row[c]
                if bm <= 0.0:
                    continue

                weight = flux.weights[c]
                if weight <= 0.0:
                    continue

                # biomass in tonnes, weight in tonnes -> abundance
                abundance = bm / weight

                # Split into n_schools if abundance is large enough
                n_schools_sp = int(ec.n_schools[flux.species_id])
                if abundance >= n_schools_sp and n_schools_sp > 0:
                    n_new = n_schools_sp
                else:
                    n_new = 1

                abund_per = abundance / n_new
                bm_per = bm / n_new

                sp_ids = np.full(n_new, flux.species_id, dtype=np.int32)
                new = SchoolState.create(n_schools=n_new, species_id=sp_ids)

                # Random ocean cell placement
                if len(self._ocean_ys) > 0:
                    cell_indices = rng.integers(0, len(self._ocean_ys), size=n_new)
                    cx = self._ocean_xs[cell_indices]
                    cy = self._ocean_ys[cell_indices]
                else:
                    cx = np.zeros(n_new, dtype=np.int32)
                    cy = np.zeros(n_new, dtype=np.int32)

                new = new.replace(
                    abundance=np.full(n_new, abund_per, dtype=np.float64),
                    biomass=np.full(n_new, bm_per, dtype=np.float64),
                    length=np.full(n_new, flux.lengths[c], dtype=np.float64),
                    weight=np.full(n_new, weight, dtype=np.float64),
                    age_dt=np.full(n_new, flux.ages_dt[c], dtype=np.int32),
                    cell_x=cx,
                    cell_y=cy,
                    is_egg=np.zeros(n_new, dtype=np.bool_),
                )
                all_new.append(new)

        if not all_new:
            return None

        result = all_new[0]
        for s in all_new[1:]:
            result = result.append(s)
        return result
