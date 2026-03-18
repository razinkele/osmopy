"""SchoolState: Structure-of-Arrays representation of all fish schools.

All school data is stored in flat NumPy arrays for vectorized operations.
This replaces Java's per-object School instances with cache-friendly
columnar storage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class MortalityCause(IntEnum):
    """Mortality cause indices for the n_dead tracking array."""

    PREDATION = 0
    STARVATION = 1
    ADDITIONAL = 2
    FISHING = 3
    OUT = 4
    FORAGING = 5
    DISCARDS = 6
    AGING = 7


@dataclass
class SchoolState:
    """Structure-of-Arrays state for all fish schools.

    Every field is a 1D NumPy array of length n_schools, except n_dead
    which is (n_schools, N_MORTALITY_CAUSES).
    """

    # Identity
    species_id: NDArray[np.int32]
    is_background: NDArray[np.bool_]

    # Demographics
    abundance: NDArray[np.float64]
    biomass: NDArray[np.float64]
    length: NDArray[np.float64]
    length_start: NDArray[np.float64]
    weight: NDArray[np.float64]
    age_dt: NDArray[np.int32]
    trophic_level: NDArray[np.float64]

    # Spatial
    cell_x: NDArray[np.int32]
    cell_y: NDArray[np.int32]
    is_out: NDArray[np.bool_]

    # Feeding / predation
    pred_success_rate: NDArray[np.float64]
    preyed_biomass: NDArray[np.float64]
    feeding_stage: NDArray[np.int32]

    # Reproduction
    gonad_weight: NDArray[np.float64]

    # Mortality tracking
    starvation_rate: NDArray[np.float64]
    n_dead: NDArray[np.float64]  # shape (n_schools, len(MortalityCause))

    # Egg state
    is_egg: NDArray[np.bool_]
    first_feeding_age_dt: NDArray[np.int32]
    egg_retained: NDArray[np.float64]  # eggs withheld from prey pool per sub-timestep

    def __len__(self) -> int:
        return len(self.species_id)

    @classmethod
    def create(
        cls,
        n_schools: int,
        species_id: NDArray[np.int32] | None = None,
    ) -> SchoolState:
        """Create a SchoolState with all arrays zeroed.

        Args:
            n_schools: Number of schools to allocate.
            species_id: Species index per school. Defaults to all zeros.
        """
        n = n_schools
        n_causes = len(MortalityCause)
        return cls(
            species_id=(species_id if species_id is not None else np.zeros(n, dtype=np.int32)),
            is_background=np.zeros(n, dtype=np.bool_),
            abundance=np.zeros(n, dtype=np.float64),
            biomass=np.zeros(n, dtype=np.float64),
            length=np.zeros(n, dtype=np.float64),
            length_start=np.zeros(n, dtype=np.float64),
            weight=np.zeros(n, dtype=np.float64),
            age_dt=np.zeros(n, dtype=np.int32),
            trophic_level=np.zeros(n, dtype=np.float64),
            cell_x=np.zeros(n, dtype=np.int32),
            cell_y=np.zeros(n, dtype=np.int32),
            is_out=np.zeros(n, dtype=np.bool_),
            pred_success_rate=np.zeros(n, dtype=np.float64),
            preyed_biomass=np.zeros(n, dtype=np.float64),
            feeding_stage=np.zeros(n, dtype=np.int32),
            gonad_weight=np.zeros(n, dtype=np.float64),
            starvation_rate=np.zeros(n, dtype=np.float64),
            n_dead=np.zeros((n, n_causes), dtype=np.float64),
            is_egg=np.zeros(n, dtype=np.bool_),
            first_feeding_age_dt=np.zeros(n, dtype=np.int32),
            egg_retained=np.zeros(n, dtype=np.float64),
        )

    def replace(self, **kwargs: NDArray) -> SchoolState:
        """Return a new SchoolState with specified fields replaced.

        Unspecified fields are shallow-copied from self.
        """
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(kwargs)
        return SchoolState(**values)

    def append(self, other: SchoolState) -> SchoolState:
        """Concatenate another SchoolState onto this one."""
        merged = {}
        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            merged[f.name] = np.concatenate([a, b], axis=0)
        return SchoolState(**merged)

    def compact(self) -> SchoolState:
        """Remove dead schools (abundance <= 0)."""
        alive = self.abundance > 0
        compacted = {}
        for f in fields(self):
            arr = getattr(self, f.name)
            compacted[f.name] = arr[alive] if arr.ndim == 1 else arr[alive, :]
        return SchoolState(**compacted)
