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
    # OSMOSE 4.4.x (MortalityCause.SPAWNING): post-spawning death of a semelparous species. Every
    # mortality output carries the column, as Java's does; it is 0 unless a species dies after
    # spawning.
    SPAWNING = 8


#: Optional boolean fields that read False for schools that lack them when states are merged.
_FILL_FALSE = frozenset({"is_mature", "has_spawned"})
#: Optional integer fields that read -1 for schools that lack them when states are merged.
_FILL_MINUS_ONE = frozenset({"spawn_step"})
_FILL = _FILL_FALSE | _FILL_MINUS_ONE


def _fill(name: str, n: int) -> NDArray:
    """The value a part without the optional field ``name`` takes when states are merged."""
    if name in _FILL_MINUS_ONE:
        return np.full(n, -1, dtype=np.int32)
    return np.zeros(n, dtype=bool)


@dataclass(frozen=True)
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
    # preyed_biomass is MODE-DEPENDENT (Task 4 review M5, deferred to and resolved by
    # Task 6 item E.10): raw eaten total when bioen is off; survivor-rescaled ingestion
    # (mortality.py's `_consume`) when bioen is on -- Java keeps these as two separate
    # accumulators (`AbstractSchool.preyedBiomass` raw vs `School.ingestion` rescaled),
    # this port collapses them into one field and switches its meaning by mode instead.
    # The two live consumers already account for this (do not add a third without
    # checking mode): `_bioen_step` (osmose/engine/simulate.py, "ingestion = ...") wants
    # the bioen-on scaled value and is only ever called under bioen; the trophic-level
    # update (osmose/engine/processes/mortality.py, `tl_denominator = raw_preyed if ...
    # else state.preyed_biomass`) wants the raw total and falls back to this field only
    # when raw_preyed is None, i.e. bioen off. A future per-school "preyed biomass"
    # output must branch on config.bioen_enabled the same way or it will silently report
    # ingestion instead of predation under bioen.
    preyed_biomass: NDArray[np.float64]
    feeding_stage: NDArray[np.int32]

    # Reproduction
    gonad_weight: NDArray[np.float64]

    # Bioenergetics
    # e_net_avg holds Java School.enet_faced (EnergyBudget.computeEnetFaced): the
    # running mean of net energy expressed PER FISH, per g^beta of somatic weight,
    # and annualised (x n_dt_per_year) — NOT the per-school tonnes E_net of this
    # step. It is the denominator of the rho function (getRho). The field keeps its
    # historical name; e_gross/e_maint/e_net below are this step's raw values in
    # tonnes PER SCHOOL, matching Java's setEGross/setEMaint/setENet.
    e_net_avg: NDArray[np.float64]
    e_gross: NDArray[np.float64]
    e_maint: NDArray[np.float64]
    e_net: NDArray[np.float64]
    rho: NDArray[np.float64]

    # Mortality tracking
    starvation_rate: NDArray[np.float64]
    n_dead: NDArray[np.float64]  # shape (n_schools, len(MortalityCause))

    # Egg state
    is_egg: NDArray[np.bool_]
    first_feeding_age_dt: NDArray[np.int32]
    egg_retained: NDArray[np.float64]  # eggs withheld from prey pool per sub-timestep

    # Evolving Imax per school (Ev-OSMOSE trait targeting ``bioen_i_max``). Set only on the
    # mortality WORKING state, from ``mortality(imax_school=...)``: the bioen ingestion cap and
    # genetic foraging mortality read it there (Java ``School.getTrait("imax")``). NaN = the
    # school carries no trait (background). None everywhere else, incl. the persistent state.
    imax_trait: NDArray[np.float64] | None = None

    # True for egg schools created from seeded (bootstrap) biomass; excluded from
    # environmental egg-survival terms. Optional (mirrors imax_trait) so raw
    # SchoolState(...) constructions that omit it stay valid; create() populates it.
    from_seeding: NDArray[np.bool_] | None = None

    # Drawn maturity (OSMOSE 4.4.0 stochastic ogive, osmose.engine.processes.maturity). None
    # unless a species uses it; where a merged or appended part lacks it, that part is "not
    # mature yet" (see _FILL_FALSE), so egg and background schools need nothing special.
    is_mature: NDArray[np.bool_] | None = None

    # Post-reproduction mortality (OSMOSE 4.4.x, species.reproduction.strategy = semelparous): set
    # at a school's first spawning (Java School.setHasSpawned) and the step of it. None unless a
    # species is semelparous; a part that lacks them has not spawned (see _FILL).
    has_spawned: NDArray[np.bool_] | None = None
    spawn_step: NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        n = len(self.species_id)
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if f.name == "n_dead":
                if val.shape != (n, len(MortalityCause)):
                    raise ValueError(f"n_dead shape {val.shape} != ({n}, {len(MortalityCause)})")
            elif val.ndim == 1 and len(val) != n:
                raise ValueError(f"SchoolState.{f.name} length {len(val)} != {n}")

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
            e_net_avg=np.zeros(n, dtype=np.float64),
            e_gross=np.zeros(n, dtype=np.float64),
            e_maint=np.zeros(n, dtype=np.float64),
            e_net=np.zeros(n, dtype=np.float64),
            rho=np.zeros(n, dtype=np.float64),
            starvation_rate=np.zeros(n, dtype=np.float64),
            n_dead=np.zeros((n, n_causes), dtype=np.float64),
            is_egg=np.zeros(n, dtype=np.bool_),
            first_feeding_age_dt=np.ones(n, dtype=np.int32),
            egg_retained=np.zeros(n, dtype=np.float64),
            from_seeding=np.zeros(n, dtype=np.bool_),
        )

    def replace(self, **kwargs: NDArray) -> SchoolState:
        """Return a new SchoolState with specified fields replaced.

        Unspecified fields are shallow-copied from self.
        """
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(kwargs)
        return SchoolState(**values)

    def append(self, other: SchoolState) -> SchoolState:
        """Concatenate another SchoolState onto this one.

        Optional (None) fields are treated uniformly: the field stays None unless
        exactly one side has an array, in which case that array is taken verbatim.
        Mixed None/array on a per-call basis is currently unreachable in the engine
        (no code path assigns imax_trait), but guarded here for robustness.
        """
        merged = {}
        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            if a is None and b is None:
                merged[f.name] = None
            elif f.name in _FILL and (a is None or b is None):
                merged[f.name] = np.concatenate(
                    [
                        a if a is not None else _fill(f.name, len(self)),
                        b if b is not None else _fill(f.name, len(other)),
                    ]
                )
            elif a is None:
                merged[f.name] = b
            elif b is None:
                merged[f.name] = a
            else:
                merged[f.name] = np.concatenate([a, b], axis=0)
        return SchoolState(**merged)

    def validate(self) -> None:
        """Check biological invariants. Opt-in: called from tests, not __post_init__.
        Raises ValueError on the first violation. Deep review v3 I-1.
        """
        if (self.abundance < 0).any():
            raise ValueError(
                f"abundance must be non-negative; found minimum {self.abundance.min()}"
            )
        if (self.length < 0).any():
            raise ValueError(f"length must be non-negative; found minimum {self.length.min()}")
        if (self.weight < 0).any():
            raise ValueError(f"weight must be non-negative; found minimum {self.weight.min()}")
        if (self.biomass < 0).any():
            raise ValueError(f"biomass must be non-negative; found minimum {self.biomass.min()}")
        if (self.cell_x < 0).any():
            raise ValueError(f"cell_x must be non-negative; found minimum {self.cell_x.min()}")
        if (self.cell_y < 0).any():
            raise ValueError(f"cell_y must be non-negative; found minimum {self.cell_y.min()}")
        live = self.abundance > 0
        if live.any():
            expected = self.abundance[live] * self.weight[live]
            rtol = 1e-6
            diff = np.abs(self.biomass[live] - expected)
            tol = rtol * np.abs(expected) + 1e-12
            bad = diff > tol
            if bad.any():
                bad_idx = np.where(live)[0][bad][0]
                raise ValueError(
                    f"biomass must equal abundance * weight for live schools "
                    f"(school {bad_idx}: biomass={self.biomass[bad_idx]}, "
                    f"abundance*weight={self.abundance[bad_idx] * self.weight[bad_idx]})"
                )

    def compact(self) -> SchoolState:
        """Remove dead schools (abundance <= 0)."""
        alive = self.abundance > 0
        compacted = {}
        for f in fields(self):
            arr = getattr(self, f.name)
            if arr is None:
                compacted[f.name] = None
                continue
            compacted[f.name] = arr[alive] if arr.ndim == 1 else arr[alive, :]
        return SchoolState(**compacted)
