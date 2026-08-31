"""Reproduction process: egg production from spawning stock biomass.

Also handles age increment for all schools (Java side-effect of reproduction).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


def apply_stock_recruitment(
    linear_eggs: NDArray[np.float64],
    ssb: NDArray[np.float64],
    ssb_half: NDArray[np.float64],
    recruitment_type: list[str],
    shepherd_beta: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply per-species density-dependent stock-recruitment.

    Multiplicative correction over the linear SSB→eggs formula. At low SSB,
    every variant approaches `linear_eggs` (preserves Java-linear regime).

    Parameters
    ----------
    linear_eggs : (n_sp,) per-step linear egg production = sex_ratio * relative_fecundity
        * SSB * season_factor * 1e6 (tonnes→grams). All non-negative.
    ssb : (n_sp,) spawning stock biomass in tonnes (per-step).
    ssb_half : (n_sp,) characteristic SSB in tonnes; ignored where type=="none".
        Per-form role:
        - beverton_holt, shepherd: half-saturation SSB (recruitment is halved
          relative to the linear formula at this SSB; for shepherd this holds
          for any beta because (ssb/ssb_half)**beta = 1 when ssb == ssb_half).
        - ricker: peak SSB (recruitment is at exp(-1) ≈ 37% of linear here).
        - hockey_stick: breakpoint SSB (recruitment is at 100% of linear at and
          below this SSB; flat cap above).
    recruitment_type : per-species, one of
        {"none","beverton_holt","ricker","hockey_stick","shepherd"}.
    shepherd_beta : (n_sp,) Shepherd exponent; only read where type=="shepherd".
        None means beta=1.0 everywhere (≡ beverton_holt).

    Returns
    -------
    (n_sp,) corrected egg counts.
    """
    n_sp = linear_eggs.shape[0]
    if not (ssb.shape[0] == ssb_half.shape[0] == len(recruitment_type) == n_sp):
        raise ValueError(
            f"apply_stock_recruitment: shape mismatch — "
            f"linear_eggs={n_sp}, ssb={ssb.shape[0]}, "
            f"ssb_half={ssb_half.shape[0]}, recruitment_type={len(recruitment_type)}"
        )
    if shepherd_beta is not None and shepherd_beta.shape[0] != n_sp:
        raise ValueError(
            f"apply_stock_recruitment: shepherd_beta length {shepherd_beta.shape[0]} != n_sp {n_sp}"
        )

    out = linear_eggs.copy()
    for sp in range(n_sp):
        t = recruitment_type[sp]
        if t == "none":
            continue
        if ssb[sp] <= 0.0:
            continue  # nothing to scale; linear_eggs is already 0
        if t == "beverton_holt":
            out[sp] = linear_eggs[sp] / (1.0 + ssb[sp] / ssb_half[sp])
        elif t == "ricker":
            out[sp] = linear_eggs[sp] * np.exp(-ssb[sp] / ssb_half[sp])
        elif t == "hockey_stick":
            if ssb[sp] > ssb_half[sp]:
                out[sp] = linear_eggs[sp] * (ssb_half[sp] / ssb[sp])
            # else: below/at breakpoint, no correction (out stays linear_eggs[sp])
        elif t == "shepherd":
            beta = 1.0 if shepherd_beta is None else shepherd_beta[sp]
            out[sp] = linear_eggs[sp] / (1.0 + (ssb[sp] / ssb_half[sp]) ** beta)
        else:
            raise ValueError(f"unknown stock-recruitment type: {t!r}")
    return out


def regulate_recruitment(
    n_eggs_linear: NDArray[np.float64],
    ssb: NDArray[np.float64],
    seeded_this_step: NDArray[np.bool_],
    config: EngineConfig,
    step: int,
) -> NDArray[np.float64]:
    """Stock-recruitment curve + RV / ceiling / thermal / depensation gates.

    Shared by BOTH reproduction paths (spec 2026-08-30 decision 5). Java's
    ``BioenReproductionProcess`` has no stock-recruitment concept, but the certified
    Baltic config depends on regulation that lives only here — the Shepherd curve with
    calibrated per-species ``ssbhalf``/``shape``, the RV gate that prescribes cod_east's
    recruitment, and the (inert in production) ceiling / thermal / depensation gates.
    Routing the bioen path's gonad-derived egg count through the same block keeps the
    bioen-on/off A/B a comparison of GROWTH structure, not of recruitment structure.

    Every branch is keyed on config the Gate-B (Bay of Biscay) config does not set, so
    the helper is the identity there and Java parity on the bioen path is preserved.

    Parameters
    ----------
    n_eggs_linear : (n_sp,) pre-regulation egg count. On the standard path this is
        ``sex_ratio * fecundity * SSB * season * 1e6``; on the bioen path it is the
        summed Java gonad release ``wEgg * sexRatio / eggWeight * N``. The season enters
        once on each path (as the annual share / as the gonad-release fraction), so this
        helper must never re-apply it.
    ssb : (n_sp,) spawning stock biomass in tonnes — the SR denominator, and the
        depensation gate's argument.
    seeded_this_step : (n_sp,) species whose SSB was replaced by the seeding biomass this
        step. Every gate is skipped for them: the bootstrap must not be gated or clipped.
    step : current time step (indexes the RV / ceiling / thermal series).
    """
    n_sp = config.n_species

    # GitHub #143: "linear" reproduces Java 4.4.1's SeedingInterface — the proportional
    # tonnes->eggs conversion, with no recruitment relationship. "stock_recruitment"
    # (default) additionally passes it through the configured curve, which saturates.
    # The switch applies only to SEEDED species: a species with real SSB always goes through
    # the recruitment relationship, since that is the running model rather than initialisation.
    n_eggs = apply_stock_recruitment(
        n_eggs_linear,
        ssb,
        config.recruitment_ssb_half[:n_sp],
        config.recruitment_type[:n_sp],
        config.shepherd_beta[:n_sp],
    )
    if config.seeding_mode == "linear" and seeded_this_step.any():
        n_eggs = np.where(seeded_this_step, n_eggs_linear, n_eggs)

    # Reproductive-volume recruitment gate (Baltic cod). Inert unless enabled;
    # skipped on steps where SSB was seeded (bootstrap must not be gated).
    if config.rv_gate_factor_by_index is not None:
        from osmose.engine.processes.recruitment_gate import rv_gate_factor

        assert config.rv_gate_enabled is not None  # invariant: set together in _load_rv_gate
        gate = rv_gate_factor(config, step)
        for sp in range(n_sp):
            if config.rv_gate_enabled[sp] and not seeded_this_step[sp]:
                n_eggs[sp] *= gate[sp]

    # Unfished-level recruitment ceiling (McGregor et al. 2019). Inert unless
    # enabled; caps recruitment at its per-season unfished-equilibrium level.
    # Skipped on seeded steps (bootstrap must not be clipped), like the RV gate.
    if config.recruitment_ceiling_by_season is not None:
        assert config.recruitment_ceiling_enabled is not None  # set together
        n_cols_ceil = config.recruitment_ceiling_by_season.shape[0]
        col = step % n_cols_ceil
        for sp in range(n_sp):
            if config.recruitment_ceiling_enabled[sp] and not seeded_this_step[sp]:
                cap = config.recruitment_ceiling_by_season[col, sp]
                if n_eggs[sp] > cap:
                    n_eggs[sp] = cap

    # Percid thermal recruitment gate (per-year summer-SST factor; spec 2026-07-05;
    # Pekcan-Hekim et al. 2011, Olin et al. 2019). Inert unless enabled. Percid-only
    # and independent of the cod-only RV gate / recruitment ceiling.
    if config.thermal_gate_factor_by_index is not None:
        from osmose.engine.processes.thermal_gate import thermal_gate_factor

        assert config.thermal_gate_enabled is not None  # set together in _load_thermal_gate
        tgate = thermal_gate_factor(config, step)
        for sp in range(n_sp):
            if config.thermal_gate_enabled[sp] and not seeded_this_step[sp]:
                n_eggs[sp] *= tgate[sp]

    # Recruitment depensation / Allee gate (SSB-dependent, not step-dependent). Inert unless
    # enabled; skipped on seeded steps so the SSB=0 bootstrap can't be trapped, like the other gates.
    if config.depensation_gate_enabled is not None:
        from osmose.engine.processes.depensation_gate import depensation_factor

        assert (
            config.depensation_s50 is not None
        )  # invariant: set together in _load_depensation_gate
        assert config.depensation_theta is not None
        dfac = depensation_factor(
            ssb, config.depensation_s50, config.depensation_theta, config.depensation_gate_enabled
        )
        for sp in range(n_sp):
            if config.depensation_gate_enabled[sp] and not seeded_this_step[sp]:
                n_eggs[sp] *= dfac[sp]

    return n_eggs


def create_egg_schools(
    n_eggs: NDArray[np.float64],
    seeded_this_step: NDArray[np.bool_],
    config: EngineConfig,
    egg_length: NDArray[np.float64] | None = None,
) -> list[SchoolState]:
    """Build ``n_schools[sp]`` UNLOCATED egg schools per species.

    Java `create_reproduction_schools` (`ReproductionProcess.java:206-217`,
    `BioenReproductionProcess.java:177-219`): ``nEgg == 0`` -> nothing; ``nEgg < nSchool``
    -> one school holding all the eggs; otherwise ``nSchool`` schools of ``nEgg/nSchool``.
    New schools are created at ``x = y = -1`` (`School.java:204-207`) on BOTH paths.

    ``egg_length`` overrides the school LENGTH only — the bioen path passes
    ``computeLength(eggWeight)``, matching `Species.getEggSize()` under bioen
    (`Species.java:327`). The egg WEIGHT is always the configured egg weight
    (``egg_weight_override``, else allometric at ``config.egg_size``); deriving it from
    the overridden length instead would double-apply the length<->weight conversion.
    """
    n_sp = config.n_species
    new_schools_list = []
    for sp in range(n_sp):
        if n_eggs[sp] <= 0:
            continue
        n_new = int(config.n_schools[sp])
        if n_new <= 0:
            continue
        # Edge case: fewer eggs than schools -> create just 1 school
        if n_eggs[sp] < n_new:
            n_new = 1
        eggs_per_school = n_eggs[sp] / n_new
        egg_len = config.egg_size[sp] if egg_length is None else float(egg_length[sp])
        egg_weight = (
            config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6
        )
        # Use egg weight override if available (already in tonnes)
        if config.egg_weight_override is not None and not np.isnan(config.egg_weight_override[sp]):
            egg_weight = config.egg_weight_override[sp]

        new = SchoolState.create(n_schools=n_new, species_id=np.full(n_new, sp, dtype=np.int32))
        new = new.replace(
            abundance=np.full(n_new, eggs_per_school, dtype=np.float64),
            length=np.full(n_new, egg_len, dtype=np.float64),
            weight=np.full(n_new, egg_weight, dtype=np.float64),
            biomass=np.full(n_new, eggs_per_school * egg_weight, dtype=np.float64),
            is_egg=np.ones(n_new, dtype=np.bool_),
            # Eggs cannot feed for their first timestep (Java convention: first_feeding_age_dt=1)
            first_feeding_age_dt=np.ones(n_new, dtype=np.int32),
        )
        # Eggs are created unlocated; movement places them on the next step.
        # Tag seeded-derived eggs so environmental egg-survival terms skip them.
        new = new.replace(
            cell_x=np.full(n_new, -1, dtype=np.int32),
            cell_y=np.full(n_new, -1, dtype=np.int32),
            from_seeding=np.full(n_new, bool(seeded_this_step[sp]), dtype=np.bool_),
        )
        new_schools_list.append(new)
    return new_schools_list


def merge_new_schools(state: SchoolState, new_schools_list: list[SchoolState]) -> SchoolState:
    """Append every new school in ONE concatenation per field.

    Used by both reproduction paths. Preferred over repeated `SchoolState.append` for two
    reasons: `append` silently takes the non-None side verbatim when one side's optional
    field is None (mis-aligning array lengths), whereas this raises; and it copies the
    whole state once per species instead of once in total.
    """
    if not new_schools_list:
        return state
    from dataclasses import fields

    merged_fields = {}
    for f in fields(state):
        existing = getattr(state, f.name)
        parts = [existing] + [getattr(s, f.name) for s in new_schools_list]
        # Skip fields that are None on every source (optional fields like
        # imax_trait are unpopulated unless genetic traits are enabled).
        non_none = [p for p in parts if p is not None]
        if not non_none:
            merged_fields[f.name] = None
        elif len(non_none) == len(parts):
            merged_fields[f.name] = np.concatenate(parts)
        else:
            # Partial population: one side has arrays, the other doesn't.
            # Currently unreachable (no code path assigns imax_trait), so
            # fail loudly rather than silently mis-align lengths.
            raise ValueError(f"SchoolState.{f.name}: cannot concatenate; some inputs are None")
    return SchoolState(**merged_fields)


def reproduction(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
    grid_ny: int = 10,
    grid_nx: int = 10,
) -> SchoolState:
    """Produce eggs from mature schools and increment age for all schools.

    Maturity condition: age_dt >= maturity_age AND length >= maturity_size.

    N_eggs = sex_ratio * relative_fecundity * SSB * season_factor
    """
    n_sp = config.n_species

    # --- Egg production ---
    # Maturity: length >= maturity_size AND age_dt >= maturity_age_dt (and abundance > 0)
    mature = (
        (state.length >= config.maturity_size[state.species_id])
        & (state.age_dt >= config.maturity_age_dt[state.species_id])
        & (state.abundance > 0)
    )

    # Spawning stock biomass per species
    ssb = np.zeros(n_sp, dtype=np.float64)
    if mature.any():
        np.add.at(ssb, state.species_id[mature], state.abundance[mature] * state.weight[mature])

    # Season factor from loaded CSV or uniform
    # Multi-year CSVs: wrap by column count (handles both single-year and multi-year)
    if config.spawning_season is not None:
        n_cols = config.spawning_season.shape[1]
        season_idx = step % n_cols
        season_factor = config.spawning_season[:, season_idx]
    else:
        season_factor = np.full(n_sp, 1.0 / config.n_dt_per_year)

    # Seeding: if SSB is zero and within seeding period, use seeding biomass
    seeded_this_step = np.zeros(n_sp, dtype=np.bool_)
    for sp in range(n_sp):
        if ssb[sp] == 0.0:
            if step < config.seeding_max_step[sp]:
                ssb[sp] = config.seeding_biomass[sp]
                seeded_this_step[sp] = True

    # Egg count per species
    # Java: nEgg = sexRatio * beta * season * SSB * 1_000_000
    # The 1e6 converts SSB from tonnes to grams (fecundity is eggs per gram)
    #
    # Slice all per-species arrays to focal-only (length n_sp). When background
    # species are configured, sex_ratio / relative_fecundity have length
    # n_focal + n_bkg (extended with zeros by _merge_focal_background in
    # config.py), but ssb is computed only for focal species. Without the
    # slice, broadcasting fails when background species are activated.
    # See `osmose/engine/config.py:701-702` for the merge that pads these.
    TONNES_TO_GRAMS = 1_000_000.0
    n_eggs_linear = (
        config.sex_ratio[:n_sp]
        * config.relative_fecundity[:n_sp]
        * ssb
        * season_factor
        * TONNES_TO_GRAMS
    )
    n_eggs = regulate_recruitment(n_eggs_linear, ssb, seeded_this_step, config, step)

    # Create new schools from eggs
    new_schools_list = create_egg_schools(n_eggs, seeded_this_step, config)

    # --- Age increment for ALL schools ---
    new_age_dt = state.age_dt + 1
    new_is_egg = new_age_dt < state.first_feeding_age_dt
    state = state.replace(age_dt=new_age_dt, is_egg=new_is_egg)

    # Append all new egg schools in one batch
    return merge_new_schools(state, new_schools_list)
