"""Build the parameter bundle the C# engine runs from (spec section 8.2, plan task 1.2).

Python resolves, C# computes (spec 8.1): every table here is what the Python engine itself reads,
already parsed, per species and, where a value can change, per step. The C# engine never sees a
config key, a file or a species name it must interpret.

Tasks 1.2a/b build the header, the static tables (contract, run, time, grid, species, growth,
predation with feeding stages, stage accessibility, non-fishing mortality, output settings) and
the per-step fishing and reproduction tables. Resources, movement and the result buffers (1.2c)
follow.

Entry names, dtypes and shapes are the contract. Every array is validated against
``n_total = n_focal + n_background`` (or ``n_focal``) here, so a mis-sized table fails at build
time with its name rather than inside the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import load_catalog, required_features
from osmose.engine_cs.philox import SCHEME as RNG_SCHEME

CONTRACT_VERSION = 1

#: ``predation.stage.metric`` codes.
STAGE_METRICS = {"size": 0, "age": 1, "weight": 2, "tl": 3}
#: ``access.mode`` codes.
ACCESS_NONE, ACCESS_SPECIES, ACCESS_STAGE = 0, 1, 2
#: ``growth.class`` codes.
GROWTH_CLASSES = {"VB": 0, "GOMPERTZ": 1}
#: ``movement.method`` codes.
MOVEMENT_METHODS = {"none": 0, "random": 1, "maps": 2}
# apply_stock_recruitment's curves (reproduction.sr.type).
SR_TYPES = {"none": 0, "beverton_holt": 1, "ricker": 2, "hockey_stick": 3, "shepherd": 4}
#: Domain separator for the random-patch generator (ASCII "PATC"): the patches get their own
#: stream, derived from the run seed, so they never share one with anything else.
PATCH_STREAM = 0x50415443


class BundleBuildError(ValueError):
    """The config cannot be expressed as a bundle (a table has the wrong size, etc.)."""


def _f8(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if arr.shape != shape:
        raise BundleBuildError(f"{name}: shape {arr.shape}, expected {shape}")
    return arr


def _i4(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int32))
    if arr.shape != shape:
        raise BundleBuildError(f"{name}: shape {arr.shape}, expected {shape}")
    return arr


def _ragged(name: str, per_species: list[Any], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-species float lists into (offsets i4 (n + 1), values f8)."""
    if len(per_species) != n:
        raise BundleBuildError(f"{name}: {len(per_species)} species, expected {n}")
    offsets = np.zeros(n + 1, dtype=np.int32)
    parts = []
    for i, values in enumerate(per_species):
        arr = np.asarray(values if values is not None else [], dtype=np.float64).ravel()
        parts.append(arr)
        offsets[i + 1] = offsets[i] + arr.size
    values = np.concatenate(parts) if parts else np.zeros(0)
    return offsets, np.ascontiguousarray(values, dtype=np.float64)


def build_bundle(
    config: Any,
    raw: Mapping[str, str],
    grid: Any,
    *,
    seed: int,
    threads: int = 1,
    deadline_seconds: float = 0.0,
    checks: bool = False,
) -> Bundle:
    """The bundle for one run of ``config`` (an ``EngineConfig``) on ``grid``.

    ``raw`` is the reader's dict the config was parsed from; the capability detectors read the
    few switches the engine takes outside ``EngineConfig``.

    Like ``simulate()``, this publishes the resource names on ``config.resource_names`` (they
    are discovered by ``ResourceState``): the diet and predator-pressure frames label their
    prey axis with them.
    """
    b = Bundle(CONTRACT_VERSION)
    _contract(b, config, raw)
    _run(b, seed, threads, deadline_seconds, checks)
    n_total = config.n_species + config.n_background
    _time_grid(b, config, grid)
    _species_growth(b, config, n_total)
    _predation(b, config, n_total)
    _access(b, config, n_total)
    _mortality(b, config, n_total)
    _fishing(b, config, grid, n_total)
    _reproduction(b, config, n_total)
    _resources(b, config, grid)
    _movement(b, config, grid, n_total, seed)
    _background(b, config, grid)
    _output(b, config, n_total)
    return b


def _contract(b: Bundle, config: Any, raw: Mapping[str, str]) -> None:
    catalog = load_catalog()
    ids = [catalog[name].id for name in required_features(config, raw)]
    b.set_array("contract.features", np.asarray(ids, dtype=np.int32))
    b.set_int("contract.rng.scheme", RNG_SCHEME)


def _run(b: Bundle, seed: int, threads: int, deadline_seconds: float, checks: bool) -> None:
    b.set_int("run.seed", int(seed))
    b.set_int("run.threads", int(threads))
    b.set_float("run.deadline_seconds", float(deadline_seconds))
    b.set_int("run.checks", int(bool(checks)))


def _time_grid(b: Bundle, config: Any, grid: Any) -> None:
    b.set_int("time.n_steps", int(config.n_steps))
    b.set_int("time.n_dt_per_year", int(config.n_dt_per_year))
    b.set_int("time.n_subdt", int(config.mortality_subdt))
    b.set_int("grid.ny", int(grid.ny))
    b.set_int("grid.nx", int(grid.nx))
    mask = np.asarray(grid.ocean_mask)
    if mask.shape != (grid.ny, grid.nx):
        raise BundleBuildError(
            f"grid.ocean_mask: shape {mask.shape}, expected {(grid.ny, grid.nx)}"
        )
    b.set_array("grid.ocean_mask", np.ascontiguousarray(mask, dtype=np.uint8))


def _species_growth(b: Bundle, config: Any, n_total: int) -> None:
    n_focal = config.n_species
    b.set_int("species.n_focal", n_focal)
    b.set_int("species.n_background", int(config.n_background))
    b.set_array(
        "species.n_schools", _i4("species.n_schools", config.n_schools[:n_focal], (n_focal,))
    )
    b.set_array("species.lifespan_dt", _i4("species.lifespan_dt", config.lifespan_dt, (n_total,)))
    unknown = sorted(set(config.growth_class) - set(GROWTH_CLASSES))
    if unknown:
        raise BundleBuildError(f"growth.class: unknown classes {unknown}")
    b.set_array(
        "growth.class",
        np.asarray([GROWTH_CLASSES[g] for g in config.growth_class], dtype=np.uint8),
    )
    for name in (
        "linf", "k", "t0", "egg_size", "vb_threshold_age", "condition_factor",
        "allometric_power", "lmax", "delta_lmax_factor", "critical_success_rate",
    ):  # fmt: skip
        b.set_array(f"growth.{name}", _f8(f"growth.{name}", getattr(config, name), (n_total,)))


def _predation(b: Bundle, config: Any, n_total: int) -> None:
    b.set_array(
        "predation.ingestion_rate",
        _f8("predation.ingestion_rate", config.ingestion_rate, (n_total,)),
    )
    b.set_array("predation.fr_shape", _i4("predation.fr_shape", config.fr_shape, (n_total,)))
    b.set_array("predation.fr_halfsat", _f8("predation.fr_halfsat", config.fr_halfsat, (n_total,)))
    n_stage = int(np.asarray(config.size_ratio_min).shape[1])
    b.set_array(
        "predation.size_ratio_min",
        _f8("predation.size_ratio_min", config.size_ratio_min, (n_total, n_stage)),
    )
    b.set_array(
        "predation.size_ratio_max",
        _f8("predation.size_ratio_max", config.size_ratio_max, (n_total, n_stage)),
    )
    metrics = list(config.feeding_stage_metric)
    unknown = sorted(set(metrics) - set(STAGE_METRICS))
    if unknown or len(metrics) != n_total:
        raise BundleBuildError(f"predation.stage.metric: {len(metrics)} entries, unknown {unknown}")
    b.set_array(
        "predation.stage.metric", np.asarray([STAGE_METRICS[m] for m in metrics], dtype=np.uint8)
    )
    # compute_feeding_stages sorts each species' thresholds before its searchsorted.
    sorted_thresholds = [
        np.sort(np.asarray(t or [], dtype=np.float64)) for t in config.feeding_stage_thresholds
    ]
    offsets, values = _ragged("predation.stage.thresholds", sorted_thresholds, n_total)
    b.set_array("predation.stage.offsets", offsets)
    b.set_array("predation.stage.thresholds", values)


def _access(b: Bundle, config: Any, n_total: int) -> None:
    sa = config.stage_accessibility
    if sa is None:
        has_species = config.accessibility_matrix is not None
        b.set_int("access.mode", ACCESS_SPECIES if has_species else ACCESS_NONE)
        if has_species:
            b.set_array(
                "access.matrix", np.ascontiguousarray(config.accessibility_matrix, dtype=np.float64)
            )
        return
    b.set_int("access.mode", ACCESS_STAGE)
    b.set_array("access.matrix", np.ascontiguousarray(sa.raw_matrix, dtype=np.float64))
    for role in ("prey", "pred"):
        stages = sa._stages_by_role.get(role, {})
        offsets = np.zeros(n_total + 1, dtype=np.int32)
        thresholds, index = [], []
        for sp in range(n_total):
            st = stages.get(sp)
            if st is not None:
                thresholds.append(np.asarray(st.thresholds, dtype=np.float64))
                index.append(np.asarray(st.matrix_indices, dtype=np.int32))
                offsets[sp + 1] = offsets[sp] + len(st.thresholds)
            else:  # absent from the CSV: an EMPTY range, which the engine resolves to -1 (full access)
                offsets[sp + 1] = offsets[sp]
        b.set_array(f"access.{role}.offsets", offsets)
        b.set_array(
            f"access.{role}.thresholds", np.concatenate(thresholds) if thresholds else np.zeros(0)
        )
        b.set_array(
            f"access.{role}.index",
            np.concatenate(index).astype(np.int32) if index else np.zeros(0, np.int32),
        )


def _mortality(b: Bundle, config: Any, n_total: int) -> None:
    n_steps, n_focal = int(config.n_steps), config.n_species
    add = _f8("mortality.additional.rate", config.additional_mortality_rate, (n_total,))
    larva = _f8("mortality.larva.rate", config.larva_mortality_rate, (n_total,))
    # Per step, the value the engine applies that step. By-step overrides are the
    # additional_rate_by_step / larva_rate_by_step features (refused until implemented), so in
    # Phase 1 every step carries the per-species constant.
    b.set_array(
        "mortality.additional.rate_by_step",
        np.ascontiguousarray(np.broadcast_to(add, (n_steps, n_total))),
    )
    b.set_array(
        "mortality.larva.rate_by_step",
        np.ascontiguousarray(np.broadcast_to(larva[:n_focal], (n_steps, n_focal))),
    )
    b.set_array(
        "mortality.out.rate", _f8("mortality.out.rate", config.out_mortality_rate, (n_total,))
    )
    b.set_array(
        "mortality.starvation.rate_max",
        _f8("mortality.starvation.rate_max", config.starvation_rate_max, (n_total,)),
    )


def _fishing(b: Bundle, config: Any, grid: Any, n_total: int) -> None:
    """What ``_precompute_effective_rates`` reads for fishing, resolved per step.

    The engine's per-school rate is ``rate * selectivity * spatial * season / n_subdt`` when
    ``fishing.season_by_step`` is present, else ``rate * selectivity * spatial / (ndt * n_subdt)``
    (the two branches of the Python expression; MPAs and by-year rates are refused features).
    Selectivity and maps apply to focal species only, as in Python's focal loop.
    """
    n_steps, ndt, n_focal = int(config.n_steps), int(config.n_dt_per_year), config.n_species
    b.set_int("fishing.enabled", int(bool(config.fishing_enabled)))
    rate = _f8("fishing.rate", config.fishing_rate, (n_total,))
    b.set_array(
        "fishing.rate_by_step", np.ascontiguousarray(np.broadcast_to(rate, (n_steps, n_total)))
    )
    if config.fishing_seasonality is not None:
        seas = _f8("fishing.seasonality", config.fishing_seasonality, (n_total, ndt))
        cols = np.arange(n_steps) % ndt  # step_in_year
        b.set_array("fishing.season_by_step", np.ascontiguousarray(seas[:, cols].T))
    b.set_array(
        "fishing.selectivity.type",
        _i4("fishing.selectivity.type", config.fishing_selectivity_type, (n_total,)),
    )
    for name in ("a50", "l50", "slope"):
        b.set_array(
            f"fishing.selectivity.{name}",
            _f8(
                f"fishing.selectivity.{name}",
                getattr(config, f"fishing_selectivity_{name}"),
                (n_total,),
            ),
        )
    maps = list(config.fishing_spatial_maps)[:n_focal]
    maps += [None] * (n_focal - len(maps))
    index = np.full(n_focal, -1, dtype=np.int32)
    stack = []
    for sp, m in enumerate(maps):
        if m is None:
            continue
        arr = np.asarray(m, dtype=np.float64)
        if arr.shape != (grid.ny, grid.nx):
            raise BundleBuildError(
                f"fishing.maps: species {sp} map shape {arr.shape}, grid {(grid.ny, grid.nx)}"
            )
        index[sp] = len(stack)
        stack.append(arr)
    b.set_array("fishing.map_index", index)
    b.set_array(
        "fishing.maps",
        np.ascontiguousarray(np.stack(stack)) if stack else np.zeros((0, grid.ny, grid.nx)),
    )
    if config.fishing_discard_rate is not None:
        b.set_array(
            "fishing.discard_rate",
            _f8("fishing.discard_rate", config.fishing_discard_rate, (n_total,)),
        )


def _reproduction(b: Bundle, config: Any, n_total: int) -> None:
    """What ``reproduction`` reads (focal species), with the season factor resolved per step and
    the egg weight resolved exactly as ``create_egg_schools`` computes it.

    Recruitment regulation (``regulate_recruitment``), in its order:
    - the stock-recruitment curve: ``sr.type`` always, ``sr.ssb_half`` and ``sr.shepherd_beta``
      exactly when a species has one (feature ``stock_recruitment``);
    - ``seeding_mode`` (1 = linear, feature ``seeding_linear``): a seeded species skips the curve;
    - the reproductive-volume gate (feature ``rv_gate``): ``rv.enabled`` and ``rv.factor_by_year``,
      the latter filled by the engine's own ``rv_gate_factor`` for each model year, so its
      offset and its clamp past the end of the series are the engine's;
    - the depensation gate (feature ``depensation_gate``): ``depensation.enabled``, ``.s50`` and
      ``.theta``, the Allee term's two numbers per species.
    The ceiling and thermal gates sit between those two in ``regulate_recruitment``. They are
    phase 3 features, and the engine refuses them.
    """
    n_steps, ndt, n_focal = int(config.n_steps), int(config.n_dt_per_year), config.n_species
    for name in ("sex_ratio", "relative_fecundity", "maturity_size", "seeding_biomass"):
        b.set_array(
            f"reproduction.{name}",
            _f8(f"reproduction.{name}", np.asarray(getattr(config, name))[:n_focal], (n_focal,)),
        )
    b.set_array(
        "reproduction.maturity_age_dt",
        _i4(
            "reproduction.maturity_age_dt", np.asarray(config.maturity_age_dt)[:n_focal], (n_focal,)
        ),
    )
    b.set_array(
        "reproduction.seeding_max_step",
        _i4(
            "reproduction.seeding_max_step",
            np.asarray(config.seeding_max_step)[:n_focal],
            (n_focal,),
        ),
    )
    if config.spawning_season is not None:
        season = np.asarray(config.spawning_season, dtype=np.float64)
        if season.shape[0] != n_focal:
            raise BundleBuildError(
                f"reproduction.season: {season.shape[0]} rows, expected {n_focal}"
            )
        cols = np.arange(n_steps) % season.shape[1]  # multi-year season files wrap by column count
        by_step = np.ascontiguousarray(season[:, cols].T)
    else:
        by_step = np.full((n_steps, n_focal), 1.0 / ndt)
    b.set_array("reproduction.season_by_step", by_step)
    egg_size = np.asarray(config.egg_size, dtype=np.float64)
    b.set_array("reproduction.egg_length", np.ascontiguousarray(egg_size[:n_focal]))
    weights = np.empty(n_focal, dtype=np.float64)
    for sp in range(n_focal):
        # create_egg_schools' expression, scalar by scalar (a vectorised ** may round differently).
        weight = (
            config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6
        )
        override = config.egg_weight_override
        if override is not None and not np.isnan(override[sp]):
            weight = override[sp]
        weights[sp] = weight
    b.set_array("reproduction.egg_weight", weights)
    types = [str(t) for t in list(config.recruitment_type)[:n_focal]]
    unknown = sorted(set(types) - set(SR_TYPES))
    if unknown:
        raise BundleBuildError(f"reproduction.sr.type: unknown stock-recruitment type {unknown}")
    b.set_array("reproduction.sr.type", np.asarray([SR_TYPES[t] for t in types], dtype=np.uint8))
    if any(t != "none" for t in types):
        for name, source in (
            ("ssb_half", "recruitment_ssb_half"),
            ("shepherd_beta", "shepherd_beta"),
        ):
            entry = f"reproduction.sr.{name}"
            b.set_array(
                entry, _f8(entry, np.asarray(getattr(config, source))[:n_focal], (n_focal,))
            )
    if config.seeding_mode not in ("stock_recruitment", "linear"):
        raise BundleBuildError(f"reproduction.seeding_mode: unknown mode {config.seeding_mode!r}")
    b.set_int("reproduction.seeding_mode", 0 if config.seeding_mode == "stock_recruitment" else 1)
    enabled = config.rv_gate_enabled
    if enabled is not None and np.asarray(enabled).any():
        from osmose.engine.processes.recruitment_gate import rv_gate_factor

        n_years = -(-n_steps // ndt)
        by_year = np.stack([rv_gate_factor(config, y * ndt)[:n_focal] for y in range(n_years)])
        b.set_array("reproduction.rv.enabled", np.asarray(enabled)[:n_focal].astype(np.uint8))
        b.set_array(
            "reproduction.rv.factor_by_year", np.ascontiguousarray(by_year, dtype=np.float64)
        )
    gated = config.depensation_gate_enabled
    if gated is not None and np.asarray(gated)[:n_focal].any():
        b.set_array(
            "reproduction.depensation.enabled", np.asarray(gated)[:n_focal].astype(np.uint8)
        )
        for name, source in (("s50", "depensation_s50"), ("theta", "depensation_theta")):
            entry = f"reproduction.depensation.{name}"
            b.set_array(
                entry, _f8(entry, np.asarray(getattr(config, source))[:n_focal], (n_focal,))
            )


def _resources(b: Bundle, config: Any, grid: Any) -> None:
    """The LTL resources as ``ResourceState.update`` computes them, non-depletable (Phase 1).

    Each step's carrying capacity is ``frame * access``: ``resources.forcing[r, frame]`` holds
    the regridded forcing already multiplied by the multiplier (Python's left-to-right
    ``forcing * multiplier * access``, so pre-multiplying is exact), ``frame_by_step`` the forcing
    frame of each step and ``accessibility_by_step`` the (capped, possibly time-varying)
    accessibility. Sizes, trophic levels and accessibility rows come from the kernel's own
    ``_precompute_resource_arrays``.

    Bottom oxygen scaling a resource's K (feature ``oxygen_benthos``): the bundle carries the
    Hill factor pre-evaluated per oxygen frame by the engine's own ``f_o2_hill`` on
    ``ResourceState._oxygen_row``, so the C# engine only multiplies. ``oxygen.coupled`` marks the
    resources it applies to (every one named ``ltl.oxygen.benthos.rsc``, as ``update`` does),
    ``oxygen.frame_by_step`` is ``step % frames`` (``PhysicalData``'s rule). The three entries
    exist exactly when the config declares the feature; with no forcing, or no resource of that
    name, nothing is coupled, as in Python.
    """
    from pathlib import Path

    from osmose.engine.processes.oxygen_function import f_o2_hill
    from osmose.engine.processes.precompute import _precompute_resource_arrays
    from osmose.engine.resources import ResourceState, oxygen_benthos_enabled

    # ResourceState as simulate() constructs it: with the bottom-oxygen forcing when the coupling
    # is on. _load_oxygen_data raises on a forcing whose frame count is not n_dt_per_year (it
    # would misalign silently), and so does this builder. simulate() loads the forcing for
    # bioenergetics too; that is feature bioenergetics, not read here.
    raw_config = config.raw_config
    coupling = oxygen_benthos_enabled(raw_config)
    oxygen = None
    if coupling:
        from osmose.engine.simulate import _load_oxygen_data

        config_dir = raw_config.get("_osmose.config.dir", "")
        oxygen = _load_oxygen_data(raw_config, Path(config_dir) if config_dir else None)
    rs = ResourceState(config=raw_config, grid=grid, oxygen=oxygen)
    n_rsc, n_cells, n_steps = rs.n_resources, grid.ny * grid.nx, int(config.n_steps)
    ndt = int(config.raw_config.get("simulation.time.ndtperyear", "24"))
    b.set_int("resources.n", n_rsc)
    # ResourceState is the single source of truth for the resources' names and number;
    # simulate() publishes them on the config for the diet's prey axis (#146), and so must
    # this path: _output sizes the diet with them and the output frames label it with them.
    config.resource_names = [r.name for r in rs.species]
    if n_rsc == 0:
        return
    size_min, size_max, tl, rows, _ = _precompute_resource_arrays(config, rs)
    b.set_array("resources.size_min", np.asarray(size_min, dtype=np.float64))
    b.set_array("resources.size_max", np.asarray(size_max, dtype=np.float64))
    b.set_array("resources.trophic_level", np.asarray(tl, dtype=np.float64))
    b.set_array("resources.access_rows", np.asarray(rows, dtype=np.int32))
    forcing = rs._forcing_data
    n_frames = max(1, int(rs._n_forcing_steps)) if forcing is not None else 1
    frames = np.zeros((n_rsc, n_frames, n_cells), dtype=np.float64)
    frame_by_step = np.zeros((n_steps, n_rsc), dtype=np.int32)
    access_by_step = np.zeros((n_steps, n_rsc), dtype=np.float64)
    for i, rsc in enumerate(rs.species):
        if forcing is not None and rsc.name in forcing:
            # The whole series at once, time first: one decode instead of one isel per frame.
            series = forcing[rsc.name].transpose("time", ...).values
            for f in range(n_frames):
                data = series[f]
                flat = rs._regrid_to_model(data).flatten() * rsc.multiplier
                frames[i, f, : len(flat)] = flat[:n_cells]
            for step in range(n_steps):
                step_in_year = step % ndt  # update()'s mapping, with its int() truncation
                frame_by_step[step, i] = min(
                    int(step_in_year * rs._n_forcing_steps / ndt), rs._n_forcing_steps - 1
                )
        elif rs._uniform_biomass[i] > 0:
            frames[i, :, :] = rsc.multiplier * (rs._uniform_biomass[i] + rsc.offset)
        for step in range(n_steps):
            if rsc.accessibility_ts is not None and len(rsc.accessibility_ts) > 0:
                access_by_step[step, i] = float(
                    rsc.accessibility_ts[step % len(rsc.accessibility_ts)]
                )
            else:
                access_by_step[step, i] = rsc.accessibility
    if coupling:
        coupled = np.zeros(n_rsc, dtype=np.uint8)
        factor = np.ones((1, n_cells), dtype=np.float64)
        o2_frame_by_step = np.zeros(n_steps, dtype=np.int32)
        if rs._oxygen is not None:
            coupled[:] = [rsc.name == rs._oxygen_rsc_name for rsc in rs.species]
            # _oxygen_row(f) is frame f (it indexes step % frames); a constant field has one.
            o2_data = rs._oxygen._data
            n_o2 = 1 if rs._oxygen.is_constant or o2_data is None else int(o2_data.shape[0])
            factor = np.stack(
                [f_o2_hill(rs._oxygen_row(f), rs._oxygen_c50, rs._oxygen_n) for f in range(n_o2)]
            ).astype(np.float64)
            o2_frame_by_step[:] = np.arange(n_steps) % n_o2
        b.set_array("resources.oxygen.coupled", coupled)
        b.set_array("resources.oxygen.factor", np.ascontiguousarray(factor))
        b.set_array("resources.oxygen.frame_by_step", o2_frame_by_step)
    rs.close()
    b.set_array("resources.forcing", frames)
    b.set_array("resources.frame_by_step", frame_by_step)
    b.set_array("resources.accessibility_by_step", access_by_step)


def _movement(b: Bundle, config: Any, grid: Any, n_total: int, seed: int) -> None:
    """Map and random movement, as the Python engine sets them up at the start of a run.

    Maps: the engine's own ``MovementMapSet`` per map-moving focal species (as ``simulate`` builds
    them) flattened by its ``_flatten_all_map_sets``. A school's map is
    ``species_offset[sp] + index[sp][age, step]`` (-1 = none), and ``_precompute_map_indices``'s
    same-map rule follows from the same table. Random: the placement domain of each random-moving
    species, built by the engine's ``build_random_patches`` on a stream of its own derived from
    the run seed (only a configured ``ncell`` smaller than the ocean draws anything).

    Salinity gate (feature ``salinity_gate``), for the gated map species: ``salinity.weight`` is
    the occupancy weight of each salinity frame, from the engine's own
    ``_movement_salinity_weight``; ``salinity.frame_by_step`` is ``step % frames``
    (``PhysicalData``'s rule); ``salinity.wmax[map, frame]`` is the largest non-NaN
    ``map * weight`` (0 when there is none), which the Python kernel recomputes for every school
    and which decides its all-zero guard. The four entries exist exactly with the feature.
    """
    from osmose.engine.movement_maps import MovementMapSet
    from osmose.engine.processes.precompute import (
        _flatten_all_map_sets,
        _movement_salinity_weight,
        build_random_patches,
    )

    n_focal = config.n_species
    methods = list(config.movement_method)
    unknown = sorted(set(methods) - set(MOVEMENT_METHODS))
    if unknown or len(methods) != n_total:
        raise BundleBuildError(f"movement.method: {len(methods)} entries, unknown {unknown}")
    b.set_array(
        "movement.method", np.asarray([MOVEMENT_METHODS[m] for m in methods], dtype=np.uint8)
    )
    b.set_array(
        "movement.walk_range", _i4("movement.walk_range", config.random_walk_range, (n_total,))
    )

    map_sets = {}
    for sp in range(n_focal):
        if methods[sp] == "maps":
            map_sets[sp] = MovementMapSet(
                config=config.raw_config,
                species_name=config.species_names[sp],
                n_dt_per_year=config.n_dt_per_year,
                n_years=config.n_year,
                lifespan_dt=int(config.lifespan_dt[sp]),
                ny=grid.ny,
                nx=grid.nx,
                config_dir=config.raw_config.get("_osmose.config.dir", ""),
                strict=config.movement_strict_coverage,
            )
    b.set_int("movement.maps.n_species", len(map_sets))
    if map_sets:
        stack, max_proba, is_null, offset = _flatten_all_map_sets(
            map_sets, n_focal, grid.ny, grid.nx
        )
        b.set_array("movement.maps.stack", np.ascontiguousarray(stack, dtype=np.float64))
        b.set_array("movement.maps.max_proba", np.asarray(max_proba, dtype=np.float64))
        b.set_array("movement.maps.is_null", np.asarray(is_null, dtype=np.uint8))
        b.set_array("movement.maps.species_offset", np.asarray(offset, dtype=np.int32))
        offsets = np.zeros(n_focal + 1, dtype=np.int64)
        shapes = np.zeros((n_focal, 2), dtype=np.int32)
        tables = []
        for sp in range(n_focal):
            im = map_sets[sp].index_maps if sp in map_sets else np.zeros((0, 0), dtype=np.int32)
            shapes[sp] = im.shape
            tables.append(np.ascontiguousarray(im, dtype=np.int32).ravel())
            offsets[sp + 1] = offsets[sp] + im.size
        b.set_array("movement.maps.index", np.concatenate(tables).astype(np.int32))
        b.set_array("movement.maps.index.offsets", offsets)
        b.set_array("movement.maps.index.shape", shapes)

    if config.salinity_gate_enabled:
        n_cells, n_steps = grid.ny * grid.nx, int(config.n_steps)
        field = config.salinity_field
        if field is None or config.salinity_gate_species is None:  # EngineConfig sets both
            raise BundleBuildError("movement.salinity: the gate is enabled without a field")
        field_data = field._data
        n_frames = 1 if field.is_constant or field_data is None else int(field_data.shape[0])
        frames: list[np.ndarray] = []
        for f in range(n_frames):  # _movement_salinity_weight(step) reads frame step % frames
            w = _movement_salinity_weight(config, grid, f)
            if w is None or w.shape != (grid.ny, grid.nx):
                raise BundleBuildError(
                    f"movement.salinity.weight: the field is not on the model grid "
                    f"({grid.ny}, {grid.nx})"
                )
            frames.append(w)
        weight = np.ascontiguousarray(np.stack(frames).reshape(n_frames, n_cells), dtype=np.float64)
        maps = (
            np.asarray(stack, dtype=np.float64).reshape(len(stack), n_cells)
            if map_sets
            else np.zeros((0, n_cells))
        )
        wmax = np.zeros((len(maps), n_frames), dtype=np.float64)
        for g in range(len(maps)):
            for f in range(n_frames):
                product = maps[g] * weight[f]  # the kernel's current_map * sal_w, cell by cell
                valid = product[~np.isnan(product)]
                wmax[g, f] = max(0.0, float(valid.max())) if valid.size else 0.0
        b.set_array(
            "movement.salinity.species",
            np.asarray(config.salinity_gate_species)[:n_focal].astype(np.uint8),
        )
        b.set_array("movement.salinity.weight", weight)
        b.set_array(
            "movement.salinity.frame_by_step", (np.arange(n_steps) % n_frames).astype(np.int32)
        )
        b.set_array("movement.salinity.wmax", wmax)

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed) & 0xFFFFFFFFFFFFFFFF, PATCH_STREAM])
    )
    domains = build_random_patches(config, grid, rng)
    offsets = np.zeros(n_focal + 1, dtype=np.int32)
    cells = []
    for sp in range(n_focal):
        d = domains.get(sp) if methods[sp] == "random" else None
        if d is not None:
            cells.append(np.asarray(d, dtype=np.int32))
            offsets[sp + 1] = offsets[sp] + len(d)
        else:
            offsets[sp + 1] = offsets[sp]
    b.set_array("movement.random.domain.offsets", offsets)
    b.set_array(
        "movement.random.domain.cells",  # (x, y) pairs, in build_random_patches' order
        np.ascontiguousarray(np.concatenate(cells)) if cells else np.zeros((0, 2), dtype=np.int32),
    )


def _background(b: Bundle, config: Any, grid: Any) -> None:
    """Background (forced) species, feature ``background_species``: what
    ``BackgroundState.get_schools`` needs to build a step's schools, one per species, class and
    ocean cell, in that order.

    The state is constructed as ``simulate()`` constructs it, and its own tables are written:
    per class the length, weight (tonnes), trophic level and age; ``proportion_by_step`` with
    the time series resolved per step (clamped at its end, as ``get_schools`` does);
    ``ocean_cells`` in the state's order; ``base``, the per-ocean-cell biomass frames (the
    forcing, already times the multiplier and on the model grid, or one uniform frame); and
    ``frame_by_step``. A school's biomass is then ``base * proportion`` and its abundance
    ``biomass / weight``: the same two operations ``get_schools`` performs, left to C#.
    The entries exist exactly when the config has background species.
    """
    n_bkg = int(config.n_background)
    if n_bkg == 0:
        return
    from osmose.engine.background import BackgroundState

    state = BackgroundState(config=config.raw_config, grid=grid, engine_config=config)
    species = state._species
    if len(species) != n_bkg:
        raise BundleBuildError(f"background: {len(species)} species parsed, config has {n_bkg}")
    n_steps, ndt = int(config.n_steps), int(config.n_dt_per_year)
    ocean = (state._ocean_ys.astype(np.int64) * grid.nx + state._ocean_xs).astype(np.int32)
    n_ocean = len(ocean)
    offsets = np.zeros(n_bkg + 1, dtype=np.int32)
    lengths: list[float] = []
    weights: list[float] = []
    levels: list[float] = []
    ages: list[int] = []
    proportion = []
    frames: list[np.ndarray] = []
    frame_by_step = np.zeros((n_steps, n_bkg), dtype=np.int32)
    for k, sp in enumerate(species):
        offsets[k + 1] = offsets[k] + sp.n_class
        ts = (
            sp.proportion_ts if sp.proportion_ts is not None and len(sp.proportion_ts) > 0 else None
        )
        for c in range(sp.n_class):
            # get_schools' own fallbacks for a list shorter than n_class.
            lengths.append(sp.lengths[c] if c < len(sp.lengths) else 0.0)
            levels.append(sp.trophic_levels[c] if c < len(sp.trophic_levels) else 2.0)
            ages.append(sp.ages_dt[c] if c < len(sp.ages_dt) else 0)
            weights.append(float(state._weights[k][c]))
            if ts is None:
                proportion.append(np.full(n_steps, sp.proportions[c], dtype=np.float64))
            else:
                rows = np.minimum(np.arange(n_steps), len(ts) - 1)  # clamped, never wrapped
                proportion.append(np.array([float(ts[r, c]) for r in rows], dtype=np.float64))
        forcing = state._forcing_data[k]
        first = len(frames)
        if forcing is not None:
            n_forcing = int(state._forcing_nsteps[k])
            frames.extend(forcing[f][state._ocean_ys, state._ocean_xs] for f in range(n_forcing))
            for step in range(n_steps):  # get_schools' step-to-frame mapping, with its int()
                idx = min(int((step % ndt) * sp.forcing_nsteps_year / ndt), n_forcing - 1)
                frame_by_step[step, k] = first + idx
        elif state._per_cell_biomass[k] < 0:
            raise BundleBuildError(
                f"background species {sp.name!r}: no species.biomass.total and no species.file"
            )
        else:
            frames.append(np.full(n_ocean, state._per_cell_biomass[k], dtype=np.float64))
            frame_by_step[:, k] = first
    b.set_array("background.class.offsets", offsets)
    b.set_array("background.class.length", np.asarray(lengths, dtype=np.float64))
    b.set_array("background.class.weight", np.asarray(weights, dtype=np.float64))
    b.set_array("background.class.trophic_level", np.asarray(levels, dtype=np.float64))
    b.set_array("background.class.age_dt", np.asarray(ages, dtype=np.int32))
    b.set_array("background.proportion_by_step", np.ascontiguousarray(np.stack(proportion, axis=1)))
    b.set_array("background.ocean_cells", ocean)
    b.set_array("background.base", np.ascontiguousarray(np.stack(frames), dtype=np.float64))
    b.set_array("background.frame_by_step", frame_by_step)


def _output(b: Bundle, config: Any, n_total: int) -> None:
    cutoff = config.output_cutoff_age
    b.set_array(
        "output.cutoff_age",
        np.zeros(n_total) if cutoff is None else _f8("output.cutoff_age", cutoff, (n_total,)),
    )
    b.set_int("output.record_frequency", int(config.output_record_frequency))
    b.set_int("output.step0_include", int(bool(config.output_step0_include)))
    families = output_families(config)
    b.set_int("output.families", families)
    # The two binning tables, exactly with their families and exactly as _collect_distributions
    # forms them.
    if families & (FAMILY_BIOMASS_BY_AGE | FAMILY_ABUNDANCE_BY_AGE):
        b.set_array("output.age_bin_offsets", age_bin_offsets(config))
    if families & (FAMILY_BIOMASS_BY_SIZE | FAMILY_ABUNDANCE_BY_SIZE):
        b.set_array("output.size_edges", size_edges(config))
    if families & FAMILY_DIET:
        b.set_int("output.diet_columns", diet_columns(config))


N_STAGES = 3  # Eggs, Juvenil, Adult
#: The life-stage buffers, named as the ``StepOutput`` fields they become.
STAGE_BUFFERS = ("abundance_by_stage", "mortality_by_cause_stage", "mortality_rate_by_cause_stage")
#: Bits of ``output.families`` (C# ``OutputFamilies``). 0 = ``output.minimal.enabled``.
FAMILY_STAGE = 1  # the life-stage split and its rates: every run that is not minimal
FAMILY_BIOMASS_BY_AGE = 2
FAMILY_ABUNDANCE_BY_AGE = 4
FAMILY_BIOMASS_BY_SIZE = 8
FAMILY_ABUNDANCE_BY_SIZE = 16
FAMILY_MEAN_TL = 32
FAMILY_YIELD_N = 64
FAMILY_MEAN_SIZE = 128
FAMILY_SSB = 256
FAMILY_DIET = 512  # diet_by_species; predator pressure is derived from it


def output_families(config: Any) -> int:
    """The families ``_collect_outputs`` computes for this config beyond the minimal set, by the
    same switches it reads. They exist exactly with feature ``full_outputs`` (a config that is
    not minimal)."""
    if config.output_minimal:
        return 0
    on = [
        (FAMILY_STAGE, True),
        (FAMILY_BIOMASS_BY_AGE, config.output_biomass_byage),
        (FAMILY_ABUNDANCE_BY_AGE, config.output_abundance_byage),
        (FAMILY_BIOMASS_BY_SIZE, config.output_biomass_bysize),
        (FAMILY_ABUNDANCE_BY_SIZE, config.output_abundance_bysize),
        (FAMILY_MEAN_TL, config.output_meantl),
        (FAMILY_YIELD_N, config.output_yield_abundance or config.output_yield_abundance_netcdf),
        (FAMILY_MEAN_SIZE, config.output_mean_size or config.output_mean_size_netcdf),
        (FAMILY_SSB, config.output_ssb or config.output_ssb_netcdf),
        (FAMILY_DIET, config.diet_output_enabled),
    ]
    return sum(bit for bit, flag in on if flag)


def age_bin_offsets(config: Any) -> np.ndarray:
    """Where each focal species' age classes start in a by-age record (``n_species + 1``
    values). A species has ``ceil(lifespan / steps per year) + 1`` classes, as
    ``_collect_distributions`` sizes them."""
    bins = [
        int(np.ceil(config.lifespan_dt[sp] / config.n_dt_per_year)) + 1
        for sp in range(config.n_species)
    ]
    return np.concatenate([[0], np.cumsum(bins)]).astype(np.int32)


def size_edges(config: Any) -> np.ndarray:
    """The left edges of the length classes, as ``_collect_distributions`` forms them."""
    edges = np.arange(
        config.output_size_min,
        config.output_size_max + config.output_size_incr,
        config.output_size_incr,
    )
    if len(edges) == 0 or not (np.diff(edges) > 0).all():
        raise BundleBuildError(
            f"output.distrib.bysize: min {config.output_size_min}, max {config.output_size_max}, "
            f"incr {config.output_size_incr} give no increasing length classes"
        )
    return np.ascontiguousarray(edges, dtype=np.float64)


def diet_columns(config: Any) -> int:
    """The diet's prey columns, as ``simulate()`` sizes its diet matrix: the focal species, the
    background species, then the resources ``ResourceState`` discovered (Java's ``DietOutput``;
    ``osmose.engine.output.diet_prey_names``). ``config.resource_names`` is published at run
    time (by ``simulate()``, and by ``build_bundle``), so call this after one of them."""
    return config.n_species + config.n_background + len(config.resource_names)


def family_shapes(config: Any) -> dict[str, tuple[int, ...]]:
    """The shape of ONE record of every family that is on, by buffer name (the ``StepOutput``
    field of the same name), in the engine's order (C# ``OutputLayout.Specs``). With the diet
    on, call it after ``build_bundle`` (see :func:`diet_columns`)."""
    families, nf = output_families(config), config.n_species
    shapes: dict[str, tuple[int, ...]] = {}
    if families & FAMILY_STAGE:
        shapes["abundance_by_stage"] = (nf, N_STAGES)
        shapes["mortality_by_cause_stage"] = (nf, N_MORTALITY_CAUSES, N_STAGES)
        shapes["mortality_rate_by_cause_stage"] = (nf, N_MORTALITY_CAUSES, N_STAGES)
    if families & (FAMILY_BIOMASS_BY_AGE | FAMILY_ABUNDANCE_BY_AGE):
        n_age = int(age_bin_offsets(config)[-1])
        if families & FAMILY_BIOMASS_BY_AGE:
            shapes["biomass_by_age"] = (n_age,)
        if families & FAMILY_ABUNDANCE_BY_AGE:
            shapes["abundance_by_age"] = (n_age,)
    if families & (FAMILY_BIOMASS_BY_SIZE | FAMILY_ABUNDANCE_BY_SIZE):
        n_size = len(size_edges(config))
        if families & FAMILY_BIOMASS_BY_SIZE:
            shapes["biomass_by_size"] = (nf, n_size)
        if families & FAMILY_ABUNDANCE_BY_SIZE:
            shapes["abundance_by_size"] = (nf, n_size)
    for name, bit in (
        ("mean_tl", FAMILY_MEAN_TL),
        ("yield_n", FAMILY_YIELD_N),
        ("mean_size", FAMILY_MEAN_SIZE),
        ("ssb", FAMILY_SSB),
    ):
        if families & bit:
            shapes[name] = (nf,)
    if families & FAMILY_DIET:
        # Predator rows: every species, background ones included (they eat).
        shapes["diet_by_species"] = (nf + config.n_background, diet_columns(config))
    return shapes


# --------------------------------------------------------------------------- result buffers

#: Columns of ``result.mortality``, in ``MortalityCause`` order.
N_MORTALITY_CAUSES = 8
CONTROL_SLOTS = 8


@dataclass
class RunBuffers:
    """The arrays the C# engine writes into, owned by Python (spec 8.4/8.5). Keep this object
    alive until ``RunSimulation`` has returned: the engine holds raw addresses."""

    record_step: np.ndarray  # i4 (n_records): last step of each window, -1 for the initial record
    window_steps: np.ndarray  # i4 (n_records): steps folded into each record
    biomass: np.ndarray  # f8 (n_records, n_total): window mean, tonnes
    abundance: np.ndarray  # f8 (n_records, n_total): window mean
    mortality: np.ndarray  # f8 (n_records, n_focal, 8): window sum of deaths by cause
    yield_: np.ndarray  # f8 (n_records, n_focal): window sum, tonnes
    control: np.ndarray  # i8 (8): [0] cancel (Python), [1] steps done, [2] records, [3] schools
    #: The families beyond the minimal set, by name (``family_shapes``): f8, one row per record.
    #: Empty for a minimal run. The stage rates are a window SUM of per-step rates (inf for an
    #: emptied stage); ``mean_tl`` and ``mean_size`` hold NaN for a species absent from a record.
    families: dict[str, np.ndarray] = field(default_factory=dict)
    #: Where each focal species' age classes start in a by-age row; None without a by-age family.
    age_bin_offsets: np.ndarray | None = None


def n_records(n_steps: int, record_frequency: int, step0_include: bool) -> int:
    """How many records a run writes: one per full window plus a trailing partial one (as
    ``simulate`` flushes it), plus the initial record when ``output.step0.include`` is set."""
    return int(bool(step0_include)) + -(-int(n_steps) // int(record_frequency))


def allocate_results(b: Bundle, config: Any) -> RunBuffers:
    """Allocate the result buffers and the control block and register them in ``b``."""
    n_total, n_focal = config.n_species + config.n_background, config.n_species
    n = n_records(config.n_steps, config.output_record_frequency, config.output_step0_include)
    buffers = RunBuffers(
        record_step=np.full(n, -2, dtype=np.int32),
        window_steps=np.zeros(n, dtype=np.int32),
        biomass=np.zeros((n, n_total)),
        abundance=np.zeros((n, n_total)),
        mortality=np.zeros((n, n_focal, N_MORTALITY_CAUSES)),
        yield_=np.zeros((n, n_focal)),
        control=np.zeros(CONTROL_SLOTS, dtype=np.int64),
    )
    for name in ("record_step", "window_steps", "biomass", "abundance", "mortality"):
        b.set_array(f"result.{name}", getattr(buffers, name))
    b.set_array("result.yield", buffers.yield_)
    for name, shape in family_shapes(config).items():
        buffers.families[name] = np.zeros((n, *shape))
        b.set_array(f"result.{name}", buffers.families[name])
    if "biomass_by_age" in buffers.families or "abundance_by_age" in buffers.families:
        buffers.age_bin_offsets = age_bin_offsets(config)
    b.set_array("control.block", buffers.control)
    return buffers
