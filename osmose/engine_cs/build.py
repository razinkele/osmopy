"""Build the parameter bundle the C# engine runs from (spec section 8.2, plan task 1.2).

Python resolves, C# computes (spec 8.1): every table here is what the Python engine itself reads,
already parsed, per species and, where a value can change, per step. The C# engine never sees a
config key, a file or a species name it must interpret.

Task 1.2a builds the header and the static tables: contract, run, time, grid, species, growth,
predation (with feeding stages), stage accessibility, non-fishing mortality and output settings.
Fishing and reproduction (1.2b) and resources, movement and the result buffers (1.2c) follow.

Entry names, dtypes and shapes are the contract. Every array is validated against
``n_total = n_focal + n_background`` (or ``n_focal``) here, so a mis-sized table fails at build
time with its name rather than inside the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def _output(b: Bundle, config: Any, n_total: int) -> None:
    cutoff = config.output_cutoff_age
    b.set_array(
        "output.cutoff_age",
        np.zeros(n_total) if cutoff is None else _f8("output.cutoff_age", cutoff, (n_total,)),
    )
    b.set_int("output.record_frequency", int(config.output_record_frequency))
    b.set_int("output.step0_include", int(bool(config.output_step0_include)))
    b.set_int("output.families", 0)  # 0 = output.minimal.enabled; full families are Phase 2
