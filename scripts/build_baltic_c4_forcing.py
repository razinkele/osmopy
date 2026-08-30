#!/usr/bin/env python
"""C4 salinity-sensitivity forcing builder (spec 2026-08-30, Task 2 of the C4 plan).

Reads the delta-spec JSON (`data/baltic/scenarios/c4_salinity_sensitivity.json`) and the
production Baltic bottom-salinity climatology
(`data/baltic/baltic_salinity_bottom_climatology.nc`) and, per arm, emits an offset salinity
NetCDF plus the sampler-aware instruments the movement gate actually transmits (spec decision
3): TV distance, newly-excluded-cell fraction, prey-overlap shift, and mean-Δw (wiring check
only).

Binding facts (review-verified, task-2-brief.md):
  * the salinity field's land convention is **NaN, not 0.0** -- the OPPOSITE of the O2 file
    B2 offsets (`scripts/build_baltic_b2_forcing.py`); wet = grid.nc's `mask > 0` AND
    `np.isfinite(sal[0])` -- 3 finite off-mask cells are excluded by the AND;
  * offset values are floored at 0, NaN-propagating, additive;
  * the written salinity NetCDF preserves dims/coords/attrs/dtype of the production file
    exactly, and MUST carry exactly `simulation.time.ndtperyear` (24 for Baltic) frames on
    both read and write (CLAUDE.md's frame-count trap: `PhysicalData.get_grid`/`get_value`
    index `step % <loaded array's frame count>`, not the declared metadata field);
  * the sourced-zero arm (dS_PSU == 0.0, `ZERO_ARM_DEF`) must emit a file that is
    value-identical (NaN-aware) to the production input -- verified by `main()`'s self-check;
  * movement-map CSVs are stored upside-down relative to the field; grids are obtained via
    the engine's OWN loader (`osmose.engine.movement_maps._load_csv_grid` +
    `._resolve_path`, imported not reimplemented -- that module carries user modifications
    and is never edited here) so the flip is never redone by hand. The orientation is pinned
    by `tests/test_build_baltic_c4_forcing.py`'s zero-map-positive-cells-on-land test.

Instruments (spec decision 3 -- the sampler renormalizes per cell/frame, so level metrics
are not what the engine sees):
  (i)   `tv_distance` -- total-variation distance between the normalized base and arm
        occupancy distributions (map * w, restricted to map>0 cells), the 24-frame mean;
  (ii)  `prey_overlap_shift` -- change in normalized cod occupancy mass over a prey
        species' footprint (its map>0 cells, unioned across all three life stages), the
        24-frame mean; computed for BOTH the adult cod stage (population-dominant, kept as
        the headline reporting row) AND the juvenile stage (age 0-1, the most
        coastal-skewed of the three maps and the stage most plausibly co-located with
        stickleback in the July chain's first link -- the max-TV/max-exclusion argument
        that picks "adult" for the (i)/(iii) headline figures does not bear on this
        instrument, which measures WHERE redistributed mass lands, not how much w changes)
        -- the per-stage machinery is general and callable on spawning too;
  (iii) `excluded_fraction` -- newly-excluded cell-frame fraction (w_base>0, w_arm==0),
        aggregated over all (frame, map-cell) pairs;
  (iv)  `mean_dw` -- wiring-only sanity check (the gate redistributes/excludes, it never
        removes fish; a near-zero reading on an already-saturated map, e.g. cod_west, is
        the expected null-control signature, not a bug).

Verified against the spec's own stated-expectations table (2026-08-30 design doc,
"Pre-computed expectations" -- computed on the real field x the real production maps):
cod_west's gate is saturated (mean w = 1.0000, all three maps, every frame) and its metrics
at dS=-1/-2 stay within the documented -0.002..-0.021 band; cod_east is 93-99% saturated per
map with mean-dw -0.016..-0.031 (dS=-1) / -0.115..-0.140 (dS=-2), TV 0.028 (dS=-1) / 0.099
(dS=-2) on the adult map specifically (the headline single-number figures in the design doc
are the adult-stage values -- the min/max ranges span the three life-stage maps), and
newly-excluded cells <= 0.23% (also the adult-map max) -- reproduced to the stated precision
by this module's own `main()` smoke run.
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from osmose.engine.processes.salinity_gate import salinity_weight

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SPEC_PATH = ROOT / "data/baltic/scenarios/c4_salinity_sensitivity.json"
DEFAULT_GRID_PATH = ROOT / "data/baltic/baltic_grid.nc"
DEFAULT_SAL_PATH = ROOT / "data/baltic/baltic_salinity_bottom_climatology.nc"
MAPS_CONFIG_DIR = ROOT / "data/baltic"

EXPECTED_FRAMES = 24
S_LOW = 3.0
S_HIGH = 6.0

GATED_SPECIES = ("cod_west", "cod_east")
PREY_SPECIES = ("stickleback", "perch", "pikeperch", "smelt")
STAGES = ("juvenile", "adult", "spawning")
# module docstring (ii): prey_overlap_shift is computed for both stages; "adult" stays the
# population-dominant headline row, "juvenile" is added per review (age 0-1, most
# coastal-skewed, most plausibly co-located with stickleback in the July chain's first link).
PREY_OVERLAP_HEADLINE_STAGE = "adult"
PREY_OVERLAP_STAGES = (PREY_OVERLAP_HEADLINE_STAGE, "juvenile")

# Single source of truth for the "all machinery engaged, zero delta" arm used by both this
# module's own zero self-check and Task 3's harness (B2 precedent: `ZERO_ARM_DEF` is a real
# module-level export, not a private local re-created per call).
ZERO_ARM_DEF = {"name": "zero", "dS_PSU": 0.0}


def _require_24_frames(n_frames: int, where: str) -> None:
    if n_frames != EXPECTED_FRAMES:
        raise ValueError(
            f"{where}: expected exactly {EXPECTED_FRAMES} frames, got {n_frames} "
            "(a mismatched frame count silently misaligns the month-to-step mapping)"
        )


def _single_data_var(ds: xr.Dataset) -> str:
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"expected exactly one data variable in {ds.encoding.get('source')}, found {data_vars}"
        )
    return data_vars[0]


def load_grid_mask(grid_path: Path) -> NDArray[np.bool_]:
    """Ocean mask from `grid.nc`'s `mask` variable (`mask > 0`) -- NOT the full wet rule
    (see `load_wet_mask`): the salinity field carries its own NaN-land convention on top."""
    with xr.open_dataset(grid_path) as ds:
        mask = ds["mask"].values
    return mask > 0


def load_wet_mask(sal_frame0: NDArray[np.float64], grid_path: Path) -> NDArray[np.bool_]:
    """Wet-cell mask: grid.nc mask>0 AND finite salinity (binding fact: this field's land
    convention is NaN, not 0.0 -- the opposite of the O2 file B2 offsets; 3 finite off-mask
    cells are excluded by the AND)."""
    return load_grid_mask(grid_path) & np.isfinite(sal_frame0)


def offset_salinity(
    sal: NDArray[np.float64], wet: NDArray[np.bool_], dS: float
) -> NDArray[np.float64]:
    """Add `dS` to `sal` on wet cells only, floored at 0; non-wet cells (land NaN, or the 3
    finite off-mask cells excluded by the mask-AND-finite rule) are preserved byte-for-byte.
    `sal` has shape (frames, ny, nx); `wet` has shape (ny, nx).

    Raises ValueError if the offset produces a non-finite value on any wet cell.
    """
    sal = np.asarray(sal, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    if sal.shape[-2:] != wet.shape:
        raise ValueError(f"sal spatial shape {sal.shape[-2:]} != wet mask shape {wet.shape}")

    wet3 = np.broadcast_to(wet, sal.shape)
    offset_vals = np.maximum(sal[wet3] + dS, 0.0)
    if not np.all(np.isfinite(offset_vals)):
        raise ValueError("offset_salinity produced non-finite value(s) on wet cells")

    out = sal.copy()
    out[wet3] = offset_vals
    return out


def ramp_w(
    sal: NDArray[np.float64], s_low: float = S_LOW, s_high: float = S_HIGH
) -> NDArray[np.float64]:
    """The production salinity-gate ramp -- a thin wrapper on the engine's own
    `salinity_weight` (imported, not reimplemented): clip((S - s_low)/(s_high - s_low), 0, 1).
    NaN-propagating: NaN salinity (land) yields NaN weight, never a silent zero."""
    return salinity_weight(sal, s_low, s_high)


def _support(map_grid: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Cells the map assigns positive occupancy to (excludes land -99 and NaN alike, since
    both compare False to `> 0`)."""
    return np.asarray(map_grid, dtype=np.float64) > 0.0


def _check_shapes(
    map_grid: NDArray[np.float64], w_base: NDArray[np.float64], w_arm: NDArray[np.float64]
) -> None:
    if w_base.shape != w_arm.shape:
        raise ValueError(f"w_base shape {w_base.shape} != w_arm shape {w_arm.shape}")
    if map_grid.shape != w_base.shape[-2:]:
        raise ValueError(
            f"map_grid shape {map_grid.shape} != weight spatial shape {w_base.shape[-2:]}"
        )


def all_zero_frames(map_grid: NDArray[np.float64], w: NDArray[np.float64]) -> list[int]:
    """Frame indices where `map_grid * w[frame]` sums to zero (or NaN) over the map's
    support cells -- the engine's all-zero guard (`salinity_weighted_map`) silently reverts
    occupancy to the unweighted map for that frame; this is how the builder turns that from
    a silent hazard into a visible, per-(map, frame) fact."""
    map_grid = np.asarray(map_grid, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    support = _support(map_grid)
    if not support.any():
        return list(range(w.shape[0]))
    sums = (map_grid[support] * w[:, support]).sum(axis=1)
    return [int(t) for t in np.flatnonzero(~(sums > 0.0))]


def tv_distance(
    map_grid: NDArray[np.float64], w_base: NDArray[np.float64], w_arm: NDArray[np.float64]
) -> float:
    """Total-variation distance between the normalized base and arm occupancy
    distributions (map * w, restricted to map>0 cells), TV = 0.5 * sum|p - q|, returned as
    the mean over frames. If an arm's (or base's) map*w sums to 0 for a frame -- the
    all-zero case, see `all_zero_frames` -- that frame contributes NaN and is excluded from
    the mean (nanmean); if every frame is all-zero, returns NaN."""
    map_grid = np.asarray(map_grid, dtype=np.float64)
    w_base = np.asarray(w_base, dtype=np.float64)
    w_arm = np.asarray(w_arm, dtype=np.float64)
    _check_shapes(map_grid, w_base, w_arm)

    support = _support(map_grid)
    if not support.any():
        return float("nan")

    base_vals = map_grid[support] * w_base[:, support]
    arm_vals = map_grid[support] * w_arm[:, support]
    base_sum = base_vals.sum(axis=1)
    arm_sum = arm_vals.sum(axis=1)
    valid = (base_sum > 0.0) & (arm_sum > 0.0)
    if not valid.any():
        return float("nan")

    p = np.zeros_like(base_vals)
    q = np.zeros_like(arm_vals)
    p[valid] = base_vals[valid] / base_sum[valid, None]
    q[valid] = arm_vals[valid] / arm_sum[valid, None]

    tv = np.full(w_base.shape[0], np.nan)
    tv[valid] = 0.5 * np.abs(p - q).sum(axis=1)[valid]
    return float(np.nanmean(tv))


def excluded_fraction(
    map_grid: NDArray[np.float64], w_base: NDArray[np.float64], w_arm: NDArray[np.float64]
) -> float:
    """Map-cell fraction with w_base>0 & w_arm==0 (newly excluded) -- the July-mechanism
    lever. The denominator is the fraction of ALL (frame, support-cell) pairs, INCLUDING
    already-closed cells (w_base==0 at baseline, which can never be "newly" excluded) --
    NOT the fraction of previously-open (w_base>0) cells that closed."""
    map_grid = np.asarray(map_grid, dtype=np.float64)
    w_base = np.asarray(w_base, dtype=np.float64)
    w_arm = np.asarray(w_arm, dtype=np.float64)
    _check_shapes(map_grid, w_base, w_arm)

    support = _support(map_grid)
    if not support.any():
        return 0.0

    base_pos = w_base[:, support] > 0.0
    arm_zero = w_arm[:, support] == 0.0
    newly_excluded = base_pos & arm_zero
    return float(newly_excluded.sum()) / float(newly_excluded.size)


def mean_dw(
    map_grid: NDArray[np.float64], w_base: NDArray[np.float64], w_arm: NDArray[np.float64]
) -> float:
    """Mean (w_arm - w_base) over the map's support cells and all frames -- a WIRING CHECK
    ONLY (monotone => ~0 iff nothing changed). Never report this beside stock responses
    without the framing sentence: the gate conserves total occupancy -- it redistributes
    and excludes, it never removes fish. Uses nanmean for robustness/consistency with the
    other instruments' NaN-aware aggregation (no NaN occurs on real production support
    cells -- this is defensive, not a behavior change)."""
    map_grid = np.asarray(map_grid, dtype=np.float64)
    w_base = np.asarray(w_base, dtype=np.float64)
    w_arm = np.asarray(w_arm, dtype=np.float64)
    _check_shapes(map_grid, w_base, w_arm)

    support = _support(map_grid)
    if not support.any():
        return 0.0

    return float(np.nanmean(w_arm[:, support] - w_base[:, support]))


def prey_overlap_shift(
    map_grid: NDArray[np.float64],
    w_base: NDArray[np.float64],
    w_arm: NDArray[np.float64],
    prey_map: NDArray[np.float64],
) -> float:
    """Change in normalized cod occupancy mass over a prey species' footprint (`prey_map >
    0` cells), the 24-frame mean -- the direct predation-chain lever. Frames where the
    normalization is undefined (see `tv_distance`) are excluded from the mean."""
    map_grid = np.asarray(map_grid, dtype=np.float64)
    prey_map = np.asarray(prey_map, dtype=np.float64)
    w_base = np.asarray(w_base, dtype=np.float64)
    w_arm = np.asarray(w_arm, dtype=np.float64)
    _check_shapes(map_grid, w_base, w_arm)
    if prey_map.shape != map_grid.shape:
        raise ValueError(f"prey_map shape {prey_map.shape} != map_grid shape {map_grid.shape}")

    support = _support(map_grid)
    if not support.any():
        return float("nan")

    base_vals = map_grid[support] * w_base[:, support]
    arm_vals = map_grid[support] * w_arm[:, support]
    base_sum = base_vals.sum(axis=1)
    arm_sum = arm_vals.sum(axis=1)
    valid = (base_sum > 0.0) & (arm_sum > 0.0)
    if not valid.any():
        return float("nan")

    prey_within_support = (prey_map > 0.0)[support]

    p = np.zeros_like(base_vals)
    q = np.zeros_like(arm_vals)
    p[valid] = base_vals[valid] / base_sum[valid, None]
    q[valid] = arm_vals[valid] / arm_sum[valid, None]

    mass_base = p[:, prey_within_support].sum(axis=1)
    mass_arm = q[:, prey_within_support].sum(axis=1)

    shift = np.full(w_base.shape[0], np.nan)
    shift[valid] = (mass_arm - mass_base)[valid]
    return float(np.nanmean(shift))


def _mean_w(map_grid: NDArray[np.float64], w: NDArray[np.float64]) -> float:
    """Reporting helper: mean occupancy weight over a map's support cells and all frames
    (used to print the cod_west/cod_east saturation figures from the expectations table)."""
    support = _support(map_grid)
    if not support.any():
        return float("nan")
    return float(np.mean(w[:, support]))


def _saturated_fraction(map_grid: NDArray[np.float64], w: NDArray[np.float64]) -> float:
    """Reporting helper: fraction of (frame, support-cell) pairs at the ramp's ceiling
    (w == 1.0, i.e. baseline salinity already >= s_high)."""
    support = _support(map_grid)
    if not support.any():
        return float("nan")
    return float(np.mean(w[:, support] == 1.0))


@lru_cache(maxsize=None)
def load_stage_map(
    species: str, stage: str, ny: int, nx: int, config_dir: str = str(MAPS_CONFIG_DIR)
) -> NDArray[np.float64]:
    """One species/stage movement map, loaded via the engine's OWN CSV grid loader
    (`osmose.engine.movement_maps._load_csv_grid` + `._resolve_path`, imported not
    reimplemented). `_load_csv_grid` flips the CSV's upside-down on-disk storage on load
    (row 0 in the CSV = northernmost = grid row ny-1) -- see that module's docstring and
    this module's orientation-pinning test."""
    from osmose.engine.movement_maps import _load_csv_grid, _resolve_path

    path = _resolve_path(f"maps/{species}_{stage}.csv", config_dir=config_dir)
    return _load_csv_grid(path, ny, nx)


def load_species_maps(
    species: str, ny: int, nx: int, config_dir: Path = MAPS_CONFIG_DIR
) -> dict[str, NDArray[np.float64]]:
    """All three life-stage maps (juvenile/adult/spawning) for one gated species."""
    return {stage: load_stage_map(species, stage, ny, nx, str(config_dir)) for stage in STAGES}


def load_prey_union_map(
    species: str, ny: int, nx: int, config_dir: Path = MAPS_CONFIG_DIR
) -> NDArray[np.float64]:
    """Prey occupancy DOMAIN for `prey_overlap_shift`: 1.0 where the species is present
    (map > 0) in ANY life stage, 0.0 elsewhere -- only the `prey_map > 0` boolean support
    matters to the instrument, not the probability magnitude, so a per-stage union is the
    most permissive (and simplest, unambiguous) footprint."""
    stages = load_species_maps(species, ny, nx, config_dir)
    presence = np.zeros((ny, nx), dtype=bool)
    for grid in stages.values():
        presence |= grid > 0.0
    return presence.astype(np.float64)


def _write_offset_salinity_netcdf(
    prod_sal_path: Path, offset_values: np.ndarray, out_path: Path
) -> None:
    """Write `offset_values` into a copy of the production salinity dataset, preserving
    dims/coords/attrs/dtype exactly (only the data variable's values change)."""
    with xr.open_dataset(prod_sal_path) as ds:
        varname = _single_data_var(ds)
        _require_24_frames(ds[varname].shape[0], f"{prod_sal_path} (read)")
        ds_out = ds.load().copy(deep=True)
    ds_out[varname].values[...] = offset_values.astype(ds_out[varname].dtype, copy=False)
    _require_24_frames(ds_out[varname].shape[0], f"{out_path} (write)")
    ds_out.to_netcdf(out_path)


def write_arm_dir(arm: dict, out_dir: Path, prod_sal_path: Path, grid_path: Path) -> dict:
    """Write one arm's forcing artifact (offset salinity NetCDF) into `out_dir` and compute
    the sampler-aware instruments (spec decision 3) for every gated-species map. Returns
    `{"sal_nc": path, "instruments": {...}, "all_zero_events": [...]}`.

    `instruments` is `{species: {stage: {"mean_w_base", "mean_dw", "tv",
    "excluded_fraction", "saturated_fraction"}}, "prey_overlap": {species: {stage: {prey:
    float}}}}` (prey overlap computed per named cod stage in `PREY_OVERLAP_STAGES` --
    "adult" is the population-dominant headline row, "juvenile" the most coastal-skewed
    stage). `all_zero_events` is a list of `{"species", "stage", "frame"}` dicts (see
    `all_zero_frames`).
    """
    out_dir = Path(out_dir).resolve()  # absolute, so Task 3's overlay never depends on cwd
    out_dir.mkdir(parents=True, exist_ok=True)

    dS = float(arm["dS_PSU"])

    with xr.open_dataset(prod_sal_path) as ds:
        varname = _single_data_var(ds)
        _require_24_frames(ds[varname].shape[0], f"{prod_sal_path} (read)")
        sal = ds[varname].values.astype(np.float64)

    ny, nx = sal.shape[-2:]
    wet = load_wet_mask(sal[0], grid_path)
    if wet.shape != (ny, nx):
        raise ValueError(f"wet mask shape {wet.shape} != salinity spatial shape {(ny, nx)}")

    offset = offset_salinity(sal, wet, dS)
    sal_nc = out_dir / "salinity_offset.nc"
    _write_offset_salinity_netcdf(prod_sal_path, offset, sal_nc)

    w_base = ramp_w(sal)
    w_arm = ramp_w(offset)

    gated_maps = {sp: load_species_maps(sp, ny, nx) for sp in GATED_SPECIES}
    prey_maps = {sp: load_prey_union_map(sp, ny, nx) for sp in PREY_SPECIES}

    instruments: dict = {}
    all_zero_events: list[dict] = []

    for species, stages in gated_maps.items():
        instruments[species] = {}
        for stage, map_grid in stages.items():
            instruments[species][stage] = {
                "mean_w_base": _mean_w(map_grid, w_base),
                "mean_dw": mean_dw(map_grid, w_base, w_arm),
                "tv": tv_distance(map_grid, w_base, w_arm),
                "excluded_fraction": excluded_fraction(map_grid, w_base, w_arm),
                "saturated_fraction": _saturated_fraction(map_grid, w_base),
            }
            for frame in all_zero_frames(map_grid, w_arm):
                all_zero_events.append({"species": species, "stage": stage, "frame": frame})

    prey_overlap: dict = {}
    for species, stages in gated_maps.items():
        prey_overlap[species] = {
            stage: {
                prey: prey_overlap_shift(stages[stage], w_base, w_arm, prey_maps[prey])
                for prey in PREY_SPECIES
            }
            for stage in PREY_OVERLAP_STAGES
        }
    instruments["prey_overlap"] = prey_overlap

    return {"sal_nc": sal_nc, "instruments": instruments, "all_zero_events": all_zero_events}


def _zero_self_check(out_root: Path, prod_sal_path: Path, grid_path: Path) -> bool:
    """Blocking self-check (spec decision 4(a)): the zero-delta arm's written salinity file
    must be value-identical (NaN-aware) to the production input it was copied from."""
    result = write_arm_dir(ZERO_ARM_DEF, out_root / "zero", prod_sal_path, grid_path)

    with (
        xr.open_dataset(prod_sal_path) as prod_ds,
        xr.open_dataset(result["sal_nc"]) as zero_ds,
    ):
        prod_var = _single_data_var(prod_ds)
        zero_var = _single_data_var(zero_ds)
        prod_vals = prod_ds[prod_var].values
        zero_vals = zero_ds[zero_var].values
        _require_24_frames(prod_vals.shape[0], f"{prod_sal_path} (self-check re-read)")
        _require_24_frames(zero_vals.shape[0], f"{result['sal_nc']} (self-check re-read)")
        return bool(np.array_equal(prod_vals, zero_vals, equal_nan=True))


def main() -> None:
    spec = json.loads(DEFAULT_SPEC_PATH.read_text())
    out_root = Path(tempfile.mkdtemp(prefix="c4_forcing_"))
    print(f"C4 salinity forcing builder: writing artifacts to {out_root}")

    for arm in spec["arms"]:
        result = write_arm_dir(arm, out_root / arm["name"], DEFAULT_SAL_PATH, DEFAULT_GRID_PATH)
        print(f"\n  {arm['name']} (dS={float(arm['dS_PSU']):+.1f} PSU):")
        for species in GATED_SPECIES:
            for stage in STAGES:
                v = result["instruments"][species][stage]
                print(
                    f"    {species:10s} {stage:10s} "
                    f"mean_w_base={v['mean_w_base']:.4f} "
                    f"mean_dw={v['mean_dw']:+.4f} tv={v['tv']:.4f} "
                    f"excluded={v['excluded_fraction']:.4%} "
                    f"saturated={v['saturated_fraction']:.1%}"
                )
        prey_overlap = result["instruments"]["prey_overlap"]
        for species in GATED_SPECIES:
            for stage in PREY_OVERLAP_STAGES:
                headline = " [headline]" if stage == PREY_OVERLAP_HEADLINE_STAGE else ""
                for prey in PREY_SPECIES:
                    print(
                        f"    prey_overlap_shift[{species} {stage}{headline} -> {prey}] = "
                        f"{prey_overlap[species][stage][prey]:+.5f}"
                    )
        if result["all_zero_events"]:
            print(
                f"    ALL-ZERO EVENTS ({len(result['all_zero_events'])}): "
                f"{result['all_zero_events']}"
            )
        else:
            print("    all-zero events: none")

    print(
        "\n  NOTE: the gate conserves total occupancy -- it redistributes and excludes, it "
        "never removes fish. mean_dw is a wiring check only, never a stock-response metric."
    )

    identical = _zero_self_check(out_root, DEFAULT_SAL_PATH, DEFAULT_GRID_PATH)
    status = "PASS" if identical else "FAIL"
    print(f"\nzero-delta self-check (NaN-aware value identity vs production input): {status}")
    if not identical:
        raise AssertionError(
            "zero-delta self-check FAILED: the written zero-arm salinity file diverges from "
            "the production input it was supposed to copy value-identically"
        )


if __name__ == "__main__":
    main()
