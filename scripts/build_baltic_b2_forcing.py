#!/usr/bin/env python
"""B2 literature-delta forcing builder (spec 2026-08-29, Task 2 of the B2 plan).

Reads the delta-spec JSON (`data/baltic/scenarios/b2_literature_deltas.json`) and the
production Baltic forcing (`data/baltic/baltic_grid.nc`, `data/baltic/baltic_oxygen_bottom.nc`)
and, per arm, emits: a constant-T knob series CSV (C1 conventions, via the imported
`write_arm_series`) and, for arms carrying a `dO2` block (including the sourced-zero
`rcp45_ref` arm), an offset O2 NetCDF.

Binding spec constraints (design §Design 2, decision 3):
  * the O2 offset is applied on WET CELLS ONLY (grid.nc's `mask` variable, mask > 0);
    land/NaN cells are left byte-for-byte untouched;
  * offset values are floored at 0;
  * the written O2 NetCDF preserves dims/coords/attrs/dtype of the production file exactly;
  * the file MUST carry exactly `simulation.time.ndtperyear` (24 for Baltic) frames, both
    on read and on write -- `PhysicalData` indexes by `step % <loaded frame count>`, so a
    mismatched frame count silently misaligns the month-to-step mapping (CLAUDE.md trap);
  * the sourced-zero arm (dO2 == 0.0) must emit a file that is value-identical (NaN-aware)
    to the production input -- verified by `main()`'s zero-delta self-check.

`predicted_k_change` reports the effect of an O2 delta on the O2->benthos-K Hill coupling
(`osmose.engine.processes.oxygen_function.f_o2_hill`) as a K-weighted mean-factor ratio.
`write_arm_dir` weights it by the REAL Benthos resource forcing field
(`data/baltic/baltic_ltl_biomass.nc`, variable `Benthos`, the same field
`osmose.engine.resources.ResourceState` scales into K -- see `species.multiplier.sp14`
(absent, defaults to 1.0) / `species.accessibility2fish.sp14` (0.8): both are per-config
SCALARS in the production Baltic config, so the raw Benthos field is proportional to the
real per-cell-frame K, and any positive scalar multiple of a weight array leaves a weighted
*ratio* unchanged -- no reconstruction of the multiplier/accessibility scaling is needed).
Verified against the design doc's own diagnostic: unweighted baseline mean Hill factor here
reproduces the documented 0.8426 (`data/baltic/baltic_param-oxygen.csv:10`); Benthos-weighted
reproduces the documented 0.866 (design doc "The O2 axis..." section) to 3 significant
figures. `predicted_k_change` itself stays a general pure function and accepts any
`k_weights` array (this reproduction is what `write_arm_dir` happens to pass it).
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.engine.processes.oxygen_function import f_o2_hill

ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent

# C1's knob-series writer + constants (herring-only thermal knob), reused verbatim via the
# established scripts/ importlib-from-path idiom (see scripts/build_cod_ew_maps.py).
_c1_spec = importlib.util.spec_from_file_location(
    "baltic_c1_knob_ab", _HERE / "baltic_c1_knob_ab.py"
)
_c1 = importlib.util.module_from_spec(_c1_spec)
_c1_spec.loader.exec_module(_c1)
write_arm_series = _c1.write_arm_series
TREFS = _c1.TREFS
BETAS = _c1.BETAS

DEFAULT_SPEC_PATH = ROOT / "data/baltic/scenarios/b2_literature_deltas.json"
DEFAULT_GRID_PATH = ROOT / "data/baltic/baltic_grid.nc"
DEFAULT_O2_PATH = ROOT / "data/baltic/baltic_oxygen_bottom.nc"
DEFAULT_LTL_PATH = ROOT / "data/baltic/baltic_ltl_biomass.nc"
BENTHOS_VAR = "Benthos"
EXPECTED_FRAMES = 24

# Single source of truth for the synthetic "all machinery engaged, zero deltas" arm used by
# both this module's own zero-delta self-check and Task 3's harness (`baltic_b2_scenario_ab
# .ZERO_ARM_DEF = build_baltic_b2_forcing.ZERO_ARM_DEF` -- controller review MINOR 1: this
# dict was previously duplicated in both files).
ZERO_ARM_DEF = {"name": "zero", "dT_C": 0.0, "dO2": {"value_mmol_m3": 0.0}}


def _require_24_frames(n_frames: int, where: str) -> None:
    if n_frames != EXPECTED_FRAMES:
        raise ValueError(
            f"{where}: expected exactly {EXPECTED_FRAMES} frames, got {n_frames} "
            "(a mismatched frame count silently misaligns the month-to-step mapping)"
        )


def load_wet_mask(grid_path: Path) -> np.ndarray:
    """Wet-cell mask from `grid.nc`'s `mask` variable (matches
    `osmose.engine.grid.Grid.from_netcdf`: ocean where `mask > 0`)."""
    with xr.open_dataset(grid_path) as ds:
        mask = ds["mask"].values
    return mask > 0


def load_benthos_k_weights(ltl_path: Path = DEFAULT_LTL_PATH) -> np.ndarray:
    """Real per-cell-frame benthos K weights: the `Benthos` variable of the LTL biomass
    forcing (`data/baltic/baltic_ltl_biomass.nc`), shape (frames, ny, nx) -- proportional to
    the K the O2->benthos coupling actually scales (see module docstring)."""
    with xr.open_dataset(ltl_path) as ds:
        return ds[BENTHOS_VAR].values.astype(np.float64)


def _single_data_var(ds: xr.Dataset) -> str:
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"expected exactly one data variable in {ds.encoding.get('source')}, found {data_vars}"
        )
    return data_vars[0]


def offset_o2(o2: np.ndarray, wet: np.ndarray, delta: float) -> np.ndarray:
    """Add `delta` to `o2` on wet cells only, floored at 0; non-wet cells are preserved
    byte-for-byte. `o2` has shape (frames, ny, nx); `wet` has shape (ny, nx).

    Raises ValueError if the offset produces a non-finite value on any wet cell.
    """
    o2 = np.asarray(o2, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    if o2.shape[-2:] != wet.shape:
        raise ValueError(f"o2 spatial shape {o2.shape[-2:]} != wet mask shape {wet.shape}")

    wet3 = np.broadcast_to(wet, o2.shape)
    offset_vals = np.maximum(o2[wet3] + delta, 0.0)
    if not np.all(np.isfinite(offset_vals)):
        raise ValueError("offset_o2 produced non-finite value(s) on wet cells")

    out = o2.copy()
    out[wet3] = offset_vals
    return out


def predicted_k_change(
    o2: np.ndarray,
    wet: np.ndarray,
    k_weights: np.ndarray,
    delta: float,
    c50: float = 60.0,
    n: float = 3.0,
) -> float:
    """K-weighted mean Hill factor with the offset applied, divided by the same mean
    without it, minus 1 -- the predicted fractional change in effective benthos K.

    Weighting/averaging is over every (frame, wet-cell) pair, `k_weights` (ny, nx)
    broadcast across frames -- matching the "cell-frames" framing used in the design doc's
    own K-weighted diagnostic.
    """
    o2 = np.asarray(o2, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    k_weights = np.asarray(k_weights, dtype=np.float64)
    offset = offset_o2(o2, wet, delta)

    wet3 = np.broadcast_to(wet, o2.shape)
    w3 = np.broadcast_to(k_weights, o2.shape)
    weights = w3[wet3]
    total_w = float(weights.sum())
    if total_w <= 0.0:
        raise ValueError("predicted_k_change: k_weights sum to zero over wet cells")

    hill_before = f_o2_hill(o2[wet3], c50, n)
    hill_after = f_o2_hill(offset[wet3], c50, n)
    mean_before = float(np.sum(hill_before * weights) / total_w)
    mean_after = float(np.sum(hill_after * weights) / total_w)
    return mean_after / mean_before - 1.0


def count_floored_cells(o2: np.ndarray, wet: np.ndarray, delta: float) -> int:
    """Count wet cell-frames where `o2 + delta` goes negative (so `offset_o2` floors them to
    0) -- the floor-asymmetry diagnostic (design doc §4's "floor asymmetry" label: additive
    offsets are floor-clipped on the negative side but uncapped on the positive side)."""
    o2 = np.asarray(o2, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    wet3 = np.broadcast_to(wet, o2.shape)
    return int(np.sum((o2[wet3] + delta) < 0.0))


def _write_offset_o2_netcdf(prod_o2_path: Path, offset_values: np.ndarray, out_path: Path) -> None:
    """Write `offset_values` into a copy of the production O2 dataset, preserving
    dims/coords/attrs/dtype exactly (only the data variable's values change)."""
    with xr.open_dataset(prod_o2_path) as ds:
        varname = _single_data_var(ds)
        _require_24_frames(ds[varname].shape[0], f"{prod_o2_path} (read)")
        ds_out = ds.load().copy(deep=True)
    ds_out[varname].values[...] = offset_values.astype(ds_out[varname].dtype, copy=False)
    _require_24_frames(ds_out[varname].shape[0], f"{out_path} (write)")
    ds_out.to_netcdf(out_path)


def write_arm_dir(
    arm: dict,
    out_dir: Path,
    prod_o2_path: Path,
    grid_path: Path,
    trefs: dict[int, float],
    betas: dict[int, float],
) -> dict:
    """Write one arm's forcing artifacts (knob series CSV + offset O2 NetCDF, when the arm
    carries a `dO2` block) into `out_dir`. Returns
    `{"series_csv": path, "o2_nc": path|None, "predicted_dK": float}`.

    `betas` is accepted (not used here) so callers can pass the same `(trefs, betas)` pair
    they pass to Task 3's `arm_overlays` -- the knob series only needs `trefs` + `dT_C`.
    """
    del betas  # not needed by the knob series or the O2 offset; kept for call-site symmetry
    out_dir = Path(out_dir).resolve()  # absolute, so Task 3's oxygen.filename overlay never
    out_dir.mkdir(parents=True, exist_ok=True)  # depends on the caller's cwd (spec §4(b))

    dT = float(arm["dT_C"])
    series_csv = out_dir / "knob_series.csv"
    write_arm_series(series_csv, trefs, dT)

    o2_nc: Path | None = None
    predicted_dK = 0.0
    dO2 = arm.get("dO2")
    if dO2 is not None:
        delta = float(dO2["value_mmol_m3"])
        wet = load_wet_mask(grid_path)
        with xr.open_dataset(prod_o2_path) as ds:
            varname = _single_data_var(ds)
            _require_24_frames(ds[varname].shape[0], f"{prod_o2_path} (read)")
            o2 = ds[varname].values.astype(np.float64)

        if o2.shape[-2:] != wet.shape:
            raise ValueError(
                f"O2 field spatial shape {o2.shape[-2:]} != grid wet-mask shape {wet.shape}"
            )

        offset = offset_o2(o2, wet, delta)
        o2_nc = out_dir / "oxygen_offset.nc"
        _write_offset_o2_netcdf(prod_o2_path, offset, o2_nc)

        k_weights = load_benthos_k_weights()
        if k_weights.shape != o2.shape:
            raise ValueError(
                f"Benthos K-weight field shape {k_weights.shape} != O2 field shape {o2.shape} "
                "(frame count or spatial grid mismatch between the two production forcing files)"
            )
        predicted_dK = predicted_k_change(o2, wet, k_weights, delta)

    return {"series_csv": series_csv, "o2_nc": o2_nc, "predicted_dK": predicted_dK}


def _zero_delta_self_check(out_root: Path, prod_o2_path: Path, grid_path: Path) -> bool:
    """Blocking self-check (spec §Design 2): a synthetic zero-delta arm's written O2 file
    must be value-identical (NaN-aware) to the production input it was copied from."""
    artifacts = write_arm_dir(
        ZERO_ARM_DEF, out_root / "zero", prod_o2_path, grid_path, TREFS, BETAS
    )

    with xr.open_dataset(prod_o2_path) as prod_ds, xr.open_dataset(artifacts["o2_nc"]) as zero_ds:
        prod_var = _single_data_var(prod_ds)
        zero_var = _single_data_var(zero_ds)
        prod_vals = prod_ds[prod_var].values
        zero_vals = zero_ds[zero_var].values
        _require_24_frames(prod_vals.shape[0], f"{prod_o2_path} (self-check re-read)")
        _require_24_frames(zero_vals.shape[0], f"{artifacts['o2_nc']} (self-check re-read)")
        return bool(np.array_equal(prod_vals, zero_vals, equal_nan=True))


def main() -> None:
    spec = json.loads(DEFAULT_SPEC_PATH.read_text())
    out_root = Path(tempfile.mkdtemp(prefix="b2_forcing_"))
    print(f"B2 forcing builder: writing artifacts to {out_root}")

    wet = load_wet_mask(DEFAULT_GRID_PATH)
    with xr.open_dataset(DEFAULT_O2_PATH) as ds:
        prod_o2 = ds[_single_data_var(ds)].values.astype(np.float64)

    for arm in spec["arms"]:
        artifacts = write_arm_dir(
            arm, out_root / arm["name"], DEFAULT_O2_PATH, DEFAULT_GRID_PATH, TREFS, BETAS
        )
        dK = artifacts["predicted_dK"]
        dO2 = arm.get("dO2")
        floored = ""
        if dO2 is not None:
            n_floored = count_floored_cells(prod_o2, wet, float(dO2["value_mmol_m3"]))
            floored = f", floored wet cell-frames = {n_floored}"
        print(f"  {arm['name']}: predicted effective-K change = {dK:+.4%}{floored}")

    identical = _zero_delta_self_check(out_root, DEFAULT_O2_PATH, DEFAULT_GRID_PATH)
    status = "PASS" if identical else "FAIL"
    print(f"zero-delta self-check (NaN-aware value identity vs production input): {status}")
    if not identical:
        raise AssertionError(
            "zero-delta self-check FAILED: the written zero-arm O2 file diverges from the "
            "production input it was supposed to copy value-identically"
        )


if __name__ == "__main__":
    main()
