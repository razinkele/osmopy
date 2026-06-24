"""Parity gate: C _apply_predation_once vs Numba _apply_predation_numba.

parity_for_cell(arrays, meta, cell) builds two independent fresh arg sets for
the same cell, runs the Numba oracle on one and the C kernel on the other,
returns {array_name: max_rel_diff} over the 7 MUTATED arrays.

assert_parity(report, bar=1e-12) raises AssertionError if any entry exceeds bar.
"""
from __future__ import annotations

import ctypes
import importlib
from typing import Any

import numpy as np

from scripts.spikes.native_predation.leaf_args import (
    LEAF_ARG_ORDER,
    MUTATED,
    build_leaf_args,
)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _get_numba_leaf():
    """Return _apply_predation_numba from mortality.py (already njit-compiled)."""
    from osmose.engine.processes import mortality  # noqa: PLC0415
    return mortality._apply_predation_numba


def _get_c_module():
    """Return the _leaf_portable cffi module (portable = reproducible, no march=native)."""
    return importlib.import_module("scripts.spikes.native_predation._leaf_portable")


# ---------------------------------------------------------------------------
# C marshaller
# ---------------------------------------------------------------------------

def _call_c_once(args: list, ffi: Any, lib: Any) -> None:
    """Call apply_predation_once with the 41-element arg list + 7 aux shape ints.

    args is the same positional list produced by build_leaf_args (LEAF_ARG_ORDER).
    All array pointers are cast to C-contiguous float64/int32 before the call.
    The 7 aux shape ints are derived from array shapes.

    CRITICAL: we keep references to every cast array alive until after the call
    so the GC cannot collect them.
    """
    # Build a name→value dict for readable access
    a: dict[str, Any] = {name: args[i] for i, name in enumerate(LEAF_ARG_ORDER)}

    # --- Ensure C-contiguous copies with correct dtype ---
    # (build_leaf_args already provides correct dtypes but we be explicit here)

    def _f64(x):
        arr = np.ascontiguousarray(x, dtype=np.float64)
        return arr, ffi.cast("double *", arr.ctypes.data)

    def _i32(x):
        arr = np.ascontiguousarray(x, dtype=np.int32)
        return arr, ffi.cast("int *", arr.ctypes.data)

    # ---- scalars (non-array args) ----
    p_idx        = int(a["p_idx"])
    n_local      = len(a["cell_indices"])
    n_dt_py      = float(a["n_dt_per_year"])   # C takes double
    n_subdt_py   = float(a["n_subdt"])          # C takes double
    has_access   = int(bool(a["has_access"]))
    use_stage_ac = int(bool(a["use_stage_access"]))
    n_resources  = int(a["n_resources"])
    n_species    = int(a["n_species"])
    cell_id      = int(a["cell_id"])
    tl_tracking  = int(bool(a["tl_tracking"]))
    diet_enabled = int(bool(a["diet_enabled"]))

    # ---- 7 aux shape ints ----
    # srm_ncol: size_ratio_min.shape[1]
    srm = np.ascontiguousarray(a["size_ratio_min"], dtype=np.float64)
    srm_ncol  = srm.shape[1]

    # acc_nrow/ncol: access_matrix.shape
    acc_arr = np.ascontiguousarray(a["access_matrix"], dtype=np.float64)
    acc_nrow, acc_ncol = acc_arr.shape

    # n_cells: rsc_biomass.shape[1]
    rsb = np.ascontiguousarray(a["rsc_biomass"], dtype=np.float64)
    n_cells = rsb.shape[1]

    # n_causes: n_dead.shape[1]
    nd = np.ascontiguousarray(a["n_dead"], dtype=np.float64)
    n_causes = nd.shape[1]

    # diet_nrow/ncol: diet_matrix.shape
    dm = np.ascontiguousarray(a["diet_matrix"], dtype=np.float64)
    diet_nrow, diet_ncol = dm.shape

    # ---- array pointers (hold refs to prevent GC) ----
    ci_arr, ci_ptr           = _i32(a["cell_indices"])
    ia_arr, ia_ptr           = _f64(a["inst_abd"])
    # n_dead already prepared above; get pointer
    nd_ptr = ffi.cast("double *", nd.ctypes.data)
    si_arr, si_ptr           = _i32(a["species_id"])
    le_arr, le_ptr           = _f64(a["length"])
    we_arr, we_ptr           = _f64(a["weight"])
    ag_arr, ag_ptr           = _i32(a["age_dt"])
    ffa_arr, ffa_ptr         = _i32(a["first_feeding_age_dt"])
    fs_arr, fs_ptr           = _i32(a["feeding_stage"])
    psr_arr, psr_ptr         = _f64(a["pred_success_rate"])
    pb_arr, pb_ptr           = _f64(a["preyed_biomass"])
    tl_arr, tl_ptr           = _f64(a["trophic_level"])
    # size_ratio_min/max already prepared
    srm_ptr = ffi.cast("double *", srm.ctypes.data)
    srmx_arr = np.ascontiguousarray(a["size_ratio_max"], dtype=np.float64)
    srmx_ptr = ffi.cast("double *", srmx_arr.ctypes.data)
    ir_arr, ir_ptr           = _f64(a["ingestion_rate"])
    frs_arr, frs_ptr         = _i32(a["fr_shape"])
    frh_arr, frh_ptr         = _f64(a["fr_halfsat"])
    # access_matrix already prepared
    am_ptr = ffi.cast("double *", acc_arr.ctypes.data)
    pai_arr, pai_ptr         = _i32(a["prey_access_idx"])
    pdi_arr, pdi_ptr         = _i32(a["pred_access_idx"])
    # rsc_biomass already prepared; need mutable pointer
    rsb_ptr = ffi.cast("double *", rsb.ctypes.data)
    rsmin_arr, rsmin_ptr     = _f64(a["rsc_size_min"])
    rsmax_arr, rsmax_ptr     = _f64(a["rsc_size_max"])
    rctl_arr, rctl_ptr       = _f64(a["rsc_tl"])
    rcar_arr, rcar_ptr       = _i32(a["rsc_access_rows"])
    tlws_arr, tlws_ptr       = _f64(a["tl_weighted_sum"])
    # diet_matrix already prepared; need mutable pointer
    dm_ptr = ffi.cast("double *", dm.ctypes.data)
    ptb_arr, ptb_ptr         = _i32(a["prey_type_buf"])
    pib_arr, pib_ptr         = _i32(a["prey_id_buf"])
    peb_arr, peb_ptr         = _f64(a["prey_eligible_buf"])
    er_arr, er_ptr           = _f64(a["egg_retained"])

    lib.apply_predation_once(
        p_idx, ci_ptr, n_local,
        ia_ptr, nd_ptr,
        si_ptr, le_ptr, we_ptr,
        ag_ptr, ffa_ptr, fs_ptr,
        psr_ptr, pb_ptr, tl_ptr,
        srm_ptr, srmx_ptr,
        ir_ptr, frs_ptr, frh_ptr,
        n_dt_py, n_subdt_py,
        am_ptr,
        has_access, use_stage_ac,
        pai_ptr, pdi_ptr,
        rsb_ptr, rsmin_ptr,
        rsmax_ptr, rctl_ptr, rcar_ptr,
        n_resources, n_species, cell_id,
        tlws_ptr, tl_tracking,
        dm_ptr, diet_enabled,
        ptb_ptr, pib_ptr, peb_ptr,
        er_ptr,
        # 7 aux shape ints
        srm_ncol, acc_nrow, acc_ncol,
        n_cells, n_causes, diet_nrow, diet_ncol,
    )

    # Write back the (possibly modified) contiguous arrays into the original arg arrays.
    # Only the MUTATED arrays need writing back; the others are read-only.
    # inst_abd, pred_success_rate, preyed_biomass, tl_weighted_sum are 1D float64.
    # n_dead is 2D float64.
    # rsc_biomass is 2D float64.
    # diet_matrix is 2D float64.
    #
    # Because we called np.ascontiguousarray on the arrays from `a` (which are fresh
    # copies from build_leaf_args), the returned contiguous arrays *are* the same
    # objects as in `args` ONLY IF they were already C-contiguous.  To be safe we
    # copy back into the original array objects.
    np.copyto(a["inst_abd"], ia_arr)
    np.copyto(a["n_dead"], nd)
    np.copyto(a["pred_success_rate"], psr_arr)
    np.copyto(a["preyed_biomass"], pb_arr)
    np.copyto(a["rsc_biomass"], rsb)
    np.copyto(a["tl_weighted_sum"], tlws_arr)
    np.copyto(a["diet_matrix"], dm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parity_for_cell(arrays: dict, meta: dict, cell: int) -> dict[str, float]:
    """Run Numba and C kernels on independent arg sets for `cell`.

    Returns {array_name: max_rel_diff} for the 7 MUTATED arrays.
    """
    _numba_fn = _get_numba_leaf()
    _mod = _get_c_module()
    ffi, lib = _mod.ffi, _mod.lib

    # Two independent fresh arg sets (both call build_leaf_args independently).
    numba_args, _p1 = build_leaf_args(arrays, meta, cell)
    c_args, _p2     = build_leaf_args(arrays, meta, cell)

    # --- Run Numba oracle ---
    _numba_fn(*numba_args)

    # --- Run C kernel (mutates c_args in-place via _call_c_once) ---
    _call_c_once(c_args, ffi, lib)

    # --- Compare MUTATED arrays ---
    numba_dict = {name: numba_args[i] for i, name in enumerate(LEAF_ARG_ORDER)}
    c_dict     = {name: c_args[i]     for i, name in enumerate(LEAF_ARG_ORDER)}

    report: dict[str, float] = {}
    for name in MUTATED:
        a_nb = np.asarray(numba_dict[name], dtype=np.float64).ravel()
        a_c  = np.asarray(c_dict[name],     dtype=np.float64).ravel()

        # NaN mask must match exactly. A divergence in WHICH positions are NaN
        # is itself a mismatch (e.g. C produced a finite value where Numba kept
        # NaN, or vice versa). Without this guard, NaN anywhere in the array
        # would poison np.max -> NaN, and `NaN > bar` is False -> a REAL finite
        # mismatch could slip through as a silent false pass. This gate is what
        # the whole spike's verdict rests on, so fail loudly on mask divergence.
        nb_nan = np.isnan(a_nb)
        c_nan  = np.isnan(a_c)
        if not np.array_equal(nb_nan, c_nan):
            raise AssertionError(
                f"NaN mask mismatch in '{name}': Numba and C disagree on which "
                f"positions are NaN (nb_nan_count={int(nb_nan.sum())}, "
                f"c_nan_count={int(c_nan.sum())}) — this is a real divergence"
            )

        # Compute max rel diff over FINITE positions only. An all-NaN array is
        # a definitional match (masks already proven equal above) -> diff 0.0.
        finite = ~nb_nan
        if not finite.any():
            report[name] = 0.0
            continue
        rel_diff = np.max(
            np.abs(a_nb[finite] - a_c[finite]) / (np.abs(a_c[finite]) + 1e-300)
        )
        report[name] = float(rel_diff)

    return report


def assert_parity(report: dict[str, float], bar: float = 1e-12) -> None:
    """Raise AssertionError if any array's max_rel_diff exceeds bar."""
    violations = {k: v for k, v in report.items() if v > bar}
    if violations:
        lines = [f"  {k}: {v:.3e} > {bar:.0e}" for k, v in sorted(violations.items())]
        raise AssertionError(
            f"C kernel parity violation (bar={bar:.0e}):\n" + "\n".join(lines)
        )
