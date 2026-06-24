"""Boundary-free benchmark: C vs Numba leaf-only throughput (Task 6).

Methodology (spec §4.2):
  leaf_time_per_call = (T_full(n_iter) - T_reset_only(n_iter)) / n_iter

Both sides use IDENTICAL reset subtraction to cancel memcpy/array-copy cost.
n_samples A/B samples are collected INTERLEAVED (alternate Numba/C per sample)
to cancel machine-state drift.

ratio = numba_med / c_med   (ratio > 1 => C is faster)
"""
from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from scripts.spikes.native_predation.leaf_args import (
    LEAF_ARG_ORDER,
    MUTATED,
    build_leaf_args,
    select_cells,
)
from scripts.spikes.native_predation.numba_driver import make_driver
from scripts.spikes.native_predation.parity import _call_c_once

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_c_lib(variant: str) -> tuple[Any, Any]:
    """Return (ffi, lib) for the chosen variant ('portable' or 'native')."""
    mod = importlib.import_module(f"scripts.spikes.native_predation._leaf_{variant}")
    return mod.ffi, mod.lib


def _build_snapshots(args: list) -> tuple:
    """Return (snap_inst_abd, snap_n_dead, snap_pred_success_rate,
               snap_preyed_biomass, snap_rsc_biomass, snap_tl_weighted_sum,
               snap_diet_matrix) as fresh copies.

    MUTATED order: ['inst_abd', 'n_dead', 'pred_success_rate', 'preyed_biomass',
                    'rsc_biomass', 'tl_weighted_sum', 'diet_matrix']
    Indices in LEAF_ARG_ORDER: 2, 3, 10, 11, 25, 33, 35.
    """
    snap_inst_abd          = np.copy(args[2])    # inst_abd
    snap_n_dead            = np.copy(args[3])    # n_dead (2D)
    snap_pred_success_rate = np.copy(args[10])   # pred_success_rate
    snap_preyed_biomass    = np.copy(args[11])   # preyed_biomass
    snap_rsc_biomass       = np.copy(args[25])   # rsc_biomass (2D)
    snap_tl_weighted_sum   = np.copy(args[33])   # tl_weighted_sum
    snap_diet_matrix       = np.copy(args[35])   # diet_matrix (2D)
    return (snap_inst_abd, snap_n_dead, snap_pred_success_rate,
            snap_preyed_biomass, snap_rsc_biomass, snap_tl_weighted_sum,
            snap_diet_matrix)


def _aux_ints(args: list) -> dict:
    """Compute the 7 aux shape ints from the arg list (same logic as parity.py)."""
    a = {name: args[i] for i, name in enumerate(LEAF_ARG_ORDER)}
    srm    = np.ascontiguousarray(a["size_ratio_min"], dtype=np.float64)
    acc    = np.ascontiguousarray(a["access_matrix"], dtype=np.float64)
    rsb    = np.ascontiguousarray(a["rsc_biomass"], dtype=np.float64)
    nd     = np.ascontiguousarray(a["n_dead"], dtype=np.float64)
    dm     = np.ascontiguousarray(a["diet_matrix"], dtype=np.float64)
    return {
        "srm_ncol":  srm.shape[1],
        "acc_nrow":  acc.shape[0],
        "acc_ncol":  acc.shape[1],
        "n_cells":   rsb.shape[1],
        "n_causes":  nd.shape[1],
        "diet_nrow": dm.shape[0],
        "diet_ncol": dm.shape[1],
    }


def _call_c_bench(args: list, ffi: Any, lib: Any,
                  snaps: tuple, n_iter: int, reset_only: bool) -> None:
    """Call apply_predation_bench or reset_only_bench via cffi.

    Mirrors the arg marshalling from parity._call_c_once, extended with the
    bench-only args (n_iter, n_schools, 7 pristine snapshot pointers).
    """
    a: dict[str, Any] = {name: args[i] for i, name in enumerate(LEAF_ARG_ORDER)}

    def _f64(x):
        arr = np.ascontiguousarray(x, dtype=np.float64)
        return arr, ffi.cast("double *", arr.ctypes.data)

    def _i32(x):
        arr = np.ascontiguousarray(x, dtype=np.int32)
        return arr, ffi.cast("int *", arr.ctypes.data)

    # scalars
    p_idx        = int(a["p_idx"])
    n_local      = len(a["cell_indices"])
    n_dt_py      = float(a["n_dt_per_year"])
    n_subdt_py   = float(a["n_subdt"])
    has_access   = int(bool(a["has_access"]))
    use_stage_ac = int(bool(a["use_stage_access"]))
    n_resources  = int(a["n_resources"])
    n_species    = int(a["n_species"])
    cell_id      = int(a["cell_id"])
    tl_tracking  = int(bool(a["tl_tracking"]))
    diet_enabled = int(bool(a["diet_enabled"]))

    # aux shape ints
    srm          = np.ascontiguousarray(a["size_ratio_min"], dtype=np.float64)
    acc_arr      = np.ascontiguousarray(a["access_matrix"], dtype=np.float64)
    rsb          = np.ascontiguousarray(a["rsc_biomass"], dtype=np.float64)
    nd           = np.ascontiguousarray(a["n_dead"], dtype=np.float64)
    dm           = np.ascontiguousarray(a["diet_matrix"], dtype=np.float64)
    srm_ncol     = srm.shape[1]
    acc_nrow, acc_ncol = acc_arr.shape
    n_cells      = rsb.shape[1]
    n_causes     = nd.shape[1]
    diet_nrow, diet_ncol = dm.shape

    n_schools = int(a["inst_abd"].shape[0])

    # array pointers — hold refs to prevent GC
    ci_arr, ci_ptr         = _i32(a["cell_indices"])
    ia_arr, ia_ptr         = _f64(a["inst_abd"])
    nd_ptr                 = ffi.cast("double *", nd.ctypes.data)
    si_arr, si_ptr         = _i32(a["species_id"])
    le_arr, le_ptr         = _f64(a["length"])
    we_arr, we_ptr         = _f64(a["weight"])
    ag_arr, ag_ptr         = _i32(a["age_dt"])
    ffa_arr, ffa_ptr       = _i32(a["first_feeding_age_dt"])
    fs_arr, fs_ptr         = _i32(a["feeding_stage"])
    psr_arr, psr_ptr       = _f64(a["pred_success_rate"])
    pb_arr, pb_ptr         = _f64(a["preyed_biomass"])
    tl_arr, tl_ptr         = _f64(a["trophic_level"])
    srm_ptr                = ffi.cast("double *", srm.ctypes.data)
    srmx_arr               = np.ascontiguousarray(a["size_ratio_max"], dtype=np.float64)
    srmx_ptr               = ffi.cast("double *", srmx_arr.ctypes.data)
    ir_arr, ir_ptr         = _f64(a["ingestion_rate"])
    frs_arr, frs_ptr       = _i32(a["fr_shape"])
    frh_arr, frh_ptr       = _f64(a["fr_halfsat"])
    am_ptr                 = ffi.cast("double *", acc_arr.ctypes.data)
    pai_arr, pai_ptr       = _i32(a["prey_access_idx"])
    pdi_arr, pdi_ptr       = _i32(a["pred_access_idx"])
    rsb_ptr                = ffi.cast("double *", rsb.ctypes.data)
    rsmin_arr, rsmin_ptr   = _f64(a["rsc_size_min"])
    rsmax_arr, rsmax_ptr   = _f64(a["rsc_size_max"])
    rctl_arr, rctl_ptr     = _f64(a["rsc_tl"])
    rcar_arr, rcar_ptr     = _i32(a["rsc_access_rows"])
    tlws_arr, tlws_ptr     = _f64(a["tl_weighted_sum"])
    dm_ptr                 = ffi.cast("double *", dm.ctypes.data)
    ptb_arr, ptb_ptr       = _i32(a["prey_type_buf"])
    pib_arr, pib_ptr       = _i32(a["prey_id_buf"])
    peb_arr, peb_ptr       = _f64(a["prey_eligible_buf"])
    er_arr, er_ptr         = _f64(a["egg_retained"])

    # pristine snapshot pointers
    (snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm) = snaps
    snap_ia_c    = np.ascontiguousarray(snap_ia,  dtype=np.float64)
    snap_nd_c    = np.ascontiguousarray(snap_nd,  dtype=np.float64)
    snap_psr_c   = np.ascontiguousarray(snap_psr, dtype=np.float64)
    snap_pb_c    = np.ascontiguousarray(snap_pb,  dtype=np.float64)
    snap_rsb_c   = np.ascontiguousarray(snap_rsb, dtype=np.float64)
    snap_tlws_c  = np.ascontiguousarray(snap_tlws, dtype=np.float64)
    snap_dm_c    = np.ascontiguousarray(snap_dm,  dtype=np.float64)

    snap_ia_ptr   = ffi.cast("double *", snap_ia_c.ctypes.data)
    snap_nd_ptr   = ffi.cast("double *", snap_nd_c.ctypes.data)
    snap_psr_ptr  = ffi.cast("double *", snap_psr_c.ctypes.data)
    snap_pb_ptr   = ffi.cast("double *", snap_pb_c.ctypes.data)
    snap_rsb_ptr  = ffi.cast("double *", snap_rsb_c.ctypes.data)
    snap_tlws_ptr = ffi.cast("double *", snap_tlws_c.ctypes.data)
    snap_dm_ptr   = ffi.cast("double *", snap_dm_c.ctypes.data)

    common_args = (
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
        rsb_ptr, rsmin_ptr, rsmax_ptr, rctl_ptr, rcar_ptr,
        n_resources, n_species, cell_id,
        tlws_ptr, tl_tracking,
        dm_ptr, diet_enabled,
        ptb_ptr, pib_ptr, peb_ptr,
        er_ptr,
        # 7 aux shape ints
        srm_ncol, acc_nrow, acc_ncol,
        n_cells, n_causes, diet_nrow, diet_ncol,
        # bench-only
        n_iter, n_schools,
        snap_ia_ptr, snap_nd_ptr, snap_psr_ptr, snap_pb_ptr,
        snap_rsb_ptr, snap_tlws_ptr, snap_dm_ptr,
    )

    fn = lib.reset_only_bench if reset_only else lib.apply_predation_bench
    fn(*common_args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bench_cell(
    arrays: dict,
    meta: dict,
    cell: int,
    variant: str = "portable",
    n_iter: int = 100_000,
    n_samples: int = 30,
) -> dict:
    """Benchmark Numba vs C for a single cell.

    Returns:
        numba_med  : median leaf-only per-call time (ns)
        numba_iqr  : IQR of leaf-only per-call times (ns)
        c_med      : median leaf-only per-call time (ns)
        c_iqr      : IQR of leaf-only per-call times (ns)
        ratio      : numba_med / c_med  (>1 => C faster)
        n_local    : number of schools in this cell

    NOTE on measurement fidelity: The reset-subtraction approach requires the
    per-iteration leaf time to be a significant fraction of the per-iteration
    total time.  With the captured fixture arrays (~384 KB of MUTATED data), the
    C memcpy reset dominates (~10 µs/iter) while the C leaf itself is ~100-300 ns.
    The Numba njit `arr[:]=snap` reset is slower (~22 µs/iter) but the Numba leaf
    is ~1500-2000 ns, so the Numba side has a higher signal-to-noise ratio.
    Use n_iter >= 100_000 and n_samples >= 30 for stable medians on C.
    """
    ffi, lib = _get_c_lib(variant)
    drv, rst = make_driver()

    args, _p_idx = build_leaf_args(arrays, meta, cell)
    snaps = _build_snapshots(args)

    n_local = len(args[1])  # cell_indices length = n_local schools

    # Unpack snapshots for the Numba driver call signature
    (snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm) = snaps

    def _call_numba():
        drv(
            *args,
            n_iter,
            snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm,
        )

    def _call_numba_rst():
        rst(
            *args,
            n_iter,
            snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm,
        )

    def _call_c():
        _call_c_bench(args, ffi, lib, snaps, n_iter, reset_only=False)

    def _call_c_rst():
        _call_c_bench(args, ffi, lib, snaps, n_iter, reset_only=True)

    # --- Warm both @njit functions (compile + cache-warm) ---
    _call_numba()
    _call_numba_rst()
    # Warm C paths (icache warm)
    _call_c()
    _call_c_rst()

    # --- Interleaved A/B timing ---
    numba_full_ns = []
    numba_rst_ns  = []
    c_full_ns     = []
    c_rst_ns      = []

    for _ in range(n_samples):
        t0 = time.perf_counter_ns(); _call_numba();     t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns(); _call_c();         t3 = time.perf_counter_ns()
        t4 = time.perf_counter_ns(); _call_numba_rst(); t5 = time.perf_counter_ns()
        t6 = time.perf_counter_ns(); _call_c_rst();     t7 = time.perf_counter_ns()

        numba_full_ns.append(t1 - t0)
        c_full_ns.append(t3 - t2)
        numba_rst_ns.append(t5 - t4)
        c_rst_ns.append(t7 - t6)

    # --- Compute leaf-only per-call times ---
    nf = np.array(numba_full_ns, dtype=np.float64)
    nr = np.array(numba_rst_ns,  dtype=np.float64)
    cf = np.array(c_full_ns,     dtype=np.float64)
    cr = np.array(c_rst_ns,      dtype=np.float64)

    numba_leaf = (nf - nr) / n_iter
    c_leaf     = (cf - cr) / n_iter

    def _med_iqr(arr: np.ndarray) -> tuple[float, float]:
        med = float(np.median(arr))
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        return med, q75 - q25

    numba_med, numba_iqr = _med_iqr(numba_leaf)
    c_med,     c_iqr     = _med_iqr(c_leaf)
    ratio = numba_med / c_med if c_med != 0 else float("inf")

    return {
        "numba_med": numba_med,
        "numba_iqr": numba_iqr,
        "c_med":     c_med,
        "c_iqr":     c_iqr,
        "ratio":     ratio,
        "n_local":   n_local,
    }


def boundary_probe(
    arrays: dict,
    meta: dict,
    cell: int,
    variant: str = "portable",
    n_samples: int = 200,
) -> dict:
    """Measure per-call overhead for the cffi noop and an empty Numba leaf call.

    Uses the 'small' cell (fewest schools) to minimise work if the leaf executes.
    Returns:
        noop_med_ns     : median cffi noop call time (ns)
        noop_iqr_ns     : IQR (ns)
        numba_empty_med_ns : median for a Numba @njit call that early-returns
        numba_empty_iqr_ns : IQR (ns)
    """
    ffi, lib = _get_c_lib(variant)
    drv, _rst = make_driver()

    args, _p_idx = build_leaf_args(arrays, meta, cell)
    snaps = _build_snapshots(args)
    (snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm) = snaps

    # Build a modified arg list that forces early return in the Numba leaf:
    # set inst_abd[p_idx] = 0 so the leaf exits immediately after `if abd_p <= 0`.
    args_early = list(args)
    ia_copy = np.copy(args[2])   # inst_abd
    p_idx   = int(args[0])
    ia_copy[p_idx] = 0.0
    args_early[2] = ia_copy

    # Warm
    drv(*args_early, 1, snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm)

    # Noop warm
    from scripts.spikes.native_predation.parity import _call_c_once  # noqa: PLC0415
    _call_c_once(list(args), ffi, lib)  # just to init icache

    noop_times  = []
    numba_times = []

    # --- One-time marshalling setup (hoisted above the timed sample loop) ---
    # The noop call only reads pointers; the per-call cost we want is the cffi
    # ABI boundary, so all array prep / casts happen exactly once here.
    a: dict = {name: args[i] for i, name in enumerate(LEAF_ARG_ORDER)}

    def _f64(x):
        arr = np.ascontiguousarray(x, dtype=np.float64)
        return arr, ffi.cast("double *", arr.ctypes.data)

    def _i32(x):
        arr = np.ascontiguousarray(x, dtype=np.int32)
        return arr, ffi.cast("int *", arr.ctypes.data)

    srm      = np.ascontiguousarray(a["size_ratio_min"], dtype=np.float64)
    acc_arr  = np.ascontiguousarray(a["access_matrix"], dtype=np.float64)
    rsb      = np.ascontiguousarray(a["rsc_biomass"], dtype=np.float64)
    nd_arr   = np.ascontiguousarray(a["n_dead"], dtype=np.float64)
    dm_arr   = np.ascontiguousarray(a["diet_matrix"], dtype=np.float64)

    ci_a, ci_p   = _i32(a["cell_indices"])
    ia_a, ia_p   = _f64(a["inst_abd"])
    si_a, si_p   = _i32(a["species_id"])
    le_a, le_p   = _f64(a["length"])
    we_a, we_p   = _f64(a["weight"])
    ag_a, ag_p   = _i32(a["age_dt"])
    ff_a, ff_p   = _i32(a["first_feeding_age_dt"])
    fs_a, fs_p   = _i32(a["feeding_stage"])
    ps_a, ps_p   = _f64(a["pred_success_rate"])
    pb_a, pb_p   = _f64(a["preyed_biomass"])
    tl_a, tl_p   = _f64(a["trophic_level"])
    sx_a         = np.ascontiguousarray(a["size_ratio_max"], dtype=np.float64)
    ir_a, ir_p   = _f64(a["ingestion_rate"])
    fr_a, fr_p   = _i32(a["fr_shape"])
    fh_a, fh_p   = _f64(a["fr_halfsat"])
    pa_a, pa_p   = _i32(a["prey_access_idx"])
    pd_a, pd_p   = _i32(a["pred_access_idx"])
    rm_a, rm_p   = _f64(a["rsc_size_min"])
    rx_a, rx_p   = _f64(a["rsc_size_max"])
    rt_a, rt_p   = _f64(a["rsc_tl"])
    ra_a, ra_p   = _i32(a["rsc_access_rows"])
    tw_a, tw_p   = _f64(a["tl_weighted_sum"])
    pt_a, pt_p   = _i32(a["prey_type_buf"])
    pi_a, pi_p   = _i32(a["prey_id_buf"])
    pe_a, pe_p   = _f64(a["prey_eligible_buf"])
    er_a, er_p   = _f64(a["egg_retained"])

    # Mutable-array pointers (used with explicit casts in the original call site).
    nd_p  = ffi.cast("double *", nd_arr.ctypes.data)
    srm_p = ffi.cast("double *", srm.ctypes.data)
    sx_p  = ffi.cast("double *", sx_a.ctypes.data)
    acc_p = ffi.cast("double *", acc_arr.ctypes.data)
    rsb_p = ffi.cast("double *", rsb.ctypes.data)
    dm_p  = ffi.cast("double *", dm_arr.ctypes.data)

    srm_ncol          = srm.shape[1]
    acc_nrow, acc_ncol = acc_arr.shape
    n_cells           = rsb.shape[1]
    n_causes          = nd_arr.shape[1]
    diet_nrow, diet_ncol = dm_arr.shape

    p_idx_i      = int(a["p_idx"])
    n_local      = len(a["cell_indices"])
    n_dt_py      = float(a["n_dt_per_year"])
    n_subdt_py   = float(a["n_subdt"])
    has_access   = int(bool(a["has_access"]))
    use_stage_ac = int(bool(a["use_stage_access"]))
    n_resources  = int(a["n_resources"])
    n_species    = int(a["n_species"])
    cell_id      = int(a["cell_id"])
    tl_tracking  = int(bool(a["tl_tracking"]))
    diet_enabled = int(bool(a["diet_enabled"]))

    for _ in range(n_samples):
        # cffi noop — only the boundary crossing is timed.
        t0 = time.perf_counter_ns()
        lib.noop(
            p_idx_i, ci_p, n_local,
            ia_p, nd_p,
            si_p, le_p, we_p,
            ag_p, ff_p, fs_p,
            ps_p, pb_p, tl_p,
            srm_p, sx_p,
            ir_p, fr_p, fh_p,
            n_dt_py, n_subdt_py,
            acc_p,
            has_access, use_stage_ac,
            pa_p, pd_p,
            rsb_p,
            rm_p, rx_p, rt_p, ra_p,
            n_resources, n_species, cell_id,
            tw_p, tl_tracking,
            dm_p, diet_enabled,
            pt_p, pi_p, pe_p,
            er_p,
            srm_ncol, acc_nrow, acc_ncol,
            n_cells, n_causes, diet_nrow, diet_ncol,
        )
        t1 = time.perf_counter_ns()
        noop_times.append(t1 - t0)

        # Numba early-exit leaf
        t2 = time.perf_counter_ns()
        drv(*args_early, 1, snap_ia, snap_nd, snap_psr, snap_pb, snap_rsb, snap_tlws, snap_dm)
        t3 = time.perf_counter_ns()
        numba_times.append(t3 - t2)

    nt = np.array(noop_times,  dtype=np.float64)
    bt = np.array(numba_times, dtype=np.float64)

    def _med_iqr(arr):
        med = float(np.median(arr))
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        return med, q75 - q25

    noop_med, noop_iqr   = _med_iqr(nt)
    nb_med,   nb_iqr     = _med_iqr(bt)

    return {
        "noop_med_ns":         noop_med,
        "noop_iqr_ns":         noop_iqr,
        "numba_empty_med_ns":  nb_med,
        "numba_empty_iqr_ns":  nb_iqr,
    }


def run_all(
    arrays: dict,
    meta: dict,
    sel: dict,
    variant: str = "portable",
    n_iter: int = 100_000,
    n_samples: int = 30,
) -> dict:
    """Run bench_cell for all cells in sel (p10/p50/p95/small).

    Also runs boundary_probe on the 'small' cell.
    Returns a dict with keys: 'p10', 'p50', 'p95', 'small', 'weighted_ratio',
    'boundary', and per-cell bench dicts.
    """
    results: dict = {}
    total_weight = 0.0
    weighted_ratio_sum = 0.0

    for label, cell in sel.items():
        r = bench_cell(arrays, meta, cell, variant=variant,
                       n_iter=n_iter, n_samples=n_samples)
        results[label] = r
        w = float(r["n_local"])
        total_weight     += w
        weighted_ratio_sum += r["ratio"] * w

    results["weighted_ratio"] = (
        weighted_ratio_sum / total_weight if total_weight > 0 else float("nan")
    )

    results["boundary"] = boundary_probe(
        arrays, meta, sel["small"], variant=variant
    )

    return results
