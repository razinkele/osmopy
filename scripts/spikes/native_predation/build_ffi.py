"""Compile kernel.c into two cffi modules: portable (-O3) and native (-march=native)."""
from __future__ import annotations

from pathlib import Path

from cffi import FFI

HERE = Path(__file__).resolve().parent

# CDEF must exactly match the three public wrapper signatures in kernel.c.
CDEF = """
void apply_predation_once(
    int p_idx, const int* cell_indices, int n_local,
    double* inst_abd, double* n_dead,
    const int* species_id, const double* length, const double* weight,
    const int* age_dt, const int* first_feeding_age_dt, const int* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const int* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const int* prey_access_idx, const int* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const int* rsc_access_rows,
    int n_resources, int n_species, int cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    int* prey_type_buf, int* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol);

void apply_predation_bench(
    int p_idx, const int* cell_indices, int n_local,
    double* inst_abd, double* n_dead,
    const int* species_id, const double* length, const double* weight,
    const int* age_dt, const int* first_feeding_age_dt, const int* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const int* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const int* prey_access_idx, const int* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const int* rsc_access_rows,
    int n_resources, int n_species, int cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    int* prey_type_buf, int* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol,
    int n_iter, int n_schools,
    const double* snap_inst_abd,
    const double* snap_n_dead,
    const double* snap_pred_success_rate,
    const double* snap_preyed_biomass,
    const double* snap_rsc_biomass,
    const double* snap_tl_weighted_sum,
    const double* snap_diet_matrix);

void noop(
    int p_idx, const int* cell_indices, int n_local,
    double* inst_abd, double* n_dead,
    const int* species_id, const double* length, const double* weight,
    const int* age_dt, const int* first_feeding_age_dt, const int* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const int* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const int* prey_access_idx, const int* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const int* rsc_access_rows,
    int n_resources, int n_species, int cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    int* prey_type_buf, int* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol);
"""


def build(variant: str) -> str:
    ffi = FFI()
    ffi.cdef(CDEF)
    flags = ["-O3"] if variant == "portable" else ["-O3", "-march=native"]
    # Use a NON-dotted module name: cffi treats dots in the module name as
    # subdirectory components under tmpdir, which would nest the .so under
    # scripts/spikes/native_predation/scripts/spikes/native_predation/.
    # HERE is already the package dir, so a bare name lands the .so directly
    # there, importable as scripts.spikes.native_predation._leaf_<variant>.
    ffi.set_source(
        f"_leaf_{variant}",
        (HERE / "kernel.c").read_text(),
        extra_compile_args=flags,
    )
    return ffi.compile(tmpdir=str(HERE))


if __name__ == "__main__":
    for v in ("portable", "native"):
        print(v, "->", build(v))
