/* scripts/spikes/native_predation/kernel.c
 * Faithful C transcription of _apply_predation_numba (mortality.py:881-1053).
 * Reduction order MUST match the Python source or parity (<=1e-12) fails.
 */
#include <stddef.h>
#include <string.h>

typedef int i32;

/* Core single-predator predation. All 2D arrays are flat row-major. */
static void leaf(
    i32 p_idx, const i32* cell_indices, i32 n_local,
    double* inst_abd, double* n_dead, int n_causes,
    const i32* species_id, const double* length, const double* weight,
    const i32* age_dt, const i32* first_feeding_age_dt, const i32* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max, int srm_ncol,
    const double* ingestion_rate, const i32* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix, i32 acc_nrow, i32 acc_ncol,
    int has_access, int use_stage_access,
    const i32* prey_access_idx, const i32* pred_access_idx,
    double* rsc_biomass, i32 n_cells, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const i32* rsc_access_rows,
    i32 n_resources, i32 n_species, i32 cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, i32 diet_nrow, i32 diet_ncol, int diet_enabled,
    i32* prey_type_buf, i32* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained)
{
    if (age_dt[p_idx] < first_feeding_age_dt[p_idx]) return;
    double abd_p = inst_abd[p_idx];
    if (abd_p <= 0) return;

    i32 sp_pred = species_id[p_idx];
    double pred_len = length[p_idx];
    i32 stage = feeding_stage[p_idx];
    double r_min = size_ratio_min[sp_pred * srm_ncol + stage];
    double r_max = size_ratio_max[sp_pred * srm_ncol + stage];

    double biomass_p = abd_p * weight[p_idx];
    double max_eatable = biomass_p * ingestion_rate[sp_pred] / (n_dt_per_year * n_subdt);
    if (max_eatable <= 0) return;

    double total_available = 0.0;
    i32 n_prey = 0;

    /* 1a: school prey (cell_indices order) */
    for (i32 q_pos = 0; q_pos < n_local; q_pos++) {
        i32 q_idx = cell_indices[q_pos];
        if (q_idx == p_idx) continue;
        double abd_q = inst_abd[q_idx] - egg_retained[q_idx];
        if (abd_q < 0.0) abd_q = 0.0;
        if (abd_q <= 0) continue;
        double prey_len = length[q_idx];
        if (prey_len <= 0) continue;
        double ratio = pred_len / prey_len;
        if (ratio < r_min || ratio >= r_max) continue;

        double access_coeff = 1.0;
        if (has_access) {
            if (use_stage_access) {
                i32 p_acc = pred_access_idx[p_idx], q_acc = prey_access_idx[q_idx];
                if (p_acc >= 0 && q_acc >= 0) {
                    if (q_acc < acc_nrow && p_acc < acc_ncol)
                        access_coeff = access_matrix[q_acc * acc_ncol + p_acc];
                    if (access_coeff <= 0) continue;
                }
            } else {
                i32 sp_prey = species_id[q_idx];
                if (sp_pred < acc_nrow && sp_prey < acc_ncol) {
                    access_coeff = access_matrix[sp_pred * acc_ncol + sp_prey];
                    if (access_coeff <= 0) continue;
                }
            }
        }
        double prey_bio = abd_q * weight[q_idx];
        if (prey_bio <= 0) continue;
        double eligible = prey_bio * access_coeff;
        prey_type_buf[n_prey] = 0;
        prey_id_buf[n_prey] = q_idx;
        prey_eligible_buf[n_prey] = eligible;
        total_available += eligible;
        n_prey++;
    }

    /* 1b: resource prey (r order) */
    for (i32 r = 0; r < n_resources; r++) {
        double rsc_bio = rsc_biomass[r * n_cells + cell_id];
        if (rsc_bio <= 0) continue;
        if (r_min <= 0 || r_max <= 0) continue;
        double prey_size_min = pred_len / r_max;
        double prey_size_max = pred_len / r_min;
        double rsmn = rsc_size_min[r], rsmx = rsc_size_max[r];
        double overlap_min = rsmn > prey_size_min ? rsmn : prey_size_min;
        double overlap_max = rsmx < prey_size_max ? rsmx : prey_size_max;
        if (overlap_max <= overlap_min) continue;
        double rsc_range = rsmx - rsmn;
        if (rsc_range <= 0) continue;
        double percent_resource = (overlap_max - overlap_min) / rsc_range;

        double access_coeff = 1.0;
        if (use_stage_access) {
            i32 rsc_row = rsc_access_rows[r], p_acc = pred_access_idx[p_idx];
            if (rsc_row >= 0 && p_acc >= 0) {
                if (rsc_row < acc_nrow && p_acc < acc_ncol) {
                    access_coeff = access_matrix[rsc_row * acc_ncol + p_acc];
                    if (access_coeff <= 0) continue;
                }
            }
        } else if (has_access) {
            i32 rsc_sp_idx = n_species + r;
            if (sp_pred < acc_nrow && rsc_sp_idx < acc_ncol) {
                access_coeff = access_matrix[sp_pred * acc_ncol + rsc_sp_idx];
                if (access_coeff <= 0) continue;
            }
        }
        double eligible_bio = rsc_bio * percent_resource * access_coeff;
        prey_type_buf[n_prey] = 1;
        prey_id_buf[n_prey] = r;
        prey_eligible_buf[n_prey] = eligible_bio;
        total_available += eligible_bio;
        n_prey++;
    }

    if (total_available <= 0) return;

    /* Phase 2: functional response */
    double eaten_total;
    if (fr_shape[sp_pred] == 1) {
        eaten_total = total_available < max_eatable ? total_available : max_eatable;
    } else {
        double rr = total_available / max_eatable;
        double k_fr = fr_halfsat[sp_pred];
        double g_form;
        if (fr_shape[sp_pred] == 2) g_form = rr / (rr + k_fr);
        else g_form = (rr * rr) / (rr * rr + k_fr * k_fr);
        double cap = rr < 1.0 ? rr : 1.0;
        double g = g_form < cap ? g_form : cap;
        eaten_total = max_eatable * g;
    }

    for (i32 k = 0; k < n_prey; k++) {
        double share = prey_eligible_buf[k] / total_available;
        double eaten_from_prey = eaten_total * share;
        if (prey_type_buf[k] == 0) {
            i32 q_idx = prey_id_buf[k];
            if (weight[q_idx] > 0) {
                double n_dead_prey = eaten_from_prey / weight[q_idx];
                n_dead[q_idx * n_causes + 0] += n_dead_prey;
                inst_abd[q_idx] -= n_dead_prey;
            }
            if (tl_tracking) {
                double prey_tl = trophic_level[q_idx];
                if (prey_tl <= 0) prey_tl = 1.0;
                tl_weighted_sum[p_idx] += prey_tl * eaten_from_prey;
            }
            if (diet_enabled) {
                i32 prey_sp = species_id[q_idx];
                if (p_idx < diet_nrow && prey_sp < diet_ncol)  /* rows=preds, cols=prey+rsc */
                    diet_matrix[p_idx * diet_ncol + prey_sp] += eaten_from_prey;
            }
        } else {
            i32 r_idx = prey_id_buf[k];
            double cur = rsc_biomass[r_idx * n_cells + cell_id] - eaten_from_prey;
            rsc_biomass[r_idx * n_cells + cell_id] = cur > 0.0 ? cur : 0.0;
            if (tl_tracking) {
                double r_tl = rsc_tl[r_idx];
                if (r_tl <= 0) r_tl = 1.0;
                tl_weighted_sum[p_idx] += r_tl * eaten_from_prey;
            }
            if (diet_enabled) {
                i32 rsc_col = n_species + r_idx;
                if (p_idx < diet_nrow && rsc_col < diet_ncol)
                    diet_matrix[p_idx * diet_ncol + rsc_col] += eaten_from_prey;
            }
        }
    }

    double success = eaten_total / max_eatable;
    if (success > 1.0) success = 1.0;
    pred_success_rate[p_idx] += success / n_subdt;
    preyed_biomass[p_idx] += eaten_total;
}

/* ---- Public wrappers ---- */

/*
 * apply_predation_once: call leaf exactly once with the provided args.
 * Used by Task 5 parity harness.
 *
 * Data args (41): same as leaf, with the 7 aux shape ints folded in.
 * n_dt_per_year and n_subdt are double per ABI contract.
 */
void apply_predation_once(
    i32 p_idx, const i32* cell_indices, i32 n_local,
    double* inst_abd, double* n_dead,
    const i32* species_id, const double* length, const double* weight,
    const i32* age_dt, const i32* first_feeding_age_dt, const i32* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const i32* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const i32* prey_access_idx, const i32* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const i32* rsc_access_rows,
    i32 n_resources, i32 n_species, i32 cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    i32* prey_type_buf, i32* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    /* 7 auxiliary shape args */
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol)
{
    leaf(
        p_idx, cell_indices, n_local,
        inst_abd, n_dead, n_causes,
        species_id, length, weight,
        age_dt, first_feeding_age_dt, feeding_stage,
        pred_success_rate, preyed_biomass, trophic_level,
        size_ratio_min, size_ratio_max, srm_ncol,
        ingestion_rate, fr_shape, fr_halfsat,
        n_dt_per_year, n_subdt,
        access_matrix, acc_nrow, acc_ncol,
        has_access, use_stage_access,
        prey_access_idx, pred_access_idx,
        rsc_biomass, n_cells, rsc_size_min,
        rsc_size_max, rsc_tl, rsc_access_rows,
        n_resources, n_species, cell_id,
        tl_weighted_sum, tl_tracking,
        diet_matrix, diet_nrow, diet_ncol, diet_enabled,
        prey_type_buf, prey_id_buf, prey_eligible_buf,
        egg_retained);
}

/*
 * apply_predation_bench: run leaf n_iter times, resetting the 7 mutated arrays
 * from pristine snapshots each iteration before the call.
 *
 * Mutated arrays (7): inst_abd, n_dead, pred_success_rate, preyed_biomass,
 *                     rsc_biomass, tl_weighted_sum, diet_matrix.
 * Pristine snapshot pointers are passed as additional args after n_iter.
 * Array byte-lengths needed for memcpy:
 *   inst_abd, pred_success_rate, preyed_biomass, tl_weighted_sum: n_schools doubles
 *   n_dead: n_schools * n_causes doubles
 *   rsc_biomass: n_resources * n_cells doubles
 *   diet_matrix: diet_nrow * diet_ncol doubles
 * n_schools is passed explicitly; the 2D sizes use shape ints already present.
 */
void apply_predation_bench(
    i32 p_idx, const i32* cell_indices, i32 n_local,
    double* inst_abd, double* n_dead,
    const i32* species_id, const double* length, const double* weight,
    const i32* age_dt, const i32* first_feeding_age_dt, const i32* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const i32* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const i32* prey_access_idx, const i32* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const i32* rsc_access_rows,
    i32 n_resources, i32 n_species, i32 cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    i32* prey_type_buf, i32* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    /* 7 auxiliary shape args */
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol,
    /* bench-only args */
    int n_iter, int n_schools,
    const double* snap_inst_abd,
    const double* snap_n_dead,
    const double* snap_pred_success_rate,
    const double* snap_preyed_biomass,
    const double* snap_rsc_biomass,
    const double* snap_tl_weighted_sum,
    const double* snap_diet_matrix)
{
    size_t sz_schools   = (size_t)n_schools * sizeof(double);
    size_t sz_n_dead    = (size_t)n_schools * (size_t)n_causes * sizeof(double);
    size_t sz_rsc       = (size_t)n_resources * (size_t)n_cells * sizeof(double);
    size_t sz_diet      = (size_t)diet_nrow * (size_t)diet_ncol * sizeof(double);

    for (int iter = 0; iter < n_iter; iter++) {
        memcpy(inst_abd,          snap_inst_abd,          sz_schools);
        memcpy(n_dead,            snap_n_dead,            sz_n_dead);
        memcpy(pred_success_rate, snap_pred_success_rate, sz_schools);
        memcpy(preyed_biomass,    snap_preyed_biomass,    sz_schools);
        memcpy(rsc_biomass,       snap_rsc_biomass,       sz_rsc);
        memcpy(tl_weighted_sum,   snap_tl_weighted_sum,   sz_schools);
        memcpy(diet_matrix,       snap_diet_matrix,       sz_diet);

        leaf(
            p_idx, cell_indices, n_local,
            inst_abd, n_dead, n_causes,
            species_id, length, weight,
            age_dt, first_feeding_age_dt, feeding_stage,
            pred_success_rate, preyed_biomass, trophic_level,
            size_ratio_min, size_ratio_max, srm_ncol,
            ingestion_rate, fr_shape, fr_halfsat,
            n_dt_per_year, n_subdt,
            access_matrix, acc_nrow, acc_ncol,
            has_access, use_stage_access,
            prey_access_idx, pred_access_idx,
            rsc_biomass, n_cells, rsc_size_min,
            rsc_size_max, rsc_tl, rsc_access_rows,
            n_resources, n_species, cell_id,
            tl_weighted_sum, tl_tracking,
            diet_matrix, diet_nrow, diet_ncol, diet_enabled,
            prey_type_buf, prey_id_buf, prey_eligible_buf,
            egg_retained);
    }
}

/*
 * noop: identical signature to apply_predation_once, empty body.
 * Used by Task 6 as a boundary-cost probe.
 */
void noop(
    i32 p_idx, const i32* cell_indices, i32 n_local,
    double* inst_abd, double* n_dead,
    const i32* species_id, const double* length, const double* weight,
    const i32* age_dt, const i32* first_feeding_age_dt, const i32* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const i32* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const i32* prey_access_idx, const i32* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const i32* rsc_access_rows,
    i32 n_resources, i32 n_species, i32 cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    i32* prey_type_buf, i32* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    /* 7 auxiliary shape args */
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol)
{
    /* intentionally empty */
    (void)p_idx; (void)cell_indices; (void)n_local;
    (void)inst_abd; (void)n_dead;
    (void)species_id; (void)length; (void)weight;
    (void)age_dt; (void)first_feeding_age_dt; (void)feeding_stage;
    (void)pred_success_rate; (void)preyed_biomass; (void)trophic_level;
    (void)size_ratio_min; (void)size_ratio_max;
    (void)ingestion_rate; (void)fr_shape; (void)fr_halfsat;
    (void)n_dt_per_year; (void)n_subdt;
    (void)access_matrix;
    (void)has_access; (void)use_stage_access;
    (void)prey_access_idx; (void)pred_access_idx;
    (void)rsc_biomass; (void)rsc_size_min;
    (void)rsc_size_max; (void)rsc_tl; (void)rsc_access_rows;
    (void)n_resources; (void)n_species; (void)cell_id;
    (void)tl_weighted_sum; (void)tl_tracking;
    (void)diet_matrix; (void)diet_enabled;
    (void)prey_type_buf; (void)prey_id_buf; (void)prey_eligible_buf;
    (void)egg_retained;
    (void)srm_ncol; (void)acc_nrow; (void)acc_ncol;
    (void)n_cells; (void)n_causes; (void)diet_nrow; (void)diet_ncol;
}
