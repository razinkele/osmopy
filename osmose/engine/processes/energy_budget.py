"""Energy budget process for the bioenergetic module.

Computes gross energy intake, maintenance costs, net energy, and
the allocation fraction rho between somatic and gonad growth.
Matches Java BioenEnergyBudget.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.processes.temp_function import arrhenius


def compute_energy_budget(
    ingestion: NDArray[np.float64],
    weight: NDArray[np.float64],
    gonad_weight: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    length: NDArray[np.float64],
    temp_c: NDArray[np.float64] | float,
    assimilation: float,
    c_m: float,
    beta: float,
    eta: float,
    r: float,
    m0: float,
    m1: float,
    e_maint_energy: float,
    phi_t: NDArray[np.float64] | float,
    f_o2: NDArray[np.float64] | float,
    n_dt_per_year: int,
    e_net_avg: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute energy budget and weight increments for one timestep.

    Parameters
    ----------
    ingestion:
        Ingested biomass in tonnes per school.
    weight:
        Somatic weight in tonnes per school.
    gonad_weight:
        Gonad weight in tonnes per school.
    age_dt:
        Age in discrete timesteps per school.
    length:
        Body length in cm per school.
    temp_c:
        Ambient temperature in Celsius (scalar or per-school array).
    assimilation:
        Assimilation efficiency (dimensionless, 0–1).
    c_m:
        c_m: Maintenance metabolic coefficient (energy_units * g^{-beta} * year^{-1}, modulated by Arrhenius temperature function).
    beta:
        Allometric scaling exponent for metabolic rate.
    eta:
        eta: Energy density ratio (grams of energy per gram of gonad tissue, dimensionless in g-equivalent framework).
    r:
        Fraction of net energy allocated to gonads when mature.
    m0:
        LMRN intercept: length at maturity = m0 + m1 * age_years (cm).
    m1:
        LMRN slope.
    e_maint_energy:
        Arrhenius activation energy for maintenance (eV).
    phi_t:
        Thermal performance factor (scalar or per-school array).
    f_o2:
        Oxygen limitation factor (scalar or per-school array).
    n_dt_per_year:
        Number of discrete timesteps per year.
    e_net_avg:
        Running average net energy per school (used for rho calculation).

    Returns
    -------
    dw_tonnes:
        Somatic weight increment in tonnes.
    dg_tonnes:
        Gonad weight increment in tonnes.
    e_net:
        Net energy this step (in grams, same units as maintenance).
    e_gross:
        Gross energy (ingestion * assimilation * phi_T * f_O2).
    e_maint:
        Maintenance energy cost this step.
    rho:
        Allocation fraction to gonads (0 for immature fish).
    """
    # Gross energy (grams-equivalent)
    e_gross = ingestion * assimilation * phi_t * f_o2

    # Maintenance: convert weight to grams, apply Arrhenius
    w_grams = weight * 1e6
    e_maint = c_m * np.power(w_grams, beta) * arrhenius(temp_c, e_maint_energy) / n_dt_per_year

    # Net energy
    e_net = e_gross - e_maint

    # LMRN maturity check
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    l_mature = m0 + m1 * age_years
    is_mature = length >= l_mature

    # Allocation fraction rho: only for mature fish with positive average energy
    # Clamp to [0, 1] to prevent over-allocation when e_net_avg is very small.
    safe_e_avg = np.where(e_net_avg > 0, e_net_avg, 1.0)
    rho_raw = np.clip(r / (eta * safe_e_avg) * np.power(w_grams, 1.0 - beta), 0.0, 1.0)
    rho = np.where(is_mature, rho_raw, 0.0)

    # Positive net energy split between soma (1-rho) and gonads (rho)
    e_pos = np.maximum(e_net, 0.0)
    dw_grams = (1.0 - rho) * e_pos
    dg_grams = rho * e_pos

    # Convert grams back to tonnes
    dw_tonnes = dw_grams * 1e-6
    dg_tonnes = dg_grams * 1e-6

    return dw_tonnes, dg_tonnes, e_net, e_gross, e_maint, rho


def update_e_net_avg(
    e_net_avg: NDArray[np.float64],
    e_net: NDArray[np.float64],
    weight: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    first_feeding_age_dt: NDArray[np.int32],
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Update the running average net energy per school.

    Schools that have not yet started feeding are left at zero.
    Active schools update as a cumulative mean over all timesteps since
    first feeding.

    Parameters
    ----------
    e_net_avg:
        Current running average (modified in place semantics — a new array
        is returned).
    e_net:
        Net energy computed this timestep.
    weight:
        Somatic weight in tonnes (used to identify live schools).
    age_dt:
        Current age in timesteps.
    first_feeding_age_dt:
        Age at first feeding in timesteps per school.
    n_dt_per_year:
        Timesteps per year (unused here, kept for API consistency).

    Returns
    -------
    NDArray[np.float64]:
        Updated running average net energy array.
    """
    is_feeding = age_dt >= first_feeding_age_dt
    steps_since_feeding = np.maximum(age_dt - first_feeding_age_dt + 1, 1).astype(np.float64)

    new_avg = e_net_avg + (e_net - e_net_avg) / steps_since_feeding
    return np.where(is_feeding, new_avg, e_net_avg)
