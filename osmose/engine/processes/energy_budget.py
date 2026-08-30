"""Energy budget process for the bioenergetic module.

Ports Java ``fr.ird.osmose.process.bioen.EnergyBudget`` (4.3.3). Java runs the
whole budget in **tonnes per school** and divides by the school's instantaneous
abundance only at the per-fish weight increments (``getDw`` / ``getDg``):

===========================  ===================================================
Java method                  Formula (units)
===========================  ===================================================
``getEgross``                ``ingestion * assimilation * phi_T * f_O2``  (t/school)
``getMaintenance``           ``c_m * (w*1e6)^beta * arrhenius / nStepYear``
                             then ``* N * 1e-6``                          (t/school)
``setENet``                  ``E_gross - E_maint``                        (t/school)
``computeEnetFaced``         per-fish, per-``g^beta``, annualised running mean
``getRho``                   ``r / (eta * enet_faced) * (w*1e6)^(1-beta)``, [0, 1]
``getDw``                    ``(1-rho) * max(E_net, 0) / N``              (t/fish)
``getDg``                    ``rho * max(E_net, 0) / N``                  (t/fish)
===========================  ===================================================

``run()`` order is E_gross, E_maint, E_net, ``computeEnetFaced``, ``getRho``,
``getDw``, ``getDg`` — so the ``enet_faced`` used by ``rho`` is the value already
updated with the current step's ``E_net``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.processes.temp_function import arrhenius


def energy_terms(
    ingestion: NDArray[np.float64],
    weight: NDArray[np.float64],
    abundance: NDArray[np.float64],
    temp_c: NDArray[np.float64] | float,
    assimilation: float,
    c_m: float,
    beta: float,
    e_maint_energy: float,
    phi_t: NDArray[np.float64] | float,
    f_o2: NDArray[np.float64] | float,
    n_dt_per_year: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Java ``EnergyBudget.getEgross`` / ``getMaintenance``: all three in TONNES PER SCHOOL.

    Parameters
    ----------
    ingestion:
        Ingested biomass, tonnes per school this timestep (survivor-scaled by mortality).
    weight:
        Somatic weight, tonnes PER FISH.
    abundance:
        Number of fish in the school (Java's instantaneous abundance, post-mortality).
    temp_c:
        Ambient temperature in Celsius (scalar or per-school array).
    assimilation:
        Assimilation efficiency (dimensionless, 0–1).
    c_m:
        Maintenance metabolic coefficient (g^{1-beta} year^{-1}, Arrhenius-modulated).
    beta:
        Allometric scaling exponent for metabolic rate.
    e_maint_energy:
        Arrhenius activation energy for maintenance (eV).
    phi_t:
        Johnson thermal performance factor (scalar or per-school array).
    f_o2:
        Oxygen limitation factor (scalar or per-school array).
    n_dt_per_year:
        Number of discrete timesteps per year.

    Returns
    -------
    tuple:
        ``(e_gross, e_maint, e_net)``, each tonnes per school.
    """
    e_gross = ingestion * assimilation * phi_t * f_o2
    w_grams = weight * 1e6
    # Java: per-fish maintenance in grams, /nStepYear, then `output *= N * 1e-6`
    # to convert the school total back into tonnes.
    e_maint = (
        c_m
        * np.power(w_grams, beta)
        * arrhenius(np.asarray(temp_c), e_maint_energy)
        / n_dt_per_year
    ) * (abundance * 1e-6)
    return e_gross, e_maint, e_gross - e_maint


def update_enet_faced(
    enet_faced_prev: NDArray[np.float64],
    e_net: NDArray[np.float64],
    abundance: NDArray[np.float64],
    weight: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    first_feeding_age_dt: NDArray[np.int32],
    larvae_thres_dt: int,
    larval_coef: float,
    beta: float,
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Java ``EnergyBudget.computeEnetFaced``: per-fish, per-``g^beta``, annualised mean.

    Java branches on ``ageDt``:

    * ``ageDt < firstFeedingAgeDt``  -> 0 (the school has never fed)
    * ``ageDt == firstFeedingAgeDt`` -> ``enet_per_fish / larval_coef`` (no averaging)
    * ``firstFeedingAgeDt < ageDt < larvaeThresDt``
      -> ``(enet_per_fish / larval_coef + prev * ageDt) / (ageDt + 1)``
    * otherwise (juvenile/adult)
      -> ``(enet_per_fish + prev * ageDt) / (ageDt + 1)``

    where ``enet_per_fish = E_net * nStepYear / N * 1e6 / (w * 1e6)^beta``.

    ``larval_coef`` is Java's ``larvaePredationRateBioen`` — in this codebase the
    canonical key ``predation.larval.ingestion.rate.increase.ratio.sp{i}``
    (``config.bioen_theta``).

    Schools with zero abundance keep their previous value: Java iterates
    ``getSchoolSet().getAliveSchools()``, so a dead school is never visited.

    One further guard has no Java counterpart: ``weight == 0`` would make Java's
    ``Math.pow(0, beta)`` zero and the division yield +/-inf, whereas here the
    denominator falls back to 1.0 and the result is finite but meaningless. The
    case is unreachable — a school at or past first feeding always has a
    positive weight — and the guard exists only to keep a degenerate input from
    poisoning a whole species slice with infinities.

    Parameters
    ----------
    enet_faced_prev:
        Previous ``enet_faced`` per school (``state.e_net_avg``).
    e_net:
        Net energy this timestep, tonnes per school.
    abundance:
        Instantaneous abundance (fish per school).
    weight:
        Somatic weight, tonnes per fish.
    age_dt:
        Age in timesteps.
    first_feeding_age_dt:
        Age at first feeding, in timesteps, per school.
    larvae_thres_dt:
        Java ``larvaeThresDt``: age (timesteps) at which the larval ingestion
        bonus stops applying.
    larval_coef:
        Java ``larvaePredationRateBioen``, the larval ingestion multiplier that
        the larval branches divide ``E_net`` by.
    beta:
        Allometric scaling exponent.
    n_dt_per_year:
        Timesteps per year (annualisation factor).

    Returns
    -------
    NDArray[np.float64]:
        Updated ``enet_faced`` per school.
    """
    alive = abundance > 0
    safe_n = np.where(alive, abundance, 1.0)
    w_b = np.power(weight * 1e6, beta)
    safe_w_b = np.where(w_b > 0, w_b, 1.0)
    per_fish = e_net * n_dt_per_year / safe_n * 1e6 / safe_w_b

    age = age_dt.astype(np.float64)
    pre_feeding = age_dt < first_feeding_age_dt
    first = age_dt == first_feeding_age_dt
    larval = (age_dt > first_feeding_age_dt) & (age_dt < larvae_thres_dt)

    running = (per_fish + enet_faced_prev * age) / (age + 1.0)
    running_larval = (per_fish / larval_coef + enet_faced_prev * age) / (age + 1.0)

    out = np.where(larval, running_larval, running)
    out = np.where(first, per_fish / larval_coef, out)
    out = np.where(pre_feeding, 0.0, out)
    # Dead schools are not in Java's alive-school loop: leave them untouched.
    return np.where(alive, out, enet_faced_prev)


def compute_energy_budget(
    ingestion: NDArray[np.float64],
    weight: NDArray[np.float64],
    abundance: NDArray[np.float64],
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
    enet_faced: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute the bioen energy budget and weight increments for one timestep.

    Follows Java ``EnergyBudget.run()``: E_gross, E_maint, E_net (all tonnes per
    school), then ``getRho``, ``getDw``, ``getDg``. ``enet_faced`` must ALREADY
    include this step's ``E_net`` (Java calls ``computeEnetFaced`` before
    ``getRho``) — the caller obtains it from :func:`update_enet_faced`, which in
    turn needs ``e_net`` from :func:`energy_terms`. The energy terms are
    therefore recomputed here rather than passed in, so that the Java formulas
    live in exactly one place.

    Parameters
    ----------
    ingestion:
        Ingested biomass, tonnes per school this timestep.
    weight:
        Somatic weight, tonnes PER FISH.
    abundance:
        Instantaneous abundance (fish per school).
    gonad_weight:
        Gonad weight, tonnes per fish (unused in the budget itself; kept for
        signature symmetry with the Java process, which reads it in starvation).
    age_dt:
        Age in timesteps.
    length:
        Body length in cm.
    temp_c:
        Ambient temperature in Celsius (scalar or per-school array).
    assimilation:
        Assimilation efficiency (dimensionless, 0–1).
    c_m:
        Maintenance metabolic coefficient.
    beta:
        Allometric scaling exponent for metabolic rate.
    eta:
        Energy density ratio (g energy per g gonad tissue).
    r:
        Gonad-allocation coefficient of the rho function.
    m0:
        LMRN intercept: length at maturity = ``m0 + m1 * age_years`` (cm).
    m1:
        LMRN slope.
    e_maint_energy:
        Arrhenius activation energy for maintenance (eV).
    phi_t:
        Johnson thermal performance factor.
    f_o2:
        Oxygen limitation factor.
    n_dt_per_year:
        Timesteps per year.
    enet_faced:
        Java's ``enet_faced`` (per fish, per ``g^beta``, annualised), already
        updated with this step's ``E_net``.

    Returns
    -------
    dw:
        Somatic weight increment in tonnes PER FISH.
    dg:
        Gonad weight increment in tonnes PER FISH.
    e_net:
        Net energy this step, tonnes per school.
    e_gross:
        Gross energy, tonnes per school.
    e_maint:
        Maintenance cost, tonnes per school.
    rho:
        Fraction of positive net energy routed to gonads (0 for immature fish).

    Notes
    -----
    ``rho`` deviates from Java in exactly one cold corner. Java's division is
    unguarded: ``enet_faced == 0`` gives ``r/0 = +inf``, which its ``rho > 1``
    clamp turns into 1 — replicated here. But ``r == 0`` **and**
    ``enet_faced == 0`` gives ``0/0 = NaN`` in Java, and both of its clamp
    comparisons are false for NaN, so Java stores NaN. That is a Java bug; we
    map it to 0, which is what ``r == 0`` (no gonad allocation) means and keeps
    ``dw`` finite.

    Maturity is recomputed every step from ``length >= m0 + m1 * age``, whereas
    Java latches it (``School.setIsMature(true)`` is never cleared). With
    ``m1 > 0`` a matured school can therefore revert here. Out of scope for this
    change; it needs a ``SchoolState.is_mature`` field.
    """
    e_gross, e_maint, e_net = energy_terms(
        ingestion,
        weight,
        abundance,
        temp_c,
        assimilation,
        c_m,
        beta,
        e_maint_energy,
        phi_t,
        f_o2,
        n_dt_per_year,
    )

    w_grams = weight * 1e6
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    is_mature = length >= (m0 + m1 * age_years)

    with np.errstate(divide="ignore", invalid="ignore"):
        rho_raw = r / (eta * enet_faced) * np.power(w_grams, 1.0 - beta)
    rho_raw = np.where(np.isnan(rho_raw), 0.0, rho_raw)  # Java's 0/0 NaN -> 0 (see Notes)
    rho = np.where(is_mature, np.clip(rho_raw, 0.0, 1.0), 0.0)

    # Java getDw/getDg: increments only when E_net > 0, divided by the school's
    # instantaneous abundance, and skipped entirely for schools that are not alive.
    e_pos = np.maximum(e_net, 0.0)
    alive = abundance > 0
    safe_n = np.where(alive, abundance, 1.0)
    dw = np.where(alive, (1.0 - rho) * e_pos / safe_n, 0.0)
    dg = np.where(alive, rho * e_pos / safe_n, 0.0)

    return dw, dg, e_net, e_gross, e_maint, rho
