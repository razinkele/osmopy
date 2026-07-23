"""Per-target log-likelihoods over the natural-log biomass residual.

Both likelihoods apply the Jensen correction ``mu = mu_emu + 0.5*sigma_seed_sq``
(the emulator target is the mean-of-logs = log-geometric-mean, biased low of the
log-arithmetic-mean the ICES targets live on). GaussianLogBiomass is the default;
BandFaithful (added alongside) is an ABC-style tolerance kernel that is
prior-dominated (flat) on the in-band plateau.
"""

from __future__ import annotations

import math

from scipy.special import log_ndtr  # type: ignore[import-untyped]

_HALF_LOG_2_OVER_PI = 0.5 * math.log(2.0 / math.pi)
_VAR_FLOOR = 1e-12


def gaussian_log_biomass(
    mu_emu: float,
    emulator_var: float,
    target: float,
    lower: float,
    upper: float,
    *,
    sigma_seed_sq: float,
    sigma_disc_sq: float,
    k: float,
) -> float:
    """Split-normal log-likelihood of one target given the emulator prediction.

    ``sigma_lo=(ln target-ln lower)/k``, ``sigma_hi=(ln upper-ln target)/k`` set the
    band's log-space widths; ``emulator_var + sigma_disc_sq`` add in quadrature and
    the result is floored. The residual ``r = (mu_emu + 0.5*sigma_seed_sq) - ln target``
    selects the lower branch when ``r<=0`` and the upper otherwise. The θ-dependent
    normalizer ``-log(sigma_eff_lo + sigma_eff_hi)`` is the two-piece-normal constant
    (keeping it is what self-penalizes high-variance regions).
    """
    mu = mu_emu + 0.5 * sigma_seed_sq
    ln_target = math.log(target)
    sig_lo = (ln_target - math.log(lower)) / k
    sig_hi = (math.log(upper) - ln_target) / k
    var_lo = max(sig_lo * sig_lo + emulator_var + sigma_disc_sq, _VAR_FLOOR)
    var_hi = max(sig_hi * sig_hi + emulator_var + sigma_disc_sq, _VAR_FLOOR)
    se_lo = math.sqrt(var_lo)
    se_hi = math.sqrt(var_hi)
    r = mu - ln_target
    var_side = var_lo if r <= 0.0 else var_hi
    return _HALF_LOG_2_OVER_PI - math.log(se_lo + se_hi) - 0.5 * r * r / var_side


def _log_interval_prob(lo_z: float, hi_z: float) -> float:
    """log(Phi(hi_z) - Phi(lo_z)) for lo_z < hi_z, stable in both tails.

    Computing a difference of CDFs underflows to -inf far from the interval; using
    log_ndtr with a reflection into the accurate left tail keeps it finite.
    """
    if lo_z + hi_z > 0.0:  # interval on the right -> reflect into the left tail
        lo_z, hi_z = -hi_z, -lo_z
    la = log_ndtr(lo_z)
    lb = log_ndtr(hi_z)
    return float(lb + math.log1p(-math.exp(la - lb)))


def band_faithful(
    mu_emu: float,
    emulator_var: float,
    target: float,
    lower: float,
    upper: float,
    *,
    sigma_seed_sq: float,
    sigma_disc_sq: float,
    k: float | None = None,
) -> float:
    """ABC tolerance kernel: log P(ln y in [ln lower, ln upper]) under the emulator's
    Jensen-corrected predictive Gaussian.

    ``k`` is accepted for signature-uniformity with ``gaussian_log_biomass`` and
    ignored — BandFaithful scores the raw band, not a coverage multiple. Requires
    ``lower < upper``. Prior-dominated (flat) wherever the prediction sits inside
    the band; decays outside.
    """
    mu = mu_emu + 0.5 * sigma_seed_sq
    se = math.sqrt(max(emulator_var + sigma_disc_sq, _VAR_FLOOR))
    lo_z = (math.log(lower) - mu) / se
    hi_z = (math.log(upper) - mu) / se
    return _log_interval_prob(lo_z, hi_z)
