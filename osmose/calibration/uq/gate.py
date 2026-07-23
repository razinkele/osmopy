"""Probabilistic emulator-calibration gate (Bastos & O'Hagan 2009 style).

Cross-validates a GPEmulator and standardizes held-out residuals by the latent
predictive variance plus the held-out seed-mean noise ``alpha`` (= s²/S). A
calibrated emulator has ~95% interval coverage, mean standardized-squared-residual
(MSSR) near 1, and PIT-uniform residuals. MSSR is the primary discriminator;
coverage and PIT are secondary. Certifies emulator fidelity only — NOT the
≤~20-effective-param sampler envelope, which Phase 2 enforces separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from osmose.calibration.uq.emulator import GPEmulator

COVERAGE_MIN = 0.85
MSSR_LOW, MSSR_HIGH = 0.4, 2.5
PIT_P_MIN = 0.02
MIN_GATE_POINTS = 8
LOO_MAX = 20
K_DEFAULT = 10


@dataclass
class GateReport:
    """Calibration verdict + metrics for one emulator output."""

    n: int
    coverage: float
    mssr: float
    pit_pvalue: float
    r2: float
    r2_ceiling: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    key: str | None = None


def _k_folds(n: int) -> int:
    """LOO for small designs, else K_DEFAULT-fold."""
    return n if n <= LOO_MAX else min(K_DEFAULT, n)


def evaluate_emulator_calibration(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray,
    *,
    key: str | None = None,
    seed: int = 0,
) -> GateReport:
    """Cross-validate the emulator and decide whether it is calibrated.

    ``X`` (n, d) sampling-space inputs; ``Y`` (n,) natural-log seed-mean targets;
    ``alpha`` (n,) per-point seed-mean noise s²/S. Returns a GateReport; a design
    with fewer than MIN_GATE_POINTS valid points is reported not-calibratable
    rather than trusting the statistics.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    n = len(X)

    if n < MIN_GATE_POINTS:
        return GateReport(
            n=n,
            coverage=float("nan"),
            mssr=float("nan"),
            pit_pvalue=float("nan"),
            r2=float("nan"),
            r2_ceiling=float("nan"),
            passed=False,
            key=key,
            reasons=[f"insufficient valid points ({n} < {MIN_GATE_POINTS})"],
        )

    cv = GPEmulator().cross_validate(X, Y, alpha, k_folds=_k_folds(n), seed=seed)
    y_true, y_pred = cv["y_true"], cv["y_pred"]
    total_var = cv["pred_var"] + alpha[cv["test_idx"]]
    resid = (y_true - y_pred) / np.sqrt(total_var)

    coverage = float(np.mean(np.abs(resid) < 1.96))
    mssr = float(np.mean(resid**2))
    pit_pvalue = float(stats.kstest(stats.norm.cdf(resid), "uniform").pvalue)

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    var_Y = float(np.var(Y, ddof=0))
    r2_ceiling = 1.0 - float(np.mean(alpha)) / var_Y if var_Y > 0 else 0.0

    reasons: list[str] = []
    if coverage < COVERAGE_MIN:
        reasons.append(f"coverage {coverage:.3f} < {COVERAGE_MIN}")
    if not (MSSR_LOW <= mssr <= MSSR_HIGH):
        reasons.append(f"MSSR {mssr:.3f} outside [{MSSR_LOW}, {MSSR_HIGH}]")
    if pit_pvalue <= PIT_P_MIN:
        reasons.append(f"PIT KS p={pit_pvalue:.3f} <= {PIT_P_MIN}")

    return GateReport(
        n=n,
        coverage=coverage,
        mssr=mssr,
        pit_pvalue=pit_pvalue,
        r2=r2,
        r2_ceiling=r2_ceiling,
        passed=not reasons,
        reasons=reasons,
        key=key,
    )
