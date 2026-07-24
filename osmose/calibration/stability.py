"""Bounded-equilibrium instability penalty for the SP-A Baltic stability calibration.

`stability_penalty()` returns 0.0 for a trajectory that stays bounded and inside its ICES envelope,
and grows with extinction (persistence), envelope violation, drift (trend), and boom-bust variability.
It is a pure function so the picklable calibration objective can call it across a ProcessPool boundary.

Design notes (see docs/superpowers/specs/2026-07-01-baltic-stability-recalibration-spA-design.md):
- persistence is a SMOOTH log10-below-floor term (commensurate with the ICES log10^2 error), not a
  flat step, so it trades off continuously against the ICES match instead of swamping it.
- the trend term takes the MAX of the full-window and late-window slopes, so a config that holds flat
  then tips in the final years is not averaged into a near-zero slope.
- the "late window" is relative (final decade of whatever horizon is run), valid for both the in-loop
  proxy and the 50-yr certification.
- documented boom-bust species (stickleback) are not charged for variability.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

_W_PERSIST, _W_ENVELOPE, _W_TREND, _W_VAR = 10.0, 1.0, 3.0, 1.0


def _series(biomass: pd.DataFrame, sp: str) -> np.ndarray | None:
    """Positive-clipped biomass series for a species, or None if the column is absent."""
    if sp not in biomass.columns:
        return None
    return np.clip(np.asarray(biomass[sp].values, float), 1e-9, None)


def _slope(a: np.ndarray) -> float:
    """OLS slope of log10(a) vs index; 0.0 for series shorter than 3 points."""
    if len(a) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(a), dtype=float), np.log10(a), 1)[0])


def stability_penalty(
    biomass: pd.DataFrame,
    targets: Iterable,
    *,
    phi: float = 0.1,
    boombust: frozenset[str] = frozenset({"stickleback"}),
    warmup_frac: float = 0.2,
) -> float:
    """Scalar instability penalty over the post-warmup window (0.0 = bounded & in-envelope).

    Parameters
    ----------
    biomass:
        WIDE biomass frame: a ``Time``/``time`` column + one numeric column per species.
    targets:
        Iterable of objects with ``.species``, ``.lower``, ``.upper``, ``.weight`` (e.g. BiomassTarget).
    phi:
        Persistence floor as a fraction of a species' ICES lower bound.
    boombust:
        Species whose natural variability is not penalised.
    warmup_frac:
        Leading fraction of the run ignored (spin-up transient).
    """
    n = len(biomass)
    if n < 5:
        return float("inf")
    start = int(round(warmup_frac * n))
    late = max(start + 1, n - 10)  # relative final-decade
    total = 0.0
    for t in targets:
        v = _series(biomass, t.species)
        if v is None:
            continue
        win = v[start:]
        lo, hi, w = float(t.lower), float(t.upper), float(t.weight)
        # persistence: smooth log10-distance of the window-min below the floor phi*lo (0 if above)
        wmin = float(win.min())
        floor = phi * lo
        persist = float(np.log10(floor / wmin) ** 2) if wmin < floor else 0.0
        # envelope: fraction of window outside [lo, hi] + final-decade mean outside
        frac_out = float(np.mean((win < lo) | (win > hi)))
        late_mean = float(np.mean(v[late:]))
        late_out = (
            0.0
            if lo <= late_mean <= hi
            else float(np.log10(max(late_mean, 1e-9) / np.clip(late_mean, lo, hi)) ** 2)
        )
        envelope = frac_out + late_out
        # trend: |slope| of log10-biomass, max of full-window and late-window (final third) slopes
        third = max(3, len(win) // 3)
        trend = max(abs(_slope(win)), abs(_slope(win[-third:])))
        # variability: CV, not charged for documented boom-bust species
        mean = float(np.mean(win))
        cv = float(np.std(win) / mean) if mean > 0 else 0.0
        variability = 0.0 if t.species in boombust else cv
        total += w * (
            _W_PERSIST * persist + _W_ENVELOPE * envelope + _W_TREND * trend + _W_VAR * variability
        )
    return float(total)
