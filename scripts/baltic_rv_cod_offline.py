#!/usr/bin/env python
"""Phase 0: offline correlation of the real 1993-2021 reproductive-volume series vs observed
eastern-Baltic cod (recruitment/SSB). Informative (soft gate) — does NOT block the engine work."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def lagged_correlations(rv_annual, cod_series, max_lag: int) -> dict[int, float]:
    """corr(rv[year], cod[year+lag]) for lag in 0..max_lag (cod responds AFTER rv)."""
    rv = np.asarray(rv_annual, float)
    cod = np.asarray(cod_series, float)
    out = {}
    for lag in range(max_lag + 1):
        a, b = rv[: len(rv) - lag], cod[lag:]
        n = min(len(a), len(b))
        out[lag] = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 2 else float("nan")
    return out


def main() -> int:
    rv = pd.read_csv(ROOT / "docs/diagnostics/baltic_rv_fraction.csv")
    rv["yr"] = pd.to_datetime(rv["time"]).dt.year
    rv_annual = rv.groupby("yr")["rv_fraction"].mean()
    cod = pd.read_csv(ROOT / "docs/diagnostics/ices_cod_2732_observed.csv", comment="#").set_index(
        "year"
    )
    years = sorted(set(rv_annual.index) & set(cod.index))
    rv_a = rv_annual.reindex(years).values
    lines = ["# Phase 0 — offline reproductive-volume vs observed cod (cod.27.24-32)\n"]
    for col in ("recruitment_thousands", "ssb_t"):
        lc = lagged_correlations(rv_a, cod[col].reindex(years).values, max_lag=4)
        best = max((k for k in lc if np.isfinite(lc[k])), key=lambda k: lc[k], default=None)
        lines.append(f"\n## RV vs {col}\n")
        lines.append("| lag (yr) | corr |\n|---|---|\n")
        lines += [f"| {k} | {lc[k]:.3f} |\n" for k in sorted(lc)]
        lines.append(f"\n**Best lag = {best} (corr {lc[best]:.3f}).**\n")
    lines.append(
        "\n*Caveat:* eastern-Baltic cod was downgraded to data-limited (~2019); SSB/R post-2014 "
        "uncertain. Soft gate — the engine hindcast (Phases 1-3) is built regardless.\n"
    )
    out = ROOT / "docs/diagnostics/baltic_rv_cod_correlation.md"
    out.write_text("".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
