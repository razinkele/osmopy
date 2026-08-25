"""Fit cod_west's temperature-dependent Beverton-Holt stock-recruitment coefficient (C1).

Implements spec decision 4 of
``docs/superpowers/specs/2026-08-25-baltic-c1-temperature-recruitment-scenario-knob-design.md``:
Voss & Quaas's Beverton-Holt form with an exponential temperature productivity term,

    ln(R) = -b0 + beta1*T + ln(SSB) - log1p(b3*SSB) + eps

fitted on the log scale by nonlinear least squares. cod.27.22-24 recruitment is age-1, so the
fit pairs assessment row R_{y+1} with SSB_y and SST-Q3_y for each hatch year y (``paired_data``).

Enable/disable of the cod_west thermal knob is PRE-REGISTERED (spec decision 4) and NOT
tunable: enable iff the primary fit's beta1 < 0 with p < 0.1 AND the sign survives refitting
against linearly detrended T (``verdict``). No sign-forcing, no threshold-shopping.

Voss, R. & Quaas, M. F. (2026). Future fishing potential of cod and herring under climate
change in the Western Baltic Sea. ICES JMS 83(4), doi:10.1093/icesjms/fsag033 -- gold OA, no
supplement. Conradt's cod coefficient (the paper's cited source) is published nowhere
accessible, so this self-fit is the SOLE source for cod_west's beta1; there is no cross-check.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = (
    ROOT / "data" / "baltic" / "reference" / "ices_snapshots" / "cod.27.22-24.assessment.json"
)
SERIES_PATH = ROOT / "data" / "baltic" / "forcing" / "baltic_thermal_sr_series.csv"
DOC_DIR = ROOT / "docs"

# scripts/ is not a package, so the sibling builder module is loaded the same way the test
# files for both scripts do (importlib.util.spec_from_file_location on the literal path).
_builder_spec = importlib.util.spec_from_file_location(
    "build_baltic_thermal_sr_series",
    Path(__file__).resolve().parent / "build_baltic_thermal_sr_series.py",
)
_builder = importlib.util.module_from_spec(_builder_spec)
_builder_spec.loader.exec_module(_builder)

# Hatch years start where the thermal series builder's historical block starts. Imported
# directly from build_baltic_thermal_sr_series.HIST_START rather than copied by value: a
# hardcoded second copy of this constant is exactly the drift class that cost Task 2's builder
# three review rounds (three independently-hardcoded year windows disagreeing with each other).
# Rows before this year are synthetic spin-up filler (constant tref), not observed temperature,
# and are excluded from the fit regardless of how many the series CSV happens to carry.
HATCH_START = _builder.HIST_START


def _bh_exp_residuals(
    params: np.ndarray, r: np.ndarray, ssb: np.ndarray, temp: np.ndarray
) -> np.ndarray:
    b0, beta1, b3 = params
    ln_r_hat = -b0 + beta1 * temp + np.log(ssb) - np.log1p(b3 * ssb)
    return np.log(r) - ln_r_hat


def fit_bh_exp(r: np.ndarray, ssb: np.ndarray, temp: np.ndarray) -> dict:
    """Fit ln(R) = -b0 + beta1*T + ln(SSB) - log1p(b3*SSB) + eps by nonlinear least squares.

    b3 is bounded >= 0 (a Beverton-Holt density-dependence term cannot be negative); b0 and
    beta1 are unbounded. Asymptotic standard errors come from the solution jacobian
    (s^2 * (J^T J)^-1, s^2 the residual variance with n-3 degrees of freedom); the p-value is
    a two-sided Student-t test on beta1 with the same n-3 dof.

    Returns a dict with keys beta1, se, p, b0, b3, n.
    """
    r = np.asarray(r, dtype=float)
    ssb = np.asarray(ssb, dtype=float)
    temp = np.asarray(temp, dtype=float)
    n = r.shape[0]
    if n <= 3:
        raise ValueError(f"need more than 3 points to fit 3 parameters, got n={n}")

    x0 = np.array([0.0, 0.0, 1e-4])
    bounds = ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])
    result = least_squares(_bh_exp_residuals, x0, args=(r, ssb, temp), bounds=bounds)
    b0, beta1, b3 = result.x

    dof = n - 3
    resid = result.fun
    s2 = float(np.sum(resid**2) / dof)
    jac = result.jac
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac)
        se_beta1 = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se_beta1 = float("nan")

    if np.isfinite(se_beta1) and se_beta1 > 0:
        t_stat = beta1 / se_beta1
        p = float(2.0 * student_t.sf(abs(t_stat), df=dof))
    else:
        p = float("nan")

    return {
        "beta1": float(beta1),
        "se": se_beta1,
        "p": p,
        "b0": float(b0),
        "b3": float(b3),
        "n": n,
        "converged": bool(result.success),
    }


def _paired_rows(
    recs: list[dict], temps: dict[int, float], hatch_years: range
) -> list[tuple[int, float, float, float]]:
    """Build (hatch_year, R_{y+1}, SSB_y, T_y) rows, dropping any with a missing value.

    recs: assessment rows as read from an ICES snapshot JSON (string-valued dict per row, keyed
      by "year"/"ssb"/"recruitment" -- the cod.27.22-24 snapshot's native shape).
    temps: {year: temperature}, historical (non-spin-up) rows only.
    hatch_years: candidate hatch years y to try; a year missing from temps, or whose y/y+1
      assessment rows are absent or blank, is silently dropped (not every candidate year need
      have full data).
    """
    by_year: dict[int, dict] = {}
    for row in recs:
        try:
            year = int(row["year"])
        except (KeyError, TypeError, ValueError):
            continue
        by_year[year] = row

    rows: list[tuple[int, float, float, float]] = []
    for hatch_year in hatch_years:
        if hatch_year not in temps:
            continue
        row_y = by_year.get(hatch_year)
        row_y1 = by_year.get(hatch_year + 1)
        if row_y is None or row_y1 is None:
            continue
        ssb_str = row_y.get("ssb", "")
        rec_str = row_y1.get("recruitment", "")
        if ssb_str in ("", None) or rec_str in ("", None):
            continue
        try:
            ssb_val = float(ssb_str)
            r_val = float(rec_str)
        except ValueError:
            continue
        rows.append((hatch_year, r_val, ssb_val, float(temps[hatch_year])))

    return rows


def paired_data(
    recs: list[dict], temps: dict[int, float], hatch_years: range
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair cod.27.22-24 recruitment/SSB/temperature with the age-1 lag.

    Recruitment is age-1: R for hatch year y is read from the assessment row for year y+1,
    while SSB and temperature come from year y's row/series entry. Returns (R, SSB, T) arrays
    with rows dropped wherever any of the three values is missing.
    """
    rows = _paired_rows(recs, temps, hatch_years)
    if not rows:
        return np.array([]), np.array([]), np.array([])
    _, r_vals, ssb_vals, t_vals = zip(*rows)
    return np.array(r_vals), np.array(ssb_vals), np.array(t_vals)


def detrended(temp: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Return the residuals of an OLS fit of temp ~ year (removes any linear trend in T)."""
    temp = np.asarray(temp, dtype=float)
    years = np.asarray(years, dtype=float)
    design = np.vstack([years, np.ones_like(years)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, temp, rcond=None)
    fitted = slope * years + intercept
    return temp - fitted


def verdict(fit: dict, fit_detrended: dict) -> dict:
    """Pre-registered enable/disable decision for the cod_west thermal knob (spec decision 4).

    enabled iff the primary fit's beta1 < 0 with p < 0.1 AND the sign survives refitting
    against linearly detrended T. Not tunable: do not adjust these thresholds to change the
    outcome of a particular fit.
    """
    enabled = bool(fit["beta1"] < 0 and fit["p"] < 0.1 and fit_detrended["beta1"] < 0)
    return {
        "enabled": enabled,
        "beta1": fit["beta1"],
        "p": fit["p"],
        "beta1_detrended": fit_detrended["beta1"],
    }


def _load_snapshot(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _load_series_temps(path: Path, sp_col: str = "temp_sp0") -> dict[int, float]:
    """Read the historical (non-spin-up) temp_sp0 rows from the thermal series CSV.

    Historical rows are those with year >= HATCH_START; earlier rows are the synthetic
    spin-up filler at tref and are excluded, per module docstring.
    """
    temps: dict[int, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row["year"])
            except (KeyError, TypeError, ValueError):
                continue
            if year < HATCH_START:
                continue
            value = row.get(sp_col, "")
            if value in ("", None):
                continue
            temps[year] = float(value)
    return temps


def _fmt(fit: dict) -> str:
    se = fit["se"]
    se_str = f"{se:.4f}" if np.isfinite(se) else "nan"
    p_str = f"{fit['p']:.4g}" if np.isfinite(fit["p"]) else "nan"
    flags = []
    if not fit["converged"]:
        flags.append("DID NOT CONVERGE")
    if not np.isfinite(fit["p"]):
        flags.append("p is non-finite -- SE could not be computed, not evidence of no signal")
    if fit["b3"] == 0.0:
        flags.append(
            "b3 at its lower bound (0) -- reported SE/p are the unconstrained 3-param "
            "asymptotics (conservative), not a profile likelihood at the bound"
        )
    flag_str = f" [{'; '.join(flags)}]" if flags else ""
    return (
        f"beta1={fit['beta1']:.4f} (se={se_str}), p={p_str}, "
        f"b0={fit['b0']:.4f}, b3={fit['b3']:.6g}, n={fit['n']}{flag_str}"
    )


def main() -> int:
    """Run the pre-registered cod_west fit and write the dated results doc.

    Loads the cod.27.22-24 ICES snapshot and the Task-4-built thermal series CSV, derives the
    hatch-year window as the OVERLAP of what both actually contain (the series' historical
    block may end wherever the CMEMS product's live end was on the day it was built -- not
    necessarily any particular year assumed in advance), then runs the primary fit, a
    detrended-T refit, and a leave-one-out-terminal refit (drops the most recent hatch year, a
    sensitivity check on the still-revisable terminal assessment year). Writes
    docs/baltic_c1_codwest_fit_YYYY-MM-DD.md with all three fits' numbers, the verdict, and the
    no-supplement/no-cross-check note (spec decision 4).
    """
    snapshot = _load_snapshot(SNAPSHOT_PATH)
    series_temps = _load_series_temps(SERIES_PATH)

    snapshot_years = set()
    for row in snapshot:
        try:
            snapshot_years.add(int(row["year"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not series_temps or not snapshot_years:
        raise RuntimeError("empty series or snapshot -- cannot fit")

    series_span = f"{min(series_temps)}-{max(series_temps)}"
    if min(series_temps) != HATCH_START:
        raise ValueError(
            f"series' historical block starts {min(series_temps)}, but HATCH_START="
            f"{HATCH_START} (imported from build_baltic_thermal_sr_series.HIST_START). "
            "The loaded series' actual first historical year has drifted from the builder's "
            "declared window -- spin-up (constant-tref) rows would otherwise be silently read "
            "as observed temperature, or real leading years would be silently dropped."
        )

    # Overlap: a hatch year y needs a temperature (from the series' historical block) AND an
    # SSB row at y AND a recruitment row at y+1 (from the snapshot). Take the widest candidate
    # range and let _paired_rows/paired_data drop anything actually missing.
    max_hatch = min(max(series_temps), max(snapshot_years) - 1)
    if max_hatch < HATCH_START:
        raise RuntimeError(
            f"no overlapping hatch years: series ends {max(series_temps)}, "
            f"snapshot ends {max(snapshot_years)}"
        )
    hatch_years = range(HATCH_START, max_hatch + 1)

    rows = _paired_rows(snapshot, series_temps, hatch_years)
    if len(rows) < 4:
        raise RuntimeError(f"only {len(rows)} paired hatch-year points -- cannot fit reliably")

    years_used = np.array([row[0] for row in rows])
    r = np.array([row[1] for row in rows])
    ssb = np.array([row[2] for row in rows])
    temp = np.array([row[3] for row in rows])

    fit_primary = fit_bh_exp(r, ssb, temp)
    fit_detrend = fit_bh_exp(r, ssb, detrended(temp, years_used))
    fit_loo = fit_bh_exp(r[:-1], ssb[:-1], temp[:-1])
    v = verdict(fit_primary, fit_detrend)

    # SSB and T can both be strongly trending over this window (cod.27.22-24 SSB fell through
    # most of the record while SST warmed) -- collinear regressors widen SE(beta1) and can push
    # p above 0.1 as a genuine consequence of the data, not a fit defect. Report the correlation
    # so a wide SE isn't misread as broken and "fixed" by adjusting x0/bounds/thresholds.
    ssb_t_corr = float(np.corrcoef(ssb, temp)[0, 1]) if len(ssb) > 1 else float("nan")

    today = date.today().isoformat()
    doc_path = DOC_DIR / f"baltic_c1_codwest_fit_{today}.md"
    lines = [
        "# cod_west thermal Beverton-Holt fit (C1, spec decision 4)",
        "",
        f"**Generated:** {today} · **n hatch years:** {len(rows)} "
        f"({years_used[0]}-{years_used[-1]}, series historical span {series_span}) · "
        f"**stock:** cod.27.22-24 (cod_west, sp0)",
        "",
        "Model: `ln(R) = -b0 + beta1*T + ln(SSB) - log1p(b3*SSB) + eps`, fit on the log scale "
        "by nonlinear least squares (`scipy.optimize.least_squares`), b3 bounded >= 0. "
        "R is age-1: R_{y+1} paired with SSB_y and SST-Q3_y (hatch year y).",
        "",
        "## Fits",
        "",
        f"- **Primary:** {_fmt(fit_primary)}",
        f"- **Detrended-T** (T ~ year OLS residuals): {_fmt(fit_detrend)}",
        f"- **Leave-one-out-terminal** (drops hatch year {years_used[-1]}): {_fmt(fit_loo)}",
        "",
        "## Verdict (pre-registered, spec decision 4 -- not tunable)",
        "",
        f"`enabled = beta1 < 0 and p < 0.1 and beta1_detrended < 0` -> **{v['enabled']}**",
        "",
        f"- primary beta1 = {v['beta1']:.4f}, p = {v['p']:.4g}",
        f"- detrended beta1 = {v['beta1_detrended']:.4f}",
        "",
        "## Data notes",
        "",
        f"- corr(SSB, T) over the fitted hatch years = {ssb_t_corr:.3f}. If |corr| is large, "
        "SSB and T are collinear regressors (both can trend over the record) and SE(beta1) "
        "widens as a genuine consequence -- not a fit defect; do not respond by adjusting x0, "
        "bounds, or the pre-registered thresholds.",
        "",
        "## No cross-check",
        "",
        "Voss & Quaas (2026, ICES JMS 83(4), doi:10.1093/icesjms/fsag033) cites Conradt (2023, "
        "Univ. Hamburg dissertation) for cod's coefficient; that value is published nowhere "
        "accessible and the paper carries no supplement. This self-fit is therefore the SOLE "
        "source for cod_west's beta1 -- there is no independent value to validate it against.",
        "",
    ]
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines))

    print(f"primary:   {_fmt(fit_primary)}")
    print(f"detrended: {_fmt(fit_detrend)}")
    print(f"loo-term:  {_fmt(fit_loo)}")
    print(f"verdict:   enabled={v['enabled']}")
    print(f"wrote {doc_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
