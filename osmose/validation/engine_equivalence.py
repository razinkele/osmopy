"""Ensemble equivalence of two engines that run the same model (C# engine spec, section 10.2).

Two in-process engines differ only in their random streams, so equivalence is judged against a
CONTROL comparison: the Python engine against itself on disjoint seeds, through the same cells
and statistics. A cell is one (metric, species): the per-replicate end-of-run values of two arms,
compared on the log10 scale by a two-one-sided t-test (TOST) at a per-metric margin.

Acceptance (spec 10.2), all of:

1. no DIVERGENT cell: its 90% interval excludes zero and ``|d|`` exceeds the margin;
2. equivalent cells of the treatment >= those of the control minus ``eq_slack``;
3. the treatment's largest ``|d|`` <= ``max_ratio`` x the control's, or every treatment cell
   within its margin;
4. each cell's variance ratio within ``var_factor`` (either way), except where the control cell
   fails it too;
5. cells with zero yield in both arms are vacuous: listed, not counted.

``calibrate`` measures the rule's false-fail rate for IDENTICAL engines by re-splitting the
pooled Python replicates, and picks the first constants on a fixed ladder whose rate is at most
5%. It is run, and its constants frozen, before any treatment data are looked at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

#: Floor for the log scale (tonnes or numbers): a collapsed species reads 1, not log(0).
COLLAPSE = 1.0
#: Per-metric log floors. Mean weight is biomass / abundance in tonnes per fish (~1e-6), so the
#: tonnes floor would clamp every value to 1 and make every mean-weight cell read d = 0.
FLOORS = {"biomass": COLLAPSE, "abundance": COLLAPSE, "yield": COLLAPSE, "mean_weight": 1e-9}
#: A standard error below this (log10 units, ~2e-12 relative) is floating-point residue, not
#: spread. Measured 2026-11-18 on Bay of Biscay: a species identical across Python seeds read a
#: C# spread of 4.7e-17 (its outputs are summed in a seed-dependent store order), and the
#: variance ratio 0 / 1e-34 failed criterion 4. Such a cell has no variance to compare.
SPREAD_FLOOR = 1e-12
#: Margins on the log10 scale (spec 10.2): 1.25x for stocks and flows, 1.05x for mean weight.
MARGINS = {
    "biomass": float(np.log10(1.25)),
    "abundance": float(np.log10(1.25)),
    "yield": float(np.log10(1.25)),
    "mean_weight": float(np.log10(1.05)),
}


def _log(a, floor: float = COLLAPSE) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(a, dtype=np.float64), floor))


def tost(x, y, delta: float, floor: float = COLLAPSE, with_ks: bool = True):
    """Formal TOST of ``mean(log10 x) - mean(log10 y)`` against ``+-delta``: returns
    ``(d, ci90_halfwidth, p_tost, equivalent, ks_p, var_ratio)``.

    ``se < SPREAD_FLOOR`` (no spread on the log scale beyond rounding) is a real comparison:
    ``d`` is genuine, ``ci90`` is the literal 0.0, ``equivalent`` is the bare ``|d| <= delta``
    test, and the KS p-value and the variance ratio are NaN (not applicable, not "not
    significant").
    """
    lx, ly = _log(x, floor), _log(y, floor)
    n1, n2 = len(lx), len(ly)
    d = float(lx.mean() - ly.mean())
    se = float(np.sqrt(lx.var(ddof=1) / n1 + ly.var(ddof=1) / n2))
    df = n1 + n2 - 2
    if se < SPREAD_FLOOR:
        eq = abs(d) <= delta
        return d, 0.0, 0.0 if eq else 1.0, eq, np.nan, np.nan
    p_lower = stats.t.sf((d + delta) / se, df)  # H0: d <= -delta
    p_upper = stats.t.cdf((d - delta) / se, df)  # H0: d >= +delta
    p_tost = float(max(p_lower, p_upper))
    ci = float(stats.t.ppf(0.95, df) * se)
    ks_p = np.nan
    if with_ks:
        ks: Any = stats.ks_2samp(x, y)  # scipy's stub does not type the result's fields
        ks_p = float(ks.pvalue)
    vy = ly.var(ddof=1)
    vr = float(lx.var(ddof=1) / vy) if vy > 0 else np.inf
    return d, ci, p_tost, p_tost < 0.05, ks_p, vr


@dataclass(frozen=True)
class Cell:
    """One (metric, species) comparison."""

    metric: str
    species: str
    d: float
    ci: float
    p: float
    equivalent: bool
    ks_p: float
    var_ratio: float
    vacuous: bool


Arm = dict[str, dict[str, np.ndarray]]  # metric -> species -> replicate values


def compare(
    x: Arm, y: Arm, margins: dict[str, float] = MARGINS, with_ks: bool = True
) -> dict[tuple[str, str], Cell]:
    """Every (metric, species) cell both arms carry, as ``x`` against ``y``."""
    cells = {}
    for metric, margin in margins.items():
        for sp in sorted(set(x.get(metric, {})) & set(y.get(metric, {}))):
            a, b = (
                np.asarray(x[metric][sp], dtype=np.float64),
                np.asarray(y[metric][sp], dtype=np.float64),
            )
            vacuous = metric == "yield" and not a.any() and not b.any()
            d, ci, p, eq, ks, vr = tost(
                a, b, margin, floor=FLOORS.get(metric, COLLAPSE), with_ks=with_ks
            )
            cells[(metric, sp)] = Cell(metric, sp, d, ci, p, bool(eq), ks, vr, vacuous)
    return cells


@dataclass(frozen=True)
class Constants:
    """The rule's tunable constants (spec 10.2: "minus one", 1.5, a factor of two)."""

    eq_slack: int = 1
    max_ratio: float = 1.5
    var_factor: float = 2.0


#: Loosening order when the default constants false-fail identical engines too often. Measured
#: on synthetic identical arms (32 cells, 32 replicates): the spec's defaults false-fail ~88%,
#: mostly through criterion 4 (a sample variance ratio leaves [1/2, 2] ~6% of the time per cell
#: at n = 32, so across 32 cells nearly always) and then criterion 2. Variance first, then slack.
LADDER = (
    Constants(1, 1.5, 2.0),
    Constants(1, 1.5, 3.0),
    Constants(2, 1.5, 3.0),
    Constants(2, 2.0, 4.0),
    Constants(3, 2.0, 4.0),
    Constants(3, 3.0, 5.0),
    Constants(4, 3.0, 6.0),
    Constants(5, 4.0, 8.0),
    Constants(6, 5.0, 10.0),
)


@dataclass
class Verdict:
    passed: bool
    failures: list[str]
    failed: frozenset[int]  # which criteria (1 to 4) failed
    divergent: list[tuple[str, str]]
    n_equivalent: int
    n_equivalent_control: int
    max_d: float
    max_d_control: float
    variance_failures: list[tuple[str, str]]
    vacuous: list[tuple[str, str]] = field(default_factory=list)


def _var_ok(vr: float, factor: float) -> bool:
    return bool(np.isnan(vr) or (1.0 / factor <= vr <= factor))


def acceptance(
    control: dict[tuple[str, str], Cell],
    treatment: dict[tuple[str, str], Cell],
    margins: dict[str, float] = MARGINS,
    constants: Constants = LADDER[0],
) -> Verdict:
    """Spec 10.2's rule. Counted cells are those present and non-vacuous in both comparisons."""
    keys = sorted(
        k for k in treatment if k in control and not treatment[k].vacuous and not control[k].vacuous
    )
    vacuous = sorted(k for k in treatment if treatment[k].vacuous)
    failures: list[str] = []
    failed: set[int] = set()

    def fail(criterion: int, message: str) -> None:
        failures.append(message)
        failed.add(criterion)

    divergent = [
        k
        for k in keys
        if abs(treatment[k].d) > treatment[k].ci and abs(treatment[k].d) > margins[k[0]]
    ]
    if divergent:
        fail(1, f"divergent cells: {divergent}")
    n_t = sum(treatment[k].equivalent for k in keys)
    n_c = sum(control[k].equivalent for k in keys)
    if n_t < n_c - constants.eq_slack:
        fail(2, f"{n_t} equivalent cells against the control's {n_c} (slack {constants.eq_slack})")
    max_t = max((abs(treatment[k].d) for k in keys), default=0.0)
    max_c = max((abs(control[k].d) for k in keys), default=0.0)
    within = all(abs(treatment[k].d) <= margins[k[0]] for k in keys)
    if not (max_t <= constants.max_ratio * max_c or within):
        fail(3, f"largest |d| {max_t:.4f} > {constants.max_ratio} x the control's {max_c:.4f}")
    f = constants.var_factor
    var_fail = [
        k
        for k in keys
        if not _var_ok(treatment[k].var_ratio, f) and _var_ok(control[k].var_ratio, f)
    ]
    if var_fail:
        fail(4, f"variance ratio outside {f}x: {var_fail}")
    return Verdict(
        not failures,
        failures,
        frozenset(failed),
        divergent,
        n_t,
        n_c,
        max_t,
        max_c,
        var_fail,
        vacuous,
    )


def _split(pool: Arm, n_first: int, rng: np.random.Generator) -> tuple[Arm, Arm]:
    """One random partition of every cell's pooled replicates, the same permutation for all
    cells (replicate i of every cell came from the same run)."""
    n = len(next(iter(next(iter(pool.values())).values())))
    perm = rng.permutation(n)
    first, second = perm[:n_first], perm[n_first:]
    a = {m: {sp: v[first] for sp, v in by_sp.items()} for m, by_sp in pool.items()}
    b = {m: {sp: v[second] for sp, v in by_sp.items()} for m, by_sp in pool.items()}
    return a, b


def calibrate(
    arm_a: Arm,
    arm_b: Arm,
    n_splits: int = 2000,
    margins: dict[str, float] = MARGINS,
    seed: int = 20261118,
    ladder: tuple[Constants, ...] = LADDER,
    target: float = 0.05,
) -> tuple[Constants, dict[Constants, float], dict[int, float]]:
    """The rule's false-fail rate for identical engines, for every rung of ``ladder``.

    ``arm_a`` and ``arm_b`` (same engine, disjoint seeds) are pooled; each split draws one
    random partition for the control comparison and an independent one for the treatment
    comparison, both at the arms' own sizes, and applies the rule. Returns the first rung whose
    false-fail rate is at most ``target`` (the last rung if none is), every rung's rate, and, at
    the chosen rung, the fraction of splits failing each criterion. Criterion 1 has no constant:
    if it alone keeps the rate above target, no rung reaches it, and the attribution says so.
    """
    pool = {
        m: {
            sp: np.concatenate([np.asarray(arm_a[m][sp]), np.asarray(arm_b[m][sp])])
            for sp in arm_a[m]
            if sp in arm_b.get(m, {})
        }
        for m in arm_a
        if m in arm_b
    }
    n_a = len(next(iter(next(iter(arm_a.values())).values())))
    rng = np.random.default_rng(seed)
    fails = dict.fromkeys(ladder, 0)
    by_criterion: dict[Constants, dict[int, int]] = {
        rung: {c: 0 for c in (1, 2, 3, 4)} for rung in ladder
    }
    for _ in range(n_splits):
        control = compare(*_split(pool, n_a, rng), margins, with_ks=False)
        treatment = compare(*_split(pool, n_a, rng), margins, with_ks=False)
        for rung in ladder:
            v = acceptance(control, treatment, margins, rung)
            fails[rung] += not v.passed
            for c in v.failed:
                by_criterion[rung][c] += 1
    rates = {rung: fails[rung] / n_splits for rung in ladder}
    chosen = next((rung for rung in ladder if rates[rung] <= target), ladder[-1])
    return chosen, rates, {c: n / n_splits for c, n in by_criterion[chosen].items()}
