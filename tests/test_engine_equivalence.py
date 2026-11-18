"""The ensemble-equivalence rule of the C# engine spec (section 10.2), on synthetic arms."""

from __future__ import annotations

import numpy as np

from osmose.validation.engine_equivalence import (
    LADDER,
    MARGINS,
    Cell,
    Constants,
    acceptance,
    calibrate,
    compare,
    tost,
)

N = 32
SPECIES = [f"sp{i}" for i in range(8)]


def _arm(rng, spread=0.03, shift=None, noisy=(), zero_yield=()):
    """Every metric x species, log10-normal around 3 (mean weight around -2)."""
    arm = {}
    for metric in MARGINS:
        arm[metric] = {}
        for sp in SPECIES:
            mu = -2.0 if metric == "mean_weight" else 3.0
            sd = 0.7 if (metric, sp) in noisy else spread
            mu += (shift or {}).get((metric, sp), 0.0)
            values = 10.0 ** rng.normal(mu, sd, N)
            if metric == "yield" and sp in zero_yield:
                values = np.zeros(N)
            arm[metric][sp] = values
    return arm


def _cell(metric="biomass", sp="sp0", d=0.0, ci=0.01, eq=True, vr=1.0, vacuous=False):
    return Cell(metric, sp, d, ci, 0.0, eq, np.nan, vr, vacuous)


# --------------------------------------------------------------------------- the criteria


def test_identical_engines_pass_under_the_calibrated_constants():
    rng = np.random.default_rng(1)
    a, b, c = _arm(rng), _arm(rng), _arm(rng)
    chosen, rates, _ = calibrate(a, b, n_splits=200)
    assert rates[chosen] <= 0.05
    v = acceptance(compare(a, b), compare(a, c), constants=chosen)
    assert v.passed, v.failures


def test_a_cell_shifted_beyond_its_margin_is_divergent():
    rng = np.random.default_rng(2)
    shifted = ("biomass", "sp3")
    a, b, c = _arm(rng), _arm(rng), _arm(rng, shift={shifted: np.log10(2.0)})
    v = acceptance(compare(a, b), compare(a, c))
    assert not v.passed and v.divergent == [shifted]


def test_mean_weight_has_the_tighter_margin():
    rng = np.random.default_rng(3)
    shift = np.log10(1.15)  # inside 1.25x, outside 1.05x
    a, b = _arm(rng), _arm(rng)
    c = _arm(rng, shift={("biomass", "sp1"): shift, ("mean_weight", "sp1"): shift})
    v = acceptance(compare(a, b), compare(a, c))
    assert v.divergent == [("mean_weight", "sp1")]


def test_noisy_cells_are_judged_against_the_control():
    """A cell too noisy to resolve at 32 replicates is non-equivalent in BOTH comparisons, so
    criterion 2 still passes: the control shows what identical engines look like."""
    rng = np.random.default_rng(4)
    noisy = {("biomass", "sp0"), ("abundance", "sp0")}
    a, b, c = _arm(rng, noisy=noisy), _arm(rng, noisy=noisy), _arm(rng, noisy=noisy)
    ctrl, treat = compare(a, b), compare(a, c)
    assert not ctrl[("biomass", "sp0")].equivalent and not treat[("biomass", "sp0")].equivalent
    v = acceptance(ctrl, treat, constants=Constants(1, 1e9, 1e9))  # criteria 3 and 4 off
    assert v.passed, v.failures


def test_criterion_two_counts_equivalent_cells_with_slack():
    ctrl = {("biomass", s): _cell(sp=s, eq=True) for s in SPECIES}
    treat = {("biomass", s): _cell(sp=s, eq=(i >= 2)) for i, s in enumerate(SPECIES)}  # two fewer
    assert not acceptance(ctrl, treat, constants=Constants(1, 1.5, 2.0)).passed
    assert acceptance(ctrl, treat, constants=Constants(2, 1.5, 2.0)).passed


def test_criterion_three_largest_difference():
    ctrl = {("biomass", s): _cell(sp=s, d=0.01) for s in SPECIES}
    treat = dict(ctrl)
    treat[("biomass", "sp0")] = _cell(d=0.09, ci=0.2)  # 9x the control's largest, inside the margin
    assert acceptance(ctrl, treat).passed  # every treatment cell within its margin
    treat[("biomass", "sp0")] = _cell(d=0.12, ci=0.2)  # beyond the margin, not divergent (ci > d)
    v = acceptance(ctrl, treat)
    assert not v.passed and not v.divergent and "largest |d|" in v.failures[0]


def test_criterion_four_variance_ratio_unless_the_control_fails_too():
    ctrl = {("biomass", s): _cell(sp=s) for s in SPECIES}
    treat = dict(ctrl)
    treat[("biomass", "sp5")] = _cell(sp="sp5", vr=3.0)
    v = acceptance(ctrl, treat)
    assert not v.passed and v.variance_failures == [("biomass", "sp5")]
    ctrl[("biomass", "sp5")] = _cell(sp="sp5", vr=0.3)
    assert acceptance(ctrl, treat).passed


def test_vacuous_yield_cells_are_listed_not_counted():
    rng = np.random.default_rng(5)
    a, b, c = (_arm(rng, zero_yield={"sp7"}) for _ in range(3))
    treat = compare(a, c)
    assert treat[("yield", "sp7")].vacuous
    v = acceptance(compare(a, b), treat, constants=Constants(99, 1e9, 1e9))  # listing/counting only
    assert v.vacuous == [("yield", "sp7")] and ("yield", "sp7") not in v.divergent
    assert v.n_equivalent <= 4 * len(SPECIES) - 1


def test_tost_zero_spread_is_a_real_comparison():
    same = np.full(N, 10.0)
    d, ci, p, eq, ks, vr = tost(same, same, MARGINS["biomass"])
    assert (d, ci, p, eq) == (0.0, 0.0, 0.0, True) and np.isnan(ks) and np.isnan(vr)
    d, ci, p, eq, *_ = tost(same * 2, same, MARGINS["biomass"])
    assert eq is False and d > MARGINS["biomass"]


def test_rounding_residue_is_not_spread():
    """A deterministic species: one arm exactly constant, the other constant to within an ulp
    (a seed-dependent summation order). No variance to compare, so criterion 4 must not fire."""
    exact = np.full(N, 1.0e6)
    # 1e-14 relative survives log10 (2**-52 would round back to exactly 6.0: a true zero).
    ulp = exact * (1.0 + np.random.default_rng(9).choice([-1, 0, 1], N) * 1e-14)
    assert 0 < np.log10(ulp).std() < 1e-13  # a residue, really present, far below any spread
    _, ci, _, eq, ks, vr = tost(exact, ulp, MARGINS["biomass"])
    assert eq and ci == 0.0 and np.isnan(vr) and np.isnan(ks)
    real = exact * 10.0 ** np.random.default_rng(10).normal(
        0, 0.01, N
    )  # a real spread still counts
    assert tost(exact, real, MARGINS["biomass"])[5] == 0.0


# --------------------------------------------------------------------------- calibration


def test_the_spec_defaults_false_fail_identical_engines_mostly_through_variance():
    """Spec 10.2 required this calibration because the defaults were a guess: on identical arms
    they fail most of the time, and the variance criterion (4) is the main reason."""
    rng = np.random.default_rng(6)
    _, rates, attribution = calibrate(_arm(rng), _arm(rng), n_splits=200, ladder=(LADDER[0],))
    assert rates[LADDER[0]] > 0.5
    assert attribution[4] == max(attribution.values())


def test_calibration_loosens_the_tunable_criteria():
    """Well-resolved identical arms: the defaults fail, a looser rung brings the false-fail rate
    to at most 5%, and rates fall down the ladder."""
    rng = np.random.default_rng(7)
    chosen, rates, _ = calibrate(_arm(rng), _arm(rng), n_splits=300)
    assert rates[LADDER[0]] > 0.05 and chosen != LADDER[0]
    assert chosen == next(r for r in LADDER if rates[r] <= 0.05)
    assert rates[LADDER[-1]] <= rates[LADDER[0]]


def test_criterion_one_cannot_be_calibrated_for_noisy_cells():
    """For a cell noisier than its margin, 'divergent' reduces to 'the 90% interval excludes
    zero', which identical engines do 10% of the time per cell; no constant touches criterion 1,
    so with several noisy cells no rung reaches 5%, and the attribution names criterion 1."""
    rng = np.random.default_rng(8)
    noisy = {(m, sp) for m in ("biomass", "abundance") for sp in SPECIES[:4]}
    chosen, rates, attribution = calibrate(
        _arm(rng, noisy=noisy), _arm(rng, noisy=noisy), n_splits=300
    )
    assert chosen == LADDER[-1] and rates[LADDER[-1]] > 0.05
    assert attribution[1] == max(attribution.values()) and attribution[1] > 0.3
