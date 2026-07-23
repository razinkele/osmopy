# Surrogate-Bayesian UQ — Phase 2a: Statistical Model (prior + likelihood + posterior) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Bayesian statistical model — a uniform prior, two pluggable likelihoods (`GaussianLogBiomass` default + `BandFaithful` ABC kernel), and a composed `log_posterior(θ) -> float` — as pure numpy/scipy functions, validated on synthetic θ*-recovery and a misspecified guard, with zero new dependencies and no sampler.

**Architecture:** Three new modules in `osmose/calibration/uq/`. `log_posterior(θ)` is the complete deliverable of this plan: it is a callable density, evaluable and testable **without any sampler** (the sampler + `run.py` + the `emcee`/`dynesty` deps are the separate Phase 2b plan). The posterior takes **injected** emulators (duck-typed `.predict(X)->(mean,var)`) so synthetic tests inject an analytic emulator and θ*-recovery is exact and fast; `fit_emulators` is the production path that builds real GPs from a `DesignResult`, tested separately. All quantities are in natural log of biomass; the Jensen correction converts the emulator's log-geometric-mean to the log-arithmetic-mean the ICES targets live on.

**Tech Stack:** Python 3.12+, NumPy, SciPy (`scipy.special.log_ndtr`, `scipy.stats.norm`). Phase 0's `GPEmulator`, Phase 1's `DesignResult`, `osmose.calibration.uq.keying.target_to_output_key`, `osmose.calibration.targets.BiomassTarget`, `osmose.calibration.problem.FreeParameter`. pytest.

## Global Constraints

- **Python 3.12+**; type hints on all public APIs. **Ruff line length 100** — run `.venv/bin/ruff format` then `.venv/bin/ruff check` on new/changed files before each commit and fix findings.
- **Test runner is `.venv/bin/python -m pytest`** (system `python` may not exist).
- **No new dependencies.** numpy/scipy only. `emcee`/`dynesty`/`arviz` and the `[uq]` extra are Phase 2b — do not add them.
- **`osmose/calibration/uq/__init__.py` stays thin** — no re-exports, no eager heavy imports; not added to `osmose/calibration/__init__.py`. Import submodules directly.
- **All quantities natural log of biomass.** The emulator's `μ_emu` is the mean-of-logs (= log-geometric-mean). ICES targets are arithmetic-scale, so **both** likelihoods apply the Jensen correction `μ = μ_emu + ½·σ_seed²` before forming the residual (converts log-geometric-mean → log-arithmetic-mean). This correction is in **both** `gaussian_log_biomass` and `band_faithful`.
- **No Jacobian.** The prior is uniform in **sampling space** and the emulator trains on and the posterior is evaluated in **sampling space** (the base-10 `10**val` simulator transform lives only in Phase 1's `point_to_overrides`, at the simulator-input boundary). Prior and posterior share the same measure, so there is no change-of-variables term — do not add one.
- **`σ_seed²` and `σ_disc²` are scalar inputs** to the likelihoods (per-key for `σ_seed²`). Pooled-constant-per-key `σ_seed²` is a deliberate Phase 2 simplification; a strongly heteroscedastic `σ_seed²(θ)` would shift the Jensen correction non-uniformly and is a documented limitation tied to the deferred non-stationarity diagnostic.
- **Numerical floors (from the Phase 1 whole-branch review discipline):** every effective variance is floored to `_VAR_FLOOR = 1e-12` before a `sqrt`/division (guards a degenerate band + `σ_disc=0` + tiny `emulator_var`). `BandFaithful` computes `log(Φ(hi)−Φ(lo))` via `scipy.special.log_ndtr` with tail reflection, never `np.log` of a difference of CDFs (which underflows to `-inf` far from the band).
- **Injected-emulator contract:** the posterior calls `emulator.predict(X)` where `X` is `(n, d)` and the return is `(mean (n,), var (n,))` — exactly Phase 0's `GPEmulator.predict` contract. Synthetic tests inject an analytic emulator implementing the same contract.
- **Do NOT mutate any Phase 0/Phase 1 file.** These are all new modules.

## Pinned knobs (adjust here at plan-review; grounded in the spec + a verified likelihood prototype)

| Knob | Value | Rationale |
|---|---|---|
| `k` (coverage multiplier) default | **1.0**, per-`reference_point_type` | Treats band edges as ±1σ — the widest, least-overconfident default. **The correct `k` depends on what `[lower, upper]` in the targets CSV actually represents** (a 95% CI → k≈1.96; a ±1σ range → k=1.0; an ICES Blim/Bpa reference point → not a probability interval, so `k` is a pure modeling knob). This is answerable from how `load_targets` / the targets CSV derived those bounds — check it there before trusting posterior widths; do not assume 1.0 is physically right, only that it is the safest default. |
| `σ_disc²` (discrepancy floor) default | 0.0 | Opt-in width inflation, NOT a structural correction (spec). |
| `_VAR_FLOOR` | 1e-12 | Positivity floor on every effective variance. |
| `_MIN_EMULATOR_POINTS` | 2 | A GP needs ≥2 points; `fit_emulators` skips keys with fewer valid points. |

These were validated against the real math: the split-normal normalizer integrates to 1.0000 and is continuous at r=0; `BandFaithful` stays finite from just-outside to 10⁶× outside the band; the Gaussian posterior recovers θ* (grid argmax exact) on an identifiable synthetic; `BandFaithful` is flat inside the feasible region and decays outside; a conflicting target lowers the max log-posterior by ~3.7.

---

### Task 1: `prior.py` — uniform prior over the sampling-space box

**Files:**
- Create: `osmose/calibration/uq/prior.py`
- Test: `tests/test_uq_prior.py`

**Interfaces:**
- Consumes: `osmose.calibration.problem.FreeParameter` (fields `lower_bound`, `upper_bound`).
- Produces: `log_prior(theta: np.ndarray, free_params: list[FreeParameter]) -> float` — `0.0` when every `theta[j]` is within `[lower_bound, upper_bound]` (inclusive), `-inf` otherwise.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_prior.py`:

```python
"""Tests for the uniform sampling-space prior."""

from __future__ import annotations

import math

import numpy as np

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.prior import log_prior


def _params():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", -2.0, 2.0, Transform.LOG),
    ]


def test_log_prior_inside_box_is_zero():
    assert log_prior(np.array([0.5, 0.0]), _params()) == 0.0


def test_log_prior_outside_box_is_neg_inf():
    assert log_prior(np.array([1.5, 0.0]), _params()) == -math.inf
    assert log_prior(np.array([0.5, -3.0]), _params()) == -math.inf


def test_log_prior_on_boundary_is_included():
    assert log_prior(np.array([0.0, 2.0]), _params()) == 0.0
    assert log_prior(np.array([1.0, -2.0]), _params()) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_prior.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.prior'`.

- [ ] **Step 3: Create `prior.py`**

Create `osmose/calibration/uq/prior.py`:

```python
"""Uniform prior over the FreeParameter sampling-space box.

Uniform in SAMPLING space (where the emulator trains and the posterior is
evaluated). The base-10 simulator transform lives only at the simulator-input
boundary (Phase 1's point_to_overrides), so prior and posterior share one
measure — there is no Jacobian term.
"""

from __future__ import annotations

import math

import numpy as np

from osmose.calibration.problem import FreeParameter


def log_prior(theta: np.ndarray, free_params: list[FreeParameter]) -> float:
    """0.0 inside the box (bounds inclusive), -inf outside."""
    for j, fp in enumerate(free_params):
        if theta[j] < fp.lower_bound or theta[j] > fp.upper_bound:
            return -math.inf
    return 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_prior.py -q`
Expected: PASS — 3 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/prior.py tests/test_uq_prior.py
.venv/bin/ruff check osmose/calibration/uq/prior.py tests/test_uq_prior.py
git add osmose/calibration/uq/prior.py tests/test_uq_prior.py
git commit -m "feat(uq): add uniform sampling-space prior (Phase 2a)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `likelihood.py` — `GaussianLogBiomass` (split-normal + Jensen + normalizer)

The default likelihood: a two-piece (split-normal) log-likelihood over the natural-log residual, with the Jensen correction, the θ-dependent coupled normalizer, and the variance floor.

**Files:**
- Create: `osmose/calibration/uq/likelihood.py`
- Test: `tests/test_uq_likelihood.py`

**Interfaces:**
- Consumes: `math`, `numpy`.
- Produces:
  - Module constants `_HALF_LOG_2_OVER_PI`, `_VAR_FLOOR = 1e-12`.
  - `gaussian_log_biomass(mu_emu, emulator_var, target, lower, upper, *, sigma_seed_sq, sigma_disc_sq, k) -> float` — split-normal log-likelihood. `σ_lo=(ln target−ln lower)/k`, `σ_hi=(ln upper−ln target)/k`; effective variances add `emulator_var + sigma_disc_sq` and are floored; residual `r = (mu_emu + ½σ_seed²) − ln target`; coupled normalizer `−log(σ_eff_lo+σ_eff_hi)`; the quadratic uses `σ_eff_lo²` when `r≤0` else `σ_eff_hi²`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_likelihood.py`:

```python
"""Tests for the UQ likelihoods (Gaussian split-normal + BandFaithful)."""

from __future__ import annotations

import numpy as np

from osmose.calibration.uq.likelihood import gaussian_log_biomass


def test_gaussian_normalizer_integrates_to_one():
    # For fixed band/var, the density over the residual (varying mu_emu) is proper.
    grid = np.linspace(-40.0, 40.0, 200001)
    vals = np.array([
        gaussian_log_biomass(m, 0.05, 100.0, 60.0, 130.0,
                             sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.96)
        for m in grid
    ])
    integral = np.trapezoid(np.exp(vals), grid)
    assert abs(integral - 1.0) < 1e-3


def test_gaussian_continuous_at_r_zero():
    # r = 0 when mu_emu = ln(target) - 0.5*sigma_seed_sq. Both branches must agree.
    mu0 = np.log(100.0) - 0.5 * 0.02
    left = gaussian_log_biomass(mu0 - 1e-7, 0.03, 100.0, 70.0, 140.0,
                                sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.0)
    right = gaussian_log_biomass(mu0 + 1e-7, 0.03, 100.0, 70.0, 140.0,
                                 sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.0)
    assert abs(left - right) < 1e-9


def test_gaussian_symmetric_band_peaks_at_jensen_corrected_target():
    # Symmetric (in log) band around target; the max over mu_emu sits at the
    # Jensen-corrected point mu_emu = ln(target) - 0.5*sigma_seed_sq.
    target, sig_seed = 100.0, 0.02
    lower, upper = target / 1.3, target * 1.3  # log-symmetric
    grid = np.linspace(np.log(50.0), np.log(200.0), 5001)
    vals = [gaussian_log_biomass(m, 0.01, target, lower, upper,
                                 sigma_seed_sq=sig_seed, sigma_disc_sq=0.0, k=1.0) for m in grid]
    mode = grid[int(np.argmax(vals))]
    assert abs(mode - (np.log(target) - 0.5 * sig_seed)) < 1e-2


def test_gaussian_var_floor_degenerate_band_is_finite():
    # lower == upper (zero-width band) + sigma_disc=0 + tiny var must not blow up.
    v = gaussian_log_biomass(np.log(100.0), 1e-15, 100.0, 100.0, 100.0,
                             sigma_seed_sq=0.0, sigma_disc_sq=0.0, k=1.0)
    assert np.isfinite(v)


def test_gaussian_asymmetric_band_uses_correct_side():
    # Prediction above target uses the upper sigma; below uses the lower. With a
    # tight upper band and wide lower band, an over-prediction is penalized harder
    # than an equal-magnitude under-prediction.
    target = 100.0
    lower, upper = 40.0, 110.0  # wide below, tight above
    lt = np.log(target)
    over = gaussian_log_biomass(lt + 0.2, 0.0, target, lower, upper,
                                sigma_seed_sq=0.0, sigma_disc_sq=0.0, k=1.0)
    under = gaussian_log_biomass(lt - 0.2, 0.0, target, lower, upper,
                                 sigma_seed_sq=0.0, sigma_disc_sq=0.0, k=1.0)
    assert over < under
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_likelihood.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.likelihood'`.

- [ ] **Step 3: Create `likelihood.py`**

Create `osmose/calibration/uq/likelihood.py`:

```python
"""Per-target log-likelihoods over the natural-log biomass residual.

Both likelihoods apply the Jensen correction ``mu = mu_emu + 0.5*sigma_seed_sq``
(the emulator target is the mean-of-logs = log-geometric-mean, biased low of the
log-arithmetic-mean the ICES targets live on). GaussianLogBiomass is the default;
BandFaithful (added alongside) is an ABC-style tolerance kernel that is
prior-dominated (flat) on the in-band plateau.
"""

from __future__ import annotations

import math

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_likelihood.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/likelihood.py tests/test_uq_likelihood.py
.venv/bin/ruff check osmose/calibration/uq/likelihood.py tests/test_uq_likelihood.py
git add osmose/calibration/uq/likelihood.py tests/test_uq_likelihood.py
git commit -m "feat(uq): add GaussianLogBiomass split-normal likelihood with Jensen correction (Phase 2a)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `likelihood.py` — `BandFaithful` (ABC tolerance kernel, tail-stable)

The alternative likelihood: the log-probability the emulator's predictive Gaussian assigns to the band, computed stably in both tails.

**Files:**
- Modify: `osmose/calibration/uq/likelihood.py` (add `_log_interval_prob` + `band_faithful`)
- Test: `tests/test_uq_likelihood.py` (append)

**Interfaces:**
- Consumes: `scipy.special.log_ndtr`.
- Produces:
  - `_log_interval_prob(lo_z: float, hi_z: float) -> float` — `log(Φ(hi_z) − Φ(lo_z))`, stable in both tails via reflection.
  - `band_faithful(mu_emu, emulator_var, target, lower, upper, *, sigma_seed_sq, sigma_disc_sq, k=None) -> float` — `log P(ln y ∈ [ln lower, ln upper])` under `N(mu_emu + ½σ_seed², emulator_var + sigma_disc_sq)`. `k` is accepted for signature-uniformity with `gaussian_log_biomass` and ignored (BandFaithful uses the raw band, not a coverage multiplier). Requires `lower < upper`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uq_likelihood.py`:

```python
from scipy.stats import norm

from osmose.calibration.uq.likelihood import _log_interval_prob, band_faithful


def test_band_faithful_high_inside_band():
    # Prediction centered in the band with tiny variance -> P ~ 1 -> loglik ~ 0.
    v = band_faithful(np.log(100.0), 1e-4, 100.0, 80.0, 120.0,
                      sigma_seed_sq=0.0, sigma_disc_sq=0.0)
    assert v > -0.05


def test_band_faithful_decays_and_stays_finite_far_outside():
    inside = band_faithful(np.log(100.0), 0.01, 100.0, 80.0, 120.0,
                           sigma_seed_sq=0.0, sigma_disc_sq=0.0)
    far = band_faithful(np.log(1e6), 0.01, 100.0, 80.0, 120.0,
                        sigma_seed_sq=0.0, sigma_disc_sq=0.0)
    very_far = band_faithful(np.log(1e-6), 0.01, 100.0, 80.0, 120.0,
                             sigma_seed_sq=0.0, sigma_disc_sq=0.0)
    assert far < inside
    assert np.isfinite(far) and np.isfinite(very_far)


def test_log_interval_prob_matches_naive_for_mild_case():
    # Mild case (band around the center) where the naive difference is stable.
    lo_z, hi_z = -1.0, 1.5
    naive = np.log(norm.cdf(hi_z) - norm.cdf(lo_z))
    assert abs(_log_interval_prob(lo_z, hi_z) - naive) < 1e-9


def test_log_interval_prob_reflection_symmetry():
    # log(Phi(b)-Phi(a)) == log(Phi(-a)-Phi(-b)) by symmetry of the normal.
    assert abs(_log_interval_prob(2.0, 4.0) - _log_interval_prob(-4.0, -2.0)) < 1e-9
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_likelihood.py -q -k "band or interval"`
Expected: FAIL — `ImportError: cannot import name 'band_faithful'` (and `_log_interval_prob`).

- [ ] **Step 3: Add `_log_interval_prob` and `band_faithful`**

In `osmose/calibration/uq/likelihood.py`, add the scipy import below the existing imports:

```python
from scipy.special import log_ndtr  # type: ignore[import-untyped]
```

Then append:

```python
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
```

- [ ] **Step 4: Run the full likelihood suite to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_uq_likelihood.py -q`
Expected: PASS — 9 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/likelihood.py tests/test_uq_likelihood.py
.venv/bin/ruff check osmose/calibration/uq/likelihood.py tests/test_uq_likelihood.py
git add osmose/calibration/uq/likelihood.py tests/test_uq_likelihood.py
git commit -m "feat(uq): add BandFaithful ABC-kernel likelihood with tail-stable interval prob (Phase 2a)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `posterior.py` — `fit_emulators` + `make_log_posterior`

Compose prior + per-target likelihood into a single `log_posterior(θ)`; provide the production emulator-fitting path.

**Files:**
- Create: `osmose/calibration/uq/posterior.py`
- Test: `tests/test_uq_posterior.py`

**Interfaces:**
- Consumes: `log_prior` (Task 1); `gaussian_log_biomass`, `band_faithful` (Tasks 2–3); `GPEmulator` (Phase 0); `DesignResult` (Phase 1); `target_to_output_key` (Phase 0); `BiomassTarget`, `FreeParameter`.
- Produces:
  - `DEFAULT_K_BY_TYPE = {"biomass": 1.0, "ssb": 1.0, "catch": 1.0}`.
  - `fit_emulators(design: DesignResult, min_points: int = 2) -> dict[str, GPEmulator]` — fits one GP per key that has `≥ min_points` valid (uncensored) points; omits keys with fewer.
  - `make_log_posterior(emulators, targets, free_params, *, sigma_seed_sq_by_key, sigma_disc_sq=0.0, k_by_type=None, likelihood="gaussian") -> Callable[[np.ndarray], float]` — validates every target's key against `emulators` at construction (raises `KeyError` on a missing one), returns `log_post(theta) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_posterior.py`:

```python
"""Tests for posterior composition + emulator fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.likelihood import gaussian_log_biomass
from osmose.calibration.uq.posterior import fit_emulators, make_log_posterior


class _AnalyticEmulator:
    """Injected emulator matching GPEmulator.predict: X (n,d) -> (mean (n,), var (n,))."""

    def __init__(self, w, b, var):
        self.w = np.asarray(w, float)
        self.b = b
        self.var = var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        mean = X @ self.w + self.b
        return mean, np.full(len(mean), self.var)


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _target(species, rpt, target, lower, upper):
    return BiomassTarget(species=species, target=target, lower=lower, upper=upper,
                         reference_point_type=rpt)


def test_fit_emulators_one_per_key_with_enough_points():
    n = 6
    X = np.linspace(0, 1, n).reshape(-1, 1)
    Y = {"cod_biomass_mean": np.log(np.full(n, 100.0)) + 0.01 * np.arange(n)}
    alpha = {"cod_biomass_mean": np.full(n, 1e-3)}
    design = DesignResult(X=X, keys=["cod_biomass_mean"], Y=Y, alpha=alpha)
    emus = fit_emulators(design)
    assert set(emus) == {"cod_biomass_mean"}
    mean, var = emus["cod_biomass_mean"].predict(np.array([[0.5]]))
    assert mean.shape == (1,) and var.shape == (1,)


def test_fit_emulators_skips_insufficient_points():
    X = np.array([[0.0], [1.0]])
    Y = {"k": np.array([np.log(100.0), np.nan])}  # only 1 valid point
    alpha = {"k": np.array([1e-3, np.nan])}
    design = DesignResult(X=X, keys=["k"], Y=Y, alpha=alpha)
    assert fit_emulators(design) == {}


def test_make_log_posterior_sums_prior_and_likelihoods():
    emu = _AnalyticEmulator([1.0, 0.0], 2.0, 0.01)
    emus = {"cod_biomass_mean": emu}
    tgt = _target("cod", "biomass", 20.0, 16.0, 24.0)
    theta = np.array([0.3, 0.5])
    logp = make_log_posterior(emus, [tgt], _fp2(),
                              sigma_seed_sq_by_key={"cod_biomass_mean": 0.02})(theta)
    mu, var = emu.predict(theta.reshape(1, -1))
    expected = 0.0 + gaussian_log_biomass(float(mu[0]), float(var[0]), 20.0, 16.0, 24.0,
                                          sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.0)
    assert abs(logp - expected) < 1e-9


def test_make_log_posterior_prior_gates_out_of_box():
    emus = {"cod_biomass_mean": _AnalyticEmulator([1.0, 0.0], 2.0, 0.01)}
    tgt = _target("cod", "biomass", 20.0, 16.0, 24.0)
    logp = make_log_posterior(emus, [tgt], _fp2(),
                              sigma_seed_sq_by_key={"cod_biomass_mean": 0.02})
    assert logp(np.array([1.5, 0.5])) == -math.inf


def test_make_log_posterior_missing_emulator_key_raises():
    tgt = _target("cod", "ssb", 20.0, 16.0, 24.0)  # key cod_ssb_mean
    with pytest.raises(KeyError, match="cod_ssb_mean"):
        make_log_posterior({}, [tgt], _fp2(), sigma_seed_sq_by_key={})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_posterior.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.posterior'`.

- [ ] **Step 3: Create `posterior.py`**

Create `osmose/calibration/uq/posterior.py`:

```python
"""Compose prior + per-target likelihood into log_posterior(theta).

The posterior takes INJECTED emulators (duck-typed predict(X)->(mean,var)), so
synthetic tests inject an analytic emulator and recovery is exact. fit_emulators
is the production path that builds real GPs from a DesignResult. Cross-target
independence (the log-likelihoods are summed) is a documented overconfidence
source: trophically-coupled species are treated as conditionally independent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from osmose.calibration.problem import FreeParameter
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.emulator import GPEmulator
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.likelihood import band_faithful, gaussian_log_biomass
from osmose.calibration.uq.prior import log_prior

DEFAULT_K_BY_TYPE = {"biomass": 1.0, "ssb": 1.0, "catch": 1.0}
_LIKELIHOODS = {"gaussian": gaussian_log_biomass, "band": band_faithful}


def fit_emulators(design: DesignResult, min_points: int = 2) -> dict[str, GPEmulator]:
    """Fit one GP per key with at least ``min_points`` valid (uncensored) points.

    Keys with too few valid points are omitted (a GP needs >=2). Fits on the
    per-key valid slice — the natural-log seed-mean targets and their noise.
    """
    emulators: dict[str, GPEmulator] = {}
    for key in design.keys:
        X, Y, alpha = design.valid(key)
        if len(X) >= min_points:
            emulators[key] = GPEmulator().fit(X, Y, alpha)
    return emulators


def make_log_posterior(
    emulators: Mapping[str, object],
    targets: Sequence[BiomassTarget],
    free_params: list[FreeParameter],
    *,
    sigma_seed_sq_by_key: Mapping[str, float],
    sigma_disc_sq: float = 0.0,
    k_by_type: Mapping[str, float] | None = None,
    likelihood: str = "gaussian",
) -> Callable[[np.ndarray], float]:
    """Return ``log_post(theta) -> float`` = log_prior + sum of per-target log-likelihoods.

    Emulators are injected (duck-typed ``predict(X)->(mean,var)``). Every target's
    key is validated against ``emulators`` at construction (raises ``KeyError``).
    ``sigma_seed_sq`` and ``k`` are resolved per target once, up front.
    """
    k_by_type = k_by_type if k_by_type is not None else DEFAULT_K_BY_TYPE
    like_fn = _LIKELIHOODS[likelihood]

    resolved = []
    for t in targets:
        key = target_to_output_key(t)
        if key not in emulators:
            raise KeyError(f"no emulator for target key {key!r}")
        resolved.append((t, key, sigma_seed_sq_by_key[key], k_by_type[t.reference_point_type]))

    def log_post(theta: np.ndarray) -> float:
        lp = log_prior(theta, free_params)
        if not math.isfinite(lp):
            return lp
        theta_2d = np.atleast_2d(theta)
        for t, key, seed_sq, k in resolved:
            mean, var = emulators[key].predict(theta_2d)
            lp += like_fn(
                float(mean[0]), float(var[0]), t.target, t.lower, t.upper,
                sigma_seed_sq=seed_sq, sigma_disc_sq=sigma_disc_sq, k=k,
            )
        return lp

    return log_post
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_posterior.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/posterior.py tests/test_uq_posterior.py
.venv/bin/ruff check osmose/calibration/uq/posterior.py tests/test_uq_posterior.py
git add osmose/calibration/uq/posterior.py tests/test_uq_posterior.py
git commit -m "feat(uq): compose prior + likelihood into log_posterior + fit_emulators (Phase 2a)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: synthetic acceptance — θ*-recovery, BandFaithful shape, misspecified guard

Validate the composed density: the Gaussian posterior **peaks at** a known θ* on an identifiable synthetic; BandFaithful is **flat inside / decays outside** (never peaked); a **misspecified** (conflicting) target measurably lowers the max log-posterior.

**Files:**
- Test: `tests/test_uq_posterior_synthetic.py`

**Interfaces:**
- Consumes: `make_log_posterior` (Task 4), `BiomassTarget`, `FreeParameter`. Reuses the `_AnalyticEmulator` pattern.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_posterior_synthetic.py`:

```python
"""Synthetic acceptance for the Phase 2a posterior density (no sampler)."""

from __future__ import annotations

import numpy as np

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.posterior import make_log_posterior

SIG_SEED = 0.02
EMU_VAR = 0.01
THETA_STAR = np.array([0.3, 0.6])


class _AnalyticEmulator:
    def __init__(self, w, b, var):
        self.w = np.asarray(w, float)
        self.b = b
        self.var = var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X @ self.w + self.b, np.full(len(X), self.var)


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


# Identifiable synthetic: 3 targets sensitive to distinct directions (n_targets >= d).
def _emulators():
    return {
        "A_biomass_mean": _AnalyticEmulator([1.0, 0.0], 2.0, EMU_VAR),
        "B_biomass_mean": _AnalyticEmulator([0.0, 1.0], 1.0, EMU_VAR),
        "C_biomass_mean": _AnalyticEmulator([1.0, 1.0], 0.5, EMU_VAR),
    }


def _targets(emulators, band=0.2, override=None):
    targets = []
    for key, emu in emulators.items():
        species = key.split("_")[0]
        mu_star, _ = emu.predict(THETA_STAR.reshape(1, -1))
        value = float(np.exp(mu_star[0] + 0.5 * SIG_SEED))  # so r(theta*) = 0
        if override and override[0] == species:
            value *= override[1]
        targets.append(BiomassTarget(species=species, target=value,
                                     lower=value * (1 - band), upper=value * (1 + band),
                                     reference_point_type="biomass"))
    return targets


def _grid():
    g = np.linspace(0.02, 0.98, 49)
    return g, [np.array([a, b]) for b in g for a in g]


def _seed_by_key(emulators):
    return {key: SIG_SEED for key in emulators}


def test_gaussian_posterior_recovers_theta_star():
    emus = _emulators()
    logp = make_log_posterior(emus, _targets(emus), _fp2(),
                              sigma_seed_sq_by_key=_seed_by_key(emus), likelihood="gaussian")
    g, points = _grid()
    vals = np.array([logp(p) for p in points])
    best = points[int(np.argmax(vals))]
    assert np.allclose(best, THETA_STAR, atol=0.05)


def test_band_faithful_flat_inside_decays_outside():
    emus = _emulators()
    logp = make_log_posterior(emus, _targets(emus, band=0.4), _fp2(),
                              sigma_seed_sq_by_key=_seed_by_key(emus), likelihood="band")
    inside_a = logp(THETA_STAR)
    inside_b = logp(THETA_STAR + np.array([0.15, -0.15]))  # separated, still feasible
    outside = logp(np.array([0.9, 0.1]))
    assert abs(inside_a - inside_b) < 1.0        # flat across the interior plateau
    assert inside_a - outside > 5.0              # decays outside the feasible region


def test_misspecified_target_lowers_max_log_posterior():
    emus = _emulators()
    g, points = _grid()
    seed = _seed_by_key(emus)
    well = make_log_posterior(emus, _targets(emus), _fp2(), sigma_seed_sq_by_key=seed)
    bad = make_log_posterior(emus, _targets(emus, override=("A", 3.0)), _fp2(),
                             sigma_seed_sq_by_key=seed)
    m_well = max(well(p) for p in points)
    m_bad = max(bad(p) for p in points)
    assert m_well - m_bad > 2.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_posterior_synthetic.py -q`
Expected: FAIL — collection succeeds but the tests fail only if the posterior is wrong; since `make_log_posterior` already exists from Task 4, this file should PASS immediately. (This task is the acceptance gate — if any test fails, do NOT weaken the assertion; the values are verified against a prototype. Report the observed numbers.)

- [ ] **Step 3: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_posterior_synthetic.py -q`
Expected: PASS — 3 passed. (Verified against the likelihood prototype: Gaussian grid-argmax = θ* exactly; BandFaithful interior pair within ~0.03 and ~6 above the exterior point (the exterior point violates only one of the three bands at `band=0.4`, so the drop is modest but comfortably clears the `>5.0` gate); misspecified drop ~3.7 > 2.0.)

- [ ] **Step 4: Lint/format and commit**

```bash
.venv/bin/ruff format tests/test_uq_posterior_synthetic.py
.venv/bin/ruff check tests/test_uq_posterior_synthetic.py
git add tests/test_uq_posterior_synthetic.py
git commit -m "test(uq): synthetic acceptance for Phase 2a posterior (recovery, band shape, misspecified guard)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 2a Done Check

After Task 5:

```bash
.venv/bin/python -m pytest tests/test_uq_prior.py tests/test_uq_likelihood.py tests/test_uq_posterior.py tests/test_uq_posterior_synthetic.py -q
.venv/bin/python -m pytest tests/test_uq_emulator.py tests/test_uq_design.py tests/test_uq_gate.py tests/test_uq_growth.py -q
```

Expected: all pass. The second command confirms Phase 2a did not disturb Phase 0/1.

Milestone: a complete, evaluable `log_posterior(θ) -> float` composed from a uniform sampling-space prior and two pluggable likelihoods, with the Jensen correction and numerical floors, validated on synthetic θ*-recovery, the BandFaithful plateau shape, and a misspecified guard — pure numpy/scipy, no new dependencies, no sampler.

## Phase 2b Watch-Points (carry forward)

- **`sampler.py` (dynesty default + emcee/multistart), `run.py`, and the `[uq]` extra deps** are Phase 2b. `run.py` is where a `DesignResult` (with its `n_seeds`) becomes the `sigma_seed_sq_by_key` this plan takes as input: `σ_seed²[key] = mean over valid points of (alpha[key] * n_seeds)`. `DesignResult` does not carry `n_seeds` — `run.py` must thread it from the design call (or Phase 2b adds it to `DesignResult`).
- **Envelope enforcement** (sampler-adequacy diagnostic + hard nominal-dimension cap that aborts regardless of a gate pass) is Phase 2b — the gate certifies emulator fidelity only.
- **`k` must be checked against the targets CSV** (`load_targets`) before any posterior width is trusted — the default 1.0 is the safe choice, not necessarily the physically correct one.
- **Cross-target independence** is a documented overconfidence source (correlated multi-output GP is the Phase 3 "open question").
- **`BiomassTarget.weight` is intentionally NOT used by the UQ posterior** (a Bayesian likelihood has no arbitrary weight term; `weight` drives only the NSGA/DE loss path). `run.py` should note this so a user with a weighted targets CSV knows the weights do not carry into the posterior.
- **Degenerate/malformed bands** (`lower >= upper`) are rejected at `make_log_posterior` construction (added in the Phase 2a whole-branch-review fix). `load_targets`/`BiomassTarget` still do not validate `lower < upper`, so `run.py` may want to surface a clearer per-target error earlier in the pipeline.

## Self-Review (completed during authoring)

- **Spec coverage (Phase 2 statistical-model rows):** `prior.py` → Task 1; `GaussianLogBiomass` (θ-dependent normalizer, split-normal, Jensen correction, σ_disc²) → Task 2; `BandFaithful` (ABC kernel) → Task 3; `posterior.py` (log_prior + Σ likelihood(emulator[key(target)])) → Task 4; well-specified synthetic recovery + misspecified guard → Task 5.
- **Deferred to Phase 2b (correctly):** `sampler.py`, `run.py`, `UQResult`, the `[uq]` deps, sampler-adequacy + dimension-cap envelope enforcement.
- **Type consistency:** the likelihood signature `(mu_emu, emulator_var, target, lower, upper, *, sigma_seed_sq, sigma_disc_sq, k)` is identical for `gaussian_log_biomass` and `band_faithful` (so `make_log_posterior` dispatches uniformly); the injected-emulator `predict(X)->(mean,var)` contract matches Phase 0's `GPEmulator` and the test `_AnalyticEmulator`; `target_to_output_key`, `DesignResult.valid`, and `GPEmulator.fit` are consumed with their real signatures.
- **Placeholder scan:** none — every code step is complete. Constants and synthetic parameters are the verified prototype values.

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Tasks 1–4 are fast; Task 5 evaluates the density on a 49×49 grid (pure arithmetic, sub-second — no GP fitting, since emulators are analytic).
2. **Inline Execution** — execute tasks in this session with checkpoints.
