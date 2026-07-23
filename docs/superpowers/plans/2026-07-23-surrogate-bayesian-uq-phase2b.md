# Surrogate-Bayesian UQ — Phase 2b: Sampler (dynesty) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the posterior over the calibrated parameters from Phase 2a's `log_posterior` using nested sampling — a `DynestySampler` producing weight-aware posterior summaries, a hard nominal-dimension cap, and a convergence flag — validated on synthetic θ*-recovery and an under-constrained equifinality-ridge case.

**Architecture:** One new module `osmose/calibration/uq/sampler.py`. The sampler takes `log_posterior` as an **injected callable** (Phase 2a's `make_log_posterior` output, or any `theta -> float`), so it tests on the synthetic posterior with zero engine runs and no `run.py`. **Dynesty was chosen over emcee empirically** (prototype: both recover θ* robustly, but dynesty is ~5× faster, has a clean native `dlogz` convergence criterion, is the spec default, and produces the weighted samples the summaries are built for). `SamplerResult` carries per-sample **weights** from the start so a later `EmceeSampler` (unweighted) drops in without reworking the summaries. The `run.py` end-to-end orchestrator (design → gate → fit_emulators → σ_seed² → sampler → `UQResult`) is a **separate later plan**.

**Tech Stack:** Python 3.12+, NumPy, and **`dynesty`** (new `[uq]` optional dependency; already installed in the working `.venv`). Phase 2a's `make_log_posterior`; `FreeParameter`, `BiomassTarget`. pytest.

## Global Constraints

- **Python 3.12+**; type hints on all public APIs. **Ruff line length 100** — run `.venv/bin/ruff format` then `.venv/bin/ruff check` on new/changed files before each commit and fix findings.
- **Test runner is `.venv/bin/python -m pytest`** (system `python` may not exist). `dynesty` is installed in this `.venv`.
- **New dependency: `dynesty` only.** Add the `[uq]` extra `= ["dynesty>=3.1"]`. Do NOT add `emcee` (deferred with `EmceeSampler`) or `arviz` (dynesty's native `logz`/`logzerr` + a Kish ESS cover the diagnostics — no arviz needed) in this plan.
- **`dynesty` is lazy-imported inside `DynestySampler.sample`**, never at `sampler.py` module top — so `SamplerResult` and `check_dimension` (Task 1) import and test without dynesty present. `sampler.py` is not added to `osmose/calibration/uq/__init__.py`.
- **The sampler takes `log_posterior` as an injected callable** (`Callable[[np.ndarray], float]`); tests inject the Phase 2a synthetic posterior. The sampler never constructs emulators or reads a design.
- **Dynesty-as-log-likelihood is valid ONLY because the prior is uniform.** The box `prior_transform` maps the unit cube onto the `[lower, upper]` box, so `log_posterior` is only ever evaluated inside the box, where Phase 2a's `log_prior == 0` and thus `log_posterior == the log-likelihood`. A non-uniform prior would silently drop the prior term — this must be stated in the code so nobody swaps one in.
- **Weight-aware summaries from the start.** `SamplerResult.posterior_mean`/`credible_interval`/`correlation`/`marginal_sd` all use the per-sample `weights` (dynesty's importance weights; uniform for a future emcee).
- **The dimension cap is deterministic** and runs BEFORE any sampling — `check_dimension` is a standalone guard, and `DynestySampler.sample` calls it first, so the cap is testable with no MCMC.
- **Stochastic tests assert the DECISION, seeded, with generous tolerance** — 90% CI covers θ*, `converged` True, posterior mean within a prototyped `atol`; never tight point values (they drift across library versions even at a fixed seed). Verified seed-robust in the prototype.
- **Do NOT mutate any Phase 0/1/2a file.** `sampler.py` is new.

## Pinned knobs (adjust here at plan-review; grounded in the sampler prototype)

| Knob | Value | Rationale |
|---|---|---|
| `MAX_NOMINAL_DIM` | 20 | Hard nominal-dimension cap — the trustworthy-envelope ceiling (concentration of measure + sampler mixing). Aborts regardless of any emulator-gate pass. |
| `nlive` | 200 | Live points; recovered θ* with ess ~580 in ~0.3s on the 2-D synthetic. |
| `dlogz` | 0.5 | Nested-sampling stopping tolerance (remaining evidence). |
| `ess_min` | 100.0 | Kish effective-sample-size floor for the `converged` flag (prototype: identifiable ~580, ridge ~470 — both clear it). |

Prototype evidence (dynesty 3.1.0, 3 seeds each): identifiable synthetic → mean ≈ [0.30, 0.58], 90% CI covers θ*=[0.3,0.6], ess ~580, converged; under-constrained ridge → marginal sd ~0.26, corr ~−0.72; dimension cap raises before sampling.

## Deferred to later plans (recorded)

- **`EmceeSampler`** — a drop-in alternative (`emcee` dep, R̂/ESS/autocorr diagnostics). `SamplerResult`'s weights are `None`/uniform for it. Only worth adding if a posterior proves hard for nested sampling.
- **`run.py`** — the end-to-end orchestrator (`run_surrogate_bayes`): runs a design, gates it, `fit_emulators`, derives `sigma_seed_sq_by_key` from the design's `alpha × n_seeds`, samples, and returns a `UQResult`. This is where real engine runs happen and where the sampler default surfaces.
- **`arviz`** — only if the reporting/predictive phase needs richer diagnostics than dynesty's native `logz`/`logzerr` + Kish ESS.
- **Sampler-adequacy beyond ESS** (cross-run mode agreement / independent-run posterior agreement) — the spec's fuller envelope check; ESS + the dimension cap are the Phase 2b subset.

---

### Task 1: `[uq]` extra + `SamplerResult` (weight-aware) + `check_dimension`

The deterministic substrate: the optional dependency, the weighted result container, and the standalone dimension cap. No dynesty run — these import and test without dynesty.

**Files:**
- Modify: `pyproject.toml` (add the `[uq]` extra)
- Create: `osmose/calibration/uq/sampler.py`
- Test: `tests/test_uq_sampler.py`

**Interfaces:**
- Produces:
  - `MAX_NOMINAL_DIM = 20`.
  - `check_dimension(n_dim: int, max_dim: int = MAX_NOMINAL_DIM) -> None` — raises `ValueError` when `n_dim > max_dim`.
  - `@dataclass SamplerResult` with `samples: np.ndarray (n,d)`, `weights: np.ndarray (n,)`, `logz: float`, `logz_err: float`, `ess: float`, `converged: bool`, and methods `posterior_mean() -> np.ndarray`, `credible_interval(level=0.9) -> tuple[np.ndarray, np.ndarray]`, `correlation() -> np.ndarray`, `marginal_sd() -> np.ndarray` — all weight-aware.

- [ ] **Step 1: Add the `[uq]` extra to `pyproject.toml`**

In `pyproject.toml`, in the `[project.optional-dependencies]` table, add (after the existing `viztest` entry):

```toml
uq = ["dynesty>=3.1"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_uq_sampler.py`:

```python
"""Tests for the UQ sampler (SamplerResult summaries, dimension cap, DynestySampler)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.uq.sampler import (
    MAX_NOMINAL_DIM,
    SamplerResult,
    check_dimension,
)


def test_check_dimension_ok_at_cap():
    check_dimension(MAX_NOMINAL_DIM)  # exactly at the cap is allowed; must not raise


def test_check_dimension_raises_above_cap():
    with pytest.raises(ValueError, match="exceeds"):
        check_dimension(MAX_NOMINAL_DIM + 1)


def _result(samples, weights):
    return SamplerResult(samples=samples, weights=weights, logz=0.0, logz_err=0.1,
                         ess=100.0, converged=True)


def test_sampler_result_posterior_mean_is_weighted():
    r = _result(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([0.25, 0.75]))
    assert np.allclose(r.posterior_mean(), [1.5, 1.5])


def test_sampler_result_credible_interval_ordered_and_brackets_mean():
    x = np.linspace(0.0, 10.0, 101)
    r = _result(np.column_stack([x, x]), np.ones_like(x))
    lo, hi = r.credible_interval(0.9)
    assert lo.shape == (2,) and hi.shape == (2,)
    assert np.all(lo < hi)
    m = r.posterior_mean()
    assert np.all(lo < m) and np.all(m < hi)


def test_sampler_result_correlation_and_marginal_sd_weighted():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    r = _result(np.column_stack([x, -x]), np.ones_like(x))  # perfectly anti-correlated
    assert r.correlation()[0, 1] == pytest.approx(-1.0)
    assert np.all(r.marginal_sd() > 0.0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.sampler'`.

- [ ] **Step 4: Create `sampler.py` (deterministic parts)**

Create `osmose/calibration/uq/sampler.py`:

```python
"""Posterior sampling from Phase 2a's log_posterior via nested sampling (dynesty).

Takes log_posterior as an INJECTED callable so it tests on the synthetic posterior
without run.py. DynestySampler was chosen over emcee empirically (faster, native
dlogz convergence, weighted samples, spec default); EmceeSampler is a deferred
drop-in. dynesty is lazy-imported inside DynestySampler.sample so SamplerResult /
check_dimension work without it.

dynesty separates prior (unit-cube transform) from likelihood. This works with
Phase 2a's combined log_posterior ONLY because the prior is uniform: the box
prior_transform evaluates log_posterior exclusively inside the box, where
log_prior == 0, so log_posterior == the log-likelihood there. A non-uniform prior
would silently drop the prior term — do not swap one in without separating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_NOMINAL_DIM = 20


def check_dimension(n_dim: int, max_dim: int = MAX_NOMINAL_DIM) -> None:
    """Hard nominal-dimension cap: raise above the trustworthy envelope, regardless
    of any emulator-gate pass (concentration of measure + ensemble-sampler mixing)."""
    if n_dim > max_dim:
        raise ValueError(
            f"n_dim ({n_dim}) exceeds the trustworthy-envelope cap ({max_dim}); "
            f"reduce dimension before sampling"
        )


def _weighted_quantile(x: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cum = np.cumsum(weights[order])
    cum /= cum[-1]
    return float(np.interp(q, cum, x[order]))


@dataclass
class SamplerResult:
    """Posterior samples + weights + evidence, with weight-aware summaries.

    ``weights`` are per-sample (dynesty's importance weights; uniform for a future
    emcee). All summaries respect them so both samplers share one result type.
    """

    samples: np.ndarray
    weights: np.ndarray
    logz: float
    logz_err: float
    ess: float
    converged: bool

    def posterior_mean(self) -> np.ndarray:
        return np.average(self.samples, axis=0, weights=self.weights)

    def credible_interval(self, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        d = self.samples.shape[1]
        lo = np.array([_weighted_quantile(self.samples[:, j], self.weights, lo_q) for j in range(d)])
        hi = np.array([_weighted_quantile(self.samples[:, j], self.weights, hi_q) for j in range(d)])
        return lo, hi

    def _cov(self) -> np.ndarray:
        return np.cov(self.samples.T, aweights=self.weights)

    def marginal_sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self._cov()))

    def correlation(self) -> np.ndarray:
        cov = self._cov()
        sd = np.sqrt(np.diag(cov))
        return cov / np.outer(sd, sd)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 6: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
.venv/bin/ruff check osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
git add pyproject.toml osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
git commit -m "feat(uq): add SamplerResult (weight-aware) + dimension cap + [uq] extra (Phase 2b)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `DynestySampler` — nested sampling of the injected posterior

The sampler itself: run dynesty on the injected `log_posterior` over the box, and build a `SamplerResult`. The dimension cap fires first.

**Files:**
- Modify: `osmose/calibration/uq/sampler.py` (add `DynestySampler`)
- Test: `tests/test_uq_sampler.py` (append)

**Interfaces:**
- Consumes: `dynesty` (lazy); `SamplerResult`, `check_dimension`; a `log_posterior: Callable[[np.ndarray], float]`; `list[FreeParameter]`.
- Produces: `DynestySampler(nlive=200, dlogz=0.5, ess_min=100.0, max_dim=MAX_NOMINAL_DIM)` with `.sample(log_posterior, free_params, *, seed=0) -> SamplerResult`. Calls `check_dimension(len(free_params))` before sampling; builds a box `prior_transform` from the `FreeParameter` bounds; `converged = ess >= ess_min` where `ess` is the Kish effective sample size `1/Σw²`.

- [ ] **Step 1: Write the failing tests**

Add these imports to the **top** of `tests/test_uq_sampler.py` (with the existing imports, so ruff's E402 stays clean), then append the helpers + tests below the Task 1 tests:

```python
from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.posterior import make_log_posterior
from osmose.calibration.uq.sampler import DynestySampler
```

Helpers + tests (append after the Task 1 tests):

```python
_THETA_STAR = np.array([0.3, 0.6])
_SIG_SEED, _EMU_VAR = 0.02, 0.01


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


def _identifiable_log_post():
    emus = {
        "A_biomass_mean": _AnalyticEmulator([1.0, 0.0], 2.0, _EMU_VAR),
        "B_biomass_mean": _AnalyticEmulator([0.0, 1.0], 1.0, _EMU_VAR),
        "C_biomass_mean": _AnalyticEmulator([1.0, 1.0], 0.5, _EMU_VAR),
    }
    targets = []
    for key, emu in emus.items():
        mu_star, _ = emu.predict(_THETA_STAR.reshape(1, -1))
        value = float(np.exp(mu_star[0] + 0.5 * _SIG_SEED))
        targets.append(BiomassTarget(species=key.split("_")[0], target=value,
                                     lower=value * 0.8, upper=value * 1.2,
                                     reference_point_type="biomass"))
    return make_log_posterior(emus, targets, _fp2(),
                              sigma_seed_sq_by_key={k: _SIG_SEED for k in emus})


def test_dynesty_recovers_theta_star():
    result = DynestySampler().sample(_identifiable_log_post(), _fp2(), seed=0)
    lo, hi = result.credible_interval(0.9)
    assert np.all((lo <= _THETA_STAR) & (_THETA_STAR <= hi))  # 90% CI covers theta*
    assert np.allclose(result.posterior_mean(), _THETA_STAR, atol=0.1)  # generous
    assert result.converged
    assert result.ess > 100.0


def test_dynesty_result_carries_weights_and_evidence():
    result = DynestySampler().sample(_identifiable_log_post(), _fp2(), seed=0)
    assert result.samples.shape[1] == 2
    assert result.weights.shape[0] == result.samples.shape[0]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(result.logz) and np.isfinite(result.logz_err)


def test_dynesty_dimension_cap_aborts_before_sampling():
    # max_dim=1 with 2 params: the cap must raise, and it must do so WITHOUT sampling
    # (a sentinel log_posterior that would fail if ever called).
    def _never(theta):
        raise AssertionError("log_posterior must not be called when the cap trips")

    with pytest.raises(ValueError, match="exceeds"):
        DynestySampler(max_dim=1).sample(_never, _fp2(), seed=0)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q -k "dynesty"`
Expected: FAIL — `ImportError: cannot import name 'DynestySampler'`.

- [ ] **Step 3: Add `DynestySampler`**

First, add these two imports at the **top** of `osmose/calibration/uq/sampler.py`, in the existing import block (they are used by `DynestySampler`'s signature annotations; `sampler.py` did not need them in Task 1):

```python
from collections.abc import Callable

from osmose.calibration.problem import FreeParameter
```

Then append the class:

```python
class DynestySampler:
    """Nested-sampling posterior sampler (dynesty) over a uniform box prior.

    ``sample`` runs the injected ``log_posterior`` as dynesty's log-likelihood with a
    box ``prior_transform`` (valid only under a uniform prior — see the module
    docstring). The dimension cap fires before any sampling.
    """

    def __init__(
        self,
        nlive: int = 200,
        dlogz: float = 0.5,
        ess_min: float = 100.0,
        max_dim: int = MAX_NOMINAL_DIM,
    ) -> None:
        self.nlive = nlive
        self.dlogz = dlogz
        self.ess_min = ess_min
        self.max_dim = max_dim

    def sample(
        self,
        log_posterior: Callable[[np.ndarray], float],
        free_params: list[FreeParameter],
        *,
        seed: int = 0,
    ) -> SamplerResult:
        n_dim = len(free_params)
        check_dimension(n_dim, self.max_dim)  # dep-independent guard, before importing/sampling
        import dynesty

        lower = np.array([fp.lower_bound for fp in free_params])
        upper = np.array([fp.upper_bound for fp in free_params])

        def prior_transform(u: np.ndarray) -> np.ndarray:
            return lower + u * (upper - lower)

        sampler = dynesty.NestedSampler(
            log_posterior, prior_transform, ndim=n_dim, nlive=self.nlive,
            rstate=np.random.default_rng(seed),
        )
        sampler.run_nested(print_progress=False, dlogz=self.dlogz)
        res = sampler.results

        weights = np.asarray(res.importance_weights())
        ess = float(1.0 / np.sum(weights**2))  # Kish ESS (version-robust)
        return SamplerResult(
            samples=np.asarray(res.samples),
            weights=weights,
            logz=float(res.logz[-1]),
            logz_err=float(res.logzerr[-1]),
            ess=ess,
            converged=ess >= self.ess_min,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q`
Expected: PASS — 8 passed. (The two dynesty runs take ~0.3s each; verified seed-robust in the prototype — do NOT loosen the `atol=0.1` / CI-coverage assertions if a run is close; report the observed mean/CI instead.)

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
.venv/bin/ruff check osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
git add osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
git commit -m "feat(uq): add DynestySampler (nested sampling, box prior, dimension cap) (Phase 2b)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: equifinality acceptance — under-constrained ridge + convergence flag

Validate Goal 1's actual product: when the targets under-constrain the parameters, the posterior is a **wide, correlated ridge** (equifinality), not a false confident point. Plus the `converged` flag responds to the ESS floor.

**Files:**
- Test: `tests/test_uq_sampler.py` (append)

**Interfaces:**
- Consumes: `DynestySampler`, the `_AnalyticEmulator`/`_fp2`/`make_log_posterior` helpers (Task 2).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uq_sampler.py`:

```python
def _ridge_log_post():
    # ONE target sensitive only to theta0 + theta1 -> the theta0+theta1=const ridge
    # is unconstrained: an equifinality direction (Goal 1's product).
    emus = {"C_biomass_mean": _AnalyticEmulator([1.0, 1.0], 0.5, _EMU_VAR)}
    mu_star, _ = emus["C_biomass_mean"].predict(_THETA_STAR.reshape(1, -1))
    value = float(np.exp(mu_star[0] + 0.5 * _SIG_SEED))
    targets = [BiomassTarget(species="C", target=value, lower=value * 0.8, upper=value * 1.2,
                             reference_point_type="biomass")]
    return make_log_posterior(emus, targets, _fp2(),
                              sigma_seed_sq_by_key={"C_biomass_mean": _SIG_SEED})


def test_underconstrained_ridge_is_wide_and_correlated():
    result = DynestySampler().sample(_ridge_log_post(), _fp2(), seed=0)
    # Wide marginals (the box is [0,1]; sd ~0.26 in the prototype) and a strong
    # negative correlation along the theta0+theta1 ridge — NOT a false point.
    assert np.all(result.marginal_sd() > 0.2)
    assert result.correlation()[0, 1] < -0.6


def test_converged_flag_false_when_ess_below_min():
    # An unreachable ESS floor forces converged=False regardless of the real ESS.
    result = DynestySampler(ess_min=1e9).sample(_identifiable_log_post(), _fp2(), seed=0)
    assert result.converged is False
    assert result.ess < 1e9  # the real ESS is finite and far below the floor
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q -k "ridge or converged_flag"`
Expected: PASS — 2 passed. (Prototype, 3 seeds: ridge marginal sd ~0.25–0.27, corr ~−0.70 to −0.74 — both clear the thresholds with margin.)

- [ ] **Step 3: Run the full sampler suite**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q`
Expected: PASS — 10 passed.

- [ ] **Step 4: Lint/format and commit**

```bash
.venv/bin/ruff format tests/test_uq_sampler.py
.venv/bin/ruff check tests/test_uq_sampler.py
git add tests/test_uq_sampler.py
git commit -m "test(uq): equifinality-ridge + convergence-flag acceptance for the sampler (Phase 2b)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 2b Done Check

After Task 3:

```bash
.venv/bin/python -m pytest tests/test_uq_sampler.py -q
.venv/bin/python -m pytest tests/test_uq_prior.py tests/test_uq_likelihood.py tests/test_uq_posterior.py tests/test_uq_posterior_synthetic.py -q
```

Expected: all pass. The second command confirms Phase 2b did not disturb Phase 2a. (Phase 0/1 are untouched by this module.)

Milestone: a `DynestySampler` that draws the posterior from an injected `log_posterior`, with weight-aware summaries, a deterministic dimension cap, and a convergence flag — validated on synthetic θ*-recovery (90% CI covers θ*, converged) and an equifinality ridge (wide + correlated, not a false point). New dependency limited to `dynesty`; `emcee`/`arviz`/`run.py` deferred.

## Phase 2c Watch-Points (carry forward)

- **`run.py`** is the remaining integration: design → gate → `fit_emulators` → `sigma_seed_sq_by_key` (from the design's `alpha × n_seeds`) → `DynestySampler.sample` → a `UQResult` bundling posterior summaries + the gate reports + censoring counts. It is where real (or synthetic-engine) runs happen and where `target.weight`-not-in-posterior and the `lower<upper` target validation (Phase 2a watch-points) should surface to the user.
- **`EmceeSampler`** drop-in (R̂/ESS/autocorr, `emcee` dep) — add only if a real posterior proves hard for nested sampling.
- **Sampler-adequacy beyond ESS** — cross-run/independent-run posterior agreement (the spec's fuller envelope diagnostic). ESS + the dimension cap are the Phase 2b subset.
- **`marginal_sd`/`correlation` use `np.cov`**, which needs ≥2 samples and non-degenerate weights — fine for real posteriors, but `run.py` should surface a clear error if a sampler returns a degenerate cloud (`correlation()` divides by a zero marginal sd → nan/inf).
- **Bound `run_nested` (from the Phase 2b whole-branch review).** `DynestySampler.sample` calls `run_nested(dlogz=self.dlogz)` with no `maxiter`/`maxcall`. Fine for the synthetic injected posteriors under test, but a pathological real posterior that never reaches `dlogz=0.5` would run unbounded — add `maxiter`/`maxcall` caps when `run.py` wires in real posteriors.
- **`converged` is an ESS-adequacy flag only** (`ess >= ess_min`). `dlogz` is the `run_nested` stopping rule so it is always reached, and the flag carries no dlogz/mixing information — the name slightly over-promises. When `run.py` wraps this, either relabel it or fold in the dlogz-reached + boundary-pileup/local-emulator-variance diagnostics the spec's convergence gate lists, so the field matches what it certifies.

## Self-Review (completed during authoring)

- **Spec coverage (Phase 2 sampler rows):** `DynestySampler` (dynesty default) → Tasks 1–2; convergence gate (dlogz + ESS) → Task 2 (`converged`) + Task 3; hard nominal-dimension cap → Task 1 (`check_dimension`) + Task 2 (wired into `sample`); synthetic posterior recovery → Task 2; the equifinality-ridge diagnostic (Goal 1's product) → Task 3.
- **Deferred (correctly):** `EmceeSampler`, `run.py`/`UQResult`, `arviz`, cross-run mode-agreement — later plans, recorded in watch-points.
- **Type consistency:** `SamplerResult`'s weighted summaries are used identically by the recovery and ridge tests; `DynestySampler.sample(log_posterior, free_params, *, seed)` matches the injected-callable contract; `check_dimension` is called with `len(free_params)` in `sample` and standalone in tests; the Phase 2a `make_log_posterior` output is consumed with its real `Callable[[np.ndarray], float]` signature.
- **Placeholder scan:** none — every code step is complete. Constants and thresholds are the verified prototype values (`nlive=200`, `dlogz=0.5`, `ess_min=100`, ridge sd>0.2 / corr<−0.6, recovery atol=0.1).

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Task 1 is deterministic (no dynesty run); Tasks 2–3 each run a couple of ~0.3s nested-sampling passes.
2. **Inline Execution** — execute tasks in this session with checkpoints.
