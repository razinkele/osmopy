# Surrogate-Bayesian UQ — Phase 3: Predictive Diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the posterior over θ into **per-species marginal predictive ranges** on the outputs (Goal 2, a labeled diagnostic) via a genuine per-θ mixture, with a per-species marginal-coverage check and a θ-mediated cross-species correlation, wired into `run_surrogate_bayes`/`UQResult`.

**Architecture:** One new module `osmose/calibration/uq/predictive.py`, plus an additive predictive step in Phase 2c's `run_surrogate_bayes`/`UQResult`. `posterior_predictive` draws posterior θ **by importance weight** (dynesty's samples are weighted dead points, not equal-weight draws), predicts each emulator, adds the single-run seed noise, and reduces to **marginal** quantiles. The labeling is **structural**: the result type `EmulatorPredictiveRanges` exposes only per-species ranges + a correlation matrix — the joint draws are computed internally and **discarded**, so a caller cannot derive the totals/ratios/`P(total>X)` the spec declares invalid for a conditionally-independent emulator. The real-data validation (held-out OSMOSE coverage, emulator-in-the-loop) needs fresh engine runs → production usage, like every prior phase's real-engine path.

**Tech Stack:** Python 3.12+, NumPy. Consumes `SamplerResult` (Phase 2b), `GPEmulator` (Phase 0), `target_to_output_key`, `BiomassTarget`; extends `UQResult`/`run_surrogate_bayes` (Phase 2c). pytest.

## Global Constraints

- **Python 3.12+**; type hints on all public APIs. **Ruff line length 100** (default ruleset — F401/E402) — run `.venv/bin/ruff format` then `.venv/bin/ruff check` on new/changed files; imports at the top.
- **Test runner is `.venv/bin/python -m pytest`**; **no new dependencies.**
- **Resample by weight.** `SamplerResult.samples` are dynesty **importance-weighted** points. `posterior_predictive` MUST draw θ with probability ∝ `weights` (`rng.choice(n, n_draws, p=weights/weights.sum())`). A uniform draw over `samples` biases toward low-weight nested-sampling points and silently corrupts every range.
- **No Jensen in the predictive `y`.** `y` is a single-run log-biomass with mean `μ_emu` (NOT `μ_emu + ½σ_seed²` — that shift is only for the arithmetic-target comparison in the likelihood). Consequence: the predictive **median** biomass `exp(μ_emu)` is the *geometric* mean, below the arithmetic target by `exp(−½σ_seed²)`. Report **quantiles only**; transform to biomass by `exp()` of the log-space quantiles (monotonic); NEVER expose a biomass mean via `exp(median)` (it would drop the ½σ²). Predictive variance per draw = `emulator_var(θ) + σ_seed²[key]`.
- **Reuse the identical `σ_seed²`.** `posterior_predictive` takes `sigma_seed_sq_by_key` as an argument; `run.py` passes the ONE `derive_sigma_seed_sq(...)` result to **both** `make_log_posterior` and `posterior_predictive`. Inference and prediction must not use different σ_seed².
- **Structural marginal-only labeling.** `EmulatorPredictiveRanges` exposes per-species `log_ranges`/`biomass_ranges` + a `cross_species_correlation` matrix — never the joint draw array. The type name carries "emulator, not reality."
- **Cross-species correlation is a θ-mediated DIAGNOSTIC, not a PPC.** There is no observed multivariate sample to check against (ICES targets are marginal), and the real trophic correlation was discarded when Phase 1's `run_design` reduced per-seed joint outputs to per-key mean/variance. So the correlation is induced *only* by shared posterior θ (zero within a fixed θ, since the likelihood is conditionally independent). Label it exactly that. The one honest cheap check is **per-species marginal coverage** (`marginal_coverage`): does each target fall within its predictive range?
- **Do NOT mutate Phase 0/1/2a/2b files.** `predictive.py` is new; the only prior change is the additive predictive step + `predictive_ranges` field in Phase 2c's `run.py`.

## Pinned knobs (from a verified prototype)

| Knob | Value | Rationale |
|---|---|---|
| `n_draws` | 4000 | Mixture draws for the marginal quantiles; cheap (emulator predict), stable quantiles. |
| `level` | 0.9 | Predictive interval level (5th/median/95th). |

Prototype evidence: weighted resample gives materially different (tighter, posterior-concentrated) ranges than a uniform draw; exp-quantiles are monotonic; all three synthetic targets fall within their predictive ranges; the cross-species correlation is a nonzero θ-mediated matrix.

## Deferred (recorded)

- **Real-data validation** — held-out OSMOSE coverage + emulator-in-the-loop at posterior mode/tail draws need fresh engine runs; production usage, not CI.
- **A genuine trophic cross-species PPC** — requires retaining per-seed **joint** design outputs (Phase 1 discarded them); ties to the correlated-multi-output-GP open question.
- **`σ_seed²(θ)` heteroscedastic model** — pooled-constant-per-key is the documented Phase 2/3 simplification.
- **Both-likelihood width report** — run the inference under `likelihood="gaussian"` and `"band"` and compare `SamplerResult.credible_interval` widths; a usage pattern over the existing pieces, not new code.

---

### Task 1: `predictive.py` — `EmulatorPredictiveRanges` + `posterior_predictive`

The mixture: weighted resample → per-emulator prediction + seed noise → marginal quantiles, with the joint draws discarded.

**Files:**
- Create: `osmose/calibration/uq/predictive.py`
- Test: `tests/test_uq_predictive.py`

**Interfaces:**
- Consumes: `SamplerResult` (Phase 2b); injected emulators (`predict(X)->(mean,var)`).
- Produces:
  - `@dataclass EmulatorPredictiveRanges` with `keys: list[str]`, `log_ranges: dict[str, tuple[float,float,float]]` (lo, median, hi natural-log), `biomass_ranges: dict[str, tuple[float,float,float]]` (`exp` of the above), `cross_species_correlation: np.ndarray`, `level: float`. NO joint-samples field.
  - `posterior_predictive(sampler_result, emulators, target_keys, sigma_seed_sq_by_key, *, n_draws=4000, seed=0, level=0.9) -> EmulatorPredictiveRanges`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_predictive.py`:

```python
"""Tests for the posterior-predictive diagnostic layer (Goal 2)."""

from __future__ import annotations

import dataclasses

import numpy as np

from osmose.calibration.uq.predictive import EmulatorPredictiveRanges, posterior_predictive
from osmose.calibration.uq.sampler import SamplerResult


class _LinEmulator:
    """Injected emulator: mean = slope * theta0 + intercept, tiny fixed variance."""

    def __init__(self, slope=5.0, intercept=0.0, var=1e-4):
        self.slope, self.intercept, self.var = slope, intercept, var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X[:, 0] * self.slope + self.intercept, np.full(len(X), self.var)


def _result(samples, weights):
    return SamplerResult(samples=samples, weights=weights, logz=0.0, logz_err=0.1,
                         ess=100.0, converged=True)


def test_posterior_predictive_resamples_by_weight():
    # theta uniformly spread; weights heavy on HIGH theta. A monotone-increasing
    # emulator maps high theta -> high y, so weighting must shift the predictive
    # median UP vs uniform weighting. (Uniform-weight code would ignore this.)
    n = 200
    samples = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    emus = {"cod_biomass_mean": _LinEmulator()}
    ss = {"cod_biomass_mean": 0.01}
    weighted = posterior_predictive(_result(samples, samples[:, 0] ** 4 + 1e-9),
                                    emus, ["cod_biomass_mean"], ss, seed=0)
    uniform = posterior_predictive(_result(samples, np.ones(n)),
                                   emus, ["cod_biomass_mean"], ss, seed=0)
    assert (weighted.log_ranges["cod_biomass_mean"][1]
            > uniform.log_ranges["cod_biomass_mean"][1] + 0.5)


def test_posterior_predictive_biomass_is_exp_of_log_and_ordered():
    n = 100
    samples = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    r = posterior_predictive(_result(samples, np.ones(n)), {"k": _LinEmulator()}, ["k"],
                             {"k": 0.02}, seed=0)
    lo, med, hi = r.log_ranges["k"]
    assert lo < med < hi
    blo, bmed, bhi = r.biomass_ranges["k"]
    assert np.allclose([blo, bmed, bhi], np.exp([lo, med, hi]))


def test_posterior_predictive_is_marginal_only_no_joint_field():
    n = 50
    samples = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    r = posterior_predictive(_result(samples, np.ones(n)), {"k": _LinEmulator()}, ["k"],
                             {"k": 0.02}, seed=0)
    field_names = {f.name for f in dataclasses.fields(r)}
    # Structural marginal-only guard: no joint/per-draw samples are exposed.
    assert field_names == {"keys", "log_ranges", "biomass_ranges",
                           "cross_species_correlation", "level"}
    assert not hasattr(r, "samples") and not hasattr(r, "draws")


def test_posterior_predictive_cross_species_correlation_matrix():
    n = 300
    samples = np.column_stack([np.linspace(0, 1, n), np.linspace(1, 0, n)])
    emus = {"a_biomass_mean": _LinEmulator(slope=5.0),
            "b_biomass_mean": _LinEmulator(slope=-5.0, intercept=5.0)}
    r = posterior_predictive(_result(samples, np.ones(n)), emus,
                             ["a_biomass_mean", "b_biomass_mean"], {"a_biomass_mean": 1e-3,
                             "b_biomass_mean": 1e-3}, seed=0)
    c = r.cross_species_correlation
    assert c.shape == (2, 2)
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_predictive.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.predictive'`.

- [ ] **Step 3: Create `predictive.py`**

Create `osmose/calibration/uq/predictive.py`:

```python
"""Posterior-predictive diagnostic (Goal 2): per-species marginal emulator-predictive
ranges from the posterior over theta.

DIAGNOSTIC, not calibrated against reality. The joint draws are computed internally
and DISCARDED — only per-species marginal ranges + a theta-mediated cross-species
correlation are returned, so totals/ratios/P(total>X) (invalid for a conditionally-
independent emulator) cannot be derived from the result. y is a single-run
log-biomass with mean mu_emu (NO Jensen shift): the predictive median biomass is
the GEOMETRIC mean, below the arithmetic target by exp(-0.5*sigma_seed_sq).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from osmose.calibration.uq.sampler import SamplerResult


@dataclass
class EmulatorPredictiveRanges:
    """Per-species MARGINAL emulator-predictive ranges (a labeled diagnostic).

    Emulator-coverage-measured, NOT calibrated against reality; marginal-only (no
    joint draws are exposed). ``log_ranges``/``biomass_ranges`` are ``(lo, median,
    hi)`` at ``level``; ``median`` is the geometric mean (below the arithmetic
    target by the Jensen factor). ``cross_species_correlation`` is theta-mediated
    only — NOT a trophic posterior-predictive check.
    """

    keys: list[str]
    log_ranges: dict[str, tuple[float, float, float]]
    biomass_ranges: dict[str, tuple[float, float, float]]
    cross_species_correlation: np.ndarray
    level: float


def posterior_predictive(
    sampler_result: SamplerResult,
    emulators: Mapping[str, object],
    target_keys: Sequence[str],
    sigma_seed_sq_by_key: Mapping[str, float],
    *,
    n_draws: int = 4000,
    seed: int = 0,
    level: float = 0.9,
) -> EmulatorPredictiveRanges:
    """Genuine per-theta mixture -> per-species marginal predictive ranges.

    Resamples theta BY IMPORTANCE WEIGHT (dynesty samples are weighted dead points,
    not equal-weight draws), predicts each emulator, adds single-run seed noise
    ``N(0, emulator_var(theta) + sigma_seed_sq[key])`` (no Jensen shift), and reduces
    to marginal quantiles. The joint draw array is internal and discarded.
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(sampler_result.weights, dtype=float)
    p = weights / weights.sum()
    n = len(sampler_result.samples)
    idx = rng.choice(n, size=n_draws, p=p)  # resample by weight -- the load-bearing step
    thetas = np.asarray(sampler_result.samples)[idx]

    keys = list(target_keys)
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    joint_log = np.empty((n_draws, len(keys)))  # internal; discarded (marginal-only guard)
    log_ranges: dict[str, tuple[float, float, float]] = {}
    biomass_ranges: dict[str, tuple[float, float, float]] = {}

    for j, key in enumerate(keys):
        mean, var = emulators[key].predict(thetas)
        sd = np.sqrt(np.asarray(var) + sigma_seed_sq_by_key[key])
        y = np.asarray(mean) + sd * rng.standard_normal(n_draws)  # single-run log-biomass, no Jensen
        joint_log[:, j] = y
        q = np.quantile(y, [lo_q, 0.5, hi_q])
        log_ranges[key] = (float(q[0]), float(q[1]), float(q[2]))
        biomass_ranges[key] = (float(np.exp(q[0])), float(np.exp(q[1])), float(np.exp(q[2])))

    corr = np.atleast_2d(np.corrcoef(joint_log, rowvar=False))
    return EmulatorPredictiveRanges(
        keys=keys, log_ranges=log_ranges, biomass_ranges=biomass_ranges,
        cross_species_correlation=corr, level=level,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_predictive.py -q`
Expected: PASS — 4 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/predictive.py tests/test_uq_predictive.py
.venv/bin/ruff check osmose/calibration/uq/predictive.py tests/test_uq_predictive.py
git add osmose/calibration/uq/predictive.py tests/test_uq_predictive.py
git commit -m "feat(uq): add posterior-predictive marginal ranges (Goal 2 diagnostic) (Phase 3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `marginal_coverage` — the one honest per-species check

Does each target fall within its predictive range? The genuine, cheap check (a real trophic PPC is deferred — no observed joint sample exists).

**Files:**
- Modify: `osmose/calibration/uq/predictive.py` (add `marginal_coverage`)
- Test: `tests/test_uq_predictive.py` (append)

**Interfaces:**
- Consumes: `EmulatorPredictiveRanges` (Task 1); `BiomassTarget`; `target_to_output_key`.
- Produces: `marginal_coverage(ranges: EmulatorPredictiveRanges, targets: Sequence[BiomassTarget]) -> dict[str, bool]` — per target key, whether `target.target` lies within its predictive biomass `[lo, hi]`.

- [ ] **Step 1: Write the failing tests**

Add these imports to the **top** of `tests/test_uq_predictive.py` (with the existing imports). NOTE: `EmulatorPredictiveRanges` must be imported here even if Task 1 left it out of the test file (it was unused there → F401) — Task 2's `_ranges` helper constructs it:

```python
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.predictive import EmulatorPredictiveRanges, marginal_coverage
```

Append the tests:

```python
def _ranges(biomass):
    return EmulatorPredictiveRanges(
        keys=list(biomass), log_ranges={k: tuple(np.log(v)) for k, v in biomass.items()},
        biomass_ranges=biomass, cross_species_correlation=np.array([[1.0]]), level=0.9,
    )


def test_marginal_coverage_target_within_range_is_covered():
    ranges = _ranges({"cod_biomass_mean": (8.0, 10.0, 13.0)})
    target = BiomassTarget("cod", 10.0, 8.0, 12.0, reference_point_type="biomass")
    assert marginal_coverage(ranges, [target]) == {"cod_biomass_mean": True}


def test_marginal_coverage_target_outside_range_is_not_covered():
    ranges = _ranges({"cod_biomass_mean": (8.0, 10.0, 13.0)})
    target = BiomassTarget("cod", 20.0, 16.0, 24.0, reference_point_type="biomass")  # 20 > 13
    assert marginal_coverage(ranges, [target]) == {"cod_biomass_mean": False}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_predictive.py -q -k "marginal_coverage"`
Expected: FAIL — `ImportError: cannot import name 'marginal_coverage'`.

- [ ] **Step 3: Add `marginal_coverage`**

In `osmose/calibration/uq/predictive.py`, add the import to the top block:

```python
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.keying import target_to_output_key
```

Then append:

```python
def marginal_coverage(
    ranges: EmulatorPredictiveRanges,
    targets: Sequence[BiomassTarget],
) -> dict[str, bool]:
    """Per-species marginal coverage: does each target fall within its predictive
    biomass ``[lo, hi]``? The one honest, cheap posterior-predictive check available
    now — a genuine trophic (joint) PPC needs per-seed joint design outputs, which
    Phase 1 discarded.
    """
    out: dict[str, bool] = {}
    for target in targets:
        key = target_to_output_key(target)
        lo, _median, hi = ranges.biomass_ranges[key]
        out[key] = lo <= target.target <= hi
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_predictive.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/predictive.py tests/test_uq_predictive.py
.venv/bin/ruff check osmose/calibration/uq/predictive.py tests/test_uq_predictive.py
git add osmose/calibration/uq/predictive.py tests/test_uq_predictive.py
git commit -m "feat(uq): add per-species marginal_coverage check (Phase 3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: wire the predictive layer into `run_surrogate_bayes`/`UQResult`

Complete `UQResult` with the predictive ranges, reusing the same emulators and σ_seed² the posterior used.

**Files:**
- Modify: `osmose/calibration/uq/run.py` (`UQResult` field + predictive step + `include_predictive` param)
- Test: `tests/test_uq_run.py` (append)

**Interfaces:**
- Consumes: `posterior_predictive`, `EmulatorPredictiveRanges` (Task 1).
- Produces: `UQResult` gains `predictive_ranges: EmulatorPredictiveRanges | None = None`; `run_surrogate_bayes` gains `include_predictive: bool = True` and, on a sampled run, populates `predictive_ranges` via `posterior_predictive(sampler_result, emulators, keys, sigma_seed_sq, seed=seed)` (the SAME `emulators` and `sigma_seed_sq` used for the posterior).

- [ ] **Step 1: Write the failing tests**

Add this import to the **top** of `tests/test_uq_run.py` (with the existing imports):

```python
from osmose.calibration.uq.predictive import EmulatorPredictiveRanges, marginal_coverage
```

Append the tests (below the existing tests):

```python
def test_run_populates_predictive_ranges_and_covers_targets():
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=100, seed=0, gate_fn=_pass_gate)
    assert result.status == "ok"
    assert isinstance(result.predictive_ranges, EmulatorPredictiveRanges)
    assert set(result.predictive_ranges.keys) == set(_MEANS)
    # The design was built at the targets, so each target sits within its predictive range.
    coverage = marginal_coverage(result.predictive_ranges, _targets())
    assert all(coverage.values())


def test_gate_failed_has_no_predictive_ranges():
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=40, seed=0, gate_fn=_fail_gate)
    assert result.status == "gate_failed"
    assert result.predictive_ranges is None


def test_include_predictive_false_skips_predictive():
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=100, seed=0, gate_fn=_pass_gate,
                                 include_predictive=False)
    assert result.status == "ok"
    assert result.predictive_ranges is None
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q -k "predictive"`
Expected: FAIL — `AttributeError: 'UQResult' object has no attribute 'predictive_ranges'` (or `TypeError` on the `include_predictive` kwarg).

- [ ] **Step 3: Wire the predictive layer into `run.py`**

In `osmose/calibration/uq/run.py`, add the import to the top block:

```python
from osmose.calibration.uq.predictive import EmulatorPredictiveRanges, posterior_predictive
```

Add the field to `UQResult` (after `posterior_mean`):

```python
    predictive_ranges: EmulatorPredictiveRanges | None = None
```

Add `include_predictive: bool = True` to `run_surrogate_bayes`'s keyword-only params (e.g. after `sampler`), and replace the final sampled-run return with a version that computes the predictive ranges:

```python
    sampler = sampler if sampler is not None else DynestySampler()
    sampler_result = sampler.sample(log_posterior, free_params, seed=seed)
    status = "ok" if sampler_result.converged else "sampled_not_converged"
    predictive_ranges = None
    if include_predictive:
        predictive_ranges = posterior_predictive(
            sampler_result, emulators, keys, sigma_seed_sq, seed=seed
        )
    return UQResult(
        status=status, gate_reports=growth.reports, design=growth.design,
        n_censored=n_censored, sampler_result=sampler_result,
        posterior_mean=sampler_result.posterior_mean(),
        predictive_ranges=predictive_ranges,
    )
```

(Update `run_surrogate_bayes`'s docstring to mention it also returns predictive ranges by default.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q`
Expected: PASS — 13 passed (10 existing + 3 new). (The `_pass_gate` runs are ~1s each; the predictive step is a cheap add.)

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/run.py tests/test_uq_run.py
.venv/bin/ruff check osmose/calibration/uq/run.py tests/test_uq_run.py
git add osmose/calibration/uq/run.py tests/test_uq_run.py
git commit -m "feat(uq): wire predictive ranges into run_surrogate_bayes/UQResult (Phase 3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 3 Done Check

After Task 3:

```bash
.venv/bin/python -m pytest tests/test_uq_predictive.py tests/test_uq_run.py -q
.venv/bin/python -m pytest tests/test_uq_prior.py tests/test_uq_likelihood.py tests/test_uq_posterior.py tests/test_uq_posterior_synthetic.py tests/test_uq_gate.py tests/test_uq_design.py tests/test_uq_growth.py tests/test_uq_sampler.py -q
```

Expected: all pass. The second command confirms Phase 3 did not disturb Phases 0–2c.

Milestone: `run_surrogate_bayes` now returns, alongside the posterior, per-species **marginal** emulator-predictive ranges (a structurally-marginal-only, weight-honoring, no-Jensen diagnostic) with a per-species marginal-coverage check and a θ-mediated cross-species correlation — completing Goal 2 as a labeled diagnostic. Real-data validation and a genuine trophic PPC are documented deferrals.

## Beyond Phase 3 (the surrogate-Bayesian UQ layer is feature-complete for the spec's v1)

- **Real-data validation** (held-out OSMOSE, emulator-in-the-loop) — the required-before-trusting-output step; needs engine runs, so it is a production/operational procedure, not library code.
- **The spec's honest-limitation caveats hold**: v1 under-coverage vs reality is real and largely unmeasurable within v1; the predictive layer is emulator-coverage-measured only. These belong in user-facing docs when the layer is surfaced (UI/CLI/report).
- **Open questions** (correlated multi-output GP, full δ(θ), dimension-reduction front-end for 30–50 params, persisted design/emulator reuse) remain future work per the spec.

## Self-Review (completed during authoring)

- **Spec coverage (Phase 3 row):** `predictive.py` genuine per-θ mixture + per-species marginal ranges → Task 1; cross-species correlation (diagnostic) → Task 1; the marginal-coverage check → Task 2; `+ UQResult` completion → Task 3. Both-likelihood width report + real-data validation → documented deferrals (usage / engine-runs).
- **Advisor-mandated points, all in the code:** resample by weight (Task 1, `rng.choice(..., p=weights)`); no Jensen in `y`; reuse the same σ_seed² (Task 3 passes the one `derive_sigma_seed_sq` result to both); structural marginal-only (no joint field, verified by `test_..._marginal_only_no_joint_field`); cross-species labeled as θ-mediated.
- **Type consistency:** `posterior_predictive` consumes `SamplerResult` (`samples`/`weights`) and injected `predict(X)->(mean,var)` emulators with their real signatures; `run.py` passes the SAME `emulators`/`sigma_seed_sq` to `posterior_predictive` as to `make_log_posterior`; `EmulatorPredictiveRanges` fields are used identically across `marginal_coverage` and the tests.
- **Placeholder scan:** none — every code step is complete. `n_draws=4000`, `level=0.9`, and the mixture math are the verified prototype values.

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Tasks 1–2 are fast (hand-built emulators/ranges, no sampling); Task 3's tests run a full injected-gate pipeline (~1s each) plus the cheap predictive step.
2. **Inline Execution** — execute tasks in this session with checkpoints.
