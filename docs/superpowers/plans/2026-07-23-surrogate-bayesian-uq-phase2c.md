# Surrogate-Bayesian UQ — Phase 2c: End-to-End Orchestrator (`run.py`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tie the whole UQ layer together — `run_surrogate_bayes(...)` runs a design, gates it, fits emulators, derives per-key seed variance, builds the posterior, samples it, and bundles a `UQResult` — with cheap-and-fatal misconfiguration failing loud **before** the expensive design, and the gate-failed case short-circuiting without sampling.

**Architecture:** One new module `osmose/calibration/uq/run.py`, plus a small `maxiter`/`maxcall` bound added to Phase 2b's `DynestySampler`. `run_surrogate_bayes` composes the existing pieces: `grow_until_calibrated` (Phase 1) → `fit_emulators` + `make_log_posterior` (Phase 2a) → `DynestySampler.sample` (Phase 2b). It takes the same **injectable evaluator** the layer uses (default: the real `make_engine_evaluator`), so the full pipeline is CI-testable with a synthetic evaluator; the real OSMOSE engine runs are production usage. **`gate_fn` is threaded through** so wiring tests use always-pass/always-fail gates (decoupling "does the orchestration compose" from "does a synthetic clear the real gate"), while one unmarked end-to-end test exercises the real-gate default path (`gate_fn=None`).

**Tech Stack:** Python 3.12+, NumPy, `dynesty` (Phase 2b). Composes `grow_until_calibrated`/`GrowthResult`/`DesignResult` (Phase 1), `fit_emulators`/`make_log_posterior` (Phase 2a), `DynestySampler`/`SamplerResult`/`check_dimension` (Phase 2b), `target_to_output_key`, `BiomassTarget`, `FreeParameter`. pytest.

## Global Constraints

- **Python 3.12+**; type hints on all public APIs. **Ruff line length 100** (default ruleset — F401/E402 active) — run `.venv/bin/ruff format` then `.venv/bin/ruff check` on new/changed files before each commit; place new imports at the **top** of each file.
- **Test runner is `.venv/bin/python -m pytest`** (system `python` may not exist). `dynesty` is installed.
- **No new dependencies.**
- **Fail fast, before the expensive design.** `run_surrogate_bayes`'s FIRST actions — before `grow_until_calibrated` (which is potentially thousands of real engine runs) — are `check_dimension(len(free_params))` and per-target validation (`lower < upper`; `reference_point_type` resolvable via `target_to_output_key`). These **raise** (caller misconfiguration), they do NOT become a `UQResult.status`.
- **`status` is for valid-run outcomes only:** `"ok"` (calibrated + sampler converged), `"gate_failed"` (design never calibrated → no posterior), `"sampled_not_converged"` (calibrated + sampled, but the sampler's convergence flag is False).
- **σ_seed² is derived in `run.py`, per-key, pooled:** `sigma_seed_sq[key] = n_seeds * mean(alpha[key]_valid)` — this is the mean per-run seed variance `s²` (since `alpha = s²/n_seeds`). Two documented assumptions: `n_seeds` is constant across the design (true today; a per-point-S future breaks the formula), and the pooled-per-key constant is the Phase 2 simplification. `DesignResult` is NOT extended — `run_surrogate_bayes` already holds `n_seeds`.
- **`gate_fn` is threaded through** `run_surrogate_bayes` to `grow_until_calibrated` (default `None` = real gate).
- **The gate-failed short-circuit is load-bearing:** a censored-out target key makes `grow` abort (`gate_failed`) before `make_log_posterior`, so its missing-key `KeyError` is never reached on a valid run.
- **Do NOT mutate any Phase 0/1/2a file.** `run.py` is new; the only prior-phase change is the additive `maxiter`/`maxcall` on `DynestySampler` (Task 1).

## Pinned knobs (from a verified end-to-end prototype)

| Knob | Value | Rationale |
|---|---|---|
| recovery `atol` | 0.12 | Recovery through the FITTED emulator (not analytic) is looser than Phase 2b's: prototype worst deviation 0.053 across 3 seeds; 0.12 is ~2× cushion (this is the highest-variance assertion in the layer — it compounds a GP hyperparameter fit with nested sampling, so it's the one most likely to drift on a dynesty/sklearn bump). |
| synthetic | `sin(1.5θ)` means | Monotonic over [0,1] (so identifiable — unlike `sin(3θ)`, which is bimodal) AND smooth (so GP-calibratable — unlike a linear mean, which pins length scales to the bound and fails the gate). |

Prototype evidence: injected always-pass gate + synthetic recovers θ* (worst dev 0.053 at N₀=40, 3 seeds); the **real-gate** default path (`gate_fn=None`) also recovers (dev 0.045–0.053, ~3–6s); σ_seed² derived 0.00083 ≈ true 0.0009; gate-failed returns no posterior; `maxcall=400` cuts samples 622→434; over-dimension and degenerate-band both raise before `grow`.

## Deferred to Phase 3 (recorded)

- **Predictive diagnostic + real-data validation** (the spec's Phase 3): per-species marginal predictive ranges, cross-species PPC, held-out / emulator-in-the-loop validation.
- **Richer convergence** than the ESS flag (dlogz-reached + boundary-pileup + local-emulator-variance diagnostics) and cross-run mode agreement.
- **A degenerate-posterior guard** (`correlation()` → nan on a zero-variance marginal) — surface a clear error if a real posterior collapses.

---

### Task 1: bound `DynestySampler` (`maxiter`/`maxcall`)

Address the Phase 2b watch-point: an unbounded `run_nested` could run forever on a pathological posterior. Add optional budget caps.

**Files:**
- Modify: `osmose/calibration/uq/sampler.py` (`DynestySampler.__init__` + `.sample`)
- Test: `tests/test_uq_sampler.py` (append)

**Interfaces:**
- Produces: `DynestySampler(nlive=200, dlogz=0.5, ess_min=100.0, max_dim=MAX_NOMINAL_DIM, maxiter=None, maxcall=None)`; `sample` passes `maxiter`/`maxcall` to `run_nested`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_uq_sampler.py`:

```python
def test_dynesty_maxcall_bounds_sample_count():
    lp = lambda t: -0.5 * float(np.sum((t - 0.5) ** 2)) / 0.05  # noqa: E731
    unbounded = DynestySampler().sample(lp, _fp2(), seed=0)
    bounded = DynestySampler(maxcall=400).sample(lp, _fp2(), seed=0)
    # A call budget cuts the run short -> fewer accumulated samples.
    assert len(bounded.samples) < len(unbounded.samples)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py::test_dynesty_maxcall_bounds_sample_count -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'maxcall'`.

- [ ] **Step 3: Add `maxiter`/`maxcall` to `DynestySampler`**

In `osmose/calibration/uq/sampler.py`, change `DynestySampler.__init__` to accept and store the two caps:

```python
    def __init__(
        self,
        nlive: int = 200,
        dlogz: float = 0.5,
        ess_min: float = 100.0,
        max_dim: int = MAX_NOMINAL_DIM,
        maxiter: int | None = None,
        maxcall: int | None = None,
    ) -> None:
        self.nlive = nlive
        self.dlogz = dlogz
        self.ess_min = ess_min
        self.max_dim = max_dim
        self.maxiter = maxiter
        self.maxcall = maxcall
```

And change the `run_nested` call in `sample` to pass them (dynesty treats `None` as unbounded):

```python
        sampler.run_nested(
            print_progress=False, dlogz=self.dlogz,
            maxiter=self.maxiter, maxcall=self.maxcall,
        )
```

- [ ] **Step 4: Run the sampler suite to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_uq_sampler.py -q`
Expected: PASS — 12 passed (11 existing + 1 new).

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
.venv/bin/ruff check osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
git add osmose/calibration/uq/sampler.py tests/test_uq_sampler.py
git commit -m "feat(uq): bound DynestySampler run_nested with maxiter/maxcall (Phase 2c)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `UQResult` + `derive_sigma_seed_sq`

The result container and the per-key seed-variance derivation — the deterministic substrate of `run.py`.

**Files:**
- Create: `osmose/calibration/uq/run.py`
- Test: `tests/test_uq_run.py`

**Interfaces:**
- Consumes: `DesignResult` (Phase 1); `GateReport` (Phase 1); `SamplerResult` (Phase 2b).
- Produces:
  - `@dataclass UQResult` with `status: str`, `gate_reports: dict[str, GateReport]`, `design: DesignResult`, `n_censored: dict[str, int]`, `sampler_result: SamplerResult | None = None`, `posterior_mean: np.ndarray | None = None`.
  - `derive_sigma_seed_sq(design: DesignResult, target_keys: Sequence[str], n_seeds: int) -> dict[str, float]` — `n_seeds * mean(alpha[key]_valid)` per key (0.0 when a key has no valid points).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_run.py`:

```python
"""Tests for the UQ end-to-end orchestrator (run_surrogate_bayes) + helpers."""

from __future__ import annotations

import numpy as np

from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.run import UQResult, derive_sigma_seed_sq


def test_derive_sigma_seed_sq_pools_per_key():
    # alpha = s^2 / n_seeds, so n_seeds * mean(alpha) recovers the pooled s^2.
    n_seeds = 5
    X = np.linspace(0, 1, 4).reshape(-1, 1)
    alpha = {"cod_biomass_mean": np.full(4, 0.02 / n_seeds)}  # s^2 = 0.02
    Y = {"cod_biomass_mean": np.zeros(4)}
    design = DesignResult(X=X, keys=["cod_biomass_mean"], Y=Y, alpha=alpha)
    out = derive_sigma_seed_sq(design, ["cod_biomass_mean"], n_seeds)
    assert out["cod_biomass_mean"] == np.float64(0.02)


def test_derive_sigma_seed_sq_uses_valid_points_only():
    # A censored (NaN) point must not enter the mean.
    n_seeds = 4
    X = np.array([[0.0], [1.0]])
    alpha = {"k": np.array([0.04 / n_seeds, np.nan])}
    Y = {"k": np.array([0.0, np.nan])}
    design = DesignResult(X=X, keys=["k"], Y=Y, alpha=alpha)
    out = derive_sigma_seed_sq(design, ["k"], n_seeds)
    assert out["k"] == np.float64(0.04)  # only the one valid point


def test_uqresult_gate_failed_has_no_posterior():
    design = DesignResult(X=np.zeros((1, 1)), keys=["k"], Y={"k": np.array([1.0])},
                          alpha={"k": np.array([0.1])})
    r = UQResult(status="gate_failed", gate_reports={}, design=design, n_censored={"k": 0})
    assert r.sampler_result is None
    assert r.posterior_mean is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.run'`.

- [ ] **Step 3: Create `run.py` (data + derivation)**

Create `osmose/calibration/uq/run.py`:

```python
"""End-to-end surrogate-Bayesian UQ orchestrator.

run_surrogate_bayes composes the layer: grow_until_calibrated (Phase 1) ->
fit_emulators + make_log_posterior (Phase 2a) -> DynestySampler (Phase 2b).
It takes the same injectable evaluator the layer uses (default: the real
make_engine_evaluator), so the pipeline is testable synthetically; real engine
runs are production usage. Cheap-and-fatal misconfiguration (over-dimension,
malformed targets) raises BEFORE the expensive design; the gate-failed case
short-circuits without sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.gate import GateReport
from osmose.calibration.uq.sampler import SamplerResult


@dataclass
class UQResult:
    """Bundle of a surrogate-Bayesian UQ run.

    ``status`` is one of ``"ok"`` (calibrated + converged), ``"gate_failed"``
    (design never calibrated -> no posterior), ``"sampled_not_converged"``
    (sampled but the convergence flag is False). ``sampler_result`` /
    ``posterior_mean`` are ``None`` when the gate failed.
    """

    status: str
    gate_reports: dict[str, GateReport]
    design: DesignResult
    n_censored: dict[str, int]
    sampler_result: SamplerResult | None = None
    posterior_mean: np.ndarray | None = None


def derive_sigma_seed_sq(
    design: DesignResult,
    target_keys: Sequence[str],
    n_seeds: int,
) -> dict[str, float]:
    """Per-key pooled seed variance ``s²`` = ``n_seeds * mean(alpha[key]_valid)``.

    ``alpha = s²/n_seeds`` (variance of the seed mean), so this recovers the mean
    per-run variance. Assumes ``n_seeds`` is constant across the design (a
    per-point-S future would break this); pooled-per-key is the Phase 2
    simplification. Censored (NaN) points are excluded via ``design.valid``.
    """
    out: dict[str, float] = {}
    for key in target_keys:
        _, _, alpha = design.valid(key)
        out[key] = n_seeds * float(np.mean(alpha)) if len(alpha) else 0.0
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q`
Expected: PASS — 3 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/run.py tests/test_uq_run.py
.venv/bin/ruff check osmose/calibration/uq/run.py tests/test_uq_run.py
git add osmose/calibration/uq/run.py tests/test_uq_run.py
git commit -m "feat(uq): add UQResult + derive_sigma_seed_sq (Phase 2c)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `run_surrogate_bayes` orchestrator + happy-path acceptance

The pipeline itself, plus the injected-always-pass-gate wiring test that proves the full composition recovers θ*.

**Files:**
- Modify: `osmose/calibration/uq/run.py` (add `run_surrogate_bayes`)
- Test: `tests/test_uq_run.py` (append)

**Interfaces:**
- Consumes: `check_dimension`, `DynestySampler` (Phase 2b); `grow_until_calibrated` (Phase 1); `fit_emulators`, `make_log_posterior` (Phase 2a); `target_to_output_key`; `derive_sigma_seed_sq`, `UQResult` (Task 2).
- Produces: `run_surrogate_bayes(evaluator, free_params, targets, n_seeds, *, n0, increment, n_max, seed=0, gate_fn=None, likelihood="gaussian", sigma_disc_sq=0.0, k_by_type=None, sampler=None) -> UQResult`. Validates dimension + targets up front (raises); grows; short-circuits to `gate_failed` if not calibrated; else fits, derives σ_seed², builds the posterior, samples, and returns `"ok"`/`"sampled_not_converged"`.

- [ ] **Step 1: Write the failing tests**

Add these imports to the **top** of `tests/test_uq_run.py` (with the existing imports):

```python
from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.gate import GateReport
from osmose.calibration.uq.run import run_surrogate_bayes
```

Append the shared synthetic + the happy-path tests (below the Task 2 tests):

```python
_THETA_STAR = np.array([0.3, 0.6])
_SIG_SEED = 0.03

# Monotonic over [0,1] (identifiable) AND smooth (GP-calibratable).
_MEANS = {
    "A_biomass_mean": lambda t: 2.0 + np.sin(1.5 * t[0]),
    "B_biomass_mean": lambda t: 1.0 + np.sin(1.5 * t[1]),
    "C_biomass_mean": lambda t: 0.5 + np.sin(1.2 * (t[0] + t[1])),
}


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _evaluator(x, seed):
    rng = np.random.default_rng(int(seed))
    return {k: float(np.exp(f(x) + rng.normal(0.0, _SIG_SEED))) for k, f in _MEANS.items()}


def _targets():
    ts = []
    for key, f in _MEANS.items():
        value = float(np.exp(f(_THETA_STAR) + 0.5 * _SIG_SEED**2))
        ts.append(BiomassTarget(species=key.split("_")[0], target=value,
                                lower=value * 0.8, upper=value * 1.2,
                                reference_point_type="biomass"))
    return ts


def _pass_gate(X, Y, alpha, **kw):
    return GateReport(len(X), 0.95, 1.0, 0.5, 0.9, 0.95, True, [])


def _fail_gate(X, Y, alpha, **kw):
    return GateReport(len(X), 0.5, 9.0, 0.001, 0.5, 0.9, False, ["synthetic-fail"])


def test_run_surrogate_bayes_recovers_theta_star():
    # Injected always-pass gate: tests the full grow->fit->sigma_seed->posterior->
    # sample composition recovers theta* WITHOUT hinging on a synthetic clearing
    # the real gate. Recovery is through a FITTED emulator (looser than Phase 2b).
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=100, seed=0, gate_fn=_pass_gate)
    assert result.status == "ok"
    assert np.allclose(result.posterior_mean, _THETA_STAR, atol=0.12)
    assert result.sampler_result.converged


def test_run_surrogate_bayes_result_fields_populated():
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=100, seed=0, gate_fn=_pass_gate)
    assert set(result.gate_reports) == set(_MEANS)
    assert set(result.n_censored) == set(_MEANS)
    assert result.sampler_result is not None
    assert result.design.X.shape[1] == 2
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q -k "recovers or fields_populated"`
Expected: FAIL — `ImportError: cannot import name 'run_surrogate_bayes'`.

- [ ] **Step 3: Add `run_surrogate_bayes`**

In `osmose/calibration/uq/run.py`, replace the top import block so it reads exactly this (Task 2's imports are extended in place — do NOT add second `from ...design import` / `from ...sampler import` lines):

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from osmose.calibration.problem import FreeParameter
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult, Evaluator, grow_until_calibrated
from osmose.calibration.uq.gate import GateReport
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.posterior import fit_emulators, make_log_posterior
from osmose.calibration.uq.sampler import DynestySampler, SamplerResult, check_dimension
```

Then append:

```python
def run_surrogate_bayes(
    evaluator: Evaluator,
    free_params: list[FreeParameter],
    targets: Sequence[BiomassTarget],
    n_seeds: int,
    *,
    n0: int,
    increment: int,
    n_max: int,
    seed: int = 0,
    gate_fn: Callable[..., GateReport] | None = None,
    likelihood: str = "gaussian",
    sigma_disc_sq: float = 0.0,
    k_by_type: dict[str, float] | None = None,
    sampler: DynestySampler | None = None,
) -> UQResult:
    """Run the full surrogate-Bayesian UQ pipeline; return a UQResult.

    Fails loud up front on caller misconfiguration (over-dimension, malformed
    target) BEFORE the expensive design. Grows a calibrated design (real gate by
    default; ``gate_fn`` injectable); short-circuits to ``"gate_failed"`` if it
    never calibrates; else fits emulators, derives per-key σ_seed², builds the
    posterior, and samples it.
    """
    # Fail-fast validation, BEFORE grow (potentially thousands of engine runs).
    check_dimension(len(free_params))
    for t in targets:
        if not (t.lower < t.upper):
            raise ValueError(
                f"target for species {t.species!r} has lower ({t.lower}) >= "
                f"upper ({t.upper}); a band requires lower < upper"
            )
    keys = [target_to_output_key(t) for t in targets]  # raises on unknown reference_point_type

    growth = grow_until_calibrated(
        evaluator, free_params, keys, n_seeds,
        n0=n0, increment=increment, n_max=n_max, seed=seed, gate_fn=gate_fn,
    )
    n_censored = {k: growth.design.n_censored(k) for k in keys}
    if growth.status != "calibrated":
        return UQResult(
            status="gate_failed", gate_reports=growth.reports,
            design=growth.design, n_censored=n_censored,
        )

    emulators = fit_emulators(growth.design)
    sigma_seed_sq = derive_sigma_seed_sq(growth.design, keys, n_seeds)
    log_posterior = make_log_posterior(
        emulators, targets, free_params,
        sigma_seed_sq_by_key=sigma_seed_sq, sigma_disc_sq=sigma_disc_sq,
        k_by_type=k_by_type, likelihood=likelihood,
    )
    sampler = sampler if sampler is not None else DynestySampler()
    sampler_result = sampler.sample(log_posterior, free_params, seed=seed)
    status = "ok" if sampler_result.converged else "sampled_not_converged"
    return UQResult(
        status=status, gate_reports=growth.reports, design=growth.design,
        n_censored=n_censored, sampler_result=sampler_result,
        posterior_mean=sampler_result.posterior_mean(),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q`
Expected: PASS — 5 passed. (The two new tests each run one full pipeline with an injected gate — ~1s each.)

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/run.py tests/test_uq_run.py
.venv/bin/ruff check osmose/calibration/uq/run.py tests/test_uq_run.py
git add osmose/calibration/uq/run.py tests/test_uq_run.py
git commit -m "feat(uq): add run_surrogate_bayes orchestrator with fail-fast validation (Phase 2c)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: acceptance — gate-failed, not-converged, fail-fast, and the real-gate path

The edge/failure paths plus the one honest end-to-end test that runs the **real** gate (`gate_fn=None`).

**Files:**
- Test: `tests/test_uq_run.py` (append)

**Interfaces:**
- Consumes: `run_surrogate_bayes`, `DynestySampler`, the synthetic helpers (Task 3).

- [ ] **Step 1: Write the failing/passing tests**

Add these imports to the **top** of `tests/test_uq_run.py` (with the existing imports):

```python
import pytest

from osmose.calibration.uq.sampler import DynestySampler
```

Append the tests (below the Task 3 tests):

```python
def test_gate_failed_returns_no_posterior():
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=40, seed=0, gate_fn=_fail_gate)
    assert result.status == "gate_failed"
    assert result.sampler_result is None and result.posterior_mean is None
    assert set(result.gate_reports) == set(_MEANS)  # reports still populated


def test_sampled_not_converged_status():
    # An unreachable ESS floor makes the sampler report not-converged; the run
    # still returns a posterior, flagged.
    sampler = DynestySampler(ess_min=1e9)
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=100, seed=0,
                                 gate_fn=_pass_gate, sampler=sampler)
    assert result.status == "sampled_not_converged"
    assert result.posterior_mean is not None  # posterior surfaced, just flagged


def test_rejects_over_dimension_before_grow():
    # 25 params > MAX_NOMINAL_DIM: must raise before running any design.
    big = [FreeParameter(f"p{i}.sp0", 0.0, 1.0) for i in range(25)]

    def _never(x, seed):
        raise AssertionError("evaluator must not run when the dimension cap trips")

    with pytest.raises(ValueError, match="exceeds"):
        run_surrogate_bayes(_never, big, _targets(), n_seeds=6, n0=40, increment=20, n_max=40)


def test_rejects_degenerate_band_before_grow():
    bad = [BiomassTarget("A", 10.0, 10.0, 10.0, reference_point_type="biomass")]

    def _never(x, seed):
        raise AssertionError("evaluator must not run when a target is malformed")

    with pytest.raises(ValueError, match="lower"):
        run_surrogate_bayes(_never, _fp2(), bad, n_seeds=6, n0=40, increment=20, n_max=40)


def test_real_gate_default_path_recovers_theta_star():
    # The honest end-to-end path: gate_fn=None runs the REAL calibration gate.
    # ~3-6s (GP cross-validation per round + nested sampling). Asserts the
    # decision (ok + recovery), not the design size (which varies with growth).
    result = run_surrogate_bayes(_evaluator, _fp2(), _targets(), n_seeds=6,
                                 n0=40, increment=20, n_max=100, seed=0)
    assert result.status == "ok"
    assert np.allclose(result.posterior_mean, _THETA_STAR, atol=0.12)
    assert result.sampler_result.converged
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q -k "gate_failed or not_converged or rejects or real_gate"`
Expected: PASS — 5 passed. (`test_real_gate_default_path_recovers_theta_star` runs the real gate + sampling, ~3–6s; the fail-fast tests are instant; if a threshold is close, do NOT weaken it — report the observed values.)

- [ ] **Step 3: Run the full run suite**

Run: `.venv/bin/python -m pytest tests/test_uq_run.py -q`
Expected: PASS — 10 passed.

- [ ] **Step 4: Lint/format and commit**

```bash
.venv/bin/ruff format tests/test_uq_run.py
.venv/bin/ruff check tests/test_uq_run.py
git add tests/test_uq_run.py
git commit -m "test(uq): acceptance for run_surrogate_bayes (gate-failed, not-converged, fail-fast, real-gate) (Phase 2c)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 2c Done Check

After Task 4:

```bash
.venv/bin/python -m pytest tests/test_uq_run.py tests/test_uq_sampler.py -q
.venv/bin/python -m pytest tests/test_uq_prior.py tests/test_uq_likelihood.py tests/test_uq_posterior.py tests/test_uq_posterior_synthetic.py tests/test_uq_gate.py tests/test_uq_design.py -q
```

Expected: all pass. The second command confirms Phase 2c did not disturb Phases 1/2a/2b.

Milestone: `run_surrogate_bayes` composes the full layer end-to-end (design → gate → fit → σ_seed² → posterior → sample → `UQResult`), fails loud on misconfiguration before the expensive design, short-circuits the gate-failed case, and recovers θ* on both the injected-gate wiring test and the real-gate default path — CI-testable via the synthetic evaluator, engine runs reserved for production.

## Phase 3 Watch-Points (carry forward)

- **`run.py` uses `make_engine_evaluator` in production** — the caller passes `make_engine_evaluator(free_params, base_config_path, species_names, enable_ssb=True)` (Phase 1) as `evaluator`. At the real scale (N₀=20·d, S=10) this is thousands of OSMOSE runs; the fail-fast dimension/target checks exist to not waste them.
- **`target.weight` still does not enter the posterior** (documented Phase 2a limitation) — `run.py` is where a user with a weighted CSV would notice; surface a note if weights are non-uniform.
- **Predictive layer (Phase 3):** `UQResult` currently carries the posterior over θ; the per-species predictive ranges + cross-species PPC + held-out/emulator-in-the-loop validation are Phase 3.
- **Convergence richness:** `status="sampled_not_converged"` is ESS-floor-only; Phase 3 should fold in dlogz-reached, boundary-pileup, and cross-run mode agreement.

## Self-Review (completed during authoring)

- **Spec coverage (Phase 2 run.py row):** `run.py` orchestrator returning a posterior-only partial `UQResult` → Tasks 2–3; the design→gate→fit→sample wiring → Task 3; fail-fast dimension cap + target validation → Task 3 (verified before `grow`); σ_seed² derivation → Task 2; gate-failed / not-converged outcomes → Task 4; the `run_nested` bound (Phase 2b watch-point) → Task 1.
- **Deferred (correctly):** the predictive diagnostic + real-data validation, richer convergence, degenerate-posterior guard — Phase 3.
- **Type consistency:** `run_surrogate_bayes` consumes `grow_until_calibrated`(→`GrowthResult.status`/`.reports`/`.design`), `fit_emulators`, `make_log_posterior`(`sigma_seed_sq_by_key`), `DynestySampler.sample`, `derive_sigma_seed_sq`, and `UQResult` with their real signatures; `gate_fn` matches `grow_until_calibrated`'s `Callable[..., GateReport]`; `Evaluator` is the Phase 1 `(np.ndarray, int) -> dict[str, float]` alias.
- **Placeholder scan:** none — every code step is complete. Synthetic (`sin(1.5θ)`), `atol=0.12`, and the derivation formula are the verified prototype values.

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Tasks 1–2 are fast; Task 3's two tests run a full injected-gate pipeline (~1s each); Task 4's `real_gate` test runs the real gate + sampling (~3–6s).
2. **Inline Execution** — execute tasks in this session with checkpoints.
