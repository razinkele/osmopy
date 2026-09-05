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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from osmose.calibration.problem import FreeParameter

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
        lo = np.array(
            [_weighted_quantile(self.samples[:, j], self.weights, lo_q) for j in range(d)]
        )
        hi = np.array(
            [_weighted_quantile(self.samples[:, j], self.weights, hi_q) for j in range(d)]
        )
        return lo, hi

    def _cov(self) -> np.ndarray:
        return np.cov(self.samples.T, aweights=self.weights)

    def marginal_sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self._cov()))

    def correlation(self) -> np.ndarray:
        cov = self._cov()
        sd = np.sqrt(np.diag(cov))
        return cov / np.outer(sd, sd)


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
        maxiter: int | None = None,
        maxcall: int | None = None,
    ) -> None:
        self.nlive = nlive
        self.dlogz = dlogz
        self.ess_min = ess_min
        self.max_dim = max_dim
        self.maxiter = maxiter
        self.maxcall = maxcall

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

        # dynesty ships no stubs, so pyright infers `NestedSampler` from the
        # source, where it is a class whose __init__ takes internal positionals
        # (live_points/sampling/bounding) built by the public factory path — the
        # documented user-facing signature (ndim/nlive/rstate) is what actually
        # runs. Same story for the `Results` attribute access below: `samples`,
        # `logz` and `logzerr` are populated dynamically, which is why the
        # neighbouring `res.importance_weights()` type-checks and these do not.
        sampler = dynesty.NestedSampler(
            log_posterior,
            prior_transform,
            ndim=n_dim,
            nlive=self.nlive,  # type: ignore[call-arg]
            rstate=np.random.default_rng(seed),
        )
        sampler.run_nested(
            print_progress=False,
            dlogz=self.dlogz,
            maxiter=self.maxiter,
            maxcall=self.maxcall,
        )
        res = sampler.results

        weights = np.asarray(res.importance_weights())
        ess = float(1.0 / np.sum(weights**2))  # Kish ESS (version-robust)
        return SamplerResult(
            samples=np.asarray(res.samples),  # type: ignore[attr-defined]
            weights=weights,
            logz=float(res.logz[-1]),  # type: ignore[attr-defined]
            logz_err=float(res.logzerr[-1]),  # type: ignore[attr-defined]
            ess=ess,
            converged=ess >= self.ess_min,
        )
