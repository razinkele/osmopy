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
