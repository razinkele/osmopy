# Surrogate-Bayesian UQ — Phase 0: Emulator GP Substrate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone, unit-tested substrate of the UQ subpackage — target→output-key mapping, per-species output-stat extraction, and a single-output GP emulator with heteroscedastic noise and a variance-returning cross-validation — with zero OSMOSE runs and no new dependencies.

**Architecture:** A new `osmose/calibration/uq/` subpackage. `keying.py` maps a `BiomassTarget` to a *distinct* emulator output key (`biomass`≠`ssb`, unlike `losses.quantity_key` which collapses both to `_mean`). `output_stats.py` reduces one `OsmoseResults` to a per-species stat dict using those keys, tolerating outputs that are not enabled. `emulator.py` wraps its **own** sklearn GP (never touching the shared `SurrogateCalibrator`) with ARD `Matern(2.5)`, per-point noise `alpha` co-scaled for `normalize_y`, fit on natural-log outputs, returning latent posterior variance and a CV routine that surfaces per-fold predictive variances for the Phase 1 gate.

**Tech Stack:** Python 3.12+, NumPy, pandas, scikit-learn (`GaussianProcessRegressor`, `Matern`, `KFold`) — all already core dependencies. pytest.

## Global Constraints

- **Python 3.12+**; type hints on all public APIs.
- **Ruff line length 100**; run `.venv/bin/ruff format` and `.venv/bin/ruff check` on new files before each commit.
- **Test runner is `.venv/bin/python -m pytest`** (system `python` may not exist).
- **Do NOT mutate `SurrogateCalibrator`** or `osmose/calibration/surrogate.py`. The emulator builds its own GP.
- **`osmose/calibration/uq/__init__.py` stays thin** — no re-exports, no eager heavy imports. It must NOT be added to `osmose/calibration/__init__.py`'s imports or `__all__`. Tests and callers import from submodules directly (`from osmose.calibration.uq.emulator import GPEmulator`). This preserves the boundary that lets Phase 2's `emcee`/`dynesty` imports stay lazy.
- **Emulator expects `Y` already in natural-log units.** The log transform and seed-mean are `design.py`'s job (Phase 1). `output_stats.py` returns **linear-scale** means.
- **`alpha` co-scaling uses `np.var(Y, ddof=0)`** — matches sklearn's internal `np.std(ddof=0)` standardization. `ddof=1` injects an `n/(n-1)` error (~5% at N=20; Phase 1 runs LOO at small N).
- **`GPEmulator.predict()` returns the LATENT (noise-free) posterior variance**, in Y-units. It does NOT add training noise back. Phase 1's gate adds held-out seed-mean noise `s²/S` itself.
- **No new dependencies.** Everything used here is already in `[project.dependencies]`.

---

### Task 1: UQ keying + output-stats

Creates the subpackage skeleton, the target→key map, and the per-species stat extractor. These two modules share one contract — `target_to_output_key(t)` must return a key that `compute_uq_stats(...)` can emit — so a round-trip test spans both and they gate together.

**Files:**
- Create: `osmose/calibration/uq/__init__.py`
- Create: `osmose/calibration/uq/keying.py`
- Create: `osmose/calibration/uq/output_stats.py`
- Test: `tests/test_uq_output_stats.py`

**Interfaces:**
- Consumes: `osmose.calibration.targets.BiomassTarget` (fields `species: str`, `reference_point_type: str`); a duck-typed `results` object exposing `biomass()`, `ssb()`, `yield_biomass()`, each returning a wide `pandas.DataFrame` (a `Time` column plus one column per species) or raising when that output is absent — matching `osmose.results.OsmoseResults`.
- Produces:
  - `target_to_output_key(target: BiomassTarget) -> str` → `"{species}_biomass_mean"` / `"{species}_ssb_mean"` / `"{species}_yield_mean"`; raises `ValueError` on an unknown `reference_point_type`.
  - `compute_uq_stats(results, species_names: Sequence[str], n_eval_years: int = 10) -> dict[str, float]` → per-species linear-scale trailing-window means under those exact keys; omits keys for absent frames/columns; never raises on a missing frame.

- [ ] **Step 1: Create the thin subpackage `__init__.py`**

Create `osmose/calibration/uq/__init__.py`:

```python
"""Surrogate-based Bayesian UQ for OSMOSE calibration.

Thin subpackage boundary: no eager re-exports and no heavy imports here, so
later phases' optional dependencies (emcee, dynesty, arviz) stay lazy. Import
from submodules directly, e.g. ``from osmose.calibration.uq.emulator import
GPEmulator``.
"""
```

- [ ] **Step 2: Write the failing tests for keying + output_stats**

Create `tests/test_uq_output_stats.py`:

```python
"""Tests for UQ target-keying and per-species output-stat extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.output_stats import compute_uq_stats


class _StubResults:
    """Duck-typed OsmoseResults exposing wide biomass/ssb/yield frames.

    Pass ``None`` for a frame to simulate an output that is not enabled: the
    getter raises, mirroring strict-mode ``OsmoseResults``.
    """

    def __init__(self, biomass=None, ssb=None, yield_biomass=None):
        self._biomass = biomass
        self._ssb = ssb
        self._yield = yield_biomass

    def biomass(self):
        if self._biomass is None:
            raise FileNotFoundError("no biomass output")
        return self._biomass

    def ssb(self):
        if self._ssb is None:
            raise FileNotFoundError("no SSB output")
        return self._ssb

    def yield_biomass(self):
        if self._yield is None:
            raise FileNotFoundError("no yield output")
        return self._yield


def _wide(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"Time": np.arange(n), **cols})


def _target(species: str, rpt: str) -> BiomassTarget:
    return BiomassTarget(
        species=species, target=1.0, lower=0.5, upper=2.0, reference_point_type=rpt
    )


def test_keying_distinct_biomass_ssb_yield():
    assert target_to_output_key(_target("cod", "biomass")) == "cod_biomass_mean"
    assert target_to_output_key(_target("cod", "ssb")) == "cod_ssb_mean"
    assert target_to_output_key(_target("cod", "catch")) == "cod_yield_mean"
    # biomass and ssb must NOT collide (they do in losses.quantity_key).
    assert target_to_output_key(_target("cod", "biomass")) != target_to_output_key(
        _target("cod", "ssb")
    )


def test_keying_unknown_reference_point_type_raises():
    with pytest.raises(ValueError, match="unknown reference_point_type"):
        target_to_output_key(_target("cod", "wat"))


def test_output_stats_all_frames_present():
    n = 20
    bio = _wide(cod=np.full(n, 100.0), herring=np.full(n, 50.0))
    ssb = _wide(cod=np.full(n, 60.0))
    yld = _wide(cod=np.full(n, 10.0))
    results = _StubResults(biomass=bio, ssb=ssb, yield_biomass=yld)
    stats = compute_uq_stats(results, ["cod", "herring"], n_eval_years=10)
    assert stats["cod_biomass_mean"] == pytest.approx(100.0)
    assert stats["herring_biomass_mean"] == pytest.approx(50.0)
    assert stats["cod_ssb_mean"] == pytest.approx(60.0)
    assert stats["cod_yield_mean"] == pytest.approx(10.0)
    # herring has no ssb/yield column -> those keys are absent.
    assert "herring_ssb_mean" not in stats
    assert "herring_yield_mean" not in stats


def test_output_stats_missing_ssb_and_yield_frames_skipped():
    n = 20
    bio = _wide(cod=np.full(n, 100.0))
    results = _StubResults(biomass=bio, ssb=None, yield_biomass=None)
    stats = compute_uq_stats(results, ["cod"])
    assert stats == {"cod_biomass_mean": pytest.approx(100.0)}


def test_output_stats_trailing_window_ignores_early_years():
    vals = np.concatenate([np.zeros(5), np.full(10, 7.0)])  # 15 years
    results = _StubResults(biomass=_wide(cod=vals))
    stats = compute_uq_stats(results, ["cod"], n_eval_years=10)
    assert stats["cod_biomass_mean"] == pytest.approx(7.0)


def test_output_stats_shorter_than_window_uses_all_years():
    results = _StubResults(biomass=_wide(cod=np.full(4, 3.0)))
    stats = compute_uq_stats(results, ["cod"], n_eval_years=10)
    assert stats["cod_biomass_mean"] == pytest.approx(3.0)


def test_output_stats_roundtrip_with_keying():
    n = 12
    results = _StubResults(biomass=_wide(cod=np.full(n, 3.0)), ssb=_wide(cod=np.full(n, 2.0)))
    stats = compute_uq_stats(results, ["cod"])
    for rpt in ("biomass", "ssb"):
        assert target_to_output_key(_target("cod", rpt)) in stats
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_output_stats.py -q`
Expected: FAIL — collection/import error `ModuleNotFoundError: No module named 'osmose.calibration.uq.keying'`.

- [ ] **Step 4: Implement `keying.py`**

Create `osmose/calibration/uq/keying.py`:

```python
"""Map a calibration target to its distinct UQ emulator output-stat key."""

from __future__ import annotations

from osmose.calibration.targets import BiomassTarget

# Distinct per reference_point_type: biomass and ssb do NOT collide here (they
# both map to "_mean" in losses.quantity_key). The emulator needs one GP per
# distinct output, so biomass and ssb must be separately keyed.
_UQ_OUTPUT_SUFFIX = {
    "biomass": "_biomass_mean",
    "ssb": "_ssb_mean",
    "catch": "_yield_mean",
}


def target_to_output_key(target: BiomassTarget) -> str:
    """Return the UQ output-stat key a target is scored against.

    ``biomass`` -> ``"{species}_biomass_mean"``; ``ssb`` -> ``"{species}_ssb_mean"``;
    ``catch`` -> ``"{species}_yield_mean"``. Raises ``ValueError`` on an unknown
    ``reference_point_type``.
    """
    rpt = getattr(target, "reference_point_type", "biomass")
    try:
        suffix = _UQ_OUTPUT_SUFFIX[rpt]
    except KeyError:
        raise ValueError(
            f"unknown reference_point_type {rpt!r}; "
            f"expected one of {sorted(_UQ_OUTPUT_SUFFIX)}"
        ) from None
    return f"{target.species}{suffix}"
```

- [ ] **Step 5: Implement `output_stats.py`**

Create `osmose/calibration/uq/output_stats.py`:

```python
"""Reduce one OsmoseResults to a per-species UQ stat dict.

UQ-scoped: emits distinct ``{sp}_biomass_mean`` / ``{sp}_ssb_mean`` /
``{sp}_yield_mean`` keys (see ``keying.py``); does not touch
``losses.quantity_key`` so NSGA/DE scoring is unaffected. Values are
linear-scale trailing-window means — the natural-log and seed-mean transforms
happen later in ``design.py`` (Phase 1). SSB extraction is net-new relative to
``scripts/calibrate_baltic.py`` (which computes only mean/yield/cv/trend).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from osmose.results import OsmoseResults

# (getter-attribute-name, output-key-suffix) for each UQ output stat.
_UQ_OUTPUTS = (
    ("biomass", "_biomass_mean"),
    ("ssb", "_ssb_mean"),
    ("yield_biomass", "_yield_mean"),
)


def _read_frame(getter: Callable[[], pd.DataFrame]) -> pd.DataFrame | None:
    """Call a results getter, returning None if its output is absent/empty."""
    try:
        frame = getter()
    except Exception:  # noqa: BLE001 -- output not enabled/absent for this run: skip its stats
        return None
    if frame is None or frame.empty:
        return None
    return frame


def _trailing_mean(frame: pd.DataFrame | None, species: str, n_eval_years: int) -> float | None:
    """Mean of a species column over the last ``n_eval_years`` rows, or None."""
    if frame is None or species not in frame.columns:
        return None
    vals = frame[species].to_numpy(dtype=float)
    window = vals[-n_eval_years:] if len(vals) > n_eval_years else vals
    if window.size == 0:
        return None
    return float(np.mean(window))


def compute_uq_stats(
    results: OsmoseResults,
    species_names: Sequence[str],
    n_eval_years: int = 10,
) -> dict[str, float]:
    """Per-species linear-scale trailing-window means keyed for the UQ emulator.

    ``results`` must expose ``biomass()``, ``ssb()`` and ``yield_biomass()``,
    each returning a wide DataFrame (``Time`` + one column per species) or
    raising when that output is not enabled — the ``OsmoseResults`` contract,
    though only those three methods are used (any duck-typed object works, which
    is how the tests pass a stub). Outputs that are absent, empty, or lack a
    species column are silently skipped — their keys are omitted rather than
    raising, so a run without SSB/yield output still yields biomass stats.
    """
    frames = {name: _read_frame(getattr(results, name)) for name, _ in _UQ_OUTPUTS}
    stats: dict[str, float] = {}
    for species in species_names:
        for name, suffix in _UQ_OUTPUTS:
            mean = _trailing_mean(frames[name], species, n_eval_years)
            if mean is not None:
                stats[f"{species}{suffix}"] = mean
    return stats
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_output_stats.py -q`
Expected: PASS — 7 passed.

- [ ] **Step 7: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/ tests/test_uq_output_stats.py
.venv/bin/ruff check osmose/calibration/uq/ tests/test_uq_output_stats.py
git add osmose/calibration/uq/__init__.py osmose/calibration/uq/keying.py osmose/calibration/uq/output_stats.py tests/test_uq_output_stats.py
git commit -m "feat(uq): add target keying + per-species output-stats (Phase 0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: GP emulator — fit/predict with co-scaled heteroscedastic noise

The core of Phase 0: one GP per output stat, ARD `Matern(2.5)`, per-point noise `alpha` co-scaled for `normalize_y`, returning the latent posterior variance. The alpha-scaling correctness is pinned by a length-scale-invariance test (with a negative control) and a c²-covariance test on the predictive variance.

**Files:**
- Create: `osmose/calibration/uq/emulator.py`
- Test: `tests/test_uq_emulator.py`

**Interfaces:**
- Consumes: `sklearn.gaussian_process.GaussianProcessRegressor`, `sklearn.gaussian_process.kernels.Matern`.
- Produces:
  - `GPEmulator(n_restarts_optimizer: int = 2, random_state: int = 42)`
  - `.fit(X: np.ndarray, Y: np.ndarray, alpha: np.ndarray | float) -> GPEmulator` — `X` is `(n, d)` sampling-space inputs; `Y` is `(n,)` **natural-log** outputs; `alpha` is per-point noise variance `s²/S` (scalar broadcast allowed).
  - `.predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(mean, var)`, each `(n,)`, in Y-units; `var` is the **latent, noise-free** posterior variance.
  - `.gp` — the fitted `GaussianProcessRegressor` (exposes `.kernel_.length_scale`); `None` before `fit`.
  - Later relied on by Task 3's `.cross_validate` and by Phase 1's gate.

- [ ] **Step 1: Write the failing fit/predict tests**

Create `tests/test_uq_emulator.py`:

```python
"""Tests for the UQ GP emulator (fit/predict, alpha co-scaling)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from osmose.calibration.uq.emulator import GPEmulator


@pytest.fixture()
def design():
    """40-point 3-D design with a smooth log-space target and tiny noise."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    Y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] - X[:, 2]
    alpha = np.full(40, 1e-3)
    return X, Y, alpha


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GPEmulator().predict(np.zeros((1, 3)))


def test_fit_predict_returns_mean_and_variance(design):
    X, Y, alpha = design
    emu = GPEmulator(n_restarts_optimizer=0).fit(X, Y, alpha)
    mean, var = emu.predict(X)
    assert mean.shape == (40,)
    assert var.shape == (40,)
    assert np.all(var >= 0.0)
    # Low-noise GP ~interpolates its training points.
    assert np.sqrt(np.mean((mean - Y) ** 2)) < 0.1


def test_fit_accepts_scalar_alpha(design):
    X, Y, _ = design
    emu = GPEmulator(n_restarts_optimizer=0).fit(X, Y, 1e-3)
    mean, var = emu.predict(X[:5])
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_length_scale_invariant_under_y_rescaling(design):
    """Co-scaled alpha makes the standardized-space fit identical under Y->c*Y,
    so the ARD length scales (which live in X-space) are Y-unit invariant."""
    X, Y, alpha = design
    c = 10.0
    emu1 = GPEmulator(n_restarts_optimizer=0).fit(X, Y, alpha)
    # Rescaling Y by c scales its noise variance by c^2, hence raw alpha by c^2.
    emu2 = GPEmulator(n_restarts_optimizer=0).fit(X, c * Y, (c**2) * alpha)
    ls1 = np.atleast_1d(emu1.gp.kernel_.length_scale)
    ls2 = np.atleast_1d(emu2.gp.kernel_.length_scale)
    assert np.allclose(ls1, ls2, rtol=1e-3)


def test_predictive_variance_scales_with_y_units(design):
    """predict() variance is in Y-units: under Y->c*Y it scales by c^2 (correct
    covariant behavior), NOT invariant. Guards the spec's 'invariant' misphrasing."""
    X, Y, alpha = design
    c = 10.0
    emu1 = GPEmulator(n_restarts_optimizer=0).fit(X, Y, alpha)
    emu2 = GPEmulator(n_restarts_optimizer=0).fit(X, c * Y, (c**2) * alpha)
    Xt = np.array([[0.3, 0.4, 0.5], [0.1, 0.9, 0.2]])
    _, var1 = emu1.predict(Xt)
    _, var2 = emu2.predict(Xt)
    assert np.allclose(var2, (c**2) * var1, rtol=1e-3)


def test_without_coscaling_length_scales_drift(design):
    """Negative control: a plain GP fed RAW alpha at Y vs c*Y drifts — this is
    the failure mode co-scaling fixes."""
    X, Y, alpha = design
    c = 10.0

    def _raw_fit(y, a):
        kernel = Matern(length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=a, normalize_y=True, n_restarts_optimizer=0, random_state=42
        )
        gp.fit(X, y)
        return np.atleast_1d(gp.kernel_.length_scale)

    ls1 = _raw_fit(Y, alpha)  # raw alpha, no co-scaling
    ls2 = _raw_fit(c * Y, (c**2) * alpha)  # raw alpha at rescaled Y
    assert not np.allclose(ls1, ls2, rtol=1e-2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_emulator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.emulator'`.

- [ ] **Step 3: Implement `emulator.py` (fit/predict only)**

Create `osmose/calibration/uq/emulator.py`:

```python
"""Single-output GP emulator for one OSMOSE output stat.

Builds its OWN sklearn GP — it never touches the shared ``SurrogateCalibrator``
(the UI and ``find_optimum`` depend on that untouched). ARD ``Matern(2.5)``
(per-dimension length scales), per-point heteroscedastic noise via ``alpha``,
``normalize_y=True``, fit on natural-log outputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import Matern  # type: ignore[import-untyped]


class GPEmulator:
    """One GP emulating a single natural-log output stat over sampling-space X.

    ``predict`` returns the LATENT (noise-free) posterior mean and variance in
    the same natural-log units as ``Y``; it does not add the training noise
    ``alpha`` back. Phase 1's calibration gate adds the held-out seed-mean noise
    ``s²/S`` itself when standardizing residuals.
    """

    def __init__(self, n_restarts_optimizer: int = 2, random_state: int = 42) -> None:
        self._n_restarts_optimizer = n_restarts_optimizer
        self._random_state = random_state
        self.gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray | float) -> "GPEmulator":
        """Fit the GP. ``X`` is (n, d); ``Y`` is (n,) natural-log; ``alpha`` is
        per-point noise variance ``s²/S`` (scalar broadcast allowed)."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(Y.shape[0], float(alpha_arr))

        # sklearn adds `alpha` to the kernel diagonal UNSCALED, but normalize_y
        # standardizes Y by its std. Co-scale the raw noise into standardized-Y
        # units so it is correct post-normalization. ddof=0 matches sklearn's
        # internal np.std(ddof=0); ddof=1 would inject an n/(n-1) error.
        var_Y = float(np.var(Y, ddof=0))
        alpha_scaled = alpha_arr / var_Y if var_Y > 0.0 else alpha_arr

        kernel = Matern(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_scaled,
            normalize_y=True,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._random_state,
        )
        gp.fit(X, Y)
        self.gp = gp
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, var)`` in Y-units; ``var`` is latent (noise-free)."""
        if self.gp is None:
            raise RuntimeError("Must call fit() before predict()")
        mean, std = self.gp.predict(np.asarray(X, dtype=float), return_std=True)
        return mean, std**2
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_emulator.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
.venv/bin/ruff check osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
git add osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
git commit -m "feat(uq): add GP emulator fit/predict with co-scaled heteroscedastic noise (Phase 0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Variance-returning cross-validation

Adds a k-fold CV to `GPEmulator` that surfaces per-fold **predictive variances** (not just RMSE/R²), which Phase 1's calibration gate needs to standardize held-out residuals.

**Files:**
- Modify: `osmose/calibration/uq/emulator.py` (add `cross_validate` method to `GPEmulator`)
- Test: `tests/test_uq_emulator.py` (append cases)

**Interfaces:**
- Consumes: `sklearn.model_selection.KFold`; `GPEmulator.fit`/`predict` from Task 2.
- Produces: `.cross_validate(X, Y, alpha, k_folds: int = 5, seed: int = 42) -> dict` with keys `y_true`, `y_pred`, `pred_var` (each `(n,)` in fold-concatenation order — NOT input order), `fold_rmse`, `fold_r2` (length `k_folds`), `mean_rmse`, `mean_r2` (floats). Raises `ValueError` if `len(X) < k_folds`.

- [ ] **Step 1: Write the failing CV tests**

Append to `tests/test_uq_emulator.py`:

```python
def test_cross_validate_returns_per_fold_variances(design):
    X, Y, alpha = design
    cv = GPEmulator(n_restarts_optimizer=0).cross_validate(X, Y, alpha, k_folds=5, seed=0)
    n = len(X)
    assert cv["y_true"].shape == (n,)
    assert cv["y_pred"].shape == (n,)
    assert cv["pred_var"].shape == (n,)
    assert np.all(cv["pred_var"] >= 0.0)
    assert len(cv["fold_rmse"]) == 5
    assert len(cv["fold_r2"]) == 5
    assert isinstance(cv["mean_rmse"], float)
    assert isinstance(cv["mean_r2"], float)
    # A smooth low-noise target should cross-validate well.
    assert cv["mean_r2"] > 0.8


def test_cross_validate_accepts_scalar_alpha(design):
    X, Y, _ = design
    cv = GPEmulator(n_restarts_optimizer=0).cross_validate(X, Y, 1e-3, k_folds=4, seed=1)
    assert cv["pred_var"].shape == (len(X),)


def test_cross_validate_too_few_samples_raises():
    X = np.zeros((3, 2))
    Y = np.zeros(3)
    alpha = np.full(3, 1e-3)
    with pytest.raises(ValueError, match="k_folds"):
        GPEmulator().cross_validate(X, Y, alpha, k_folds=5)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_emulator.py -q -k cross_validate`
Expected: FAIL — `AttributeError: 'GPEmulator' object has no attribute 'cross_validate'`.

- [ ] **Step 3: Add the `KFold` import**

In `osmose/calibration/uq/emulator.py`, add below the existing sklearn imports:

```python
from sklearn.model_selection import KFold  # type: ignore[import-untyped]
```

- [ ] **Step 4: Implement `cross_validate`**

Add this method to the `GPEmulator` class in `osmose/calibration/uq/emulator.py` (after `predict`):

```python
    def cross_validate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: np.ndarray | float,
        k_folds: int = 5,
        seed: int = 42,
    ) -> dict:
        """K-fold CV returning per-fold predictive variances for the Phase 1 gate.

        Fits a fresh emulator per fold and predicts on the held-out points.
        Returns ``y_true``/``y_pred``/``pred_var`` (each concatenated in fold
        order, not input order) plus per-fold and mean RMSE/R². ``pred_var`` is
        the latent predictive variance; the gate adds held-out seed-mean noise.
        Raises ``ValueError`` if ``len(X) < k_folds``.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(Y.shape[0], float(alpha_arr))
        if len(X) < k_folds:
            raise ValueError(f"Need at least k_folds={k_folds} samples, got {len(X)}")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        pred_var: list[np.ndarray] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []

        for train_idx, test_idx in kf.split(X):
            fold = GPEmulator(self._n_restarts_optimizer, self._random_state)
            fold.fit(X[train_idx], Y[train_idx], alpha_arr[train_idx])
            mean, var = fold.predict(X[test_idx])
            truth = Y[test_idx]

            y_true.append(truth)
            y_pred.append(mean)
            pred_var.append(var)
            fold_rmse.append(float(np.sqrt(np.mean((truth - mean) ** 2))))
            ss_res = float(np.sum((truth - mean) ** 2))
            ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
            fold_r2.append(1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0)

        return {
            "y_true": np.concatenate(y_true),
            "y_pred": np.concatenate(y_pred),
            "pred_var": np.concatenate(pred_var),
            "fold_rmse": fold_rmse,
            "fold_r2": fold_r2,
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_r2": float(np.mean(fold_r2)),
        }
```

- [ ] **Step 5: Run the full emulator test file to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_uq_emulator.py -q`
Expected: PASS — 9 passed.

- [ ] **Step 6: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
.venv/bin/ruff check osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
git add osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
git commit -m "feat(uq): add variance-returning cross-validation to GP emulator (Phase 0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 0 Done Check

After Task 3, run the whole UQ suite and confirm the subpackage boundary is clean:

```bash
.venv/bin/python -m pytest tests/test_uq_output_stats.py tests/test_uq_emulator.py -q
.venv/bin/python -m pytest tests/test_calibration_surrogate.py tests/test_calibration_losses.py -q
```

Expected: all pass. The second command confirms Phase 0 did not disturb the existing surrogate/loss paths.

Milestone (from the spec): a distinct-keyed, natural-log emulator with a variance-returning CV, unit-tested, with no change to `SurrogateCalibrator` and `uq/` absent from the parent package's eager imports.

## Phase 1 Watch-Points (deferred here — NOT Phase 0 defects)

- **Emulator input scaling.** `GPEmulator` uses fixed `length_scale_bounds=(1e-2, 1e2)` and does not normalize `X`. Phase 0 tests use `X ∈ [0, 1]`, so this is fine. When `design.py` (Phase 1) builds real sampling-space `X`, `LINEAR`-transform parameters can have large natural ranges that push the fitted length scale against the upper bound. Phase 1 should either scale `X` to the unit hypercube before fitting or widen/relax the bounds, and the coverage/PIT gate will catch a badly-fit emulator regardless.
- **Wide-frame format assumption.** `compute_uq_stats` reads species as *columns* (`frame[species]`), matching how `scripts/calibrate_baltic.py:263` already consumes `results.biomass()`. This holds for single-file, per-species-column OSMOSE output (the Baltic setup). If Phase 1 ever encounters per-species-*file* output (species carried as a row tag, values in a differently-named column), the per-species column lookup would silently miss — add a format check when `design.py` first wires a real `OsmoseResults` through.
- **Intentional spec deviation (emulator unit test).** The spec's Testing row states "predictive variance invariant to arbitrary y-rescaling when `alpha` is co-scaled." That is mathematically wrong for this implementation: with co-scaled `alpha`, the *length scales* are Y-unit invariant but the predictive *variance* correctly scales by `c²` (it carries Y's units). Task 2 asserts the true invariants — length-scale invariance + `c²`-covariance of predictive variance + a no-co-scaling negative control — instead of the spec's literal (incorrect) claim. Empirically verified: length scales identical to machine precision, variance scales by exactly `c²`.

### Added by the Phase 0 whole-branch review (carry into Phase 1)

- **`cross_validate` returns `pred_var` without a test-index mapping.** The gate must standardize each held-out residual by `emulator_var(θᵢ) + sᵢ²/S`, which requires knowing which input row each concatenated `pred_var` entry came from. Today that mapping is only recoverable by re-running `KFold(n_splits, shuffle=True, random_state=seed).split(X)`. It is deterministic and correct, but Phase 1 should have `cross_validate` also return the concatenated test-index array (or per-entry fold ids) so the gate aligns variances explicitly instead of re-deriving the split.
- **Defuse the broad-except silence at the gate.** `output_stats._read_frame` intentionally swallows any getter failure as "output absent" (user-adjudicated to keep). Phase 1 should (a) add one `compute_uq_stats` test against a real disk- or in-memory-backed `OsmoseResults`, and (b) have the calibration gate flag "expected output key missing" rather than proceed silently, so a genuine extraction failure cannot pass as a bogus low score.
- **Extinction/zero-biomass censoring at the log seam.** `_trailing_mean` passes NaN through and returns a linear `0` for a collapsed species; when `design.py` applies the natural-log transform, `log(0) = -inf`. Wire the spec's extinction censoring / seed-mean exclusion at that same seam so linear zeros never reach `log()`.

## Self-Review (completed during authoring)

- **Spec coverage (Phase 0 rows):** `output_stats.py` → Task 1; `keying.py` → Task 1; `emulator.py` ARD/`alpha`/log/var GP → Task 2; variance-returning CV → Task 3; "no change to `SurrogateCalibrator`" → Global Constraints + Done Check; distinct `_biomass_mean`/`_ssb_mean`/`_yield_mean` keys → Task 1 keying + tests; natural-log emulator contract → Task 2 docstring/constraints.
- **Out of Phase 0 scope (correctly deferred):** `design.py` seed loop / SSB-enabled runs, the coverage/PIT gate, `prior.py`, likelihoods, samplers, predictive layer, `[uq]` extra deps — all Phase 1–3.
- **Type consistency:** `GPEmulator.fit` returns `GPEmulator`; `predict` returns `(mean, var)` used identically in Task 3's `cross_validate`; `alpha` accepts array-or-scalar in `fit` and `cross_validate` alike; `compute_uq_stats` key strings match `target_to_output_key` suffixes (`_biomass_mean`/`_ssb_mean`/`_yield_mean`), verified by the round-trip test.
- **Placeholder scan:** none — every code step contains complete, runnable content.

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints for review.
