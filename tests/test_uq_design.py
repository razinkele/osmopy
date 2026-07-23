"""Tests for the UQ design executor (helpers, seed reduction, engine evaluator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.design import (
    DesignResult,
    lhs_design,
    make_engine_evaluator,
    point_to_overrides,
    run_design,
)


def _params():
    return [
        FreeParameter("mortality.fishing.rate.sp0", 0.0, 2.0, Transform.LINEAR),
        FreeParameter("species.larva.mortality.rate.sp0", -3.0, 0.0, Transform.LOG),
    ]


def test_point_to_overrides_linear_passthrough():
    ov = point_to_overrides(np.array([1.5, -1.0]), _params())
    assert ov["mortality.fishing.rate.sp0"] == "1.5"


def test_point_to_overrides_log_is_base10():
    ov = point_to_overrides(np.array([1.5, -2.0]), _params())
    # LOG param: 10**(-2.0) == 0.01
    assert float(ov["species.larva.mortality.rate.sp0"]) == pytest.approx(0.01)


def test_point_to_overrides_all_keys_stringified():
    ov = point_to_overrides(np.array([0.3, -1.0]), _params())
    assert set(ov) == {"mortality.fishing.rate.sp0", "species.larva.mortality.rate.sp0"}
    assert all(isinstance(v, str) for v in ov.values())


def test_lhs_design_shape_and_bounds():
    X = lhs_design(_params(), n_points=25, seed=0)
    assert X.shape == (25, 2)
    assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 2.0)
    assert np.all(X[:, 1] >= -3.0) and np.all(X[:, 1] <= 0.0)


def test_lhs_design_deterministic():
    a = lhs_design(_params(), n_points=25, seed=7)
    b = lhs_design(_params(), n_points=25, seed=7)
    assert np.array_equal(a, b)
    c = lhs_design(_params(), n_points=25, seed=8)
    assert not np.array_equal(a, c)


_EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _linear_fp():
    return [FreeParameter("mortality.fishing.rate.sp0", 0.0, 1.0, Transform.LINEAR)]


def _const_evaluator(value_by_key):
    """Evaluator returning fixed per-key values with a tiny seed-dependent wobble."""

    def ev(x, seed):
        rng = np.random.default_rng(int(seed))
        out = {}
        for k, v in value_by_key.items():
            out[k] = float(v * np.exp(rng.normal(0.0, 0.05)))
        return out

    return ev


def test_run_design_reduces_log_mean_and_ddof1_alpha():
    # Deterministic evaluator: value depends only on seed, so we can hand-check.
    def ev(x, seed):
        return {"cod_biomass_mean": float(np.exp(0.1 * int(seed)))}

    res = run_design(
        ev, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=4, seed=0, seed_offset=0
    )
    # Point 0 uses run seeds 0..3 -> logs = [0, 0.1, 0.2, 0.3].
    logs = np.array([0.0, 0.1, 0.2, 0.3])
    assert res.Y["cod_biomass_mean"][0] == pytest.approx(logs.mean())
    assert res.alpha["cod_biomass_mean"][0] == pytest.approx(logs.var(ddof=1) / 4)


def test_run_design_censors_extinction_per_point():
    def ev(x, seed):
        # Point index is encoded via x[0]; extinct (0.0) only at x[0] < 0.5.
        val = 0.0 if x[0] < 0.5 else 10.0
        return {"cod_biomass_mean": float(val)}

    X = np.array([[0.2], [0.8], [0.9]])
    res = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=2, X=X)
    assert np.isnan(res.Y["cod_biomass_mean"][0])  # extinct -> censored
    assert not np.isnan(res.Y["cod_biomass_mean"][1])
    assert res.n_censored("cod_biomass_mean") == 1


def test_run_design_per_key_independent_censoring():
    def ev(x, seed):
        # cod extinct everywhere; herring healthy everywhere.
        return {"cod_ssb_mean": 0.0, "herring_biomass_mean": 100.0}

    res = run_design(
        ev,
        _linear_fp(),
        ["cod_ssb_mean", "herring_biomass_mean"],
        n_points=4,
        n_seeds=2,
        seed=1,
    )
    assert res.n_censored("cod_ssb_mean") == 4
    assert res.n_censored("herring_biomass_mean") == 0
    Xv, Yv, av = res.valid("herring_biomass_mean")
    assert len(Xv) == 4 and not np.any(np.isnan(Yv))


def test_run_design_reproducible():
    ev = _const_evaluator({"cod_biomass_mean": 10.0})
    a = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=5, n_seeds=3, seed=2)
    b = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=5, n_seeds=3, seed=2)
    assert np.array_equal(a.X, b.X)
    assert np.allclose(a.Y["cod_biomass_mean"], b.Y["cod_biomass_mean"], equal_nan=True)
    assert np.allclose(a.alpha["cod_biomass_mean"], b.alpha["cod_biomass_mean"], equal_nan=True)


def test_run_design_requires_two_seeds():
    ev = _const_evaluator({"cod_biomass_mean": 10.0})
    with pytest.raises(ValueError, match="n_seeds"):
        run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=1)


def test_design_result_valid_filters_censored_rows():
    Y = {"k": np.array([1.0, np.nan, 3.0])}
    alpha = {"k": np.array([0.1, np.nan, 0.3])}
    res = DesignResult(X=np.arange(3).reshape(-1, 1).astype(float), keys=["k"], Y=Y, alpha=alpha)
    Xv, Yv, av = res.valid("k")
    assert Xv.shape == (2, 1)
    assert np.array_equal(Yv, np.array([1.0, 3.0]))
    assert np.array_equal(av, np.array([0.1, 0.3]))


# ---- parallel design loop (opt-in batch-capable evaluator) ----


class _RecordingBatchEvaluator:
    """In-process double exposing both the callable and batch protocol; records use."""

    def __init__(self):
        self.batch_calls = 0

    def __call__(self, x, seed):
        return {"cod_biomass_mean": 10.0}

    def evaluate_batch(self, tasks):
        self.batch_calls += 1
        return [{"cod_biomass_mean": 10.0} for _ in tasks]


def test_run_design_dispatches_via_evaluate_batch_when_present():
    rec = _RecordingBatchEvaluator()
    res = run_design(rec, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=2, seed=0)
    assert rec.batch_calls >= 1  # routed through the batch API, not the serial loop
    assert not np.isnan(res.Y["cod_biomass_mean"]).any()


def _analytic_factory():
    """Picklable factory -> pure (x, seed) -> stats evaluator (no engine, real pool)."""

    def ev(x, seed):
        return {"cod_biomass_mean": float(np.exp(0.1 * (float(x[0]) + int(seed))))}

    return ev


def test_parallel_evaluator_batch_matches_serial_and_preserves_order():
    from osmose.calibration.uq.design import _ParallelEngineEvaluator

    tasks = [(i, np.array([0.1 * i]), i % 3) for i in range(6)]
    par = _ParallelEngineEvaluator(_analytic_factory, n_workers=2)
    try:
        got = par.evaluate_batch(tasks)
    finally:
        par.close()
    serial = _analytic_factory()
    expected = [serial(x, s) for _, x, s in tasks]
    assert got == expected  # values correct AND aligned to input task order


def test_run_design_parallel_equals_serial():
    from osmose.calibration.uq.design import _ParallelEngineEvaluator

    fps, keys = _linear_fp(), ["cod_biomass_mean"]
    serial = run_design(_analytic_factory(), fps, keys, n_points=5, n_seeds=3, seed=1)
    par_ev = _ParallelEngineEvaluator(_analytic_factory, n_workers=2)
    try:
        par = run_design(par_ev, fps, keys, n_points=5, n_seeds=3, seed=1)
    finally:
        par_ev.close()
    assert np.array_equal(serial.X, par.X)
    for k in keys:
        assert np.array_equal(serial.Y[k], par.Y[k], equal_nan=True)
        assert np.array_equal(serial.alpha[k], par.alpha[k], equal_nan=True)


def test_make_engine_evaluator_serial_has_no_batch_api():
    ev = make_engine_evaluator(_linear_fp(), _EXAMPLE_CONFIG, ["cod"], nyear=1)
    assert not hasattr(ev, "evaluate_batch")  # serial path unchanged


def test_make_engine_evaluator_parallel_exposes_batch_api():
    # n_workers>1 returns the batch-capable evaluator WITHOUT eagerly reading the
    # config or starting the pool (both are lazy / worker-side).
    ev = make_engine_evaluator(_linear_fp(), _EXAMPLE_CONFIG, ["cod"], nyear=1, n_workers=2)
    try:
        assert hasattr(ev, "evaluate_batch")
    finally:
        ev.close()


@pytest.mark.slow
def test_make_engine_evaluator_parallel_runs_real_engine():
    from osmose.config import OsmoseConfigReader

    cfg = OsmoseConfigReader().read(_EXAMPLE_CONFIG)
    n_sp = int(cfg.get("simulation.nspecies", "0"))
    species = [cfg.get(f"species.name.sp{i}") for i in range(n_sp)]
    fps = _linear_fp()
    with make_engine_evaluator(fps, _EXAMPLE_CONFIG, species, nyear=1, n_workers=2) as ev:
        res = run_design(ev, fps, [f"{species[0]}_biomass_mean"], n_points=2, n_seeds=2, seed=0)
    # Real engine + real process pool: keys produced, at least one point uncensored.
    assert not np.isnan(res.Y[f"{species[0]}_biomass_mean"]).all()


def test_engine_evaluator_emits_biomass_and_ssb_keys():
    from osmose.config import OsmoseConfigReader

    cfg = OsmoseConfigReader().read(_EXAMPLE_CONFIG)
    n_sp = int(cfg.get("simulation.nspecies", "0"))
    species = [cfg.get(f"species.name.sp{i}") for i in range(n_sp)]
    ev = make_engine_evaluator(_linear_fp(), _EXAMPLE_CONFIG, species, enable_ssb=True, nyear=1)
    stats = ev(np.array([0.3]), seed=1)
    # Biomass is always collected and non-zero.
    biomass_keys = [k for k in stats if k.endswith("_biomass_mean")]
    assert biomass_keys and all(stats[k] >= 0.0 for k in biomass_keys)
    # SSB plumbing: enabling output.ssb.enabled makes .ssb() readable, so _ssb_mean
    # keys are emitted. Values are 0.0 on this fixture at nyear=1 — assert presence,
    # NOT magnitude.
    assert any(k.endswith("_ssb_mean") for k in stats)
