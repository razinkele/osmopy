"""Opt-in common random numbers for the NSGA-II calibration problem (perf plan item 5a).

By default ``OsmoseCalibrationProblem`` runs candidate i of a generation on ``seed=i``, its
position in the population. So two candidates differ by their parameters AND by their seed, and
the seed noise follows population position, not anything about the candidate. ``crn_seeds`` runs
every candidate in every generation on the same seed set and scores the mean, so candidates are
compared under common random numbers. The default is unchanged (past runs stay reproducible), and
the seed set enters the cache key only in this mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.calibration.objectives import BiomassRMSEObjective, _biomass_long
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem

_MINIMAL = Path(__file__).resolve().parent.parent / "data" / "minimal" / "osm_all-parameters.csv"
_FREE = [FreeParameter("predation.efficiency.critical.sp0", 0.3, 0.5)]


class _FakeResults:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _recording_problem(tmp_path, crn_seeds=None):
    p = OsmoseCalibrationProblem(
        free_params=_FREE,
        objective_fns=[lambda r: r.value],
        base_config_path=_MINIMAL,
        work_dir=tmp_path,
        crn_seeds=crn_seeds,
    )
    calls: list[tuple[str, int]] = []

    def fake_run(overrides, run_id, seed=None):
        used = run_id if seed is None else seed
        calls.append((overrides["predation.efficiency.critical.sp0"], used))
        return _FakeResults(10.0 * used)  # a seed-dependent objective

    p._run_python_engine = fake_run  # type: ignore[method-assign]
    return p, calls


def test_default_keeps_the_position_seed(tmp_path):
    p, calls = _recording_problem(tmp_path)
    out: dict = {}
    p._evaluate(np.array([[0.35], [0.40], [0.45]]), out)
    assert [s for _, s in calls] == [0, 1, 2]  # seed = position in the population
    np.testing.assert_array_equal(out["F"][:, 0], [0.0, 10.0, 20.0])


def test_crn_runs_every_candidate_on_the_same_seed_set_and_scores_the_mean(tmp_path):
    p, calls = _recording_problem(tmp_path, crn_seeds=(5, 7))
    out: dict = {}
    p._evaluate(np.array([[0.35], [0.40], [0.45]]), out)
    by_candidate: dict[str, list[int]] = {}
    for value, seed in calls:
        by_candidate.setdefault(value, []).append(seed)
    assert list(by_candidate.values()) == [[5, 7]] * 3  # same seeds, whatever the position
    np.testing.assert_array_equal(out["F"][:, 0], [60.0] * 3)  # mean(50, 70)


def test_crn_seed_set_enters_the_cache_key_only_in_crn_mode(tmp_path):
    plain = OsmoseCalibrationProblem(_FREE, [lambda r: 0.0], _MINIMAL, tmp_path)
    crn_a = OsmoseCalibrationProblem(_FREE, [lambda r: 0.0], _MINIMAL, tmp_path, crn_seeds=(1, 2))
    crn_b = OsmoseCalibrationProblem(_FREE, [lambda r: 0.0], _MINIMAL, tmp_path, crn_seeds=(1, 3))
    ov = {"predation.efficiency.critical.sp0": "0.4"}
    keys = {plain._cache_key(ov), crn_a._cache_key(ov), crn_b._cache_key(ov)}
    assert len(keys) == 3
    # The default key is the pre-CRN formula, so existing caches stay valid.
    import hashlib

    from osmose import __version__

    raw = f"{sorted(ov.items())}|python-{__version__}|{plain._base_config_hash}"
    assert plain._cache_key(ov) == hashlib.sha256(raw.encode()).hexdigest()


def test_crn_is_python_engine_only(tmp_path):
    with pytest.raises(ValueError, match="crn_seeds"):
        OsmoseCalibrationProblem(
            _FREE,
            [lambda r: 0.0],
            _MINIMAL,
            tmp_path,
            jar_path=tmp_path / "x.jar",
            use_java_engine=True,
            crn_seeds=(1, 2),
        )
    with pytest.raises(ValueError, match="crn_seeds"):
        OsmoseCalibrationProblem(_FREE, [lambda r: 0.0], _MINIMAL, tmp_path, crn_seeds=())


@pytest.mark.skipif(not _MINIMAL.exists(), reason="minimal config absent")
def test_crn_thread_and_process_backends_agree_bit_for_bit(tmp_path):
    base = OsmoseCalibrationProblem(_FREE, [lambda r: 0.0], _MINIMAL, tmp_path / "b")
    res = base._run_python_engine({}, run_id=0)
    with res as r:
        obs = _biomass_long(r.biomass())

    def make(backend, sub):
        return OsmoseCalibrationProblem(
            _FREE,
            [BiomassRMSEObjective(obs)],
            _MINIMAL,
            tmp_path / sub,
            n_parallel=2,
            parallel_backend=backend,
            crn_seeds=(3, 11),
        )

    X = np.array([[0.35], [0.45]])
    pt, pp = make("thread", "t"), make("process", "p")
    out_t: dict = {}
    out_p: dict = {}
    try:
        pt._evaluate(X, out_t)
        pp._evaluate(X, out_p)
    finally:
        pp.shutdown_pool()
    np.testing.assert_array_equal(out_t["F"], out_p["F"])
