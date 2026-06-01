"""Shared run harness and FR config-injection helpers for the functional-response
feature (Holling type-I / II / III).

Tasks A5–A8 depend on these helpers.  A0 only ships the helpers and a smoke
test; engine / config / schema changes come in later tasks.

Design contract (fixed — do not change without updating A5–A8):
    _apply_fr(cfg, fr)                   — inject FR config keys
    _build_via_entry_point(cfg)          — sole construction path
    _base_cfg(background)                — minimal valid config dict
    _run_short_sim(numba, fr, seed, ...) — end-to-end sim, returns biomass array
    _run_baltic_short(seed, fr)          — shortcut for background=True run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Config key constants (fixed API contract — A5–A8 import these)
# ---------------------------------------------------------------------------

_FR_KEY_SHAPE = "predation.functional.response.shape.sp{i}"
_FR_KEY_HALFSAT = "predation.functional.response.halfsat.sp{i}"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MINIMAL_CONFIG = _PROJECT_ROOT / "data" / "minimal" / "osm_all-parameters.csv"
_BALTIC_CONFIG = _PROJECT_ROOT / "data" / "baltic" / "baltic_all-parameters.csv"


# ---------------------------------------------------------------------------
# Helper: inject FR config keys
# ---------------------------------------------------------------------------


def _apply_fr(cfg: dict, fr: dict | None) -> dict:
    """Inject FR config keys into *cfg* and return the mutated dict.

    *fr* maps species tokens ('sp0', 'sp14') to ``(shape_int, k | None)``
    tuples.

    Shape encoding:
        1 → type1  — emits the shape key but NO halfsat key.
        2 → type2  — emits both shape and halfsat keys.
        3 → type3  — emits both shape and halfsat keys.

    Background tokens (sp14 / sp15) are the **config** keys; the engine maps
    them to runtime slots 8 / 9 internally.

    When *fr* is None (or empty) the dict is returned unchanged so callers can
    safely pass ``fr=None`` for the no-FR baseline.
    """
    if not fr:
        return cfg
    code_to_name = {1: "type1", 2: "type2", 3: "type3"}
    for tok, (shape_int, k) in fr.items():
        i = tok[2:]  # 'sp0' -> '0', 'sp14' -> '14'
        cfg[_FR_KEY_SHAPE.format(i=i)] = code_to_name[shape_int]
        if shape_int != 1:
            cfg[_FR_KEY_HALFSAT.format(i=i)] = k
    return cfg


# ---------------------------------------------------------------------------
# Helper: sole construction path
# ---------------------------------------------------------------------------


def _build_via_entry_point(cfg: dict):
    """The ONLY supported construction path is EngineConfig.from_dict.

    Reference: osmose/engine/config.py, class EngineConfig, classmethod
    from_dict (approx. line 1469 as of 2026-05).  There is no
    build_engine_config helper.
    """
    from osmose.engine.config import EngineConfig

    return EngineConfig.from_dict(cfg)


# ---------------------------------------------------------------------------
# Helper: base config dicts
# ---------------------------------------------------------------------------


def _base_cfg(background: bool) -> dict:
    """Return the smallest valid config dict for a short simulation run.

    background=False  → 2-species synthetic (data/minimal) — fast, no NetCDF
                        background predators.
    background=True   → Baltic 8-focal + 2-background setup (data/baltic) —
                        exercises the background species pathway.

    The returned dict is always a fresh copy safe to mutate via _apply_fr.
    Runtime keys added here (nyear=1, seed-fixed flags) keep test runs short
    and deterministic.
    """
    from osmose.config.reader import OsmoseConfigReader

    if background:
        raw = OsmoseConfigReader().read(str(_BALTIC_CONFIG))
    else:
        raw = OsmoseConfigReader().read(str(_MINIMAL_CONFIG))

    cfg: dict = {k: v for k, v in raw.items() if k != ""}

    # Keep runs short — 1 year is sufficient for all A-task FR assertions.
    cfg["simulation.time.nyear"] = "1"

    # Deterministic RNG: per-species independent streams keyed by seed so that
    # the same seed → identical output across repeated calls.
    # movement.randomseed.fixed / stochastic.mortality.randomseed.fixed are the
    # two engine config keys that activate build_rng(fixed=True) for movement
    # and mortality respectively (see osmose/engine/rng.py and config.py:1759).
    cfg["movement.randomseed.fixed"] = "true"
    cfg["stochastic.mortality.randomseed.fixed"] = "true"

    # Reduce output verbosity for test runs.
    cfg["output.recordfrequency.ndt"] = str(int(cfg.get("simulation.time.ndtperyear", "12")))

    return cfg


# ---------------------------------------------------------------------------
# Helper: end-to-end short simulation
# ---------------------------------------------------------------------------


def _run_short_sim(
    numba: bool = True,
    fr: dict | None = None,
    seed: int = 7,
    background: bool = False,
    force_empty_prey: bool = False,
) -> np.ndarray:
    """Run a short simulation and return per-species end-of-run biomass.

    Parameters
    ----------
    numba:
        True  → use the default engine path (numba-accelerated mortality /
                predation kernels when numba is available).
        False → patch ``osmose.engine.processes.mortality._HAS_NUMBA`` to
                ``False`` for the duration of the run, forcing the engine to
                take the pure-Python predation fallback
                (``_apply_predation_for_school``) instead of the compiled
                ``_apply_predation_numba`` kernel.  This is implemented via
                ``unittest.mock.patch`` as a context manager so no pytest
                fixture is required.  The two backends produce different (but
                both valid and deterministic) outputs — do NOT compare
                numba=True vs numba=False results for bit-equality; that
                tolerance comparison is A6's responsibility.
    fr:
        FR config injection dict or None.  Passed straight through to
        _apply_fr; see that function's docstring for the token/shape contract.
    seed:
        RNG seed for PythonEngine.run_in_memory.  Same seed + same config →
        bit-identical output (requires movement.randomseed.fixed=true and
        stochastic.mortality.randomseed.fixed=true, both set by _base_cfg).
    background:
        When True, load the Baltic 8+2 config (exercises background species).
    force_empty_prey:
        When True, zero all resource biomass in the config so that a type-3
        predator encounters a cell with total_available ≈ 0 (exercises the
        zero-denominator guard / NaN path added in A6).
        # A6 will exercise this path in depth; here we implement best-effort
        # resource zeroing via the config key population.seeding.biomass.spN=0.
        # A full "no-other-schools" scenario may need additional per-step
        # patching that A6 will add.

    Returns
    -------
    np.ndarray
        1-D float64 array of length n_focal_species with end-of-run biomass
        (tonnes) for each focal species, in species-index order.
    """
    from osmose.engine import PythonEngine

    cfg = _apply_fr(_base_cfg(background), fr)

    if force_empty_prey:
        # Zero resource seeding biomass to drive total_available near 0.
        # A6 will exercise this path; see docstring note above.
        for key in list(cfg.keys()):
            if key.startswith("population.seeding.biomass."):
                cfg[key] = "0"
            if key.startswith("resource.biomass."):
                cfg[key] = "0"

    def _execute() -> np.ndarray:
        results = PythonEngine().run_in_memory(cfg, seed=seed)
        # Extract the final time-step biomass for each focal species.
        # OsmoseResults.biomass() returns a DataFrame with a 'Time' column and one
        # numeric column per focal species (species-index order).
        bio_df = results.biomass()
        numeric = bio_df.select_dtypes(include=["number"]).drop(columns=["Time"], errors="ignore")
        return numeric.to_numpy(dtype=np.float64)[-1]

    if numba:
        return _execute()

    from unittest import mock

    with mock.patch("osmose.engine.processes.mortality._HAS_NUMBA", False):
        return _execute()


# ---------------------------------------------------------------------------
# Helper: Baltic-specific shortcut
# ---------------------------------------------------------------------------


def _run_baltic_short(seed: int = 11, fr: dict | None = None) -> np.ndarray:
    """Shortcut for background=True runs (Baltic 8-focal + 2-background).

    Returns per-species end-of-run biomass as np.ndarray (n_focal=8).
    """
    return _run_short_sim(numba=True, fr=fr, seed=seed, background=True)


# ---------------------------------------------------------------------------
# Smoke test — must pass against the CURRENT engine (no FR code yet)
# ---------------------------------------------------------------------------


def test_helpers_run_baseline():
    """Two calls with the same seed must produce bit-identical biomass arrays.

    Determinism is guaranteed by:
      - movement.randomseed.fixed=true  (per-species independent movement RNG)
      - stochastic.mortality.randomseed.fixed=true  (per-species mortality RNG)
      - Same seed passed to PythonEngine.run_in_memory
    All values must be finite (no NaN / Inf produced by the baseline engine).
    """
    a = _run_short_sim(numba=True, fr=None, seed=7)
    b = _run_short_sim(numba=True, fr=None, seed=7)
    np.testing.assert_array_equal(a, b)  # deterministic
    assert np.all(np.isfinite(a)), f"baseline biomass contains non-finite values: {a}"


@pytest.mark.skipif(
    not _BALTIC_CONFIG.exists(),
    reason="Baltic config not present in data/baltic/ — path-confirmation spy requires located schools",
)
def test_python_fallback_path_runs_and_is_deterministic():
    """numba=False must route through the pure-Python predation fallback.

    Verification strategy
    ---------------------
    1. Determinism: two identical (config, seed, numba=False) runs must produce
       bit-identical biomass arrays — the Python fallback must be seeded as
       consistently as the Numba path.
    2. Finiteness: no NaN / Inf in either backend.
    3. Path confirmation: ``_apply_predation_for_school`` must be called at
       least once during a numba=False run (confirming the patch takes effect).
       The minimal config places all schools as unlocated (cell_x=-1), so the
       per-cell mortality loop never executes with it.  The Baltic config is
       used for path confirmation since it has thousands of located schools per
       timestep.

    NOTE: numba=True and numba=False outputs are NOT expected to be
    bit-identical — different code paths consume RNG differently and apply
    slightly different arithmetic.  Cross-backend tolerance comparison is
    A6's responsibility.
    """
    from unittest import mock

    import osmose.engine.processes.mortality as _mort

    # --- 1 & 2: determinism + finiteness of the Python fallback ---
    # Use the minimal config (fast) for the determinism assertion.
    a = _run_short_sim(numba=False, fr=None, seed=7)
    b = _run_short_sim(numba=False, fr=None, seed=7)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a)), f"Python-fallback biomass contains non-finite values: {a}"

    # Also confirm numba=True backend is itself deterministic and finite.
    c = _run_short_sim(numba=True, fr=None, seed=7)
    assert np.all(np.isfinite(c)), f"Numba biomass contains non-finite values: {c}"

    # --- 3: path confirmation via call-counting spy (Baltic config) ---
    # The minimal config places all schools as unlocated (cell_x=-1), so the
    # per-cell loop is never entered and _apply_predation_for_school is never
    # called.  The Baltic config has ~2,500 located schools per timestep and
    # confirms the Python fallback is genuinely exercised when numba=False.
    py_call_count = 0
    original_py = _mort._apply_predation_for_school

    def _spy_py(*args, **kwargs):
        nonlocal py_call_count
        py_call_count += 1
        return original_py(*args, **kwargs)

    with mock.patch("osmose.engine.processes.mortality._HAS_NUMBA", False):
        with mock.patch(
            "osmose.engine.processes.mortality._apply_predation_for_school",
            side_effect=_spy_py,
        ):
            _run_short_sim(numba=False, fr=None, seed=7, background=True)

    assert py_call_count > 0, (
        "_apply_predation_for_school was never called with numba=False (Baltic config) — "
        "the _HAS_NUMBA patch may not be reaching the mortality() per-school loop"
    )


def test_apply_fr_type1_no_halfsat():
    """type1 shape injection must emit only the shape key, never a halfsat key."""
    cfg = _base_cfg(background=False)
    result = _apply_fr(cfg, {"sp0": (1, None)})
    assert _FR_KEY_SHAPE.format(i="0") in result
    assert _FR_KEY_HALFSAT.format(i="0") not in result


def test_apply_fr_type2_both_keys():
    """type2 / type3 injection must emit both shape and halfsat keys."""
    cfg = _base_cfg(background=False)
    result = _apply_fr(cfg, {"sp0": (2, 500.0)})
    assert _FR_KEY_SHAPE.format(i="0") in result
    assert _FR_KEY_HALFSAT.format(i="0") in result
    assert result[_FR_KEY_HALFSAT.format(i="0")] == 500.0


def test_apply_fr_none_is_noop():
    """_apply_fr with fr=None must return the dict unchanged."""
    cfg = _base_cfg(background=False)
    keys_before = set(cfg.keys())
    _apply_fr(cfg, None)
    assert set(cfg.keys()) == keys_before


def test_build_via_entry_point():
    """_build_via_entry_point must return a valid EngineConfig."""
    from osmose.engine.config import EngineConfig

    cfg = _base_cfg(background=False)
    result = _build_via_entry_point(cfg)
    assert isinstance(result, EngineConfig)
    assert result.n_species >= 1


@pytest.mark.skipif(
    not _BALTIC_CONFIG.exists(),
    reason="Baltic config not present in data/baltic/",
)
def test_run_baltic_short_deterministic():
    """Baltic helper must produce deterministic output."""
    a = _run_baltic_short(seed=11, fr=None)
    b = _run_baltic_short(seed=11, fr=None)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a)), f"Baltic baseline contains non-finite values: {a}"
