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


# ---------------------------------------------------------------------------
# A1: schema field registration tests
# ---------------------------------------------------------------------------


def test_fr_schema_fields_registered():
    from osmose.schema import build_registry

    reg = build_registry()
    shape = reg.get_field("predation.functional.response.shape.sp{idx}")
    assert shape.param_type.value == "enum"
    assert shape.default == "type1"
    assert set(shape.choices) == {"type1", "type2", "type3"}
    halfsat = reg.get_field("predation.functional.response.halfsat.sp{idx}")
    assert halfsat.min_val == 0.1
    assert halfsat.max_val == 5.0


# ---------------------------------------------------------------------------
# A2: focal-species parse + strict validation tests
# ---------------------------------------------------------------------------


def test_fr_halfsat_required_when_shape_not_type1():
    cfg = _apply_fr(_base_cfg(background=False), {"sp0": (3, None)})  # type3, no halfsat
    with pytest.raises(ValueError, match="is required when"):
        _build_via_entry_point(cfg)


def test_fr_halfsat_out_of_range_raises():
    cfg = _base_cfg(background=False)
    cfg["predation.functional.response.shape.sp0"] = "type3"
    cfg["predation.functional.response.halfsat.sp0"] = 0.0
    with pytest.raises(ValueError, match="out of range"):
        _build_via_entry_point(cfg)


def test_fr_shape_invalid_enum_raises():
    cfg = _base_cfg(background=False)
    cfg["predation.functional.response.shape.sp0"] = "type9"
    with pytest.raises(ValueError, match="(?i)type9|not.*one of|invalid"):
        _build_via_entry_point(cfg)


# ---------------------------------------------------------------------------
# A3: background-species FR parse tests
# ---------------------------------------------------------------------------


def test_fr_shape_code_parity_across_modules():
    """_FR_SHAPE_CODE in background.py must match config.py — guards against drift."""
    from osmose.engine.background import _FR_SHAPE_CODE as B
    from osmose.engine.config import _FR_SHAPE_CODE as A

    assert A == B


def test_background_parse_sets_fr_fields():
    """parse_background_species populates fr_shape/fr_halfsat on BackgroundSpeciesInfo.

    This is the A3 acceptance criterion: background parsing works independently
    of A4 (EngineConfig concat).  We call parse_background_species directly with
    a minimal cfg that has sp14 declared as background + type3 FR.
    """
    from osmose.engine.background import parse_background_species

    cfg = {
        "species.type.sp14": "background",
        "simulation.nbackground": "1",
        "species.name.sp14": "GreySeal",
        "species.nclass.sp14": "1",
        "species.length.sp14": "100",
        "species.size.proportion.sp14": "1",
        "species.trophic.level.sp14": "3.5",
        "species.age.sp14": "1",
        "species.length2weight.condition.factor.sp14": "0.01",
        "species.length2weight.allometric.power.sp14": "3.0",
        "predation.predprey.sizeratio.max.sp14": "3.5",
        "predation.predprey.sizeratio.min.sp14": "1.0",
        "predation.ingestion.rate.max.sp14": "3.5",
        "predation.functional.response.shape.sp14": "type3",
        "predation.functional.response.halfsat.sp14": "1.0",
        "species.biomass.multiplier.sp14": "1.0",
        "species.biomass.offset.sp14": "0.0",
        "species.biomass.total.sp14": "1000.0",
    }

    species_list = parse_background_species(cfg, n_focal=8, n_dt_per_year=12)
    assert len(species_list) == 1
    sp = species_list[0]
    assert sp.fr_shape == 3, f"expected fr_shape=3 (type3), got {sp.fr_shape}"
    assert sp.fr_halfsat == 1.0, f"expected fr_halfsat=1.0, got {sp.fr_halfsat}"


@pytest.mark.skipif(
    not _BALTIC_CONFIG.exists(),
    reason="Baltic config not present in data/baltic/",
)
def test_fr_background_enum_maps_to_runtime_slot():
    """Background FR config key sp14=type3 must reach runtime slot 8 (n_focal+bkg_idx).

    NOTE: This test exercises the full A3+A4 pipeline.  It will fail until A4
    wires fr_shape/fr_halfsat into EngineConfig arrays — expect AttributeError
    or a missing-field error until then.  The background *parse* is validated
    independently by test_background_parse_sets_fr_fields (A3-only).
    """
    cfg = _apply_fr(_base_cfg(background=True), {"sp14": (3, 1.0)})
    ecfg = _build_via_entry_point(cfg)
    assert ecfg.fr_shape[8] == 3
    assert ecfg.fr_halfsat[8] == 1.0
    assert ecfg.fr_shape[9] == 1


# ---------------------------------------------------------------------------
# A4: EngineConfig fr_shape / fr_halfsat array wiring + validation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _BALTIC_CONFIG.exists(),
    reason="Baltic config not present in data/baltic/",
)
def test_fr_arrays_sized_n_total_and_registered():
    ecfg = _build_via_entry_point(_base_cfg(background=True))  # 8 focal + 2 bkg
    assert ecfg.fr_shape.dtype == np.int32
    assert ecfg.fr_halfsat.dtype == np.float64
    assert len(ecfg.fr_shape) == ecfg.n_species + ecfg.n_background == 10
    assert len(ecfg.fr_halfsat) == 10


@pytest.mark.skipif(
    not _BALTIC_CONFIG.exists(),
    reason="Baltic config not present in data/baltic/",
)
def test_fr_mis_sized_array_raises():
    ecfg = _build_via_entry_point(_base_cfg(background=True))
    kwargs = dict(ecfg.__dict__)
    kwargs["fr_shape"] = np.ones(ecfg.n_species, dtype=np.int32)  # wrong length
    with pytest.raises(ValueError, match="fr_shape"):
        type(ecfg)(**kwargs)  # re-runs __post_init__ length check


# ---------------------------------------------------------------------------
# A5: functional-response branch in the pure-Python predation kernel
# ---------------------------------------------------------------------------
#
# Oracle (mirrors the intended math in _apply_predation_for_school):
#   type-I (1): min(r, 1)
#   type-II (2): r/(r+k), then conservation-clamped to min(r, 1)
#   type-III(3): r²/(r²+k²), then conservation-clamped to min(r, 1)


def _g_ref(r, shape, k):
    if shape == 1:
        return min(r, 1.0)
    g = (r / (r + k)) if shape == 2 else ((r * r) / (r * r + k * k))
    return min(g, min(r, 1.0))  # conservation clamp


@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 5.0])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 0.9, 1.0, 2.0, 10.0])
def test_oracle_conservation(r, k):
    for shape in (2, 3):
        assert _g_ref(r, shape, k) <= min(r, 1.0) + 1e-12


def test_oracle_clamp_is_load_bearing():
    r, k = 0.5, 0.1
    raw = (r * r) / (r * r + k * k)
    assert raw > min(r, 1.0)  # raw type-III violates conservation here
    assert _g_ref(r, 3, k) == min(r, 1.0)  # clamp pulls it back to the ration cap


def test_oracle_anchors_and_limits():
    assert _g_ref(1.0, 2, 1.0) == pytest.approx(0.5)
    assert _g_ref(1.0, 3, 1.0) == pytest.approx(0.5)
    assert _g_ref(0.3, 1, 999) == 0.3 and _g_ref(5.0, 1, 999) == 1.0
    assert _g_ref(1e6, 2, 1.0) == pytest.approx(1.0, abs=1e-3)
    assert _g_ref(1e6, 3, 1.0) == pytest.approx(1.0, abs=1e-3)
    xs = [0.05, 0.2, 0.5, 1.0, 2.0]
    for shape in (2, 3):
        gs = [_g_ref(x, shape, 1.0) for x in xs]
        assert all(b >= a - 1e-12 for a, b in zip(gs, gs[1:]))


def test_oracle_type3_refuge_ratio_increasing():
    k = 1.0
    rs = [0.05, 0.1, 0.2, 0.4]
    ratios = [((x * x) / (x * x + k * k)) / x for x in rs]
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    assert _g_ref(0.05, 3, k) < _g_ref(0.05, 2, k)


def _run_single_predation_step_python(r: float, shape: int, k: float):
    """Drive _apply_predation_for_school once with a hand-built one-predator /
    one-prey scenario and return (eaten_total, max_eatable).

    Scenario construction (reuses the harness pattern from
    tests/test_engine_mortality_loop.py::TestUnifiedPredation):
      - sp1 = single big predator school, sp0 = single small prey school,
        co-located in cell (0, 0), no resources (n_resources == 0).
      - access_coeff defaults to 1.0 (has_access=False), so the predator's
        accessible pool is exactly total_available = prey_abundance * prey_weight.
      - max_eatable = pred_biomass * ingestion_rate / (n_dt_per_year * n_subdt).
      - Prey abundance is solved so total_available == r * max_eatable EXACTLY,
        giving the kernel a known ratio r at the injection point.
      - FR config arrays are mutated directly on the EngineConfig:
        fr_shape[sp_pred] = shape, fr_halfsat[sp_pred] = k.
      - State is freshly zeroed (preyed_biomass starts at 0), so eaten_total is
        read back as state.preyed_biomass[p_idx] (it is += from zero).
    """
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.processes.mortality import _apply_predation_for_school
    from osmose.engine.resources import ResourceState
    from osmose.engine.state import SchoolState

    n_subdt = 10
    cfg_dict = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "1",
        "simulation.nschool.sp1": "1",
        "species.name.sp0": "SmallPrey",
        "species.name.sp1": "BigPredator",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "50.0",
        "species.k.sp0": "0.5",
        "species.k.sp1": "0.2",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.0",
        "species.lifespan.sp0": "5",
        "species.lifespan.sp1": "10",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": str(n_subdt),
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
        "predation.predPrey.sizeRatio.min.sp0": "1.0",
        "predation.predPrey.sizeRatio.min.sp1": "1.0",
        "predation.predPrey.sizeRatio.max.sp0": "0.3",
        "predation.predPrey.sizeRatio.max.sp1": "0.3",
        "mortality.additional.rate.sp0": "0.0",
        "mortality.additional.rate.sp1": "0.0",
        "mortality.starvation.rate.max.sp0": "0.0",
        "mortality.starvation.rate.max.sp1": "0.0",
        "simulation.fishing.mortality.enabled": "false",
    }
    cfg = EngineConfig.from_dict(cfg_dict)

    # Inject FR config on the predator species (sp1 == runtime slot 1).
    sp_pred = 1
    cfg.fr_shape[sp_pred] = shape
    cfg.fr_halfsat[sp_pred] = k

    grid = Grid.from_dimensions(ny=1, nx=1)
    rs = ResourceState(config=cfg.raw_config, grid=grid)
    assert rs.n_resources == 0, "scenario assumes no resources so total_available is school-only"

    # schools: index 0 = predator (sp1), index 1 = prey (sp0).
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    pred_w = 0.006 * 30**3 * 1e-6  # tonnes per individual
    prey_w = 0.006 * 10**3 * 1e-6
    pred_abundance = 100.0
    pred_biomass = pred_abundance * pred_w

    # max_eatable for the predator (matches kernel formula at mortality.py:378).
    ingestion_rate = 3.5
    n_dt_per_year = 24
    max_eatable = pred_biomass * ingestion_rate / (n_dt_per_year * n_subdt)

    # total_available = prey_abundance * prey_w == r * max_eatable  -> solve abundance.
    prey_abundance = (r * max_eatable) / prey_w

    state = state.replace(
        abundance=np.array([pred_abundance, prey_abundance]),
        length=np.array([30.0, 10.0]),  # ratio 3.0, within [1.0, 1/0.3)
        weight=np.array([pred_w, prey_w]),
        biomass=np.array([pred_biomass, prey_abundance * prey_w]),
        age_dt=np.array([48, 24], dtype=np.int32),
        cell_x=np.array([0, 0], dtype=np.int32),
        cell_y=np.array([0, 0], dtype=np.int32),
        feeding_stage=np.array([0, 0], dtype=np.int32),
    )

    rng = np.random.default_rng(42)
    cell_indices = np.array([0, 1], dtype=np.int32)
    _apply_predation_for_school(
        0,  # p_idx = predator school
        cell_indices,
        state,
        cfg,
        rs,
        0,  # cell_y
        0,  # cell_x
        rng,
        n_subdt,
        None,  # access_matrix
        False,  # has_access
        False,  # use_stage_access
        None,
        None,
        inst_abd=state.abundance.copy(),
    )
    eaten = float(state.preyed_biomass[0])
    return eaten, max_eatable


@pytest.mark.parametrize("shape,k", [(1, 1.0), (2, 1.0), (3, 1.0), (3, 0.1), (3, 5.0)])
@pytest.mark.parametrize("r", [0.05, 0.5, 0.95, 2.0])
def test_python_kernel_matches_oracle(shape, k, r):
    eaten, max_eatable = _run_single_predation_step_python(r=r, shape=shape, k=k)
    assert eaten == pytest.approx(max_eatable * _g_ref(r, shape, k), rel=1e-12)


def test_python_kernel_type3_reduces_eaten_at_low_r():
    eaten1, me1 = _run_single_predation_step_python(r=0.3, shape=1, k=1.0)
    eaten3, me3 = _run_single_predation_step_python(r=0.3, shape=3, k=1.0)
    assert eaten3 < eaten1  # type-III refuge eats less at low r
