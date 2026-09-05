"""Bioen fixture checks for `data/baltic_ev` (the augmented Baltic config used by the
FIE/genetics integration tests) plus a viability pre-flight gating those downstream tests.

For the realistic-config bioen regression against PRODUCTION `data/baltic` + the C3 overlay
(`data/baltic/scenarios/c3_bioen/`), see `tests/test_baltic_c3_bioen_smoke.py` instead --
that is the test that exercises gonad-derived spawning past
`population.seeding.year.max`, i.e. the C3 bioen work's own smoke regression. This module's
`test_baltic_ev_baseline_viable_for_fie` pre-flight below stays self-skipping (Task 7.4's
un-tuned-fixture contract): it gates only the FIE/genetics demo tests on `data/baltic_ev`,
not the C3 bioen work, and is unaffected by anything C3 does to `data/baltic`.
"""

from pathlib import Path
import pytest

from tests._ev_preflight import ensure_preflight_result


def test_baltic_ev_all_parameters_exists() -> None:
    assert (Path("data/baltic_ev") / "baltic_ev_all-parameters.csv").exists()


def test_baltic_ev_has_bioen_enabled() -> None:
    # native 4.4.0: the bioen toggle is now module.bioenergetics.enabled (renamed from
    # simulation.bioen.enabled by RENAMES_440)
    text = (Path("data/baltic_ev") / "baltic_ev_param-simulation.csv").read_text()
    assert "module.bioenergetics.enabled" in text
    assert "true" in text.split("module.bioenergetics.enabled")[1].split("\n")[0].lower()


def test_baltic_ev_cod_has_bioen_imax() -> None:
    # cod is sp0 in baltic; bioen ingestion key (real path used by reader at
    # config.py:1796) must exist
    all_text = "\n".join(p.read_text() for p in Path("data/baltic_ev").rglob("*.csv"))
    assert "predation.ingestion.rate.max.bioen.sp0" in all_text


def test_baltic_ev_bioen_subfile_is_included() -> None:
    """OSMOSE only recurses into sub-files referenced via
    `osmose.configuration.<key>;<file>` lines (reader.py:56-68). A bare
    `include;<file>` is silently ignored."""
    master = (Path("data/baltic_ev") / "baltic_ev_all-parameters.csv").read_text()
    assert "osmose.configuration.bioen;baltic_ev_param-bioen.csv" in master


@pytest.mark.integration
def test_baltic_ev_runs_5_years_without_genetics() -> None:
    """Smoke: baltic_ev with bioen on must run end-to-end for 5y."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "5"
    # Temporarily disable genetics for this bioen-only smoke (Task 8 enables it).
    # The reader canonicalizes to the NEW 4.4.0 key, so set that one (setting the
    # old key would be a no-op the from_dict merge silently drops).
    cfg["module.genetics.enabled"] = "false"
    result = PythonEngine().run_in_memory(cfg, seed=0)
    biomass = result.biomass()
    # `biomass()` returns wide-form: columns `[Time, <species1>, <species2>, ...]`
    # with a trailing `species` column from the loader that equals "all" for
    # the unified biomass CSV. The docstring at osmose/results.py:343 currently
    # documents long-form but the implementation is wide; the docstring fix is
    # out of scope for this plan.
    assert "cod" in biomass.columns, (
        f"biomass output missing 'cod' column; got columns={list(biomass.columns)}"
    )
    cod_final = float(biomass.sort_values("Time")["cod"].iloc[-1])
    assert cod_final > 0, f"cod biomass at end of 5y is {cod_final}, expected > 0"


@pytest.mark.integration
def test_baltic_ev_baseline_viable_for_fie() -> None:
    """Single viability pre-flight gating the FIE / genetics integration tests.

    Runs the baltic_ev baseline once (bioen on, genetics off, no fishing) for
    50y and checks BOTH preconditions the downstream FIE demo depends on:

    1. Size — cod biomass in size bins >=35cm at the final year is > 0, so the
       l50=35cm gear catches a non-empty share (otherwise the FIE selection
       differential on imax is structurally zero).
    2. Stability — cod biomass at year 50 stays within [0.5, 2.0]x its year-5
       (post-burnin) level, so the demo runs on a non-degenerate population
       rather than one dominated by founder-effect drift or selection collapse.

    The actual probe (and the shared cache/lock that makes it safe under
    pytest-xdist) lives in `tests/_ev_preflight.py::ensure_preflight_result` —
    see that module's docstring for why. This test is a thin wrapper: it just
    reports the same (viable, detail) result every downstream dependant
    reads via `require_baltic_ev_preflight()`. If ANOTHER caller in this
    pytest run already computed the result (e.g. it lost the race to a
    downstream test on a different xdist worker), this test reuses that
    cached answer instead of re-running the 50y simulation — the point of the
    shared cache is that it does not matter who computes it first, only that
    everyone agrees on the same answer within one run.
    """
    viable, detail = ensure_preflight_result()
    if not viable:
        pytest.skip(detail)
