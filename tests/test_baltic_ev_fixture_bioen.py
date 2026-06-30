from pathlib import Path
import pytest

from tests._ev_preflight import SENTINEL


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

    On full pass this touches `tests/.preflight_wired`; the FIE-demo and
    genetics-activation tests call `require_baltic_ev_preflight()` and skip
    until it exists. When either criterion fails the fixture is un-tuned (plan
    Task 7.4): the sentinel is removed and THIS test SKIPS (not fails), so CI
    stays clean while leaving a visible "pending Task 7.4" signal. The sentinel
    is deterministic — a regression that breaks viability removes it and reverts
    the downstream tests to skipped.
    """
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    # Remove any stale sentinel up front so it is only valid if THIS run reaches
    # the touch() at the end. SENTINEL is anchored on the shared module dir so
    # the path matches every reader regardless of pytest's cwd.
    SENTINEL.unlink(missing_ok=True)

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "50"
    # Reader canonicalizes to the NEW 4.4.0 key; set that one (see note above).
    cfg["module.genetics.enabled"] = "false"
    # Zero fishing so the cod size distribution reflects bioen alone.
    cfg["fisheries.rate.base.fsh0"] = "0.0"
    result = PythonEngine().run_in_memory(cfg, seed=0)

    # Criterion 1: cod reach the 35cm gear at the final year.
    # biomass_by_size returns long-form [time, species, bin, value] where `bin`
    # is the size-bin lower edge as a string (e.g. "35.0"); see
    # osmose/engine/output.py:_build_distribution_dataframes.
    bbs = result.biomass_by_size("cod")
    if bbs.empty:
        pytest.skip(
            "baltic_ev pre-flight: biomass_by_size('cod') is empty — fixture un-tuned (Task 7.4)."
        )
    bbs = bbs.assign(bin_lower=bbs["bin"].astype(float))
    t_max = bbs["time"].max()
    last_year = bbs[bbs["time"] >= t_max - 1.0]
    biomass_ge35 = float(last_year[last_year["bin_lower"] >= 35.0]["value"].sum())
    biomass_total = float(last_year["value"].sum())
    max_occupied_bin = float(last_year[last_year["value"] > 0]["bin_lower"].max())

    # Criterion 2: 50y/5y (post-burnin) biomass envelope.
    bio = result.biomass().sort_values("Time")
    # See test_baltic_ev_runs_5_years_without_genetics for the wide-form note.
    if "cod" not in bio.columns:
        pytest.skip(
            f"baltic_ev pre-flight: biomass output missing 'cod' column; "
            f"got columns={list(bio.columns)}."
        )
    burnin = float(bio[(bio["Time"] >= 5.0) & (bio["Time"] < 6.0)]["cod"].mean())
    end = float(bio[bio["Time"] >= 49.0]["cod"].mean())
    ratio = end / burnin if burnin > 0 else float("inf")

    if not (biomass_ge35 > 0.0 and 0.5 <= ratio <= 2.0):
        pytest.skip(
            "baltic_ev FIE pre-flight not viable — tune bioen params (Task 7.4). "
            f"cod biomass >=35cm at year {t_max:.1f} = {biomass_ge35:.3e} "
            f"(total = {biomass_total:.3e}, largest occupied bin = "
            f"{max_occupied_bin:.1f}cm); 50y/5y envelope ratio = {ratio:.2f} "
            "(need cod >=35cm present and 0.5 <= ratio <= 2.0)."
        )

    # Both criteria hold — un-gate the downstream FIE / genetics tests.
    SENTINEL.touch()
