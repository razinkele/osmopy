from pathlib import Path
import pytest


def test_baltic_ev_all_parameters_exists() -> None:
    assert (Path("data/baltic_ev") / "baltic_ev_all-parameters.csv").exists()


def test_baltic_ev_has_bioen_enabled() -> None:
    text = (Path("data/baltic_ev") / "baltic_ev_param-simulation.csv").read_text()
    assert "simulation.bioen.enabled" in text
    assert "true" in text.split("simulation.bioen.enabled")[1].split("\n")[0].lower()


def test_baltic_ev_cod_has_bioen_imax() -> None:
    # cod is sp0 in baltic; bioen ingestion key (real path used by reader at
    # config.py:1796) must exist
    all_text = "\n".join(
        p.read_text() for p in Path("data/baltic_ev").rglob("*.csv")
    )
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
    cfg["simulation.genetic.enabled"] = "false"
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
def test_baltic_ev_cod_reaches_fishery_l50_in_baseline(tmp_path: Path) -> None:
    """Baseline (bioen on, genetics off, no fishing) must produce cod
    that grow past 35cm in adult life-stage, otherwise the FIE demo's
    l50=35cm gear catches nothing and produces a null FIE signal for
    structural reasons rather than the science."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "20"
    cfg["simulation.genetic.enabled"] = "false"
    # Zero fishing so cod size distribution reflects bioen alone
    cfg["fisheries.rate.base.fsh0"] = "0.0"
    PythonEngine().run(cfg, tmp_path, seed=0)

    raise AssertionError(
        "PRE-FLIGHT NOT WIRED. This stub must be replaced with a real "
        "cod-final-length >= 35cm assertion sourced from the engine output "
        "BEFORE running Task 11. Until wired, the FIE-direction test can "
        "silently pass on a null signal — cod that never reach the gear "
        "l50 produce structurally-zero selection regardless of trait "
        "variance. See plan §Task 7.8."
    )


@pytest.mark.integration
def test_baltic_ev_cod_biomass_within_2x_envelope_over_50y(tmp_path: Path) -> None:
    """Baseline (bioen on, genetics off, no fishing) cod biomass at year 50
    must stay within [0.5, 2.0] × year-5 (post-burnin) biomass. Outside
    this envelope, the FIE demo (Task 11) runs on a degenerate population
    where founder-effect drift or selection collapse swamps the FIE signal."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "50"
    cfg["simulation.genetic.enabled"] = "false"
    cfg["fisheries.rate.base.fsh0"] = "0.0"

    result = PythonEngine().run_in_memory(cfg, seed=0)
    bio = result.biomass().sort_values("Time")
    # See test_baltic_ev_runs_5_years_without_genetics for the wide-form
    # biomass() return-shape note.
    assert "cod" in bio.columns, (
        f"biomass output missing 'cod' column; got columns={list(bio.columns)}"
    )
    burnin = float(bio[(bio["Time"] >= 5.0) & (bio["Time"] < 6.0)]["cod"].mean())
    end = float(bio[bio["Time"] >= 49.0]["cod"].mean())
    ratio = end / burnin if burnin > 0 else float("inf")
    assert 0.5 <= ratio <= 2.0, (
        f"cod biomass at year 50 = {end:.2e}, year 5 = {burnin:.2e}, "
        f"ratio = {ratio:.2f}. Expected 0.5 <= ratio <= 2.0 under no-fishing, "
        "no-genetics. Outside this envelope the FIE demo runs on a degenerate "
        "population; tune bioen params (Task 7.4) before relying on Task 11."
    )
