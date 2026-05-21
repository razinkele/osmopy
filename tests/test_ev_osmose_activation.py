from pathlib import Path
import pytest


@pytest.mark.integration
def test_baltic_ev_runs_15_years_with_genetics_on(tmp_path: Path) -> None:
    """End-to-end smoke: baltic_ev with genetics on must run 15y (5y past
    evolution.seeding.year=10) and produce non-empty genetic_trait_means CSV.

    The 15y window is deliberate: with evolution.seeding.year=10, the first
    10y are seed phase where offspring genotypes are RANDOMLY REDRAWN from
    population donors (per inheritance.py:61-68). A test that runs only
    nyear<=10 cannot distinguish working inheritance from broken inheritance
    because the variance pattern is identical in both cases during seed phase.
    Running 15y and asserting that the variance pattern continues post-year-10
    confirms the inheritance pipeline did NOT degenerate at the seed-phase
    boundary (i.e., variance does not crash to zero or NaN once redraws stop)."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.results import read_genetic_trait_means

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "15"
    PythonEngine().run(cfg, tmp_path, seed=0)

    # The engine uses prefix="osm" (default) regardless of output.file.prefix
    # in the config (write_outputs in engine/__init__.py:108 omits prefix arg).
    csv_path = tmp_path / "osm_genetic_trait_means_Simu0.csv"
    assert csv_path.exists(), "genetic_trait_means CSV not produced"

    ds = read_genetic_trait_means(tmp_path, prefix="osm")
    assert "trait_name" in ds.coords
    assert "imax" in set(ds["trait_name"].values)

    # Trait expression must be non-degenerate at all times.
    cod_var_series = (
        ds["variance"].sel(species_id=0, trait_name="imax").to_pandas()
    )
    assert (cod_var_series > 1e-4).all(), (
        f"cod imax variance must stay > 1e-4 at all timesteps; "
        f"got min={cod_var_series.min():.6f} at time={cod_var_series.idxmin()}. "
        "Either genetics is silently disabled or the inheritance pipeline "
        "degenerated post-seed-phase."
    )

    # Specifically check post-seed-phase (year > 10) variance is healthy.
    # Inheritance kicks in at year 10; variance should NOT collapse.
    post_seed = cod_var_series[cod_var_series.index > 10]
    assert len(post_seed) > 0, "no post-seed-phase samples; expected ~5y worth"
    assert (post_seed > 1e-4).all(), (
        f"variance collapsed post-year-10 (inheritance phase): "
        f"min={post_seed.min():.6f}. inheritance.py may be returning empty "
        "or degenerate parts."
    )
