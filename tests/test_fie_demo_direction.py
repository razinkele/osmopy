from pathlib import Path
import pytest


def _require_preflight() -> None:
    """Block Task 11 from running until Task 7.8 is wired + passing."""
    # Anchor on this file's directory so the sentinel resolves to the same
    # absolute path regardless of pytest's cwd, matching the writer in
    # test_baltic_ev_fixture_bioen.py.
    sentinel = Path(__file__).parent / ".preflight_wired"
    if not sentinel.exists():
        pytest.skip(
            "Pre-flight viability check (Task 7.8) is not wired or has not "
            "been run successfully. Wire test_baltic_ev_cod_reaches_fishery_l50 "
            "to the engine size output and run it; on success it should "
            "`tests/.preflight_wired`.touch(). See plan §Task 7.8."
        )


@pytest.mark.slow
def test_high_f_drives_lower_cod_imax_than_low_f(tmp_path: Path) -> None:
    """Direction-only assertion: mean-across-3-seeds end-of-run cod imax
    must be lower under high F than low F by >=2%.

    Threshold defense.
    Per-generation FIE response on growth-rate at moderate F clusters at
    0.02-0.93%/yr across modelling studies (Audzijonyte et al., 2013,
    https://doi.org/10.1111/eva.12044), with a theoretical envelope of
    0.1-0.6%/yr (Andersen & Brander, 2009,
    https://doi.org/10.1073/pnas.0901690106). Over ~8 selecting generations
    (cod gen time ~=5y per Eero et al. 2015, https://doi.org/10.1093/icesjms/fsv109;
    Task 8 sets evolution.seeding.year=10 so only year>10 contributes), the
    expected cumulative high-F response is 1-4%. The paired (high-F minus
    low-F) contrast is ~2/3 of that = ~0.7-2.7%.

    Multi-seed drift floor (back-of-envelope): with sigma_A^2=0.018
    (config line: evolution.trait.imax.var.sp0=0.018), sigma_A~=0.134; for
    N_e ~= 0.1*N marine fish (Marty et al., 2015,
    https://doi.org/10.1111/eva.12220) and N~=10^3-10^4 schools, per-arm drift
    SD over 8 generations ~= sigma_A*sqrt(2g/N_e)/mu ~= 0.6% of trait mean.
    With 3 seeds, multi-seed-mean drift SD ~= 0.35%. A 1% threshold sits
    at ~2sigma -> ~5% false-pass risk under null. 2% sits at ~6sigma -> resilient.

    If the engine produces only ~1%, the right response is escalating to
    100y BEFORE relaxing the threshold (the test prints `drop_pct` so the
    implementer can decide).
    """
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.results import read_genetic_trait_means

    _require_preflight()  # see Task 7.8 — refuses to run until pre-flight wired

    def _cfg(fsh0_rate: str) -> dict[str, str]:
        cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
        cfg["simulation.time.nyear"] = "50"
        cfg["fisheries.rate.base.fsh0"] = fsh0_rate
        # Force size-selectivity for FIE; baltic default is age-knife-edge
        # which would make the FIE selection differential on imax zero.
        cfg["fisheries.selectivity.type.fsh0"] = "1"
        cfg["fisheries.selectivity.l50.fsh0"] = "35.0"
        cfg["fisheries.selectivity.slope.fsh0"] = "2.0"
        return cfg

    def _final_mean(out_dir: Path) -> float:
        ds = read_genetic_trait_means(out_dir, prefix="osm")
        s = ds.sel(species_id=0, trait_name="imax")["mean"].to_pandas()
        return float(s.iloc[-1])

    seeds = [42, 43, 44]
    high_ends, low_ends = [], []
    for s in seeds:
        out_high = tmp_path / f"high_{s}"
        out_low = tmp_path / f"low_{s}"
        out_high.mkdir()
        out_low.mkdir()
        PythonEngine().run(_cfg("0.6"), out_high, seed=s)
        PythonEngine().run(_cfg("0.1"), out_low, seed=s)
        high_ends.append(_final_mean(out_high))
        low_ends.append(_final_mean(out_low))

    import statistics
    high_mean = statistics.mean(high_ends)
    low_mean = statistics.mean(low_ends)
    drop_pct = (low_mean - high_mean) / low_mean
    assert high_mean < low_mean, (
        f"expected high-F imax < low-F imax across seeds, "
        f"got {high_mean=:.4f} vs {low_mean=:.4f} "
        f"(per-seed high={high_ends}, low={low_ends})"
    )
    assert drop_pct >= 0.02, (
        f"expected >= 2% drop in mean-across-seeds; got {drop_pct*100:.2f}% "
        f"(per-seed high={high_ends}, low={low_ends}). "
        "If close to 1%, the response is at multi-seed drift noise floor — "
        "escalate to nyear=100 (~16 generations) BEFORE relaxing the threshold."
    )
