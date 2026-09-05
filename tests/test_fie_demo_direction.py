import os
from pathlib import Path
import pytest

from tests._ev_preflight import require_baltic_ev_preflight


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OSMOSE_EV_FIE_DEMO"),
    reason=(
        "~6.8 min (six 50y baltic_ev bioen runs: 3 seeds x 2 F-arms), on top of a "
        "one-time ~60s cost priming the shared baltic_ev viability pre-flight if "
        "nothing else in this run has already done so (tests/_ev_preflight.py). "
        "`pytest.mark.slow` alone does not exclude this from a bare `pytest` run in "
        "this repo (`addopts` only filters e2e/visual, and CI's `pytest -n auto ...` "
        "passes no -m override), so this ALSO gates on an opt-in env var, matching "
        "tests/test_egg_retention_java_parity.py's OSMOSE_JAR and "
        "tests/test_engine_bioen_numba_kernel.py's OSMOSE_BIOEN_WHOLE_RUN_SMOKE "
        "precedent. Opt in with OSMOSE_EV_FIE_DEMO=1. WHEN to opt in: after any "
        "change to trait-expression / mortality step ordering in "
        "osmose/engine/simulate.py or osmose/engine/processes/mortality.py that "
        "could populate state.imax_trait before _mortality runs (see the xfail "
        "reason below for the exact gap) — this is the test that flips loudly to "
        "XPASS (strict=True) the moment that plumbing exists. Otherwise this is a "
        "known, root-caused, out-of-scope result (ruling R24) that does not need "
        "re-confirming on every CI invocation."
    ),
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN GAP, ruling R24 (C3 Stage-1 Task 6, 2026-08-30). Measured "
        "high_mean=3.0012 vs low_mean=2.9965 (measured at b46d599, BEFORE 20fdd05 "
        "changed seeding allele sourcing -- these exact figures may not reproduce, "
        "but the diagnosis does not depend on them: imax_trait is None whatever the "
        "allele values are) -- a 0.16% move in the WRONG "
        "direction against the >=2% expectation, with per-seed values "
        "(high=[2.998, 3.005, 3.000], low=[2.991, 2.991, 3.008]) clustered at "
        "3.00 well inside seed spread. Not a threshold miss: the FIE response is "
        "absent. Root cause, confirmed at source: state.imax_trait "
        "(osmose/engine/state.py) is READ by "
        "osmose/engine/processes/mortality.py:394,406,996 but ASSIGNED BY "
        "NOTHING -- grep for assignments turns up only the field declaration and "
        "comments noting it is unreachable. simulate.py runs _mortality at :1749 "
        "but express_traits (which would populate the phenotype) at :1818, i.e. "
        "AFTER mortality, so imax_trait is always None when the ingestion cap is "
        "applied in mortality.py. The genetic imax value is tracked and "
        "inherited but phenotypically inert, so it drifts neutrally under "
        "fishing and selection cannot act on it. Java DOES honour the "
        "per-school trait (getMaxPredationRate: existsTrait('imax')), so this is "
        "a real parity gap, not a modelling choice -- this follows directly from "
        "this plan's own spec decision 14 (moving the allometric cap into the "
        "mortality loop for Java parity). The fix needs BOTH making phenotypes "
        "available before mortality runs (express_traits before _mortality in "
        "simulate.py) AND new wiring to populate state.imax_trait from the "
        "expressed phenotypes -- reordering alone would not fix it, since "
        "state.imax_trait is assigned nowhere in the codebase today (confirmed "
        "by a full-repo grep, ruling R26). Both changes are an Ev-OSMOSE "
        "step-order-and-wiring change, out of scope for Task 6 and for Stage 1. "
        "Do NOT relax the 2% threshold instead: this test's own "
        "docstring says escalate to nyear=100 before relaxing it, and relaxing "
        "it would convert a measured capability gap into a test that passes "
        "while asserting nothing. Sibling xfail owning the same gap: "
        "tests/test_genetics_bioen_integration.py:86 "
        "(test_trait_overrides_affect_growth). strict=True so this flips loudly "
        "the moment the plumbing is restored."
    ),
)
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

    require_baltic_ev_preflight()  # skips until the viability pre-flight passes

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
        f"expected >= 2% drop in mean-across-seeds; got {drop_pct * 100:.2f}% "
        f"(per-seed high={high_ends}, low={low_ends}). "
        "If close to 1%, the response is at multi-seed drift noise floor — "
        "escalate to nyear=100 (~16 generations) BEFORE relaxing the threshold."
    )
