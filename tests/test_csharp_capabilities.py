"""Plan task 1.1: can the C# engine run a config? The catalogue, the detectors, key coverage.

Pure Python, except the last test, which compares the C# engine's embedded catalogue with this
package's (skipped without the built engine unless OSMOSE_REQUIRE_CSHARP=1).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.config import EngineConfig
from osmose.engine.config_validation import _ALLOWLIST_JAVA_ONLY, build_known_keys
from osmose.engine_cs import capabilities as cap
from osmose.engine_cs import csharp_engine_available

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demos(tmp_path_factory) -> dict:
    """(EngineConfig, raw) per (demo, minimal). The demo copies stay on disk for the module:
    csharp_engine_block_reason re-parses the raw config, which reads its files."""
    out = {}
    for demo in (
        "eec_full", "bay_of_biscay", "baltic", "benguela", "baltic_depensation", "baltic_a2"
    ):  # fmt: skip
        info = osmose_demo(demo, tmp_path_factory.mktemp(demo))
        base = dict(OsmoseConfigReader().read(Path(info["config_file"])))
        for minimal in (False, True):
            raw = base | ({"output.minimal.enabled": "true"} if minimal else {})
            out[(demo, minimal)] = (EngineConfig.from_dict(raw), raw)
    return out


# --------------------------------------------------------------------------- catalogue


def test_catalogue_ids_are_dense_and_core_is_first():
    feats = list(cap.load_catalog().values())
    assert [f.id for f in feats] == list(range(len(feats)))
    assert feats[0].name == "core" and feats[0].phase == 1
    assert {f.status for f in feats} <= {"planned", "refused", "not_planned"}
    assert all((f.phase in (1, 2, 3, 4)) == (f.status == "planned") for f in feats)
    from osmose.engine_cs.build import CONTRACT_VERSION

    assert CONTRACT_VERSION in cap.catalog_contract_versions()


def test_every_catalogue_feature_has_exactly_one_detector():
    assert set(cap.DETECTORS) == set(cap.load_catalog())


def test_the_implemented_features_are_exactly_these():
    # Flip a feature's flag only in the PR that makes it run: core ran end to end in task 1.8,
    # oxygen_benthos in task 2.5, the recruitment three in task 2.3, salinity_gate in task 2.4,
    # background_species in task 2.1, full_outputs in task 2.6c (with its last family, the diet),
    # lmax_cap and movement_none in the follow-up to task 2.2 (the code was in the Phase 1 core;
    # what they waited for was a test of a whole step and of a whole run), depensation_gate in
    # task 3.6 (that gate only: the ceiling and thermal gates of the same task are not ported),
    # depletable_ltl in task 3.1.
    implemented = {f.name for f in cap.load_catalog().values() if f.implemented}
    assert implemented == {
        "core", "oxygen_benthos", "stock_recruitment", "rv_gate", "seeding_linear",
        "salinity_gate", "background_species", "full_outputs", "lmax_cap", "movement_none",
        "depensation_gate", "depletable_ltl",
    }  # fmt: skip


# --------------------------------------------------------------------------- key coverage


def _engine_read_patterns() -> list[str]:
    return sorted(p for p in build_known_keys().patterns if p not in _ALLOWLIST_JAVA_ONLY)


def test_every_key_the_python_engine_reads_is_classified():
    """Plan rule 7, mechanically: a new engine key must be classified before the C# engine
    will accept a config that sets it."""
    patterns = _engine_read_patterns()
    assert len(patterns) > 300  # the instrument sees the real key set
    unclassified = [p for p in patterns if cap.classify_pattern(p) is None]
    assert not unclassified, f"classify these in KEY_RULES: {unclassified}"


#: The output keys the Python engine reads and full_outputs covers, one by one.
FULL_OUTPUT_KEYS = {
    "output.abundance.byage.enabled", "output.abundance.byage.netcdf.enabled",
    "output.abundance.bysize.enabled", "output.abundance.bysize.netcdf.enabled",
    "output.abundance.netcdf.enabled",
    "output.biomass.byage.enabled", "output.biomass.byage.netcdf.enabled",
    "output.biomass.bysize.enabled", "output.biomass.bysize.netcdf.enabled",
    "output.biomass.netcdf.enabled",
    "output.diet.composition.enabled",
    "output.distrib.bysize.incr", "output.distrib.bysize.max", "output.distrib.bysize.min",
    "output.meantl.enabled", "output.tl.enabled",
    "output.mortality.netcdf.enabled",
    "output.size.enabled", "output.size.netcdf.enabled",
    "output.ssb.enabled", "output.ssb.netcdf.enabled",
    "output.yield.abundance.enabled", "output.yield.abundance.netcdf.enabled",
    "output.yield.biomass.netcdf.enabled",
}  # fmt: skip


def test_full_outputs_owns_exactly_the_output_keys_listed_here():
    """full_outputs is implemented, so a key it owns is ACCEPTED. The rule must therefore name
    its keys, not catch every output.* key: a new Python-side output would be accepted and
    silently missing from a C# run. Widening the rule means editing this list, on purpose."""
    owned = {p for p in _engine_read_patterns() if cap.classify_pattern(p) == "full_outputs"}
    assert owned == FULL_OUTPUT_KEYS
    for made_up in ("output.foo.enabled", "output.biomass.byweight.netcdf.enabled", "output.x"):
        assert cap.classify_pattern(made_up) is None, made_up


def test_a_java_only_key_with_an_index_like_segment_is_not_an_engine_key(demos):
    """``output.abundance.age1.enabled`` (Benguela sets it) is Java-only, but ``age1`` looks
    like an indexed segment and normalises to ``age{idx}``, which the Java-only list does not
    hold. While a catch-all owned every output key this went unnoticed; without one it made the
    C# engine refuse a config the Python engine runs. The key itself is looked up as well."""
    key = "output.abundance.age1.enabled"
    assert key in _ALLOWLIST_JAVA_ONLY and cap.classify_pattern(key) is None
    raw = demos[("eec_full", True)][1] | {key: "true"}
    assert cap.unclassified_keys(raw) == []
    assert cap.csharp_engine_block_reason(raw) is None


@pytest.mark.parametrize(
    "master", sorted(p for p in (ROOT / "data").rglob("*.csv") if "all-parameters" in p.name)
)
def test_no_shipped_config_has_an_unclassified_output_key(master):
    """The sweep behind the rule above: every output key of every shipped master config is
    classified (or Java-only), so removing the catch-all refused nothing that ran before."""
    from osmose.config.reader import OsmoseConfigReader

    raw = dict(OsmoseConfigReader().read(master))
    assert [k for k in cap.unclassified_keys(raw) if k.startswith("output.")] == []


def test_a_new_output_key_the_python_engine_reads_blocks_the_engine(demos, monkeypatch):
    """What the rule above is for: an output key the Python engine starts reading, which nobody
    has classified, makes the C# engine refuse the config instead of running without it."""
    import osmose.engine.config_validation as cv

    real = cv.build_known_keys

    def with_a_new_output_key():
        known = real()
        return type(known)(
            patterns=known.patterns | {"output.newfamily.enabled"},
            literals=known.literals,
            regexes=known.regexes,
        )

    monkeypatch.setattr(cv, "build_known_keys", with_a_new_output_key)
    raw = demos[("eec_full", False)][1] | {"output.newfamily.enabled": "true"}
    assert cap.unclassified_keys(raw) == ["output.newfamily.enabled"]
    reason = cap.csharp_engine_block_reason(raw)
    assert reason is not None and "output.newfamily.enabled" in reason


def test_rule_owners_are_catalogue_features_or_core_classes():
    valid = set(cap.load_catalog()) | {"phase1", "inert", "internal"}
    for _, owner in cap.KEY_RULES:
        assert set(owner.split(",")) <= valid, owner


def test_an_unclassified_engine_key_blocks_and_is_named(demos, monkeypatch):
    _config, raw = demos[("baltic", True)]
    assert not cap.unclassified_keys(raw)
    kept = tuple((rx, o) for rx, o in cap._COMPILED if o != "salinity_gate")
    monkeypatch.setattr(cap, "_COMPILED", kept)
    blocked = cap.unclassified_keys(raw)
    assert blocked and all(k.startswith("movement.salinity.") for k in blocked)
    assert "movement.salinity." in (cap.csharp_engine_block_reason(raw) or "")


def test_keys_the_python_engine_never_reads_follow_python(demos):
    _, raw = demos[("eec_full", True)]
    noisy = raw | {"made.up.key": "1", "simulation.restart.file": "x.nc"}  # unknown; Java-only
    assert cap.unclassified_keys(noisy) == []


# --------------------------------------------------------------------------- the demos

EXPECTED = {
    "eec_full": {"core"},
    "bay_of_biscay": {"core"},
    # Not incoming_flux: Benguela switches it on and gives no species a flux file, which does
    # nothing (tests/test_engine_incoming_flux.py::TestOnButUnused).
    "benguela": {"core"},
    # Not lmax_cap or movement_none: Baltic sets no species.lmax, and only its two background
    # species are immobile (their padded entries must not count; see capabilities._focal).
    "baltic": {
        "core", "background_species", "stock_recruitment", "rv_gate", "salinity_gate",
        "oxygen_benthos",
    },
    # The Baltic demo with the depensation gate on cod_west, and WITHOUT the oxygen coupling
    # (the variant does not include it: CLAUDE.md, the oxygen gotcha).
    "baltic_depensation": {
        "core", "background_species", "stock_recruitment", "rv_gate", "salinity_gate",
        "depensation_gate",
    },
    # The Baltic demo with depletable resources, likewise without the oxygen coupling.
    "baltic_a2": {
        "core", "background_species", "stock_recruitment", "rv_gate", "salinity_gate",
        "depletable_ltl",
    },
}  # fmt: skip


@pytest.mark.parametrize("demo", sorted(EXPECTED))
@pytest.mark.parametrize("minimal", [True, False])
def test_demo_needs_exactly_these_features(demos, demo, minimal):
    config, raw = demos[(demo, minimal)]
    want = EXPECTED[demo] | (set() if minimal else {"full_outputs"})
    assert set(cap.required_features(config, raw)) == want
    assert cap.unclassified_keys(raw) == []


@pytest.mark.parametrize("demo", sorted(EXPECTED))
@pytest.mark.parametrize("minimal", [True, False])
def test_every_demo_runs_as_shipped_and_with_minimal_outputs(demos, demo, minimal):
    """EEC and Bay of Biscay need nothing beyond core, and Baltic only implemented features
    since task 2.1 (its background species were the last). Since task 2.6c the full outputs are
    implemented too, so the stock demos run as shipped. The two Baltic variants run since
    their features were ported: the depensation gate (task 3.6), depletable resources (3.1)."""
    assert cap.csharp_engine_block_reason(demos[(demo, minimal)][1]) is None


@pytest.mark.parametrize(
    ("key", "feature"),
    [
        ("output.spatial.enabled", "spatial_outputs (phase 3)"),
        ("output.carbon.enabled", "carbon_output (phase 3)"),
    ],
)
def test_the_output_families_the_engine_does_not_collect_are_still_refused(demos, key, feature):
    """full_outputs covers the life-stage split, the distributions, SSB, the two means, yield
    in numbers and the diet. The spatial and the carbon-flux outputs are features of their own
    and stay refused, by name."""
    reason = cap.csharp_engine_block_reason(demos[("eec_full", False)][1] | {key: "true"})
    assert reason is not None and feature in reason and "full_outputs" not in reason


def test_incoming_flux_is_refused_when_used_not_when_merely_switched_on(demos):
    """Benguela ships with the switch on and no flux file. That does nothing in the Python
    engine, so it is not a reason to refuse. A flux file is: the C# engine creates no incoming
    schools, and says so by the feature's name (not as an unclassified key)."""
    raw = demos[("eec_full", True)][1]
    on = raw | {"simulation.incoming.flux.enabled": "true"}
    assert cap.csharp_engine_block_reason(on) is None
    for key in ("flux.incoming.bydt.byage.file.sp0", "flux.incoming.bydt.bysize.file.sp2"):
        used = on | {key: "flux.csv"}
        assert cap.unclassified_keys(used) == []
        assert cap.classify_pattern(key.rsplit("sp", 1)[0] + "sp{idx}") == "incoming_flux"
        reason = cap.csharp_engine_block_reason(used)
        assert reason is not None and "not implemented yet: incoming_flux (phase 3)" in reason
        # The file key without the switch is not read by the Python engine either.
        assert cap.csharp_engine_block_reason(raw | {key: "flux.csv"}) is None
    # Keys are read in their lowercase form (the reader's), by the Python engine and by this
    # check alike: a camelCase key that did not go through the reader is not a flux file to
    # either, so the two cannot disagree about it.
    camel = on | {"flux.incoming.byDt.byAge.file.sp0": "flux.csv"}
    assert cap.csharp_engine_block_reason(camel) is None
    from osmose.engine.grid import Grid
    from osmose.engine.incoming_flux import IncomingFluxState

    state = IncomingFluxState(
        config=camel,
        engine_config=EngineConfig.from_dict(camel),
        grid=Grid.from_dimensions(ny=3, nx=3),
    )
    assert state.enabled and state._fluxes == []
    benguela = demos[("benguela", False)][1]
    assert benguela["simulation.incoming.flux.enabled"].lower() == "true"
    assert cap.csharp_engine_block_reason(benguela | {"flux.incoming.bydt.byage.file.sp0": "f.csv"})


# --------------------------------------------------------------------------- one positive case per feature


def _flip(arr, value, dtype):
    a = np.array(arr, dtype=dtype, copy=True)
    a.flat[0] = value
    return a


def _recipes(c: EngineConfig) -> dict:
    """For each feature, a (config, raw-overrides) change that turns on EXACTLY that feature."""
    n = len(c.species_names)
    nones = [None] * n
    sel = np.asarray(c.fishing_selectivity_type)
    two_stage = int(np.argmax(np.asarray(c.n_feeding_stages) > 1))

    def R(base: EngineConfig, **changes):
        # A shallow copy with attributes set directly: detectors only read attributes, and
        # dataclasses.replace would re-run __post_init__'s cross-field validation (e.g. array
        # lengths against n_background), which a one-field change is not meant to satisfy.
        new = copy.copy(base)
        for k, v in changes.items():
            setattr(new, k, v)
        return new

    return {
        "full_outputs": (R(c, output_minimal=False), {}),
        "background_species": (R(c, n_background=1), {}),
        "lmax_cap": (R(c, lmax=_flip(c.lmax, 50.0, np.float64)), {}),
        "stock_recruitment": (
            R(c, recruitment_type=["shepherd"] + list(c.recruitment_type)[1:]),
            {},
        ),
        "rv_gate": (R(c, rv_gate_enabled=np.array([True] + [False] * (n - 1))), {}),
        "salinity_gate": (R(c, salinity_gate_enabled=True), {}),
        "oxygen_benthos": (c, {"ltl.oxygen.benthos.enabled": "true"}),
        "movement_none": (R(c, movement_method=["none"] + list(c.movement_method)[1:]), {}),
        "depletable_ltl": (c, {"ltl.depletable.enabled": "true"}),
        "incoming_flux": (
            c,
            {
                "simulation.incoming.flux.enabled": "true",
                "flux.incoming.bydt.byage.file.sp0": "flux.csv",
            },
        ),
        "mpa": (R(c, mpa_zones=[object()]), {}),
        "fishing_rate_by_year": (R(c, fishing_rate_by_year=[np.ones(2)] + nones[1:]), {}),
        "additional_rate_by_step": (R(c, additional_mortality_by_dt=[np.ones(2)] + nones[1:]), {}),
        "larva_rate_by_step": (R(c, larva_mortality_by_dt=[np.ones(2)] + nones[1:]), {}),
        "additional_maps": (R(c, additional_mortality_spatial=[np.ones((2, 2))] + nones[1:]), {}),
        "selectivity_logistic": (R(c, fishing_selectivity_type=_flip(sel, 1, sel.dtype)), {}),
        "selectivity_length": (
            R(
                c,
                fishing_selectivity_type=_flip(sel, -1, sel.dtype),
                fishing_selectivity_l50=_flip(c.fishing_selectivity_l50, 12.0, np.float64),
            ),
            {},
        ),
        "recruitment_ceiling": (
            R(c, recruitment_ceiling_enabled=np.array([True] + [False] * (n - 1))),
            {},
        ),
        "thermal_gate": (R(c, thermal_gate_enabled=np.array([True] + [False] * (n - 1))), {}),
        "depensation_gate": (
            R(c, depensation_gate_enabled=np.array([True] + [False] * (n - 1))),
            {},
        ),
        "rv_spatial": (R(c, rv_spatial_enabled=np.array([True] + [False] * (n - 1))), {}),
        "functional_response_2_3": (R(c, fr_shape=_flip(c.fr_shape, 2, np.int32)), {}),
        "species_accessibility": (R(c, stage_accessibility=None), {}),
        "dynamic_accessibility": (R(c, dynamic_accessibility_enabled=True), {}),
        "gompertz_growth": (R(c, growth_class=["GOMPERTZ"] + list(c.growth_class)[1:]), {}),
        "feeding_stage_metric": (
            R(
                c,
                feeding_stage_metric=[
                    ("age" if i == two_stage else m) for i, m in enumerate(c.feeding_stage_metric)
                ],
            ),
            {},
        ),
        "seeding_linear": (R(c, seeding_mode="linear"), {}),
        "warm_start": (c, {"module.population.initialisation.enabled": "true"}),
        "spatial_outputs": (R(c, output_spatial_enabled=True), {}),
        "carbon_output": (R(c, output_carbon=True), {}),
        "bioenergetics": (R(c, bioen_enabled=True), {}),
        "genetics": (R(c, genetics_enabled=True), {}),
        "economics": (R(c, economics_enabled=True), {}),
        "java_compat_rng": (R(c, java_compat_rng=True), {}),
        "kill_if_no_school": (R(c, kill_if_no_school=True), {}),
        "resource_percent_log": (R(c, resource_percent_log=True), {}),
        "maturity_stochastic": (R(c, maturity_stochastic=np.ones(n, dtype=bool)), {}),
        "larva_density_dependence": (R(c, larva_reference_density=np.full(n, 100.0)), {}),
        "mortality_rate_factor": (R(c, larva_mortality_factor=np.ones((n, 4))), {}),
        "movement_gradient": (
            R(
                c,
                movement_method=["gradient"] + list(c.movement_method)[1:],
                movement_is_gradient=np.array(
                    [True] + [False] * (len(c.movement_method) - 1), dtype=bool
                ),
            ),
            {},
        ),
        "semelparous": (R(c, reproduction_semelparous=np.ones(n, dtype=bool)), {}),
        "fishing_catches": (R(c, fishing_catches=np.full(n, 10.0)), {}),
        "rate_by_class": (R(c, fishing_rate_by_dt_by_class=[object()] + nones[1:]), {}),
        "selectivity_type_2_3": (R(c, fishing_selectivity_type=_flip(sel, 2, sel.dtype)), {}),
    }


def test_background_padding_does_not_trip_focal_detectors(demos):
    config, raw = demos[("baltic", True)]
    n = config.n_species
    assert np.asarray(config.lmax)[n:].tolist() == [0.0, 0.0]  # the padding the detector must skip
    assert list(config.movement_method)[n:] == ["none", "none"]
    needed = set(cap.required_features(config, raw))
    assert "lmax_cap" not in needed and "movement_none" not in needed


def test_every_feature_but_core_has_a_positive_case(demos):
    config, _ = demos[("eec_full", True)]
    assert set(_recipes(config)) == set(cap.load_catalog()) - {"core"}


@pytest.mark.parametrize("feature", [f for f in cap.load_catalog() if f != "core"])
def test_turning_on_one_feature_adds_exactly_that_feature(demos, feature):
    config, raw = demos[("eec_full", True)]
    assert cap.required_features(config, raw) == ["core"]
    changed, overrides = _recipes(config)[feature]
    assert set(cap.required_features(changed, raw | overrides)) == {"core", feature}


# --------------------------------------------------------------------------- one source for both engines


def test_csharp_engine_embeds_the_same_catalogue():
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "e = bridge.load(); "
        "print(json.dumps([[f.Id, f.Name, f.Phase, f.Status, bool(f.Implemented)] "
        "for f in e.FeatureCatalog.All]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    csharp = json.loads(proc.stdout.strip().splitlines()[-1])
    python = [[f.id, f.name, f.phase, f.status, f.implemented] for f in cap.load_catalog().values()]
    assert csharp == python
