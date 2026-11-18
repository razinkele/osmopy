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
    for demo in ("eec_full", "bay_of_biscay", "baltic"):
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
    assert 1 in cap.catalog_contract_versions()


def test_every_catalogue_feature_has_exactly_one_detector():
    assert set(cap.DETECTORS) == set(cap.load_catalog())


def test_the_implemented_features_are_exactly_these():
    # Flip a feature's flag only in the PR that makes it run: core ran end to end in task 1.8,
    # oxygen_benthos in task 2.5, the recruitment three in task 2.3, salinity_gate in task 2.4,
    # background_species in task 2.1.
    implemented = {f.name for f in cap.load_catalog().values() if f.implemented}
    assert implemented == {
        "core", "oxygen_benthos", "stock_recruitment", "rv_gate", "seeding_linear",
        "salinity_gate", "background_species",
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
    # Not lmax_cap or movement_none: Baltic sets no species.lmax, and only its two background
    # species are immobile (their padded entries must not count; see capabilities._focal).
    "baltic": {
        "core", "background_species", "stock_recruitment", "rv_gate", "salinity_gate",
        "oxygen_benthos",
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
def test_every_demo_runs_with_minimal_outputs(demos, demo):
    """With output.minimal on, EEC and Bay of Biscay need nothing beyond core, and Baltic needs
    only implemented features since task 2.1 (its background species were the last). The stock
    demos, without output.minimal, are still refused: see the next test."""
    assert cap.csharp_engine_block_reason(demos[(demo, True)][1]) is None


def test_stock_demos_are_refused_naming_the_full_outputs(demos):
    reason = cap.csharp_engine_block_reason(demos[("eec_full", False)][1])
    assert reason is not None and "full_outputs (phase 2)" in reason


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
        "incoming_flux": (c, {"simulation.incoming.flux.enabled": "true"}),
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
