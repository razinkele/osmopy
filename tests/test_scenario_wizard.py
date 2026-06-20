import pytest

from osmose.scenario_wizard import (
    Basics,
    apply_basics,
    default_description,
    parse_source,
    read_basics,
    resolve_source,
    source_choices,
    validate_name,
)


def test_apply_basics_sets_exactly_the_four_keys_and_copies():
    cfg = {"simulation.time.nyear": "100", "species.name.sp0": "cod"}
    out = apply_basics(cfg, Basics(nyear=50, ndtperyear=12, reproducible_rng=True))
    assert out["simulation.time.nyear"] == "50"
    assert out["simulation.time.ndtperyear"] == "12"
    assert out["movement.randomseed.fixed"] == "true"
    assert out["stochastic.mortality.randomseed.fixed"] == "true"
    assert out["species.name.sp0"] == "cod"
    assert cfg["simulation.time.nyear"] == "100"


def test_apply_basics_false_rng():
    out = apply_basics({}, Basics(nyear=10, ndtperyear=24, reproducible_rng=False))
    assert out["movement.randomseed.fixed"] == "false"
    assert out["stochastic.mortality.randomseed.fixed"] == "false"


def test_read_basics_roundtrips_with_apply_basics():
    cfg = apply_basics({}, Basics(nyear=33, ndtperyear=12, reproducible_rng=True))
    assert read_basics(cfg) == Basics(nyear=33, ndtperyear=12, reproducible_rng=True)


def test_read_basics_falls_back_on_missing_or_garbage():
    assert read_basics({}) == Basics(nyear=10, ndtperyear=24, reproducible_rng=False)
    assert read_basics({"simulation.time.nyear": "x"}).nyear == 10


def test_read_basics_rng_true_only_when_both_booleans_true():
    assert read_basics({"movement.randomseed.fixed": "true"}).reproducible_rng is False
    both = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
    assert read_basics(both).reproducible_rng is True


def test_parse_source():
    assert parse_source("demo:baltic") == ("demo", "baltic")
    assert parse_source("scenario:my_run") == ("scenario", "my_run")
    with pytest.raises(ValueError):
        parse_source("bogus")


def test_source_choices_groups_and_prefixes():
    ch = source_choices(["baltic", "eec"], ["my_run"])
    assert ch["Bundled demos"] == {"demo:baltic": "baltic", "demo:eec": "eec"}
    assert ch["Saved scenarios"] == {"scenario:my_run": "my_run"}


def test_source_choices_omits_saved_group_when_empty():
    ch = source_choices(["baltic"], [])
    assert "Saved scenarios" not in ch
    assert ch["Bundled demos"] == {"demo:baltic": "baltic"}


def test_validate_name():
    existing = {"baltic_run"}
    assert validate_name("new_run", existing) == []
    assert validate_name("", existing)
    assert validate_name("   ", existing)
    assert validate_name("../evil", existing)
    assert validate_name("a/b", existing)
    assert validate_name("a\\b", existing)
    assert validate_name("baltic_run", existing)


def test_default_description():
    b = Basics(nyear=50, ndtperyear=24, reproducible_rng=False)
    assert default_description("demo", "baltic", b) == "Created from baltic demo, 50 yr"
    assert default_description("scenario", "my_run", b) == "Created from scenario 'my_run', 50 yr"


def test_resolve_source_demo(tmp_path):
    dest = tmp_path / "demo_dest"
    dest.mkdir()
    r = resolve_source("demo", "baltic", scenarios_dir=tmp_path / "scen", dest_dir=dest)
    assert r.kind == "demo" and r.name == "baltic"
    assert r.parent is None
    assert r.config_dir is not None and r.config_dir.exists()
    assert "grid.nlon" in r.config and r.case_map


def test_resolve_source_scenario(tmp_path):
    from osmose.scenarios import Scenario, ScenarioManager

    scen_dir = tmp_path / "scen"
    mgr = ScenarioManager(scen_dir)
    mgr.save(Scenario(name="base", config={"simulation.nspecies": "2"}, key_case_map={"a": "A"}))
    r = resolve_source("scenario", "base", scenarios_dir=scen_dir, dest_dir=None)
    assert r.kind == "scenario" and r.name == "base"
    assert r.config_dir is None
    assert r.parent == "base"
    assert r.config["simulation.nspecies"] == "2"


def test_resolve_source_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        resolve_source("bogus", "x", scenarios_dir=tmp_path, dest_dir=None)


def test_resolve_source_demo_requires_dest_dir(tmp_path):
    with pytest.raises(ValueError, match="dest_dir"):
        resolve_source("demo", "baltic", scenarios_dir=tmp_path, dest_dir=None)
