import pytest

from osmose.scenario_wizard import (
    Basics,
    apply_basics,
    default_description,
    parse_source,
    read_basics,
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
