from osmose.scenario_wizard import Basics, apply_basics


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
