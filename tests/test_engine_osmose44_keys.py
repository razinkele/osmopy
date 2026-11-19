"""Two OSMOSE 4.4.x keys of issue #137's first phase.

``simulation.fixed.seed.enabled`` (4.4.0) replaces the per-process fixed-seed keys by one key.
``simulation.kill.if.no.school.enabled`` (4.4.1) stops a run in which a species has no fish
left once the seeding period is over. Java reads them in ``MapDistribution`` and its siblings,
and in ``SimulationStep.step`` (lines 263-289 at v4.4.1).

Every config here goes through the real reader: the keys are looked up as it leaves them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine, SpeciesCollapsed
from osmose.engine.config import EngineConfig
from osmose.engine.config_validation import build_known_keys
from osmose.engine_cs import capabilities as cap

MINIMAL = Path(__file__).resolve().parents[1] / "data" / "minimal"
ONE_KEY = "simulation.fixed.seed.enabled"
OLD_KEYS = ("movement.randomseed.fixed", "stochastic.mortality.randomseed.fixed")
KILL = "simulation.kill.if.no.school.enabled"


def _minimal(tmp_path: Path, *lines: str, **over: str) -> dict:
    """``data/minimal`` (Anchovy and Hake, 12 steps a year), with lines appended to its master
    file and read by the real reader. ``over`` replaces values the files already set: a second
    definition of a key in a file is not the one the reader keeps."""
    dst = tmp_path / "minimal"
    shutil.copytree(MINIMAL, dst)
    master = dst / "osm_all-parameters.csv"
    master.write_text(master.read_text() + "".join(f"\n{line}" for line in lines) + "\n")
    raw = dict(OsmoseConfigReader().read(str(master)))
    for line in lines:  # the appended keys are new to the config, so they are what was read
        key, value = (part.strip() for part in line.split(";"))
        assert raw[key] == value, line
    return raw | {k.replace("__", "."): v for k, v in over.items()}


# ----------------------------------------------------------------------- the one seed key


@pytest.mark.parametrize(
    ("lines", "movement", "mortality"),
    [
        ((), False, False),
        ((f"{ONE_KEY} ; true",), True, True),
        ((f"{OLD_KEYS[0]} ; true",), True, False),
        ((f"{OLD_KEYS[1]} ; true",), False, True),
        # A config of either generation is run, so either spelling fixes the seeds.
        ((f"{ONE_KEY} ; false", f"{OLD_KEYS[0]} ; true"), True, False),
    ],
)
def test_the_one_seed_key_fixes_both_streams(tmp_path, lines, movement, mortality):
    config = EngineConfig.from_dict(_minimal(tmp_path, *lines))
    assert (config.movement_seed_fixed, config.mortality_seed_fixed) == (movement, mortality)


def test_the_one_key_hands_a_run_the_generators_the_two_old_keys_do(tmp_path):
    """What the flags decide is which generators a run is handed: one shared by all species,
    or one of its own per species, for movement and for mortality. The one key must give
    exactly what the two keys it replaces give, the same streams included."""

    def handed(name: str, *lines: str) -> tuple[list, list]:
        _config, _grid, _rng, movement, mortality = PythonEngine()._prepare_run(
            _minimal(tmp_path / name, *lines), 7
        )
        return movement, mortality

    def distinct(generators: list) -> int:
        return len({id(g) for g in generators})

    def first_draws(generators: list) -> list[float]:
        return [float(g.random()) for g in generators]

    plain = handed("plain")
    one = handed("one", f"{ONE_KEY} ; true")
    old = handed("old", *(f"{key} ; true" for key in OLD_KEYS))
    assert [distinct(g) for g in plain] == [1, 1]  # one generator shared by both species
    assert [distinct(g) for g in one] == [2, 2] == [distinct(g) for g in old]
    for mine, theirs in zip(one, old, strict=True):
        draws = first_draws(mine)
        assert draws == first_draws(theirs) and draws[0] != draws[1]


def test_the_validator_and_the_csharp_gate_know_the_one_seed_key(tmp_path):
    assert ONE_KEY in build_known_keys().patterns
    # The C# engine is reproducible from its seed whatever these keys say: they change nothing.
    assert cap.classify_pattern(ONE_KEY) == "inert"
    raw = _minimal(tmp_path, f"{ONE_KEY} ; true")
    assert cap.unclassified_keys(raw) == []
    assert cap.csharp_engine_block_reason(raw) == cap.csharp_engine_block_reason(
        _minimal(tmp_path / "b")
    )


# ----------------------------------------------------------------------- the kill switch

SHORT_SEEDING = "population.seeding.year.max ; 0.25"  # 3 steps
NO_HAKE = {"population__seeding__biomass__sp1": "0"}  # Hake is never seeded: no fish at all


def _run(raw: dict, years: int = 1):
    return PythonEngine().run_in_memory(raw | {"simulation.time.nyear": str(years)}, seed=3)


def test_a_species_with_no_fish_after_the_seeding_period_stops_the_run(tmp_path):
    raw = _minimal(tmp_path, f"{KILL} ; true", SHORT_SEEDING, **NO_HAKE)
    assert EngineConfig.from_dict(raw).kill_if_no_school is True
    with pytest.raises(SpeciesCollapsed, match=r"Hake .* step 4\b") as stopped:
        _run(raw)
    assert "Anchovy" not in str(stopped.value)  # the species that still has fish is not named


def test_without_the_switch_the_same_run_goes_on(tmp_path):
    """The control: the species is as absent, and nothing stops."""
    res = _run(_minimal(tmp_path, SHORT_SEEDING, **NO_HAKE))
    last = res.abundance().iloc[-1]
    assert float(last["Hake"]) == 0.0 and float(last["Anchovy"]) > 0.0


def test_the_seeding_period_is_spared(tmp_path):
    """Java compares the step with the seeding period, which defaults to the LONGEST lifespan:
    Hake's 12 years here, Anchovy's being 4. A five-year run is past the shorter and inside
    the longer, so the switch does not act yet."""
    raw = _minimal(tmp_path, f"{KILL} ; true", **NO_HAKE)
    config = EngineConfig.from_dict(raw)
    assert (config.lifespan_dt[: config.n_species]).tolist() == [48, 144]
    assert config.seeding_max_step[: config.n_species].tolist() == [144, 144]  # issue #334
    res = _run(raw, years=5)
    assert float(res.abundance().iloc[-1]["Hake"]) == 0.0


def test_a_run_in_which_every_species_has_fish_is_not_stopped(tmp_path):
    res = _run(_minimal(tmp_path, f"{KILL} ; true", SHORT_SEEDING), years=2)
    last = res.abundance().iloc[-1]
    assert float(last["Hake"]) > 0.0 and float(last["Anchovy"]) > 0.0


def test_the_csharp_engine_refuses_the_kill_switch_by_name(tmp_path):
    """The C# engine has no such check. Running the config on it would ignore the switch
    without a word, so the config is refused, by the feature's name."""
    assert KILL in build_known_keys().patterns
    assert cap.classify_pattern(KILL) == "kill_if_no_school"
    on = _minimal(tmp_path, f"{KILL} ; true")
    assert cap.unclassified_keys(on) == []
    reason = cap.csharp_engine_block_reason(on)
    assert reason is not None and "kill_if_no_school" in reason
    off = _minimal(tmp_path / "off", f"{KILL} ; false")
    assert cap.csharp_engine_block_reason(off) == cap.csharp_engine_block_reason(
        _minimal(tmp_path / "plain")
    )
