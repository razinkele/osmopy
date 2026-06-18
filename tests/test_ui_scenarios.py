"""Tests for scenarios page logic."""

from osmose.scenarios import Scenario, ScenarioManager


def test_scenario_save_and_list(tmp_path):
    mgr = ScenarioManager(tmp_path)
    s = Scenario(name="test1", config={"a": "1"}, tags=["v1"])
    mgr.save(s)
    scenarios = mgr.list_scenarios()
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "test1"


def test_scenario_load_config(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="s1", config={"x": "42"}))
    loaded = mgr.load("s1")
    # load() canonicalizes to 4.4.0 keys (stamps osmose.version); other values preserved.
    assert {k: v for k, v in loaded.config.items() if k != "osmose.version"} == {"x": "42"}


def test_scenario_fork(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="base", config={"a": "1"}))
    forked = mgr.fork("base", "derived")
    # fork() canonicalizes to 4.4.0 keys (stamps osmose.version); other values preserved.
    assert {k: v for k, v in forked.config.items() if k != "osmose.version"} == {"a": "1"}
    assert forked.parent_scenario == "base"


def test_scenario_delete(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="del_me", config={}))
    assert len(mgr.list_scenarios()) == 1
    mgr.delete("del_me")
    assert len(mgr.list_scenarios()) == 0


def test_scenario_compare_shows_diffs(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="a", config={"x": "1", "y": "2"}))
    mgr.save(Scenario(name="b", config={"x": "1", "y": "9"}))
    diffs = mgr.compare("a", "b")
    assert len(diffs) == 1
    assert diffs[0].key == "y"
