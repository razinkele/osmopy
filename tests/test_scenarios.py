import pytest

from osmose.scenarios import Scenario, ScenarioManager


@pytest.fixture
def manager(tmp_path):
    return ScenarioManager(tmp_path / "scenarios")


@pytest.fixture
def sample_scenario():
    return Scenario(
        name="baltic_smelt_baseline",
        description="Baseline Baltic smelt configuration",
        config={
            "simulation.nspecies": "3",
            "species.name.sp0": "Smelt",
            "species.linf.sp0": "25.0",
        },
        tags=["baseline", "baltic"],
    )


def test_save_and_load(manager, sample_scenario):
    manager.save(sample_scenario)
    loaded = manager.load("baltic_smelt_baseline")
    assert loaded.name == "baltic_smelt_baseline"
    assert loaded.config["simulation.nspecies"] == "3"
    assert loaded.tags == ["baseline", "baltic"]


def test_save_creates_directory(manager, sample_scenario):
    path = manager.save(sample_scenario)
    assert path.exists()
    assert (path / "scenario.json").exists()


def test_list_scenarios(manager, sample_scenario):
    manager.save(sample_scenario)
    manager.save(Scenario(name="another", config={"x": "1"}))
    listing = manager.list_scenarios()
    assert len(listing) == 2
    names = [s["name"] for s in listing]
    assert "baltic_smelt_baseline" in names
    assert "another" in names


def test_delete_scenario(manager, sample_scenario):
    manager.save(sample_scenario)
    assert len(manager.list_scenarios()) == 1
    manager.delete("baltic_smelt_baseline")
    assert len(manager.list_scenarios()) == 0


def test_compare_scenarios(manager):
    manager.save(Scenario(name="a", config={"x": "1", "y": "2", "z": "3"}))
    manager.save(Scenario(name="b", config={"x": "1", "y": "99", "w": "4"}))
    diffs = manager.compare("a", "b")
    keys = [d.key for d in diffs]
    assert "y" in keys  # different value
    assert "z" in keys  # only in a
    assert "w" in keys  # only in b
    assert "x" not in keys  # same value


def test_compare_finds_value_changes(manager):
    manager.save(Scenario(name="a", config={"species.linf.sp0": "25.0"}))
    manager.save(Scenario(name="b", config={"species.linf.sp0": "30.0"}))
    diffs = manager.compare("a", "b")
    assert len(diffs) == 1
    assert diffs[0].value_a == "25.0"
    assert diffs[0].value_b == "30.0"


def test_fork_scenario(manager, sample_scenario):
    manager.save(sample_scenario)
    forked = manager.fork("baltic_smelt_baseline", "high_fishing", "High fishing scenario")
    assert forked.name == "high_fishing"
    assert forked.parent_scenario == "baltic_smelt_baseline"
    assert forked.config == sample_scenario.config
    # Verify it was saved
    loaded = manager.load("high_fishing")
    assert loaded.name == "high_fishing"


def test_fork_is_independent(manager, sample_scenario):
    manager.save(sample_scenario)
    forked = manager.fork("baltic_smelt_baseline", "variant")
    forked.config["species.linf.sp0"] = "999"
    manager.save(forked)
    # Original should be unchanged
    original = manager.load("baltic_smelt_baseline")
    assert original.config["species.linf.sp0"] == "25.0"


def test_scenario_timestamps(manager, sample_scenario):
    manager.save(sample_scenario)
    loaded = manager.load("baltic_smelt_baseline")
    assert loaded.created_at
    assert loaded.modified_at


def test_load_nonexistent_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.load("nonexistent")


def test_export_all_creates_zip(tmp_path):
    mgr = ScenarioManager(tmp_path / "scenarios")
    mgr.save(Scenario(name="alpha", config={"x": "1"}))
    mgr.save(Scenario(name="beta", config={"y": "2"}))
    zip_path = tmp_path / "export.zip"
    mgr.export_all(zip_path)
    assert zip_path.exists()
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "alpha.json" in names
        assert "beta.json" in names


def test_import_all_from_zip(tmp_path):
    src_mgr = ScenarioManager(tmp_path / "src")
    src_mgr.save(Scenario(name="gamma", config={"z": "3"}))
    zip_path = tmp_path / "bundle.zip"
    src_mgr.export_all(zip_path)
    dst_mgr = ScenarioManager(tmp_path / "dst")
    count = dst_mgr.import_all(zip_path)
    assert count == 1
    loaded = dst_mgr.load("gamma")
    assert loaded.config == {"z": "3"}


def test_save_overwrites_existing_scenario(tmp_path):
    manager = ScenarioManager(tmp_path)
    s1 = Scenario(name="test", config={"a": "1"})
    manager.save(s1)
    s2 = Scenario(name="test", config={"a": "2"})
    manager.save(s2)
    loaded = manager.load("test")
    assert loaded.config["a"] == "2"


def test_save_creates_new_scenario(tmp_path):
    manager = ScenarioManager(tmp_path)
    s = Scenario(name="brand_new", config={"x": "1"})
    manager.save(s)
    loaded = manager.load("brand_new")
    assert loaded.config["x"] == "1"


def test_save_backup_restored_on_rename_failure(tmp_path):
    """If os.rename fails putting new data in place, backup is restored to target."""
    import os
    import json as _json
    from unittest.mock import patch

    manager = ScenarioManager(tmp_path)
    s1 = Scenario(name="test", config={"a": "original"})
    manager.save(s1)

    call_count = 0
    original_rename = os.rename

    def failing_rename(src, dst):
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # fail on the second rename (new -> target)
            raise OSError("Simulated failure")
        return original_rename(src, dst)

    s2 = Scenario(name="test", config={"a": "updated"})
    with patch("os.rename", side_effect=failing_rename):
        with pytest.raises(OSError, match="Simulated failure"):
            manager.save(s2)

    # Backup should be restored to target so original data is accessible
    target = tmp_path / "test" / "scenario.json"
    assert target.exists(), "Target should be restored from backup after failure"
    with open(target) as f:
        restored = _json.load(f)
    assert restored["config"]["a"] == "original"


def test_save_with_existing_stale_backup(tmp_path):
    """Save should handle leftover backup from a previous failure."""
    manager = ScenarioManager(tmp_path)
    manager.save(Scenario(name="test", config={"a": "1"}))
    # Simulate leftover backup from previous crash
    stale = tmp_path / "test.bak"
    stale.mkdir()
    (stale / "scenario.json").write_text("{}")

    # Second save should succeed despite stale backup
    manager.save(Scenario(name="test", config={"a": "2"}))
    loaded = manager.load("test")
    assert loaded.config["a"] == "2"
    assert not stale.exists()


def test_save_rejects_path_traversal(tmp_path):
    from osmose.scenarios import Scenario, ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")  # noqa: F841
    with pytest.raises(ValueError, match="invalid characters"):
        Scenario(name="../../etc/evil", config={})


def test_delete_rejects_path_traversal(tmp_path):
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.delete("../../etc")


def test_load_rejects_path_traversal(tmp_path):
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.load("../../etc/passwd")


def test_fork_rejects_path_traversal(tmp_path):
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.fork("../../etc/passwd", "new_name")


def test_import_all_rejects_path_traversal_in_zip(tmp_path):
    """ZIP containing scenario with traversal name should be skipped."""
    import zipfile
    import json
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    zip_path = tmp_path / "evil.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # The scenario name inside the JSON is what's validated — use a traversal name.
        data = json.dumps({"name": "../escape", "config": {}, "description": ""})
        zf.writestr("escape.json", data)

    count = mgr.import_all(zip_path)
    assert count == 0
    assert not (tmp_path / "escape").exists()


def test_save_rejects_empty_name(tmp_path):
    from osmose.scenarios import Scenario, ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")  # noqa: F841
    with pytest.raises(ValueError, match="[Ee]mpty"):
        Scenario(name="", config={})


def test_import_all_rejects_oversized_zip_entries(tmp_path, caplog):
    """ZIP entries larger than 10 MB must be skipped with a warning, not read."""
    import json
    import zipfile
    import logging
    from osmose.scenarios import ScenarioManager

    storage = tmp_path / "scenarios"
    storage.mkdir()
    mgr = ScenarioManager(storage)

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # A legitimate small scenario
        small = {
            "name": "ok",
            "description": "",
            "config": {},
            "tags": [],
            "parent_scenario": None,
        }
        zf.writestr("ok.json", json.dumps(small))
        # An oversized entry: 11 MB of valid JSON (above the 10 MB cap)
        big = {"name": "big", "filler": "x" * (11 * 1024 * 1024)}
        zf.writestr("big.json", json.dumps(big))

    with caplog.at_level(logging.WARNING):
        count = mgr.import_all(zip_path)

    assert count == 1, f"Only the small scenario should import, got {count}"
    # ScenarioManager.save() writes to storage_dir/<name>/scenario.json
    assert (storage / "ok" / "scenario.json").exists(), (
        "Small scenario should have been saved to storage/ok/scenario.json"
    )
    assert any("oversized" in rec.message.lower() for rec in caplog.records), (
        "An oversize warning should have been logged"
    )
