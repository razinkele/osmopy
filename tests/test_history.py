import pytest

from osmose.history import RunRecord, RunHistory


def test_save_and_list_records(tmp_path):
    history = RunHistory(tmp_path)
    record = RunRecord(
        config_snapshot={"simulation.nspecies": "3"},
        duration_sec=120.5,
        output_dir=str(tmp_path / "output"),
        summary={"total_biomass": 1000.0},
    )
    history.save(record)
    records = history.list_runs()
    assert len(records) == 1
    assert records[0].duration_sec == 120.5
    assert records[0].summary["total_biomass"] == 1000.0


def test_list_sorted_newest_first(tmp_path):
    import time

    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    time.sleep(0.01)
    r2 = RunRecord(config_snapshot={"a": "2"}, duration_sec=20, output_dir="", summary={})
    history.save(r2)
    records = history.list_runs()
    assert len(records) == 2
    assert records[0].duration_sec == 20  # newest first


def test_load_run(tmp_path):
    history = RunHistory(tmp_path)
    record = RunRecord(
        config_snapshot={"x": "1"},
        duration_sec=5,
        output_dir="",
        summary={"y": 2},
    )
    history.save(record)
    records = history.list_runs()
    loaded = history.load_run(records[0].timestamp)
    assert loaded.config_snapshot == {"x": "1"}


def test_compare_runs(tmp_path):
    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1", "b": "2"}, duration_sec=10, output_dir="", summary={})
    r2 = RunRecord(
        config_snapshot={"a": "1", "b": "3", "c": "4"}, duration_sec=20, output_dir="", summary={}
    )
    history.save(r1)
    history.save(r2)
    records = history.list_runs()
    diffs = history.compare_runs(records[0].timestamp, records[1].timestamp)
    assert len(diffs) > 0


def test_compare_runs_multi_two_runs(tmp_path):
    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1", "b": "2"}, duration_sec=10, output_dir="", summary={})
    r2 = RunRecord(
        config_snapshot={"a": "1", "b": "3", "c": "4"}, duration_sec=20, output_dir="", summary={}
    )
    history.save(r1)
    history.save(r2)
    records = history.list_runs()
    timestamps = [r.timestamp for r in records]
    diffs = history.compare_runs_multi(timestamps)
    # 'a' is the same, 'b' and 'c' differ
    diff_keys = {d["key"] for d in diffs}
    assert "b" in diff_keys
    assert "c" in diff_keys
    assert "a" not in diff_keys


def test_compare_runs_multi_three_runs(tmp_path):
    import time as time_mod

    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1", "b": "2"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    time_mod.sleep(0.01)
    r2 = RunRecord(config_snapshot={"a": "1", "b": "3"}, duration_sec=20, output_dir="", summary={})
    history.save(r2)
    time_mod.sleep(0.01)
    r3 = RunRecord(
        config_snapshot={"a": "1", "b": "2", "d": "5"}, duration_sec=30, output_dir="", summary={}
    )
    history.save(r3)
    records = history.list_runs()
    timestamps = [r.timestamp for r in records]
    diffs = history.compare_runs_multi(timestamps)
    diff_keys = {d["key"] for d in diffs}
    assert "b" in diff_keys  # differs across runs
    assert "d" in diff_keys  # only in r3
    assert "a" not in diff_keys  # same in all


def test_compare_runs_multi_values_list(tmp_path):
    import time as time_mod

    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"x": "1"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    time_mod.sleep(0.01)
    r2 = RunRecord(config_snapshot={"x": "2"}, duration_sec=20, output_dir="", summary={})
    history.save(r2)
    records = history.list_runs()
    timestamps = [r.timestamp for r in records]
    diffs = history.compare_runs_multi(timestamps)
    assert len(diffs) == 1
    assert diffs[0]["key"] == "x"
    assert len(diffs[0]["values"]) == 2
    # Verify values are actually different
    assert set(diffs[0]["values"]) == {"1", "2"}


def test_compare_runs_multi_empty(tmp_path):
    history = RunHistory(tmp_path)
    diffs = history.compare_runs_multi([])
    assert diffs == []


def test_compare_runs_multi_single_run(tmp_path):
    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    records = history.list_runs()
    diffs = history.compare_runs_multi([records[0].timestamp])
    assert diffs == []  # Nothing to compare


def test_load_run_rejects_path_traversal(tmp_path):
    from osmose.history import RunHistory

    history = RunHistory(tmp_path / "history")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        history.load_run("../../etc/passwd")
