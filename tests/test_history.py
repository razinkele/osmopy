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
