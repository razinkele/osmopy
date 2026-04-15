"""Tests for calibration history persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmose.calibration.history import delete_run, list_runs, load_run, save_run


@pytest.fixture()
def sample_run() -> dict:
    return {
        "version": 1,
        "timestamp": "2026-04-15T14:30:00",
        "algorithm": "nsga2",
        "settings": {"population_size": 50, "generations": 100, "n_parallel": 4},
        "parameters": [{"key": "species.k.sp0", "lower": 0.01, "upper": 1.0}],
        "objectives": {
            "biomass_rmse": True,
            "diet_distance": False,
            "banded_loss": {"enabled": False},
        },
        "results": {
            "best_objective": 0.342,
            "n_evaluations": 5000,
            "duration_seconds": 847,
            "objective_names": ["Biomass RMSE"],
            "convergence": [[0, 12.5], [1, 8.3], [2, 5.1]],
            "pareto_X": [[0.3, 100.0], [0.4, 120.0]],
            "pareto_F": [[0.34], [0.51]],
        },
    }


class TestSaveRun:
    def test_creates_json_file(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_filename_contains_timestamp_and_algorithm(
        self, tmp_path: Path, sample_run: dict
    ) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        assert "2026-04-15" in path.name
        assert "nsga2" in path.name

    def test_content_roundtrips(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        loaded = json.loads(path.read_text())
        assert loaded["version"] == 1
        assert loaded["algorithm"] == "nsga2"
        assert loaded["results"]["best_objective"] == 0.342

    def test_creates_directory_if_missing(self, tmp_path: Path, sample_run: dict) -> None:
        history_dir = tmp_path / "nested" / "history"
        path = save_run(sample_run, history_dir=history_dir)
        assert path.exists()
        assert history_dir.is_dir()


class TestLoadRun:
    def test_loads_saved_run(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        loaded = load_run(path)
        assert loaded["timestamp"] == "2026-04-15T14:30:00"
        assert loaded["results"]["pareto_X"] == [[0.3, 100.0], [0.4, 120.0]]

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run(tmp_path / "nonexistent.json")


class TestListRuns:
    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list_runs(history_dir=tmp_path) == []

    def test_lists_runs_sorted_by_timestamp_desc(self, tmp_path: Path) -> None:
        run1 = {
            "version": 1,
            "timestamp": "2026-04-14T09:00:00",
            "algorithm": "surrogate",
            "settings": {},
            "parameters": [],
            "objectives": {},
            "results": {
                "best_objective": 0.5,
                "n_evaluations": 100,
                "duration_seconds": 60,
                "objective_names": [],
                "convergence": [],
                "pareto_X": [],
                "pareto_F": [],
            },
        }
        run2 = {
            "version": 1,
            "timestamp": "2026-04-15T14:30:00",
            "algorithm": "nsga2",
            "settings": {},
            "parameters": [{"key": "a", "lower": 0, "upper": 1}],
            "objectives": {},
            "results": {
                "best_objective": 0.3,
                "n_evaluations": 200,
                "duration_seconds": 120,
                "objective_names": [],
                "convergence": [],
                "pareto_X": [],
                "pareto_F": [],
            },
        }
        save_run(run1, history_dir=tmp_path)
        save_run(run2, history_dir=tmp_path)
        runs = list_runs(history_dir=tmp_path)
        assert len(runs) == 2
        assert runs[0]["timestamp"] == "2026-04-15T14:30:00"
        assert runs[1]["timestamp"] == "2026-04-14T09:00:00"

    def test_list_entry_fields(self, tmp_path: Path, sample_run: dict) -> None:
        save_run(sample_run, history_dir=tmp_path)
        runs = list_runs(history_dir=tmp_path)
        entry = runs[0]
        assert "path" in entry
        assert entry["timestamp"] == "2026-04-15T14:30:00"
        assert entry["algorithm"] == "nsga2"
        assert entry["best_objective"] == 0.342
        assert entry["n_params"] == 1
        assert entry["duration_seconds"] == 847

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert list_runs(history_dir=tmp_path / "nonexistent") == []


class TestDeleteRun:
    def test_deletes_file(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        assert path.exists()
        delete_run(path)
        assert not path.exists()

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            delete_run(tmp_path / "nonexistent.json")
