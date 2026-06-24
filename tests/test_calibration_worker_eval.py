"""_worker_eval must let unexpected (programming) errors propagate, not swallow
them to inf — expected model errors are already handled in _evaluate_candidate.
Deep review 2026-06-22 (HIGH-3, narrowed)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import osmose.calibration.problem as problem


def test_worker_eval_propagates_unexpected_error(monkeypatch):
    stub = MagicMock(n_obj=1)
    stub._evaluate_candidate.side_effect = TypeError("objective bug")
    monkeypatch.setattr(problem, "_WORKER_PROBLEM", stub)

    with pytest.raises(TypeError):
        problem._worker_eval(0, np.zeros(2))


def test_worker_eval_returns_evaluate_candidate_result(monkeypatch):
    # Expected-error handling lives in _evaluate_candidate; _worker_eval just
    # returns whatever it produces (here, a normal objective vector).
    stub = MagicMock(n_obj=1)
    stub._evaluate_candidate.return_value = [1.5]
    monkeypatch.setattr(problem, "_WORKER_PROBLEM", stub)

    assert problem._worker_eval(0, np.zeros(2)) == [1.5]
