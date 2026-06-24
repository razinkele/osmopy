from pathlib import Path

import pytest

from scripts.spikes.native_predation.provenance import assert_provenance

WORKTREE = Path(__file__).resolve().parents[4]  # .../native_predation/tests -> repo root


def test_assert_provenance_passes_in_worktree():
    info = assert_provenance(WORKTREE)
    assert info["has_numba"] is True
    assert str(WORKTREE) in info["mortality_file"]
    assert info["numba_version"]  # non-empty


def test_assert_provenance_rejects_wrong_root():
    with pytest.raises(RuntimeError, match="not under worktree"):
        assert_provenance(Path("/nonexistent/elsewhere"))
