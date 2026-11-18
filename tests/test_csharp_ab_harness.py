"""``scripts/csharp/ab_eec.py``: which gate a ``--config`` is judged against.

The Baltic gate is go/no-go (spec 10.4: 1.15x at one thread, 1.5x at 14). A run named by PATH
must get the same gate as one named ``baltic``, and a relative path must mean the caller's
directory although the workers run in the repo root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ab_eec", ROOT / "scripts" / "csharp" / "ab_eec.py")
assert spec is not None and spec.loader is not None
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

BALTIC = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"


def test_a_fixture_name_and_a_path_to_its_file_get_the_same_gate(tmp_path, monkeypatch):
    by_name = ab.resolve_config("baltic")
    assert by_name == (BALTIC, "baltic")
    assert ab.resolve_config(str(BALTIC)) == by_name
    monkeypatch.chdir(BALTIC.parent)  # a relative path, from the caller's directory
    assert ab.resolve_config(BALTIC.name) == by_name
    monkeypatch.chdir(ROOT)
    assert ab.resolve_config("data/baltic/../baltic/baltic_all-parameters.csv") == by_name
    assert ab.GATES[by_name[1]] == {1: 1.15, 14: 1.5}


def test_another_file_is_resolved_and_has_no_fixture_gate(tmp_path, monkeypatch):
    other = tmp_path / "copy.csv"
    other.write_text(BALTIC.read_text())
    monkeypatch.chdir(tmp_path)
    path, fixture = ab.resolve_config("copy.csv")
    assert path == other.resolve() and path.is_absolute() and fixture is None
    assert ab.GATES.get(fixture or "", {}) == {}


def test_every_fixture_resolves_to_itself_and_exists():
    for name, path in ab.FIXTURES.items():
        assert ab.resolve_config(name) == (path.resolve(), name)
        assert path.is_file(), name
