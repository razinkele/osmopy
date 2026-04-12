"""Tests for pure helpers used by ui/pages/fishing.py."""

from ui.pages._helpers import collect_resolved_keys


class _FakeField:
    def __init__(self, pattern):
        self._pattern = pattern

    def resolve_key(self, idx):
        return self._pattern.replace("{idx}", str(idx))


def test_collect_resolved_keys_basic():
    fields = [_FakeField("fishing.rate.fsh{idx}"), _FakeField("fishing.name.fsh{idx}")]
    result = collect_resolved_keys(fields, count=2)
    assert result == [
        "fishing.rate.fsh0", "fishing.name.fsh0",
        "fishing.rate.fsh1", "fishing.name.fsh1",
    ]


def test_collect_resolved_keys_zero_count():
    fields = [_FakeField("fishing.rate.fsh{idx}")]
    assert collect_resolved_keys(fields, count=0) == []


def test_collect_resolved_keys_start_idx():
    fields = [_FakeField("mpa.file.mpa{idx}")]
    result = collect_resolved_keys(fields, count=2, start_idx=3)
    assert result == ["mpa.file.mpa3", "mpa.file.mpa4"]
