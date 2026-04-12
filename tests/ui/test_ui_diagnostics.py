"""Tests for pure helpers extracted from ui/pages/diagnostics.py."""

from ui.pages._helpers import format_timing_pairs


def test_format_timing_pairs_sorts_by_name():
    timing = {"predation": 1.5, "growth": 0.3, "movement": 2.1}
    result = format_timing_pairs(timing)
    assert result == [("growth", "0.300s"), ("movement", "2.100s"), ("predation", "1.500s")]


def test_format_timing_pairs_empty():
    assert format_timing_pairs({}) == []


def test_format_timing_pairs_single():
    result = format_timing_pairs({"init": 0.001})
    assert result == [("init", "0.001s")]
