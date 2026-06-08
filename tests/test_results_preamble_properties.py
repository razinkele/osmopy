"""Property-based tests: _detect_preamble_lines header/preamble detection."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import event, given
from hypothesis import strategies as st

from osmose.results import _detect_preamble_lines
from tests.strategies import csv_text_pairs, csv_texts


@given(tk=csv_texts())
def test_detects_planted_header(tk):
    text, k, _ncols = tk
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        p.write_text(text)
        assert _detect_preamble_lines(p) == k


@given(text=st.text(max_size=40))
def test_never_raises_on_degenerate_input(text):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.csv"
        p.write_text(text)
        assert isinstance(_detect_preamble_lines(p), int)


@given(pair=csv_text_pairs())
def test_cache_invalidates_on_file_change(pair):
    text_a, k_a, text_b, k_b = pair
    event(f"k_a={k_a} k_b={k_b}")  # confirm differing-k cases are generated
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.csv"
        p.write_text(text_a)
        assert _detect_preamble_lines(p) == k_a
        # Overwrite the SAME path; the (mtime_ns, size) cache key must change so a
        # stale cached value is not returned. (k_a != k_b AND byte size differs by
        # construction — see csv_text_pairs; size alone flips the cache key.)
        p.write_text(text_b)
        assert _detect_preamble_lines(p) == k_b
