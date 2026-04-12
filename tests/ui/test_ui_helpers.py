"""Tests for shared UI page helpers."""

from ui.pages._helpers import parse_nspecies


def test_parse_nspecies_valid():
    assert parse_nspecies({"simulation.nspecies": "5"}) == 5


def test_parse_nspecies_float_string():
    assert parse_nspecies({"simulation.nspecies": "3.0"}) == 3


def test_parse_nspecies_missing_key():
    assert parse_nspecies({}) == 0


def test_parse_nspecies_empty_string():
    assert parse_nspecies({"simulation.nspecies": ""}) == 0


def test_parse_nspecies_invalid():
    assert parse_nspecies({"simulation.nspecies": "abc"}) == 0
