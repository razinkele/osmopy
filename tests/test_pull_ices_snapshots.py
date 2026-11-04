"""Guards in the ICES SAG snapshot puller (no network: the helpers take SAG payloads)."""

from __future__ import annotations

import pytest

from scripts._pull_ices_snapshots import _f_type, _unit_for


def _rows(desc: str, f_values: list[str], units: str = "") -> list[dict]:
    return [
        {"FishingPressureDescription": desc, "FishingPressureUnits": units, "FishingPressure": v}
        for v in f_values
    ]


def test_blank_f_description_raises_when_the_stock_publishes_f():
    # The validator reads anything but "F" as relative: a blank would strip an absolute stock
    # of its verdict without a word.
    with pytest.raises(RuntimeError, match="blank FishingPressureDescription"):
        _f_type("x", 2024, _rows("", ["", "0.3"]))


def test_blank_f_description_is_recorded_when_the_stock_publishes_no_f():
    # bzq.27.2425: biomass index and catches only, no F values at all — nothing to strip.
    assert _f_type("x", 2024, _rows("", ["", None])) == {"description": "", "units": ""}


def test_f_description_and_units_recorded_verbatim():
    assert _f_type("x", 2024, _rows(" F/Fmsy ", ["0.4"], "ratio")) == {
        "description": "F/Fmsy",
        "units": "ratio",
    }


@pytest.mark.parametrize(
    ("blim", "units", "expected"),
    [
        ("0.3", "tonnes", "index"),  # fle.27.2223: says tonnes, is B/Bmsy; Blim decides
        ("150000", "tonnes", "tonnes"),
        (None, "tonnes/h", "index"),  # bzq.27.2425 survey index
        (None, "Kilograms per gillnet station", "index"),  # bwp.27.2729-32
        (None, "tonnes", "unknown"),  # "tonnes" alone is not trusted without a Blim
        ("", "", "unknown"),
    ],
)
def test_unit_classification(blim, units, expected):
    assert _unit_for([{"StockSizeUnits": units}], blim) == expected
