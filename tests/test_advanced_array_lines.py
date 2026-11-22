"""Advanced tab: multi-value entries edit one element per line and round-trip exactly."""

import pytest

from ui.pages.advanced import array_to_lines, lines_to_array


@pytest.mark.parametrize(
    "value",
    ["2.3;1.8", "2.3; 1.8 ;0.5", "1;2;3;", "0.5,1.5,3", "a;b,c", ";", "12"],
)
def test_round_trip_is_exact(value):
    text, sep = array_to_lines(value)
    assert lines_to_array(text, sep) == value


def test_semicolon_wins_and_lines_split():
    assert array_to_lines("1,2;3") == ("1,2\n3", ";")
    assert array_to_lines("1,2,3") == ("1\n2\n3", ",")


def test_crlf_tolerated_and_trailing_line_is_a_separator():
    assert lines_to_array("1\r\n2\r\n3", ";") == "1;2;3"
    assert lines_to_array("1\n2\n", ";") == "1;2;"
