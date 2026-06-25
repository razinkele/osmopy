from osmose.validation import fisheries as fis

# NOTE: Task 5 will add `from osmose.validation import stock_status as ss` and
# `from osmose.validation.fisheries_reference import ReferencePoint` HERE (at the
# top, with these imports) when it appends its tests — do not import them mid-file.


def test_annual_by_year_sum_and_mean():
    # 2 saved rows in year 0 (Time 0.0, 0.5), 1 in year 1 (Time 1.0)
    time = [0.0, 0.5, 1.0]
    assert fis.annual_by_year([2.0, 3.0, 10.0], time, how="sum") == {0: 5.0, 1: 10.0}
    assert fis.annual_by_year([2.0, 4.0, 10.0], time, how="mean") == {0: 3.0, 1: 10.0}


def test_annual_by_year_one_row_per_year_identity():
    assert fis.annual_by_year([10.0, 20.0], [0.0, 1.0], how="sum") == {0: 10.0, 1: 20.0}
