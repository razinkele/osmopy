import numpy as np
import pandas as pd
import pytest

from osmose.validation import fisheries as fis
from osmose.validation import stock_status as ss
from osmose.validation.fisheries_reference import ReferencePoint


# NOTE: imports at top per brief Step 1 (stock_status + ReferencePoint
# are used by later tasks; kept here to avoid E402)


def test_annual_by_year_sum_and_mean():
    # 2 saved rows in year 0 (Time 0.0, 0.5), 1 in year 1 (Time 1.0)
    time = [0.0, 0.5, 1.0]
    assert fis.annual_by_year([2.0, 3.0, 10.0], time, how="sum") == {0: 5.0, 1: 10.0}
    assert fis.annual_by_year([2.0, 4.0, 10.0], time, how="mean") == {0: 3.0, 1: 10.0}


def test_annual_by_year_one_row_per_year_identity():
    assert fis.annual_by_year([10.0, 20.0], [0.0, 1.0], how="sum") == {0: 10.0, 1: 20.0}
