import numpy as np
import pytest

from osmose.forcing.percid_thermal import summer_sst_by_year


def test_averages_masked_cells_over_selected_months():
    # 2 years x months 6,7,8 ; 2x2 grid ; mask selects one cell
    temp = np.zeros((6, 2, 2))
    times_year = np.array([2000, 2000, 2000, 2001, 2001, 2001])
    times_month = np.array([6, 7, 8, 6, 7, 8])
    temp[:, 0, 0] = [10, 20, 30, 40, 50, 60]  # only this cell is unmasked
    mask = np.array([[True, False], [False, False]])
    years, means = summer_sst_by_year(temp, times_year, times_month, mask, months=(6, 7))
    assert list(years) == [2000, 2001]
    assert means[0] == pytest.approx(15.0)  # mean(10,20)
    assert means[1] == pytest.approx(45.0)  # mean(40,50)


def test_ignores_nan_ocean_fill():
    temp = np.full((2, 1, 2), np.nan)
    temp[:, 0, 0] = [12.0, 14.0]
    times_year = np.array([2000, 2000])
    times_month = np.array([6, 7])
    mask = np.array([[True, True]])
    years, means = summer_sst_by_year(temp, times_year, times_month, mask, months=(6, 7))
    assert means[0] == pytest.approx(13.0)  # nanmean over the one valid cell across 2 months
