"""Build a per-year percid summer surface-temperature index from CMEMS thetao.

The pure core (summer_sst_by_year) is grid/data-source agnostic and unit-tested
with synthetic arrays. The script scripts/build_percid_thermal_series.py wires
it to the real CMEMS baltic_phy monthly reanalysis (variable thetao, surface
layer) and the percid habitat masks, writing the sidecar CSV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def summer_sst_by_year(
    temp_tyx: NDArray[np.float64],
    times_year: NDArray[np.int_],
    times_month: NDArray[np.int_],
    mask_yx: NDArray[np.bool_],
    months: tuple[int, ...],
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Mean surface temperature over habitat cells for the given summer months.

    Returns (years_sorted, mean_per_year). NaN ocean-fill is ignored via nanmean.
    """
    sel_month = np.isin(times_month, months)
    years_sorted = np.array(sorted(set(times_year[sel_month].tolist())), dtype=int)
    means = np.empty(years_sorted.shape[0], dtype=np.float64)
    for i, yr in enumerate(years_sorted):
        sel = sel_month & (times_year == yr)
        block = temp_tyx[sel][:, mask_yx]  # (n_selected_months, n_masked_cells)
        means[i] = float(np.nanmean(block))
    return years_sorted, means
