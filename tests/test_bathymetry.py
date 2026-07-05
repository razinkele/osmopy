# tests/test_bathymetry.py
import numpy as np
from osmose.forcing.bathymetry import shallow_fraction
from osmose.maps.builder import GridSpec


def _grid():
    return GridSpec(
        nlon=1, nlat=1, upleft_lat=54.2, upleft_lon=10.0, lowright_lat=54.0, lowright_lon=10.2
    )


def test_shallow_fraction_and_ocean():
    lat_hi = np.array([54.15, 54.05])
    lon_hi = np.array([10.05, 10.15])
    elev = np.array([[-5.0, -50.0], [3.0, -10.0]])  # (lat, lon)
    frac, ocean = shallow_fraction(elev, lat_hi, lon_hi, _grid(), depth_max_m=15.0)
    assert ocean[0, 0]
    assert frac[0, 0] == np.float64(2) / 3  # wet: -5,-50,-10; <=15m: -5,-10


def test_all_land_is_not_ocean_and_zero_fraction():
    frac, ocean = shallow_fraction(
        np.array([[5.0]]), np.array([54.1]), np.array([10.1]), _grid(), 15.0
    )
    assert not ocean[0, 0] and frac[0, 0] == 0.0
