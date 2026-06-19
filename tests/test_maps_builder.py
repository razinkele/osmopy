from osmose.maps.builder import GridSpec


def test_gridspec_cell_polygon_and_center():
    g = GridSpec(
        nlon=50, nlat=40, upleft_lat=66.0, upleft_lon=10.0, lowright_lat=54.0, lowright_lon=30.0
    )
    poly = g.cell_polygon(0, 0)
    assert poly == [[10.0, 66.0], [10.4, 66.0], [10.4, 65.7], [10.0, 65.7]]
    lat, lon = g.cell_center(0, 0)
    assert abs(lat - 65.85) < 1e-9 and abs(lon - 10.2) < 1e-9
    polys = g.cell_polygons()
    assert polys.shape == (40, 50, 4, 2)
    assert polys[0, 0].tolist() == g.cell_polygon(0, 0)
    assert polys[39, 49].tolist() == g.cell_polygon(39, 49)


def test_gridspec_from_config():
    from osmose.maps.builder import GridSpec

    cfg = {
        "grid.nlon": "50",
        "grid.nlat": "40",
        "grid.upleft.lat": "66",
        "grid.upleft.lon": "10",
        "grid.lowright.lat": "54",
        "grid.lowright.lon": "30",
    }
    g = GridSpec.from_config(cfg)
    assert (g.nlon, g.nlat) == (50, 40) and g.upleft_lat == 66.0
