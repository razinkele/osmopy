import numpy as np
from hypothesis import given, strategies as st

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


def test_rasterize_polygon_centers_inside():
    from osmose.maps.builder import GridSpec, rasterize_polygon

    g = GridSpec(nlon=4, nlat=4, upleft_lat=4.0, upleft_lon=0.0, lowright_lat=0.0, lowright_lon=4.0)
    ring = [[0.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0]]
    cells = set(rasterize_polygon(g, ring, mask=None))
    assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_rasterize_excludes_masked_cells():
    from osmose.maps.builder import GridSpec, rasterize_polygon

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    mask = np.zeros((4, 4))
    mask[0, 0] = -99
    ring = [[0.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0]]
    cells = set(rasterize_polygon(g, ring, mask=mask))
    assert (0, 0) not in cells and (0, 1) in cells


def test_rasterize_polygon_outside_grid_empty():
    from osmose.maps.builder import GridSpec, rasterize_polygon

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    assert rasterize_polygon(g, [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0]], mask=None) == []


def test_lonlat_to_cell():
    from osmose.maps.builder import GridSpec, lonlat_to_cell

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    assert lonlat_to_cell(g, 0.5, 3.5) == (0, 0)
    assert lonlat_to_cell(g, 3.5, 0.5) == (3, 3)
    assert lonlat_to_cell(g, -1.0, 3.5) is None


@given(
    lons=st.lists(st.floats(0.1, 3.9), min_size=3, max_size=6),
    lats=st.lists(st.floats(0.1, 3.9), min_size=3, max_size=6),
)
def test_rasterize_matches_center_membership(lons, lats):
    from osmose.maps.builder import GridSpec, rasterize_polygon, _point_in_ring, _open_ring

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    n = min(len(lons), len(lats))
    ring = [[lons[i], lats[i]] for i in range(n)]
    got = set(rasterize_polygon(g, ring, mask=None))
    expected = {
        (r, c)
        for r in range(4)
        for c in range(4)
        if _point_in_ring(*(lambda la, lo: (lo, la))(*g.cell_center(r, c)), _open_ring(ring))
    }
    assert got == expected


def test_mapgrid_apply_erase_mask():
    from osmose.maps.builder import GridSpec, MapGrid

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    mg = MapGrid.blank(g)
    mg.apply_cells([(0, 0), (1, 1)], 1.0)
    assert mg.array[0, 0] == 1.0 and mg.array[1, 1] == 1.0
    mg.erase([(0, 0)])
    assert mg.array[0, 0] == 0.0
    mg.set_mask([(2, 2)], True)
    assert mg.array[2, 2] == -99
    mg.set_mask([(2, 2)], False)
    assert mg.array[2, 2] == 0.0


def test_mapgrid_blank_seeds_base_mask():
    from osmose.maps.builder import GridSpec, MapGrid

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    base = np.zeros((4, 4))
    base[0, 0] = -99
    mg = MapGrid.blank(g, base_mask=base)
    assert mg.array[0, 0] == -99 and mg.array[1, 1] == 0.0


def test_mapgrid_apply_polygon():
    from osmose.maps.builder import GridSpec, MapGrid

    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    mg = MapGrid.blank(g)
    mg.apply_polygon(g, [[0.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0]], 1.0)
    assert mg.array[0, 0] == 1.0 and mg.array[3, 3] == 0.0


def test_csv_roundtrip_through_engine_loader(tmp_path):
    from osmose.maps.builder import GridSpec, MapGrid, to_csv_text
    from osmose.engine.movement_maps import _load_csv_grid

    g = GridSpec(3, 2, 2.0, 0.0, 0.0, 3.0)
    mg = MapGrid.blank(g)
    mg.apply_cells([(0, 0)], 1.0)
    f = tmp_path / "m.csv"
    f.write_text(to_csv_text(mg))
    loaded = _load_csv_grid(f, 2, 3)
    assert np.array_equal(loaded, mg.array)


def test_from_csv_text_roundtrip_and_dim_validation():
    from osmose.maps.builder import GridSpec, MapGrid, to_csv_text, from_csv_text
    import pytest

    g = GridSpec(3, 2, 2.0, 0.0, 0.0, 3.0)
    mg = MapGrid.blank(g)
    mg.apply_cells([(1, 2)], 5.0)
    back = from_csv_text(to_csv_text(mg), g)
    assert np.array_equal(back.array, mg.array)
    with pytest.raises(ValueError):
        from_csv_text("1;2;3;4\n1;2;3;4\n", g)
