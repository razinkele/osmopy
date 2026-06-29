"""Unit tests for Baltic 4.4.1 staging helpers.

Tests:
- inline_biomass_series: domain-total per-step series from the predator biomass NetCDF
- augment_accessibility: adds background predator columns + apex prey rows to the matrix CSV
"""


def test_inline_biomass_from_nc():
    from scripts.baltic_440_smoke import inline_biomass_series

    # domain-total per-step series, length = ndt
    series = inline_biomass_series("data/baltic/baltic_predator_biomass.nc", "GreySeal")
    assert len(series) == 24
    assert abs(series[0] - 4500.0) < 1.0  # documented standing biomass


def test_augment_accessibility_adds_predator_columns(tmp_path):
    from scripts.baltic_440_smoke import augment_accessibility

    src = tmp_path / "predation-accessibility.csv"
    src.write_text("v Prey / Predator >;cod;herring\ncod;0.05;0\nherring;0.4;0\n")
    augment_accessibility(
        src, {"GreySeal": {"herring": 0.4, "cod": 0.3}, "Cormorant": {"herring": 0.3}}
    )
    lines = src.read_text().splitlines()
    header = lines[0].split(";")
    assert "GreySeal" in header and "Cormorant" in header  # predator columns added
    # prey rows added (apex -> 0 to all predators)
    assert any(ln.startswith("GreySeal;") for ln in lines)
    # authored value present: herring row, GreySeal column
    gs_col = header.index("GreySeal")
    herring_row = next(ln.split(";") for ln in lines if ln.startswith("herring;"))
    assert float(herring_row[gs_col]) == 0.4
