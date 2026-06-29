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


def test_augment_accessibility_does_not_touch_source(tmp_path):
    """Assert that augment_accessibility writes ONLY to its target path and leaves source untouched.

    This guards the invariant that harness matrix augmentation never modifies the canonical
    source config in data/baltic/.
    """
    import hashlib
    import shutil
    from pathlib import Path

    from scripts.baltic_440_smoke import augment_accessibility

    # Read source file hash BEFORE
    source_path = Path("data/baltic/predation-accessibility.csv")
    assert source_path.exists(), f"Source file {source_path} not found"
    source_bytes_before = source_path.read_bytes()
    source_hash_before = hashlib.sha256(source_bytes_before).hexdigest()

    # Copy source to temp location
    tmp_copy = tmp_path / "predation-accessibility.csv"
    shutil.copy(source_path, tmp_copy)

    # Call augment_accessibility on the temp copy only
    augment_accessibility(tmp_copy, {"GreySeal": {"herring": 0.4}, "Cormorant": {"sprat": 0.3}})

    # Verify temp copy CHANGED (has new predator columns)
    tmp_lines = tmp_copy.read_text().splitlines()
    tmp_header = tmp_lines[0].split(";")
    assert "GreySeal" in tmp_header, "Temp copy should have GreySeal column"
    assert "Cormorant" in tmp_header, "Temp copy should have Cormorant column"

    # Verify source is UNCHANGED (byte-identical)
    source_bytes_after = source_path.read_bytes()
    source_hash_after = hashlib.sha256(source_bytes_after).hexdigest()
    assert source_hash_before == source_hash_after, (
        f"Source {source_path} was modified! Hash before: {source_hash_before}, after: {source_hash_after}"
    )
    assert source_bytes_before == source_bytes_after, (
        f"Source {source_path} bytes changed; content was mutated"
    )
