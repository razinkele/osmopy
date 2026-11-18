"""The map CSV loader's fast path equals the original per-part parser, bit for bit.

The loader gained a NumPy fast path (2026-11-18, C# engine task 1.11: parsing was most of the
remaining bundle build). Both engines read these grids, so a C#-vs-Python comparison cannot
catch an error here; this compares against the original parser on every shipped map file and on
a synthetic file with every special token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.engine.movement_maps import _load_csv_grid

ROOT = Path(__file__).resolve().parents[1]


def _original(path: Path, ny: int, nx: int) -> np.ndarray:
    """The loader before the fast path, verbatim in its logic."""
    grid = np.empty((ny, nx), dtype=np.float64)
    for i, line in enumerate(path.read_text().splitlines(keepends=True)):
        line = line.strip()
        if not line:
            continue
        values = []
        for part in line.split(";"):
            p = part.strip().lower()
            values.append(np.nan if p in ("na", "nan", "") else float(p))
        if len(values) < nx:
            raise ValueError("short row")
        grid[ny - 1 - i, :] = values[:nx]
    return grid


def _written_rows(path: Path, ny: int) -> list[int]:
    lines = path.read_text().splitlines(keepends=True)
    return sorted({ny - 1 - i for i, line in enumerate(lines) if line.strip()})


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.tobytes() == b.tobytes()  # NaN bit patterns included


MAP_FILES = sorted(p for p in (ROOT / "data").rglob("*.csv") if "maps" in p.parts)


def test_every_shipped_map_file_parses_identically():
    assert len(MAP_FILES) > 100
    for path in MAP_FILES:
        lines = path.read_text().splitlines(keepends=True)
        first = next(line for line in lines if line.strip())
        ny, nx = len(lines), len(first.strip().split(";"))
        rows = _written_rows(path, ny)
        assert _same(_load_csv_grid(path, ny, nx)[rows], _original(path, ny, nx)[rows]), path


def test_special_tokens_take_the_original_rule(tmp_path):
    path = tmp_path / "special.csv"
    path.write_text(
        "1;0;-99;0.5\n"
        "na;NaN; 2 ;-nan\n"
        "\n"  # a blank line still counts as a row
        "1e3;inf;;-0.0\n"
        " 3.25 ;  -1 ;7;8\n"
        # A fast-path row (no special token) that float32 cannot hold: the shipped maps are
        # all float32-exact (0/1/-99), so without it a precision loss would pass this file.
        "0.1;0.123456789012345;1e-300;2.5e-7\n"
    )
    ny, nx = 6, 4
    rows = _written_rows(path, ny)
    got, want = _load_csv_grid(path, ny, nx)[rows], _original(path, ny, nx)[rows]
    assert _same(got, want)
    assert np.isnan(got).sum() == 4 and np.isinf(got).sum() == 1  # na, NaN, -nan, empty; inf


def test_a_malformed_value_still_fails_as_before(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1;2;abc;4\n")
    with pytest.raises(ValueError):
        _load_csv_grid(path, 1, 4)
    with pytest.raises(ValueError):
        _original(path, 1, 4)
