import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.schema import build_registry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import baltic_rv_overshoot_diagnostic as diag  # noqa: E402


def test_rv_gate_keys_registered():
    reg = build_registry()
    keys = {f.key_pattern for f in reg.all_fields()}
    assert "reproduction.rv.gate.enabled" in keys
    assert "reproduction.rv.gate.mode" in keys
    assert "reproduction.rv.gate.series.file" in keys
    assert "reproduction.rv.gate.ref" in keys
    assert "reproduction.rv.gate.floor" in keys
    assert "reproduction.rv.gate.start.year" in keys
    assert "reproduction.rv.gate.species.enabled.sp{idx}" in keys


def _rv_dict(years, vals, both=True):
    # 12 monthly steps/year; put the annual value in the Mar-Aug months.
    times, frac = [], []
    for y, v in zip(years, vals):
        for m in range(1, 13):
            times.append(np.datetime64(f"{y}-{m:02d}-01"))
            frac.append(v if m in diag.SPAWNING_MONTHS else 0.0)
    return {
        "available": True,
        "both_criteria": both,
        "times": np.array(times),
        "fraction": np.array(frac),
    }


def test_build_rv_gate_series_writes_rows(tmp_path):
    rv = _rv_dict([1993, 1994, 1995], [0.00, 0.07, 0.12])
    out = diag.build_rv_gate_series(rv, tmp_path / "series.csv")
    text = out.read_text().strip().splitlines()
    assert text[0] == "year,spawning_rv"
    assert text[1].startswith("1993,")
    assert len(text) == 4  # header + 3 years
    # spawning value round-trips (Mar-Aug mean == the injected value)
    assert abs(float(text[2].split(",")[1]) - 0.07) < 1e-6


def test_build_rv_gate_series_requires_both_criteria(tmp_path):
    rv = _rv_dict([1993, 1994], [0.0, 0.07], both=False)
    with pytest.raises(ValueError, match="both"):
        diag.build_rv_gate_series(rv, tmp_path / "series.csv")
