import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.config import _load_rv_gate
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


def _write_series(tmp_path, years, vals, name="s.csv"):
    # `name` MUST be unique per file so tests that need two different series in
    # the same tmp_path do not overwrite each other (a good series written by
    # _cfg would otherwise clobber a bad series written for a validation test).
    p = tmp_path / name
    rows = ["year,spawning_rv"] + ["%d,%.6f" % (y, v) for y, v in zip(years, vals)]
    p.write_text("\n".join(rows) + "\n")
    return p


def _cfg(tmp_path, **over):
    series = _write_series(
        tmp_path, range(1993, 1998), [0.0, 0.10, 0.20, 0.05, 0.15], name="good.csv"
    )
    base = {
        "reproduction.rv.gate.enabled": "true",
        "reproduction.rv.gate.mode": "mean_preserving",
        "reproduction.rv.gate.series.file": str(series),
        "reproduction.rv.gate.start.year": "1993",
        "reproduction.rv.gate.species.enabled.sp0": "true",
        "_osmose.config.dir": str(tmp_path),
    }
    base.update(over)
    return base


def test_load_rv_gate_disabled_returns_none():
    fac, mask, off = _load_rv_gate({"reproduction.rv.gate.enabled": "false"}, 3, 24, 5)
    assert fac is None and mask is None and off == 0


def test_load_rv_gate_mean_preserving_full_window(tmp_path):
    # 5-year run over the full 5-year series -> window == all rows -> mean(fac) == 1.
    fac, mask, off = _load_rv_gate(_cfg(tmp_path), n_species=1, n_dt_per_year=24, n_year=5)
    assert off == 0
    assert mask.tolist() == [True]
    assert abs(float(np.mean(fac)) - 1.0) < 1e-9


def test_load_rv_gate_mean_preserving_window_subset(tmp_path):
    # 3-year run over a 5-row series with asymmetric values: the denominator MUST
    # be the mean over the SAMPLED window (rows 0..2), not the whole array. This
    # is the test that actually proves the windowing (the full-window test above
    # would pass even for a whole-array denominator).
    series = _write_series(
        tmp_path, range(1993, 1998), [0.10, 0.20, 0.30, 0.40, 0.50], name="sub.csv"
    )
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.gate.series.file"] = str(series)
    fac, _, _ = _load_rv_gate(cfg, 1, 24, 3)  # window = rows 0,1,2 -> D = mean(.1,.2,.3) = 0.20
    assert fac.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])
    assert abs(float(np.mean(fac[[0, 1, 2]])) - 1.0) < 1e-9  # window mean == 1
    assert float(np.mean(fac)) == pytest.approx(1.5)  # whole-array mean != 1 -> scoping proven


def test_load_rv_gate_offset_indexes_window(tmp_path):
    # start_year 1995 -> offset 2; 3-year window = rows 2,3,4 -> D = mean(.3,.4,.5) = 0.40.
    series = _write_series(
        tmp_path, range(1993, 1998), [0.10, 0.20, 0.30, 0.40, 0.50], name="off.csv"
    )
    cfg = _cfg(tmp_path, **{"reproduction.rv.gate.start.year": "1995"})
    cfg["reproduction.rv.gate.series.file"] = str(series)
    fac, _, off = _load_rv_gate(cfg, 1, 24, 3)
    assert off == 2
    assert fac[2] == pytest.approx(0.75)  # 0.30 / 0.40
    assert fac[4] == pytest.approx(1.25)  # 0.50 / 0.40


def test_load_rv_gate_raw_cap_clips(tmp_path):
    cfg = _cfg(
        tmp_path, **{"reproduction.rv.gate.mode": "raw_cap", "reproduction.rv.gate.ref": "0.10"}
    )
    fac, _, _ = _load_rv_gate(cfg, 1, 24, 5)
    assert fac.min() >= 0.0 and fac.max() <= 1.0
    assert fac[0] == 0.0  # rv=0.0 -> 0
    assert fac[1] == pytest.approx(1.0)  # rv=0.10 == ref -> 1


@pytest.mark.parametrize(
    "bad,exc",
    [
        ({"reproduction.rv.gate.mode": "nope"}, "mode"),
        ({"reproduction.rv.gate.species.enabled.sp0": "false"}, "no species"),
        ({"reproduction.rv.gate.mode": "raw_cap", "reproduction.rv.gate.ref": "0"}, "ref"),
        ({"reproduction.rv.gate.floor": "2.0"}, "floor"),
    ],
)
def test_load_rv_gate_fail_fast_config(tmp_path, bad, exc):
    with pytest.raises(ValueError, match=exc):
        _load_rv_gate(_cfg(tmp_path, **bad), 1, 24, 5)


def test_load_rv_gate_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        _load_rv_gate(_cfg(tmp_path, **{"reproduction.rv.gate.series.file": ""}), 1, 24, 5)


def test_load_rv_gate_nan_rv_raises(tmp_path):
    p = tmp_path / "nan.csv"
    p.write_text("year,spawning_rv\n1993,0.1\n1994,nan\n1995,0.2\n")
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.gate.series.file"] = str(p)
    with pytest.raises(ValueError, match="NaN|negative"):
        _load_rv_gate(cfg, 1, 24, 3)


def test_load_rv_gate_zero_denominator_raises(tmp_path):
    # all-zero window under mean_preserving -> D == 0.
    series = _write_series(tmp_path, range(1993, 1996), [0.0, 0.0, 0.0], name="zero.csv")
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.gate.series.file"] = str(series)
    with pytest.raises(ValueError, match="denominator"):
        _load_rv_gate(cfg, 1, 24, 3)


def test_load_rv_gate_nonascending_years_raises(tmp_path):
    cfg = _cfg(tmp_path)  # writes good.csv and points series.file at it
    bad = _write_series(tmp_path, [1993, 1995, 1994], [0.1, 0.1, 0.1], name="bad.csv")
    cfg["reproduction.rv.gate.series.file"] = str(bad)  # now point at the bad file
    with pytest.raises(ValueError, match="contiguous"):
        _load_rv_gate(cfg, 1, 24, 3)
