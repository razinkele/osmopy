import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from apply_calibration import apply_calibration, set_key  # noqa: E402


def test_set_key_updates_existing_line_preserving_comments(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("# a comment\nstock.recruitment.type.sp0;beverton_holt\nother.key;9\n")
    set_key(f, "stock.recruitment.type.sp0", "shepherd")
    lines = f.read_text().splitlines()
    assert "# a comment" in lines  # comment preserved
    assert "stock.recruitment.type.sp0;shepherd" in lines
    assert "other.key;9" in lines  # untouched


def test_set_key_appends_when_absent(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("existing;1\n")
    set_key(f, "stock.recruitment.shape.sp2", "1.5")
    lines = f.read_text().splitlines()
    assert "stock.recruitment.shape.sp2;1.5" in lines
    assert "existing;1" in lines


def test_apply_calibration_roundtrips_through_reader(tmp_path):
    cfg = tmp_path
    (cfg / "baltic_param-reproduction.csv").write_text("stock.recruitment.type.sp0;beverton_holt\n")
    (cfg / "baltic_param-additional-mortality.csv").write_text(
        "mortality.additional.rate.sp0;0.1\n"
    )
    (cfg / "baltic_param-fishing.csv").write_text("fisheries.rate.base.sp0;0.2\n")
    results = cfg / "r.json"
    results.write_text(
        json.dumps(
            {
                "parameters": {
                    "mortality.additional.rate.sp0": 3.7,
                    "fisheries.rate.base.sp0": 0.077,
                    "stock.recruitment.shape.sp0": 1.88,
                    "stock.recruitment.ssbhalf.sp0": 120000.0,
                }
            }
        )
    )
    apply_calibration(results, cfg)
    repro = (cfg / "baltic_param-reproduction.csv").read_text().splitlines()
    assert "stock.recruitment.type.sp0;shepherd" in repro
    assert "stock.recruitment.shape.sp0;1.88" in repro
    mort = (cfg / "baltic_param-additional-mortality.csv").read_text().splitlines()
    assert "mortality.additional.rate.sp0;3.7" in mort
    # all 8 species switched to shepherd
    assert sum(1 for line in repro if line.startswith("stock.recruitment.type.sp")) == 8


def test_apply_calibration_scales_larval_rate_by_ndtperyear(tmp_path):
    # The reader divides larval rate by ndtperyear on read; the file must therefore
    # store authored_value * ndt so the reader recovers the authored value.
    cfg = tmp_path
    (cfg / "baltic_param-simulation.csv").write_text("simulation.time.ndtperyear;24\n")
    (cfg / "baltic_param-reproduction.csv").write_text("")
    (cfg / "baltic_param-additional-mortality.csv").write_text("")
    (cfg / "baltic_param-fishing.csv").write_text("")
    results = cfg / "r.json"
    results.write_text(
        json.dumps(
            {
                "parameters": {
                    "mortality.additional.larva.rate.sp0": 10.0,  # authored -> file must be 240.0
                    "mortality.additional.rate.sp0": 0.5,  # adult: identity
                }
            }
        )
    )
    apply_calibration(results, cfg)
    mort = (cfg / "baltic_param-additional-mortality.csv").read_text().splitlines()
    assert "mortality.additional.larva.rate.sp0;240.0" in mort
    assert "mortality.additional.rate.sp0;0.5" in mort  # adult unscaled
