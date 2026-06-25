"""Tests for osmose.validation.fisheries_reference — per-species reference-point resolver."""

import json
from pathlib import Path

import pytest

from osmose.validation import fisheries_reference as fr

ICES = Path("data/baltic/reference/ices_snapshots")


def test_autofill_fmsy_from_primary_tonnes_stock():
    refs, unmatched = fr.load_reference_points(
        Path("/nonexistent"), ["sprat", "cod", "perch"], ices_snapshot_dir=ICES
    )
    # sprat: single tonnes stock spr.27.22-32 fmsy=0.34
    assert refs["sprat"].fmsy == pytest.approx(0.34)
    assert refs["sprat"].fmsy_stock == "spr.27.22-32"
    assert refs["sprat"].has_f_axis and not refs["sprat"].has_b_axis
    # cod: cod.27.22-24 (tonnes) chosen over cod.27.24-32 (index, null fmsy)
    assert refs["cod"].fmsy_stock == "cod.27.22-24"
    # perch: empty stock list -> no F-axis
    assert not refs["perch"].has_f_axis
    assert refs["perch"].b_ref_kind == "none"


def test_herring_multistock_deterministic_primary_with_caveat():
    refs, _ = fr.load_reference_points(Path("/nonexistent"), ["herring"], ices_snapshot_dir=ICES)
    # 3 tonnes stocks; primary = largest msy_btrigger (her.27.3031 = 613355) -- DETERMINISTIC
    assert refs["herring"].fmsy_stock == "her.27.3031"
    assert refs["herring"].fmsy == pytest.approx(0.218)
    assert any("stock" in c.lower() for c in refs["herring"].caveats)


def test_user_bmsy_and_override(tmp_path):
    (tmp_path / "fisheries_reference_points.json").write_text(
        json.dumps({"sprat": {"bmsy": 600000.0, "fmsy": 0.4}, "ghostfish": {"fmsy": 1.0}})
    )
    refs, unmatched = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert refs["sprat"].bmsy == pytest.approx(600000.0) and refs["sprat"].b_ref_kind == "bmsy_user"
    assert refs["sprat"].fmsy == pytest.approx(0.4)  # user overrides ICES
    assert "ghostfish" in unmatched  # key with no matching species


def test_save_roundtrip(tmp_path):
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    refs["sprat"].bmsy = 500000.0
    fr.save_reference_points(tmp_path, refs)
    reloaded, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert reloaded["sprat"].bmsy == pytest.approx(500000.0)


def test_save_does_not_persist_ices_autofill_fmsy(tmp_path):
    """save_reference_points must NOT freeze ICES-auto-filled fmsy -- only user-supplied."""
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    # fmsy was auto-filled from ICES snapshot (source = "ices:...")
    assert refs["sprat"].fmsy is not None
    assert refs["sprat"].source.startswith("ices:")
    fr.save_reference_points(tmp_path, refs)
    saved = json.loads((tmp_path / "fisheries_reference_points.json").read_text())
    # sprat should NOT appear in saved (no bmsy, no user-supplied fmsy)
    assert "sprat" not in saved or "fmsy" not in saved.get("sprat", {})


def test_ecosystem_of():
    assert fr.ecosystem_of(Path("/x/data/baltic")) == "baltic"
    assert fr.ecosystem_of(Path("/x/data/eec_full")) == "eec_full"


# Task 5: model sidecar tests


def _write_model(ref_dir, payload):
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "fisheries_model_reference_points.json").write_text(json.dumps(payload))


def test_model_fills_fmsy_and_bmsy(tmp_path):
    _write_model(tmp_path, {"sprat": {"fmsy": 0.5, "bmsy": 600000.0}})
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert refs["sprat"].fmsy == 0.5 and refs["sprat"].bmsy == 600000.0
    assert refs["sprat"].b_ref_kind == "bmsy_model"
    assert refs["sprat"].b_ref_label == "Bmsy [model]"
    assert "model" in refs["sprat"].source


def test_precedence_user_over_model_over_ices(tmp_path):
    _write_model(tmp_path, {"sprat": {"fmsy": 0.5, "bmsy": 600000.0}})
    (tmp_path / "fisheries_reference_points.json").write_text(
        json.dumps({"sprat": {"bmsy": 999000.0}})
    )  # user Bmsy wins
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert refs["sprat"].bmsy == 999000.0 and refs["sprat"].b_ref_kind == "bmsy_user"
    assert refs["sprat"].fmsy == 0.5  # model Fmsy kept (no user/ICES override of fmsy here)
