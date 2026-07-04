import pytest

from osmose.engine.config import _load_recruitment_ceiling
from osmose.schema import build_registry


def test_ceiling_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.recruitment.ceiling.enabled" in keys
    assert "reproduction.recruitment.ceiling.series.file" in keys
    assert "reproduction.recruitment.ceiling.species.enabled.sp{idx}" in keys


def _write_ceiling_csv(path, n_cols, cols):
    # cols: dict {species_index: [value per season_idx]}
    sp_ids = sorted(cols)
    header = "season_idx," + ",".join(f"ceiling_sp{i}" for i in sp_ids)
    lines = [header]
    for s in range(n_cols):
        row = [str(s)] + [f"{cols[i][s]:.6f}" for i in sp_ids]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _cfg(tmp_path, csv_path, enabled_species=(0,)):
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.recruitment.ceiling.enabled": "true",
        "reproduction.recruitment.ceiling.series.file": csv_path.name,
    }
    for sp in enabled_species:
        cfg[f"reproduction.recruitment.ceiling.species.enabled.sp{sp}"] = "true"
    return cfg


def test_ceiling_off_returns_none(tmp_path):
    ceil, mask = _load_recruitment_ceiling({}, 3, 12, None)
    assert ceil is None and mask is None


def test_ceiling_loads_shape_and_mask(tmp_path):
    csv = _write_ceiling_csv(
        tmp_path / "c.csv", 12, {0: [10.0] * 12, 1: [20.0] * 12, 2: [30.0] * 12}
    )
    ceil, mask = _load_recruitment_ceiling(_cfg(tmp_path, csv, (0, 2)), 3, 12, None)
    assert ceil.shape == (12, 3)
    assert list(mask) == [True, False, True]
    assert ceil[5, 1] == 20.0


def test_ceiling_row_count_must_match_ncols(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 6, {0: [10.0] * 6})
    with pytest.raises(ValueError, match="season"):
        _load_recruitment_ceiling(_cfg(tmp_path, csv), 1, 12, None)


def test_ceiling_rejects_negative(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [-1.0] + [10.0] * 11})
    with pytest.raises(ValueError, match="negative|NaN"):
        _load_recruitment_ceiling(_cfg(tmp_path, csv), 1, 12, None)


def test_ceiling_requires_enabled_species(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [10.0] * 12})
    cfg = _cfg(tmp_path, csv, enabled_species=())
    with pytest.raises(ValueError, match="no species enabled"):
        _load_recruitment_ceiling(cfg, 1, 12, None)


def test_ceiling_missing_file_raises(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        _load_recruitment_ceiling(cfg, 1, 12, None)
