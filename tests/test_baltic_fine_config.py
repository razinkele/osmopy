from pathlib import Path
from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig


def _cfg(variant):
    p = sorted(Path("data/baltic-fine").glob(f"*{variant}_all-parameters*.csv"))[0]
    return dict(OsmoseConfigReader().read(str(p)))


def test_both_variants_load_4x_and_construct():
    for v in ["upsampled", "real"]:
        cfg = _cfg(v)
        assert cfg["grid.nlon"] == "200" and cfg["grid.nlat"] == "160"
        assert EngineConfig.from_dict(cfg).n_species >= 8


def test_variants_differ_only_in_percid_maps_after_include_resolution():
    up, real = _cfg("upsampled"), _cfg("real")
    diffs = {k for k in set(up) | set(real) if up.get(k) != real.get(k)}
    # only the 6 percid movement.file.map values (map13..18) may differ
    assert diffs, "variants identical -> rung2==rung3 (the C1 bug)"
    percid_map_keys = {f"movement.file.map{n}" for n in range(13, 19)}
    assert all(k in percid_map_keys for k in diffs), f"unexpected non-percid diffs: {diffs}"
    assert any("upsampled" in up[k] for k in diffs)  # up points at *_upsampled.csv
    assert all("upsampled" not in real[k] for k in diffs)  # real points at the binary maps
