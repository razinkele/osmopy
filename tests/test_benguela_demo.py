from pathlib import Path

import numpy as np
import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import demo_info, list_demos, osmose_demo
from osmose.engine import PythonEngine
from osmose.runner import java_engine_block_reason


def test_benguela_registered():
    assert "benguela" in list_demos()
    info = demo_info("benguela")
    assert info is not None
    for field in ("title", "region", "species", "resources", "engine", "summary"):
        assert info.get(field), f"DEMO_INFO['benguela'] missing {field}"


def test_benguela_generates_and_copies(tmp_path: Path):
    out = osmose_demo("benguela", tmp_path)
    cfg = Path(out["config_file"])
    assert cfg.name == "benguela_all-parameters.csv" and cfg.exists()
    cfgdir = cfg.parent
    assert (cfgdir / "input" / "roms_climatological_merged.nc").exists()
    assert (cfgdir / "input" / "reproduction").is_dir()
    assert list((cfgdir / "maps").glob("*.csv"))
    assert not (cfgdir / "input" / "fisheries").exists()
    raw = dict(OsmoseConfigReader().read(str(cfg)))
    assert raw["simulation.nspecies"] == "10"


def test_benguela_blocks_java_engine():
    raw = {"output.file.prefix": "benguela", "simulation.nbackground": "0"}
    reason = java_engine_block_reason(raw)
    assert reason is not None and "benguela" in reason.lower()


EXPECTED_SEED = {
    0: 3129213,
    1: 3888750,
    2: 3029155,
    3: 1286364,
    4: 1138339,
    5: 1439984,
    6: 198865,
    7: 81054,
    8: 575361,
    9: 591907,
}


def _run(tmp_path):
    out = osmose_demo("benguela", tmp_path)
    raw = dict(OsmoseConfigReader().read(str(out["config_file"])))
    return PythonEngine().run_in_memory(raw, seed=42)


def test_benguela_smoke_bounded_and_positive(tmp_path):
    b = _run(tmp_path).biomass()
    sp_cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    assert len(sp_cols) == 10
    for i, c in enumerate(sp_cols):
        v = b[c].to_numpy(dtype=float)
        assert np.all(np.isfinite(v)), f"{c} has NaN/inf"
        assert v[-1] > 0, f"{c} collapsed to 0"
        assert np.all(v <= 1000.0 * EXPECTED_SEED[i]), f"{c} exceeds 1000x seed (explosion)"
        assert np.all(v <= 1e9), f"{c} exceeds 1e9 t (explosion)"

    # Nine of ten species stay healthy at the pinned 15-yr horizon; mesopelagic (sp5) is a known,
    # documented uncalibrated-example decliner (ends ~1e-10 of seed). This guard keeps the gate
    # honest: `v[-1] > 0` alone can't distinguish healthy from collapsing, so require >=9/10 species
    # above a meaningful floor — a regression collapsing a SECOND species then fails here.
    n_healthy = sum(
        b[c].to_numpy(dtype=float)[-1] >= 1e-3 * EXPECTED_SEED[i] for i, c in enumerate(sp_cols)
    )
    assert n_healthy >= 9, f"only {n_healthy}/10 species above the healthy floor (expected >=9)"


def test_benguela_resources_load_nonzero(tmp_path):
    ds = xr.open_dataset(
        Path(osmose_demo("benguela", tmp_path)["config_file"]).parent
        / "input"
        / "roms_climatological_merged.nc"
    )
    try:
        for v in ("sphy", "lphy", "szoo", "lzoo"):
            assert float(np.nansum(ds[v].values)) > 0, f"resource {v} is empty"
    finally:
        ds.close()


def test_benguela_deterministic(tmp_path):
    a = _run(tmp_path / "a").biomass()
    cols = [c for c in a.columns if c not in ("Time", "time", "species")]
    va = a[cols].to_numpy(dtype=float)
    vb = _run(tmp_path / "b").biomass()[cols].to_numpy(dtype=float)
    assert np.array_equal(va, vb), "fixed-seed runs diverge"
