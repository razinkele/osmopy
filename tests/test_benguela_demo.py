from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import demo_info, list_demos, osmose_demo
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
