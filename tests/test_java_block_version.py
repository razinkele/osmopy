"""C2: java_engine_block_reason is version-aware (allow staging-supported nbackground>0 on >=4.4.0)."""

from osmose.config.reader import OsmoseConfigReader
from osmose.runner import java_engine_block_reason


def _baltic():
    return dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))


def test_block_matrix():
    focal = {"simulation.nbackground": "0"}
    assert java_engine_block_reason(focal, "4.4.1") is None  # no background -> always allowed

    bal = _baltic()
    assert java_engine_block_reason(bal, "4.3.3") is not None  # bg + 4.3.3 jar -> block
    assert (
        java_engine_block_reason(bal, None) is not None
    )  # bg + unknown jar -> block (conservative)
    assert java_engine_block_reason(bal, "4.4.1") is None  # bg + 4.4.1 + Baltic-supported -> allow

    unknown = {
        "simulation.nbackground": "1",
        "species.type.sp9": "background",
        "species.name.sp9": "Yeti",
    }
    block = java_engine_block_reason(unknown, "4.4.1")  # bg + 4.4.1 + unsupported species -> block
    assert block is not None
    assert "Yeti" in block


def test_block_back_compat_default_jar_version_none():
    # default jar_version=None preserves the old (conservative) behaviour for existing callers
    assert java_engine_block_reason(_baltic()) is not None
    assert java_engine_block_reason({"simulation.nbackground": "0"}) is None


def test_gate_wiring_jar_to_version_unblocks_baltic():
    # mirrors the UI gate: java_engine_block_reason(config, target_version_for_jar(jar_path))
    from osmose.config.aliases import target_version_for_jar

    bal = _baltic()
    default_jar = "osmose-java/osmose-4.4.1-jar-with-dependencies.jar"  # ui/state.py default
    assert java_engine_block_reason(bal, target_version_for_jar(default_jar)) is None
    legacy_jar = "osmose-java/osmose_4.3.3-jar-with-dependencies.jar"
    assert java_engine_block_reason(bal, target_version_for_jar(legacy_jar)) is not None
