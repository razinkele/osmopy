from osmose.engine_capabilities import EngineCapability, _is_enabled


def test_is_enabled_truthiness():
    assert _is_enabled({"k": "true"}, "k") is True
    assert _is_enabled({"k": "True"}, "k") is True
    assert _is_enabled({"k": "1"}, "k") is True
    assert _is_enabled({"k": "false"}, "k") is False
    assert _is_enabled({"k": ""}, "k") is False
    assert _is_enabled({}, "k") is False


def test_capability_dataclass_fields():
    cap = EngineCapability(
        engine="python",
        can_run=True,
        block_reason=None,
        pages_populated=["Results"],
        pages_empty=["Genetics"],
        notable_outputs="x",
    )
    assert cap.engine == "python"
    assert cap.can_run is True
    assert cap.pages_populated == ["Results"]
