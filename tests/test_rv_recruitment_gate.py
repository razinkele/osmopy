from osmose.schema import build_registry


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
