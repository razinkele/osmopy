from osmose.schema import build_registry


def test_rv_spatial_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.rv.spatial.enabled" in keys
    assert "reproduction.rv.spatial.field.file" in keys
    assert "reproduction.rv.spatial.field.varname" in keys
    assert "reproduction.rv.spatial.ref" in keys
    assert "reproduction.rv.spatial.species.enabled.sp{idx}" in keys
