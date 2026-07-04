from osmose.schema import build_registry


def test_ceiling_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.recruitment.ceiling.enabled" in keys
    assert "reproduction.recruitment.ceiling.series.file" in keys
    assert "reproduction.recruitment.ceiling.species.enabled.sp{idx}" in keys
