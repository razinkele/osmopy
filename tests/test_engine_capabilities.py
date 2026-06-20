from osmose.engine_capabilities import EngineCapability, _is_enabled, describe_engine


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


def test_python_base_pages_always_populate():
    cap = describe_engine("python", {})
    assert cap.engine == "python"
    assert cap.can_run is True
    assert cap.block_reason is None
    assert "Results" in cap.pages_populated
    assert "Diagnostics" in cap.pages_populated
    # disabled-by-default modules are empty
    assert "Genetics" in cap.pages_empty
    assert "Economic" in cap.pages_empty
    assert "Spatial Results" in cap.pages_empty


def test_python_genetics_gated_on_module_flag():
    cap = describe_engine("python", {"module.genetics.enabled": "true"})
    assert "Genetics" in cap.pages_populated
    assert "Genetics" not in cap.pages_empty


def test_python_economics_and_spatial_gates():
    cap = describe_engine(
        "python",
        {"module.bioeconomics.enabled": "true", "output.spatial.enabled": "true"},
    )
    assert "Economic" in cap.pages_populated
    assert "Spatial Results" in cap.pages_populated


def test_python_notable_outputs_mentions_java_only_families():
    cap = describe_engine("python", {})
    assert "sizeSpectrum" in cap.notable_outputs


def test_java_plain_config_runs_results_only():
    cap = describe_engine("java", {})
    assert cap.engine == "java"
    assert cap.can_run is True
    assert cap.block_reason is None
    assert cap.pages_populated == ["Results"]
    for page in ("Diagnostics", "Genetics", "Economic", "Spatial Results"):
        assert page in cap.pages_empty


def test_java_background_species_blocked():
    cap = describe_engine("java", {"simulation.nbackground": "2"})
    assert cap.can_run is False
    assert cap.block_reason is not None
    assert "background" in cap.block_reason.lower()


def test_java_notable_outputs_mentions_equivalence():
    cap = describe_engine("java", {})
    assert "statistically equivalent" in cap.notable_outputs


def test_unknown_engine_returns_total_fallback():
    cap = describe_engine("rust", {})
    assert cap.engine == "rust"
    assert cap.can_run is False
    assert cap.block_reason is not None
    assert "rust" in cap.block_reason
    assert cap.pages_populated == []
    assert cap.pages_empty == []


def test_describe_engine_does_not_share_list_state_across_calls():
    first = describe_engine("java", {})
    first.pages_empty.append("MUTATED")
    second = describe_engine("java", {})
    assert "MUTATED" not in second.pages_empty


def test_is_enabled_only_true_and_one_count_as_enabled():
    # Truthy-looking but not the sanctioned tokens → disabled.
    assert _is_enabled({"k": "yes"}, "k") is False
    assert _is_enabled({"k": "0"}, "k") is False
    # Non-string values are coerced safely.
    assert _is_enabled({"k": 1}, "k") is True
    assert _is_enabled({"k": 0}, "k") is False
    assert _is_enabled({"k": True}, "k") is True
