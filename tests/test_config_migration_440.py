from osmose.config.aliases import canonicalize_config, to_target_keys
from osmose.demo import migrate_config
from osmose.engine.config import EngineConfig


def _min_bioen_cfg(extra: dict) -> dict[str, str]:
    base = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
        "species.k.sp0": "0.364",
        "species.t0.sp0": "-0.70",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.06",
        "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "0",
        "mortality.subdt": "10",
        "predation.efficiency.critical.sp0": "0.57",
    }
    base.update(extra)
    return base


def test_from_dict_accepts_old_module_toggle_key():
    cfg_old = _min_bioen_cfg(
        {"simulation.bioen.enabled": "false", "predation.ingestion.rate.max.sp0": "3.5"}
    )
    config = EngineConfig.from_dict(cfg_old)
    assert config.bioen_enabled is False


def test_from_dict_unified_ingestion_read():
    cfg = _min_bioen_cfg(
        {
            "simulation.bioen.enabled": "false",
            "predation.ingestion.rate.max.bioen.sp0": "4.2",
        }
    )
    config = EngineConfig.from_dict(cfg)
    assert config.ingestion_rate[0] == 4.2


def test_440_clean_renames():
    cfg = {
        "osmose.version": "4.3.3",
        "simulation.bioen.enabled": "true",
        "fisheries.enabled": "false",
        "output.restart.spinup": "5",
        "output.fishery.enabled": "true",
        "output.spatial.fishery.enabled": "true",
        "species.bioen.maturity.eta.sp0": "1.2",
        "species.bioen.maturity.m0.sp3": "0.5",
    }
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["module.multispecies.fisheries.enabled"] == "false"
    assert out["simulation.restart.spinup.nyear"] == "5"
    assert out["output.fisheries.enabled"] == "true"
    assert out["output.spatial.fisheries.enabled"] == "true"
    assert out["species.maturity.eta.sp0"] == "1.2"
    assert out["species.maturity.m0.sp3"] == "0.5"
    assert "simulation.bioen.enabled" not in out
    assert "species.bioen.maturity.eta.sp0" not in out
    assert out["osmose.version"] == "4.4.0"


def test_440_ingestion_merge_skip_if_target_exists():
    cfg = {
        "osmose.version": "4.3.3",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.bioen.sp0": "3.0",
        "predation.ingestion.rate.max.bioen.sp1": "4.0",
    }
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"  # base kept (skip-if-exists)
    assert "predation.ingestion.rate.max.bioen.sp0" not in out  # bioen dropped
    assert out["predation.ingestion.rate.max.sp1"] == "4.0"  # bioen-only -> renamed


def test_440_idempotent_on_new_keys():
    cfg = {"osmose.version": "4.4.0", "module.bioenergetics.enabled": "true"}
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" not in out


def test_canonicalize_reports_deprecated_and_canonicalizes():
    cfg = {"osmose.version": "4.3.3", "simulation.bioen.enabled": "true"}
    out, deprecated = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" in deprecated


def test_canonicalize_missing_version_does_not_corrupt_new_keys():
    # No osmose.version, but config already uses NEW keys -> must stay new.
    cfg = {"module.bioenergetics.enabled": "true", "predation.ingestion.rate.max.sp0": "3.5"}
    out, _ = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"


def test_canonicalize_snapshot_version_handled():
    cfg = {"osmose.version": "4.4.0-SNAPSHOT", "simulation.bioen.enabled": "true"}
    out, _ = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"


def test_to_target_keys_reverses_to_4_3_3():
    cfg = {
        "osmose.version": "4.4.0",
        "module.bioenergetics.enabled": "true",
        "simulation.restart.spinup.nyear": "5",
        "species.maturity.eta.sp0": "1.2",
        "predation.ingestion.rate.max.sp0": "3.5",
        # pre-existing GROWTH maturity keys that must NOT be touched by the inverse:
        "species.maturity.size.sp0": "20.0",
        "species.maturity.age.sp0": "2.0",
    }
    out = to_target_keys(cfg, target_version="4.3.3")
    assert out["simulation.bioen.enabled"] == "true"
    assert out["output.restart.spinup"] == "5"
    assert out["species.bioen.maturity.eta.sp0"] == "1.2"
    assert out["species.maturity.size.sp0"] == "20.0"  # untouched
    assert out["species.maturity.age.sp0"] == "2.0"  # untouched
    assert out["osmose.version"] == "4.3.3"


def test_to_target_keys_4_4_0_is_identity_plus_stamp():
    cfg = {"osmose.version": "4.3.3", "module.bioenergetics.enabled": "true"}
    out = to_target_keys(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["osmose.version"] == "4.4.0"


def test_to_target_keys_ingestion_merge_is_lossy_to_legacy_key():
    cfg = {"osmose.version": "4.4.0", "predation.ingestion.rate.max.sp0": "3.5"}
    out = to_target_keys(cfg, target_version="4.3.3")
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"  # stays the legacy key
    assert not any(".bioen." in k for k in out)  # no .bioen fabricated


def test_inverse_is_faithful_inverse_of_renames():
    # DRIFT GUARD: _INVERSE_440 hand-maintained alongside RENAMES_440. Every forward rename
    # — except the lossy merge and the maturity prefix that fans out to leaves — must have a
    # faithful inverse; no inverse may be orphaned.
    from osmose.config.aliases import _INVERSE_440, RENAMES_440

    LOSSY = {"predation.ingestion.rate.max.bioen"}  # merge -> not invertible
    LEAF_FANOUT = {"species.bioen.maturity"}  # prefix -> 4 leaves
    for old, new in RENAMES_440.items():
        if old in LOSSY or old in LEAF_FANOUT:
            continue
        assert _INVERSE_440.get(new) == old, f"{new} has no faithful inverse"
    for leaf in ("eta", "r", "m0", "m1"):
        assert _INVERSE_440[f"species.maturity.{leaf}"] == f"species.bioen.maturity.{leaf}"
    forward_new = set(RENAMES_440.values()) | {
        f"species.maturity.{leaf}" for leaf in ("eta", "r", "m0", "m1")
    }
    for new in _INVERSE_440:
        assert any(new == f or new.startswith(f) for f in forward_new), f"orphan inverse {new}"


def test_config_validation_clean_on_old_keys():
    # An old-key config validates warning-free (old keys canonicalized before the
    # unknown-key check; new keys recognized via the allowlist).
    from osmose.engine.config_validation import validate

    cfg = {
        "osmose.version": "4.3.3",
        "simulation.bioen.enabled": "true",
        "output.restart.enabled": "false",
        "economy.enabled": "false",
    }
    unknowns = validate(cfg, mode="warn")  # list[UnknownKey], each with .key
    flagged = {u.key for u in unknowns}
    assert "simulation.bioen.enabled" not in flagged
    assert "economy.enabled" not in flagged
    assert "output.restart.enabled" not in flagged


def test_bioen_ingestion_unification_is_consistent():
    """After canonicalization, a bioen config with BOTH ingestion keys uses the SINGLE
    unified predation.ingestion.rate.max for both predation and the energy budget — the
    intended 4.4.0 behavior. Base value wins (skip-if-target-exists). This is an intended
    result change vs 4.3.x for bioen/Ev-OSMOSE configs (changelog note lands in PR2)."""
    cfg = _min_bioen_cfg(
        {
            "simulation.bioen.enabled": "true",
            "predation.ingestion.rate.max.sp0": "3.5",  # legacy/base
            "predation.ingestion.rate.max.bioen.sp0": "3.0",  # bioen — dropped on merge
        }
    )
    canon, _ = canonicalize_config(cfg)
    assert canon["predation.ingestion.rate.max.sp0"] == "3.5"  # base wins
    assert "predation.ingestion.rate.max.bioen.sp0" not in canon  # bioen dropped
    config = EngineConfig.from_dict(cfg)
    # the unified value (3.5) drives the engine's ingestion read used by both paths:
    assert config.ingestion_rate[0] == 3.5


def test_schema_uses_new_440_key_patterns():
    from osmose.schema import build_registry  # exported from osmose/schema/__init__.py

    reg = build_registry()
    patterns = {f.key_pattern for f in reg.all_fields()}
    assert "module.bioenergetics.enabled" in patterns
    assert "module.multispecies.fisheries.enabled" in patterns
    assert "module.genetics.enabled" in patterns
    assert "module.bioeconomics.enabled" in patterns
    assert "simulation.restart.enabled" in patterns
    assert "species.maturity.eta.sp{idx}" in patterns
    assert "predation.ingestion.rate.max.bioen.sp{idx}" not in patterns  # old gone
    assert "simulation.bioen.enabled" not in patterns
    assert "fisheries.enabled" not in patterns
    assert "economy.enabled" not in patterns
    assert "output.restart.enabled" not in patterns
    assert "output.fishery.enabled" not in patterns
    assert "simulation.genetic.enabled" not in patterns
    assert "species.bioen.maturity.eta.sp{idx}" not in patterns


def test_reader_canonicalizes_and_rebuilds_case_map(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    f = tmp_path / "osm_param-simulation.csv"
    f.write_text("simulation.bioen.enabled ; true\noutput.restart.spinup ; 5\n")
    reader = OsmoseConfigReader()
    cfg = reader.read_file(f)
    assert cfg["module.bioenergetics.enabled"] == "true"  # canonicalized on read
    assert cfg["simulation.restart.spinup.nyear"] == "5"
    assert "simulation.bioen.enabled" not in cfg


def test_reader_case_map_preserves_renamed_key_source_casing(tmp_path):
    # REGRESSION: a renamed key that is camelCase in the source must survive a
    # read -> canonicalize(new) -> to_target_keys(old) round-trip with ORIGINAL casing.
    from osmose.config.reader import OsmoseConfigReader
    from osmose.config.aliases import to_target_keys

    f = tmp_path / "osm_param-output.csv"
    f.write_text("output.fishery.byAge.enabled ; true\n")
    reader = OsmoseConfigReader()
    cfg = reader.read_file(f)  # cfg holds the NEW (4.4.0) key
    old_cfg = to_target_keys(cfg, target_version="4.3.3")  # back to the OLD key for the jar
    (old_key,) = [k for k in old_cfg if k.endswith("byage.enabled")]
    assert reader.key_case_map.get(old_key) == "output.fishery.byAge.enabled"


def test_appstate_load_config_canonicalizes():
    from shiny import reactive

    from ui.state import AppState

    st = AppState()
    st.load_config({"osmose.version": "4.3.3", "simulation.bioen.enabled": "true"})
    with reactive.isolate():
        cfg = st.config.get()
    assert cfg["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" not in cfg


def test_grid_load_surfaces_deprecation_notification():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "ui" / "pages" / "grid.py").read_text()
    assert "load_config" in src and "notification_show" in src
    assert "reader.deprecated_keys" in src


def test_reader_exposes_deprecated_keys(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    f = tmp_path / "osm_all-parameters.csv"
    f.write_text(
        "osmose.version ; 4.3.3\nsimulation.bioen.enabled ; true\noutput.restart.enabled ; false\n"
    )
    reader = OsmoseConfigReader()
    reader.read(f)
    assert "simulation.bioen.enabled" in reader.deprecated_keys
    assert "output.restart.enabled" in reader.deprecated_keys


def test_to_target_keys_collapses_mixed_old_and_new():
    from osmose.config.aliases import to_target_keys

    mixed = {"module.bioenergetics.enabled": "true", "simulation.bioen.enabled": "false"}
    out = to_target_keys(mixed, target_version="4.3.3")
    assert "module.bioenergetics.enabled" not in out  # redundant NEW form dropped
    assert out["simulation.bioen.enabled"] == "false"  # existing OLD value wins (base-wins)


def test_writer_default_target_emits_old_keys(tmp_path):
    from osmose.config.writer import OsmoseConfigWriter

    cfg = {
        "osmose.version": "4.4.0",
        "module.bioenergetics.enabled": "true",
        "simulation.restart.spinup.nyear": "5",
    }
    OsmoseConfigWriter().write(cfg, tmp_path)  # default target_version="4.3.3"
    # The writer routes keys to sub-files by prefix, so read across all CSVs.
    raw = "".join(p.read_text() for p in sorted(tmp_path.glob("*.csv")))
    assert "simulation.bioen.enabled" in raw  # reverse-mapped to old
    assert "output.restart.spinup" in raw  # routed to osm_param-output.csv
    assert "module.bioenergetics.enabled" not in raw


def test_write_temp_config_default_target_emits_old_keys(tmp_path):
    from ui.pages.run import write_temp_config

    master = write_temp_config(
        {
            "module.multispecies.fisheries.enabled": "false",
            "module.bioenergetics.enabled": "true",
        },
        tmp_path,
    )
    raw = master.read_text()
    assert "fisheries.enabled" in raw
    assert "module.multispecies.fisheries.enabled" not in raw


def test_export_writes_target_format(tmp_path):
    from osmose.config.writer import OsmoseConfigWriter

    OsmoseConfigWriter().write({"module.bioenergetics.enabled": "true"}, tmp_path)
    raw = (tmp_path / "osm_all-parameters.csv").read_text()
    assert "simulation.bioen.enabled" in raw  # export inherits the 4.3.3 reverse-map


def test_writer_roundtrip_of_canonical_config_is_faithful(tmp_path):
    """A fully-canonical config survives write (target 4.3.3) -> read losslessly.

    The writer stamps ``osmose.version=4.3.3`` and the reader's canonicalize is
    version-gated by that stamp, so pre-4.3.3 migrations are skipped on read-back.
    A canonical config (the production reality — state.config is always canonical)
    round-trips faithfully because its keys are already in 4.4.0 form.

    NOTE: the input here deliberately carries NO ``osmose.version`` so that
    ``canonicalize_config`` applies the full migration chain and produces the
    canonical ``mortality.additional.rate.sp0`` (the pre-4.3.3 rename of
    ``mortality.natural.rate``). Stamping ``osmose.version=4.3.3`` on the input
    would gate that rename and leave the legacy key — which is exactly the
    anachronistic, non-production scenario we are guarding against.
    """
    from osmose.config.aliases import canonicalize_config
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import write_temp_config

    canon, _ = canonicalize_config(
        {
            "simulation.bioen.enabled": "true",
            "mortality.natural.rate.sp0": "0.2",
        }
    )
    # canon holds NEW keys + mortality.additional.rate.sp0 (fully migrated).
    assert canon["mortality.additional.rate.sp0"] == "0.2"
    assert "mortality.natural.rate.sp0" not in canon

    master = write_temp_config(canon, tmp_path)
    back = OsmoseConfigReader().read(master)
    assert back["module.bioenergetics.enabled"] == "true"
    assert back["mortality.additional.rate.sp0"] == "0.2"  # survives the canonical round-trip
    assert "mortality.natural.rate.sp0" not in back


def test_calibration_java_cmd_reverse_maps_override_keys(tmp_path):
    from unittest.mock import MagicMock, patch

    from osmose.calibration.problem import FreeParameter
    from tests.test_calibration_problem import _make_problem  # reuse the existing helper

    problem = _make_problem(
        tmp_path, free_params=[FreeParameter("species.maturity.eta.sp0", 0.1, 0.5)]
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)  # short-circuit after cmd build
        try:
            problem._run_single({"species.maturity.eta.sp0": 0.3}, run_id=0)
        except Exception:
            pass
    cmd = mock_run.call_args[0][0]
    p_args = [s for s in cmd if s.startswith("-P")]
    assert any(s.startswith("-Pspecies.bioen.maturity.eta.sp0=") for s in p_args)  # reverse-mapped
    assert not any("species.maturity.eta.sp0" in s for s in p_args)  # NEW key gone
    assert not any(s.startswith("-Posmose.version=") for s in p_args)  # stamp skipped


def test_pr2_load_write_roundtrip_coherent(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import write_temp_config

    src = tmp_path / "src"
    src.mkdir()
    (src / "osm_all-parameters.csv").write_text(
        "osmose.version ; 4.3.3\nsimulation.bioen.enabled ; true\nfisheries.enabled ; false\n"
    )
    cfg = OsmoseConfigReader().read(src / "osm_all-parameters.csv")
    assert cfg["module.bioenergetics.enabled"] == "true"  # reader canonicalized
    out = tmp_path / "out"
    master = write_temp_config(cfg, out)  # default target 4.3.3
    raw = master.read_text()
    assert "simulation.bioen.enabled" in raw  # reverse-mapped to old for the jar
    assert "module.bioenergetics.enabled" not in raw


def test_larva_rate_scaled_to_per_year_on_4_4_0_write():
    from osmose.config.aliases import to_target_keys

    cfg = {"simulation.time.ndtperyear": "24", "mortality.additional.larva.rate.sp0": "2.145"}
    out = to_target_keys(cfg, "4.4.0")
    assert abs(float(out["mortality.additional.larva.rate.sp0"]) - 2.145 * 24) < 1e-6


def test_larva_rate_value_migration_roundtrips_through_reader(tmp_path):
    from osmose.config.aliases import to_target_keys
    from osmose.config.reader import OsmoseConfigReader

    cfg = {"simulation.time.ndtperyear": "24", "mortality.additional.larva.rate.sp0": "2.145"}
    native_440 = to_target_keys(cfg, "4.4.0")  # ×24 + osmose.version=4.4.0
    f = tmp_path / "osm_all-parameters.csv"
    f.write_text("\n".join(f"{k} ; {v}" for k, v in native_440.items()) + "\n")
    back = OsmoseConfigReader().read(f)  # master version 4.4.0 -> ÷24
    assert abs(float(back["mortality.additional.larva.rate.sp0"]) - 2.145) < 1e-9


def test_larva_rate_not_scaled_for_4_3_3_target():
    from osmose.config.aliases import to_target_keys

    cfg = {"simulation.time.ndtperyear": "24", "mortality.additional.larva.rate.sp0": "2.145"}
    out = to_target_keys(cfg, "4.3.3")
    assert out["mortality.additional.larva.rate.sp0"] == "2.145"


def test_reader_does_not_scale_larva_for_4_3_x_source(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    f = tmp_path / "osm_all-parameters.csv"
    f.write_text(
        "osmose.version ; 4.3.3\nsimulation.time.ndtperyear ; 24\n"
        "mortality.additional.larva.rate.sp0 ; 2.145\n"
    )
    back = OsmoseConfigReader().read(f)
    assert back["mortality.additional.larva.rate.sp0"] == "2.145"


def test_reader_scales_larva_for_snapshot_4_4_0_source(tmp_path):
    """A '4.4.0-SNAPSHOT' source must trigger the read-side divide-back.

    Regression: the read gate used the suffix-intolerant _version_tuple, so
    '4.4.0-SNAPSHOT' parsed to (0,) and the ÷ndt was silently skipped — leaving
    larval mortality mis-scaled ~24x. The gate now uses _numeric_version.
    """
    from osmose.config.reader import OsmoseConfigReader

    f = tmp_path / "osm_all-parameters.csv"
    f.write_text(
        "osmose.version ; 4.4.0-SNAPSHOT\nsimulation.time.ndtperyear ; 24\n"
        f"mortality.additional.larva.rate.sp0 ; {2.145 * 24}\n"
    )
    back = OsmoseConfigReader().read(f)
    assert abs(float(back["mortality.additional.larva.rate.sp0"]) - 2.145) < 1e-9


def test_scale_rate_value_passes_through_sentinels():
    """Sentinel/unset components must pass through verbatim, not raise ValueError."""
    from osmose.config.aliases import _scale_rate_value

    assert _scale_rate_value("0.5;null;0.3", 2.0) == "1;null;0.6"
    assert _scale_rate_value("null", 24.0) == "null"
    assert _scale_rate_value("", 24.0) == ""
    assert _scale_rate_value("2.145", 24.0) == f"{2.145 * 24:.10g}"


def test_reader_does_not_scale_larva_when_version_absent(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    f = tmp_path / "osm_all-parameters.csv"
    f.write_text("simulation.time.ndtperyear ; 24\nmortality.additional.larva.rate.sp0 ; 2.145\n")
    back = OsmoseConfigReader().read(f)
    assert back["mortality.additional.larva.rate.sp0"] == "2.145"


def test_larva_rate_only_scalar_key_bydt_file_left_alone():
    from osmose.config.aliases import to_target_keys

    cfg = {
        "simulation.time.ndtperyear": "24",
        "mortality.additional.larva.rate.bydt.file.sp0": "larva_sp0.csv",
        "mortality.additional.larva.rate.seasonality.file.sp0": "season_sp0.csv",
    }
    out = to_target_keys(cfg, "4.4.0")
    assert out["mortality.additional.larva.rate.bydt.file.sp0"] == "larva_sp0.csv"
    assert out["mortality.additional.larva.rate.seasonality.file.sp0"] == "season_sp0.csv"


def test_larva_migration_skipped_when_ndt_absent():
    from osmose.config.aliases import to_target_keys

    cfg = {"mortality.additional.larva.rate.sp0": "2.145"}  # no ndtperyear
    out = to_target_keys(cfg, "4.4.0")
    assert out["mortality.additional.larva.rate.sp0"] == "2.145"


def test_larva_migration_skipped_when_ndt_zero():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys(
        {"simulation.time.ndtperyear": "0", "mortality.additional.larva.rate.sp0": "2.145"}, "4.4.0"
    )
    assert out["mortality.additional.larva.rate.sp0"] == "2.145"  # not scaled by a bogus factor


def test_semicolon_separated_larva_rate_scaled_componentwise():
    from osmose.config.aliases import to_target_keys

    cfg = {"simulation.time.ndtperyear": "10", "mortality.additional.larva.rate.sp0": "2.0;3.0"}
    out = to_target_keys(cfg, "4.4.0")
    assert out["mortality.additional.larva.rate.sp0"] == "20;30"


def test_4_4_0_write_drops_lmax_growth_cap():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys({"species.lmax.sp0": "120"}, "4.4.0")
    assert "species.lmax.sp0" not in out  # 4.4.0 removed the lmax cap


def test_4_4_0_write_drops_nonbioen_species_beta():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys(
        {"species.beta.sp0": "2.0", "module.bioenergetics.enabled": "false"}, "4.4.0"
    )
    assert "species.beta.sp0" not in out  # non-bioen beta would feed 4.4.0's predation exponent


def test_4_4_0_write_keeps_species_beta_when_bioen_on():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys(
        {"species.beta.sp0": "2.0", "module.bioenergetics.enabled": "true"}, "4.4.0"
    )
    assert out["species.beta.sp0"] == "2.0"


def test_4_3_3_write_does_not_drop_lmax_or_beta():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys({"species.lmax.sp0": "120", "species.beta.sp0": "2.0"}, "4.3.3")
    assert out["species.lmax.sp0"] == "120"  # 4.3.3 still honors lmax + beta
    assert out["species.beta.sp0"] == "2.0"


def test_4_4_0_write_never_emits_computepercent_legacy_false():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys({"module.bioenergetics.enabled": "true"}, "4.4.0")
    assert out.get("simulation.resources.computepercent.legacy") != "false"


def test_to_target_keys_snapshot_version_is_native():
    from osmose.config.aliases import to_target_keys

    # A 4.4.x-family target (incl. a -SNAPSHOT suffix) must take the native (identity) branch,
    # NOT the reverse branch that would corrupt a native config back to 4.3.x key names.
    out = to_target_keys({"module.bioenergetics.enabled": "true"}, "4.4.0-SNAPSHOT")
    assert "simulation.bioen.enabled" not in out  # NOT reverse-mapped
    assert out["module.bioenergetics.enabled"] == "true"


def test_to_target_keys_4_4_1_is_native():
    from osmose.config.aliases import to_target_keys

    out = to_target_keys({"module.bioenergetics.enabled": "true"}, "4.4.1")
    assert "simulation.bioen.enabled" not in out


# --- Phase 1 Task 1.1: 4.4.x NETCDF_BIOMASS resource-forcing required keys ---
# 4.4.x ResourceForcing.init() hard-requires species.biomass.{file,varname,nsteps.year}.spN
# for resource species; OSMOPY 4.3.x configs carry only species.{type,name,file}.spN, so
# 4.4.x fails "NETCDF_BIOMASS resource forcing ... parameters are missing". The migration
# emits them additively (varname = species.name, NOT NetCDF-sniffed).


def _resource_cfg() -> dict[str, str]:
    return {
        "species.type.sp14": "resource",
        "species.name.sp14": "Dinoflagellates",
        "species.file.sp14": "eec_ltlbiomassTons.nc",
        "simulation.time.ndtperyear": "24",
    }


def test_4_4_x_write_adds_resource_biomass_forcing_keys():
    out = to_target_keys(_resource_cfg(), "4.4.0")
    assert out["species.biomass.mode.sp14"] == "NETCDF_BIOMASS"
    assert out["species.biomass.file.sp14"] == "eec_ltlbiomassTons.nc"  # the path 4.4.x reads
    assert out["species.biomass.varname.sp14"] == "Dinoflagellates"  # = species.name
    assert out["species.biomass.nsteps.year.sp14"] == "24"  # from ndtperyear


def test_4_4_1_target_also_adds_resource_forcing():
    out = to_target_keys(_resource_cfg(), "4.4.1")
    assert out["species.biomass.varname.sp14"] == "Dinoflagellates"
    assert out["osmose.version"] == "4.4.1"


def test_resource_forcing_global_nsteps_fallback():
    cfg = _resource_cfg()
    del cfg["simulation.time.ndtperyear"]
    cfg["species.biomass.nsteps.year"] = "24"  # global fallback
    out = to_target_keys(cfg, "4.4.0")
    assert out["species.biomass.nsteps.year.sp14"] == "24"


def test_resource_forcing_is_additive_not_overwriting():
    cfg = _resource_cfg()
    cfg["species.biomass.varname.sp14"] = "ExplicitVar"  # pre-existing wins
    out = to_target_keys(cfg, "4.4.0")
    assert out["species.biomass.varname.sp14"] == "ExplicitVar"


def test_non_resource_species_get_no_forcing_keys():
    cfg = {"species.type.sp0": "focal", "species.name.sp0": "Cod", "simulation.time.ndtperyear": "24"}
    out = to_target_keys(cfg, "4.4.0")
    assert not any(k.startswith("species.biomass.") for k in out)


def test_reverse_to_4_3_3_does_not_add_forcing_keys():
    out = to_target_keys(_resource_cfg(), "4.3.3")
    assert not any(k.startswith("species.biomass.") for k in out)
