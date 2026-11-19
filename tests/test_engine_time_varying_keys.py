"""Time-varying mortality files must be found when the config comes from the reader (issue #297).

``OsmoseConfigReader`` lowercases every key. Two engine readers looked their keys up in the
config-file spelling (``byDt``, ``byAge``, ``bySize``), so a config read from disk never matched:
the time-varying larval and by-class additional mortality were dropped without a word and the
run used the constant rate. The older tests built dicts with camelCase keys by hand, which is
the one path that worked.

Every config here is written to a file and read back with the real reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import (
    EngineConfig,
    _detect_larva_by_dt_key,
    _load_additional_mortality_by_dt_by_class,
    _load_larva_mortality_by_dt,
)

ROOT = Path(__file__).resolve().parents[1]
N_DT = 24


def _series(path: Path, values) -> Path:
    path.write_text("step;rate\n" + "\n".join(f"{i};{v}" for i, v in enumerate(values)) + "\n")
    return path


def _by_class(path: Path, thresholds, rows) -> Path:
    header = "step;" + ";".join(str(t) for t in thresholds)
    body = "\n".join(f"{i};" + ";".join(str(v) for v in row) for i, row in enumerate(rows))
    path.write_text(header + "\n" + body + "\n")
    return path


def _read(tmp_path: Path, lines: list[str]) -> dict[str, str]:
    """Write ``key;value`` lines as a config file and read them with the real reader."""
    cfg_file = tmp_path / "extra.csv"
    cfg_file.write_text("\n".join(lines) + "\n")
    return dict(OsmoseConfigReader().read(cfg_file))


@pytest.mark.parametrize("spelling", ["byDt", "bytDt"])  # the second is Java's own typo
def test_a_larval_by_dt_file_set_in_a_config_file_is_loaded(tmp_path, spelling):
    rates = np.linspace(0.5, 2.8, N_DT)
    csv = _series(tmp_path / "larva.csv", rates)
    cfg = _read(tmp_path, [f"mortality.additional.larva.rate.{spelling}.file.sp0;{csv}"])
    assert list(cfg) and all(key == key.lower() for key in cfg), "the reader lowercases keys"
    assert _detect_larva_by_dt_key(cfg, 0) == str(csv)
    assert _detect_larva_by_dt_key(cfg, 1) is None
    loaded = _load_larva_mortality_by_dt(cfg, n_species=2, n_dt_per_year=N_DT, n_dt_simu=2 * N_DT)
    assert loaded is not None and loaded[1] is None
    assert np.allclose(loaded[0].values, np.tile(rates, 2))


@pytest.mark.parametrize(("spelling", "by_age"), [("byDt.byAge", True), ("byDt.bySize", False)])
def test_a_by_class_additional_mortality_file_set_in_a_config_file_is_loaded(
    tmp_path, spelling, by_age
):
    rows = [[0.1, 0.2, 0.3]] * N_DT
    csv = _by_class(tmp_path / "byclass.csv", [0, 1, 3], rows)
    cfg = _read(tmp_path, [f"mortality.additional.rate.{spelling}.file.sp1;{csv}"])
    loaded = _load_additional_mortality_by_dt_by_class(
        cfg, n_species=2, n_dt_per_year=N_DT, n_dt_simu=N_DT
    )
    assert loaded is not None and loaded[0] is None and loaded[1] is not None
    # Age thresholds are given in years and held in time steps; size thresholds stay in cm.
    want = [0.0, 24.0, 72.0] if by_age else [0.0, 1.0, 3.0]
    assert loaded[1].classes.tolist() == want


def test_the_engine_config_carries_them(tmp_path):
    """The whole path: a shipped config plus the two keys, through the reader, into the fields
    the mortality processes read."""
    base = dict(OsmoseConfigReader().read(ROOT / "data" / "minimal" / "osm_all-parameters.csv"))
    n_dt = int(base["simulation.time.ndtperyear"])
    larva = _series(tmp_path / "larva.csv", np.linspace(1.0, 2.0, n_dt))
    by_class = _by_class(tmp_path / "byclass.csv", [0, 2], [[0.05, 0.4]] * n_dt)
    extra = _read(tmp_path, [
        f"mortality.additional.larva.rate.byDt.file.sp0;{larva}",
        f"mortality.additional.rate.byDt.bySize.file.sp1;{by_class}",
    ])  # fmt: skip
    plain = EngineConfig.from_dict(dict(base))
    assert plain.larva_mortality_by_dt is None and plain.additional_mortality_by_dt_by_class is None
    config = EngineConfig.from_dict(base | extra)
    assert config.larva_mortality_by_dt is not None and config.larva_mortality_by_dt[0] is not None
    assert config.larva_mortality_by_dt[1] is None
    by = config.additional_mortality_by_dt_by_class
    assert by is not None and by[0] is None and by[1].classes.tolist() == [0.0, 2.0]


def test_a_dict_with_the_config_file_spelling_still_works():
    """Callers that build a dict by hand (tests, overlays) may use the spelling of config files.
    With both spellings present, the reader's (lowercase) wins."""
    camel = {"mortality.additional.larva.rate.byDt.file.sp0": "/camel.csv"}
    assert _detect_larva_by_dt_key(camel, 0) == "/camel.csv"
    both = camel | {"mortality.additional.larva.rate.bydt.file.sp0": "/lower.csv"}
    assert _detect_larva_by_dt_key(both, 0) == "/lower.csv"


def test_the_validator_knows_these_keys():
    from osmose.engine.config_validation import _check, build_known_keys

    known = build_known_keys()
    for key in (
        "mortality.additional.larva.rate.bydt.file.sp0",
        "mortality.additional.larva.rate.bytdt.file.sp0",
        "mortality.additional.rate.bydt.byage.file.sp0",
        "mortality.additional.rate.bydt.bysize.file.sp3",
    ):
        assert _check(key, known) is None, key


def test_the_csharp_engine_refuses_such_a_config_by_the_feature_s_name(tmp_path):
    """Until the lookups were fixed these files never reached the engine config, so the C#
    engine's detectors never saw them either: it would have run such a config and ignored the
    files, as the Python engine did. Now both see them, and the C# engine, which does not
    implement time-varying rates, says so."""
    from osmose.engine_cs.capabilities import classify_pattern, csharp_engine_block_reason

    base = dict(OsmoseConfigReader().read(ROOT / "data" / "eec_full" / "eec_all-parameters.csv"))
    base["output.minimal.enabled"] = "true"
    assert csharp_engine_block_reason(base) is None
    n_dt = int(base["simulation.time.ndtperyear"])
    larva = _series(tmp_path / "larva.csv", np.linspace(1.0, 2.0, n_dt))
    by_class = _by_class(tmp_path / "byclass.csv", [0, 2], [[0.05, 0.4]] * n_dt)
    for line, feature, pattern in (
        (f"mortality.additional.larva.rate.byDt.file.sp0;{larva}", "larva_rate_by_step",
         "mortality.additional.larva.rate.bydt.file.sp{idx}"),
        (f"mortality.additional.larva.rate.bytDt.file.sp0;{larva}", "larva_rate_by_step",
         "mortality.additional.larva.rate.bytdt.file.sp{idx}"),
        (f"mortality.additional.rate.byDt.byAge.file.sp1;{by_class}", "rate_by_class",
         "mortality.additional.rate.bydt.byage.file.sp{idx}"),
        (f"mortality.additional.rate.byDt.bySize.file.sp1;{by_class}", "rate_by_class",
         "mortality.additional.rate.bydt.bysize.file.sp{idx}"),
    ):  # fmt: skip
        reason = csharp_engine_block_reason(base | _read(tmp_path, [line]))
        assert reason is not None and feature in reason, (line, reason)
        assert "has not classified" not in reason, reason
        assert classify_pattern(pattern) == feature


def test_the_validator_knows_the_economics_fleet_keys():
    """Read in ``osmose/engine/economics/fleet.py``, outside ``config.py``, and so never listed:
    a config with a fleet was told its keys were unknown. The C# engine files them under
    economics, which it refuses on the module switch."""
    from osmose.engine.config_validation import _check, build_known_keys
    from osmose.engine.economics.fleet import parse_fleets
    from osmose.engine_cs.capabilities import classify_pattern

    known = build_known_keys()
    fleet = {
        "economic.fleet.number": "1",
        "economic.fleet.name.fsh0": "trawlers",
        "economic.fleet.nvessels.fsh0": "12",
        "economic.fleet.homeport.y.fsh0": "1",
        "economic.fleet.homeport.x.fsh0": "2",
        "economic.fleet.gear.fsh0": "trawl",
        "economic.fleet.max.days.fsh0": "180",
        "economic.fleet.fuel.cost.fsh0": "3.5",
        "economic.fleet.operating.cost.fsh0": "1.5",
        "economic.fleet.target.species.fsh0": "0, 1",
        "economic.fleet.price.sp0.fsh0": "1200",
        "economic.fleet.price.sp1.fsh0": "800",
        "economic.fleet.stock.elasticity.sp0.fsh0": "0.4",
        "economic.fleet.stock.elasticity.sp1.fsh0": "0.6",
    }
    for key in fleet:
        assert _check(key, known) is None, key
    # Every key the parser reads is in the dict above: parsing with one removed changes the fleet.
    parsed = parse_fleets(fleet, n_species=2)[0]
    assert (parsed.name, parsed.n_vessels, parsed.gear_type, parsed.max_days_at_sea) == (
        "trawlers", 12, "trawl", 180,
    )  # fmt: skip
    assert parsed.price_per_tonne.tolist() == [1200.0, 800.0]
    for pattern in ("economic.fleet.number", "economic.fleet.price.sp{idx}.fsh{idx}"):
        assert classify_pattern(pattern) == "economics"


def test_a_series_with_no_rows_is_an_error_not_a_hang(tmp_path):
    """A file with a header and no data. Cycling an empty series to a positive length never
    advanced, so loading the config hung. The lowercase lookups make that reachable from a
    config file."""
    from osmose.engine.timeseries import ByYearTimeSeries, _cycle_to_length, load_timeseries

    with pytest.raises(ValueError, match="empty"):
        _cycle_to_length([], 3)
    assert _cycle_to_length([], 0).tolist() == []  # nothing asked for: nothing to cycle
    assert _cycle_to_length([1.0, 2.0], 5).tolist() == [1.0, 2.0, 1.0, 2.0, 1.0]
    empty = tmp_path / "empty.csv"
    empty.write_text("year;rate\n")
    with pytest.raises(ValueError, match="empty"):
        ByYearTimeSeries.from_csv(empty, n_years=3)
    cfg = _read(tmp_path, [f"mortality.fishing.rate.byYear.file.sp0;{empty}"])
    with pytest.raises(ValueError, match="empty"):
        load_timeseries(cfg, "mortality.fishing.rate", 0, ndt_per_year=N_DT, ndt_simu=3 * N_DT)
