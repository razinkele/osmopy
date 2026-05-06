"""H10: reproduction-parameter bounds (closes Phase 3 / H10).

Three regression tests:
1. species.sexratio out of [0, 1] -> ValueError on EngineConfig.from_dict.
2. species.relativefecundity < 0 -> ValueError. (Zero is allowed; tests
   in test_engine_background.py exercise it as a degenerate
   no-reproduction path.)
3. reproduction.season.file with non-unit per-year sum -> UserWarning,
   suppressed when reproduction.normalisation.enabled=true.

Pre-change sweep (plan r6) verified that all five fixtures
(eec/baltic/eec_full/examples/minimal) ship clean values for sexratio
and relativefecundity, so these bounds don't break shipped configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig

REPO = Path(__file__).resolve().parents[1]


def _load_eec_config() -> dict[str, str]:
    matches = sorted((REPO / "data" / "eec").glob("*_all-parameters.csv"))
    cfg = OsmoseConfigReader().read(str(matches[0]))
    return {k: v for k, v in cfg.items() if k != ""}


def test_sexratio_above_one_raises() -> None:
    cfg = _load_eec_config()
    cfg["species.sexratio.sp0"] = "1.5"
    with pytest.raises(ValueError, match=r"species\.sexratio\.sp0 must be in \[0, 1\]"):
        EngineConfig.from_dict(cfg)


def test_sexratio_negative_raises() -> None:
    cfg = _load_eec_config()
    cfg["species.sexratio.sp1"] = "-0.1"
    with pytest.raises(ValueError, match=r"species\.sexratio\.sp1 must be in \[0, 1\]"):
        EngineConfig.from_dict(cfg)


def test_sexratio_zero_and_one_accepted() -> None:
    """Boundary values are valid; zero = all-male, one = all-female."""
    cfg = _load_eec_config()
    cfg["species.sexratio.sp0"] = "0.0"
    cfg["species.sexratio.sp1"] = "1.0"
    EngineConfig.from_dict(cfg)  # must not raise


def test_relativefecundity_negative_raises() -> None:
    cfg = _load_eec_config()
    cfg["species.relativefecundity.sp0"] = "-1"
    with pytest.raises(ValueError, match=r"species\.relativefecundity\.sp0 must be >= 0"):
        EngineConfig.from_dict(cfg)


def test_relativefecundity_zero_accepted() -> None:
    """Zero is a valid degenerate 'no reproduction' case (regression: see
    tests/test_engine_background.py — calibrations sometimes set this)."""
    cfg = _load_eec_config()
    cfg["species.relativefecundity.sp0"] = "0"
    EngineConfig.from_dict(cfg)  # must not raise


def test_spawning_season_non_unit_sum_warns(tmp_path: Path) -> None:
    """A spawning-season vector that doesn't sum to 1.0 / year must warn."""
    cfg = _load_eec_config()
    n_dt = int(float(cfg.get("simulation.time.ndtperyear", "12")))
    # Build a season CSV that sums to 2.0 (way off unit) and point sp0 at it.
    season_csv = tmp_path / "bad_season.csv"
    rows = ["step;value\n"] + [f"{i};{2.0 / n_dt}\n" for i in range(n_dt)]
    season_csv.write_text("".join(rows))
    cfg["reproduction.season.file.sp0"] = str(season_csv)
    cfg["reproduction.normalisation.enabled"] = "false"
    with pytest.warns(UserWarning, match=r"reproduction\.season\.file\.sp0.*per-year sum is 2"):
        EngineConfig.from_dict(cfg)


def test_spawning_season_normalisation_suppresses_warning(tmp_path: Path) -> None:
    """When reproduction.normalisation.enabled=true, the off-unit warning is suppressed."""
    cfg = _load_eec_config()
    n_dt = int(float(cfg.get("simulation.time.ndtperyear", "12")))
    season_csv = tmp_path / "bad_season.csv"
    rows = ["step;value\n"] + [f"{i};{2.0 / n_dt}\n" for i in range(n_dt)]
    season_csv.write_text("".join(rows))
    cfg["reproduction.season.file.sp0"] = str(season_csv)
    cfg["reproduction.normalisation.enabled"] = "true"
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)  # any UserWarning becomes an exception
        # Filter out unrelated UserWarnings (e.g. predation-size-ratio-swap from eec sp10-13)
        # by checking the message only when the H10 source is the trigger.
        try:
            EngineConfig.from_dict(cfg)
        except UserWarning as exc:
            if "reproduction.season.file" in str(exc):
                raise AssertionError(f"H10 warning fired despite normalisation: {exc}") from None
            # any other UserWarning is unrelated — accept (the test is about H10's silence)
