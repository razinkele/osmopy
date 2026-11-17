"""``output.minimal.enabled``: collect only what a calibration objective reads (perf plan item 4).

A calibration evaluation needs per-species biomass, abundance and yield. The engine also collected
every other output each step: the life-stage split, age/size distributions, mortality-rate tables,
mean TL and size, SSB, and, the largest cost, the diet tracked inside the mortality kernel. On a
one-thread 10-yr Baltic run diet tracking alone was 5.4% of the wall time, and the unread collectors
another ~4% (docs/perf/2026-11-17-output-minimal.md).

The mode is opt-in. It must not change what it keeps (biomass, abundance and yield stay
bit-identical), and what it skips must be ABSENT, never zeros that a reader would take for data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from tests._tutorial_config import build_config


@pytest.fixture(scope="module")
def cfg(tmp_path_factory) -> dict[str, str]:
    c = dict(build_config(tmp_path_factory.mktemp("minimal"), n_year=1))
    assert c.get("output.diet.composition.enabled") == "true", "fixture must exercise diet tracking"
    return c


def _minimal(cfg: dict[str, str]) -> dict[str, str]:
    return {**cfg, "output.minimal.enabled": "true"}


def test_minimal_is_off_by_default(cfg):
    assert EngineConfig.from_dict(dict(cfg)).output_minimal is False


def test_minimal_turns_every_optional_output_off(cfg):
    full, minimal = EngineConfig.from_dict(dict(cfg)), EngineConfig.from_dict(_minimal(cfg))
    assert minimal.output_minimal is True
    assert full.diet_output_enabled and not minimal.diet_output_enabled
    assert full.output_ssb and not minimal.output_ssb
    assert full.output_biomass_byage and not minimal.output_biomass_byage
    # What decides which steps are recorded is NOT an output to skip.
    assert minimal.output_step0_include == full.output_step0_include
    assert minimal.output_record_frequency == full.output_record_frequency


def test_minimal_keeps_biomass_abundance_and_yield_bit_identical(cfg):
    full = PythonEngine().run_in_memory(dict(cfg), seed=7)
    minimal = PythonEngine().run_in_memory(_minimal(cfg), seed=7)
    pd.testing.assert_frame_equal(minimal.biomass(), full.biomass(), check_exact=True)
    pd.testing.assert_frame_equal(minimal.abundance(), full.abundance(), check_exact=True)
    pd.testing.assert_frame_equal(minimal.yield_biomass(), full.yield_biomass(), check_exact=True)


@pytest.mark.parametrize("getter", ["biomass_by_age", "diet_matrix", "mean_size", "ssb"])
def test_minimal_leaves_the_other_outputs_absent_not_zero(cfg, getter):
    """A skipped output must not come back as zeros a reader would take for data."""
    minimal = PythonEngine().run_in_memory(_minimal(cfg), seed=7)
    try:
        frame = getattr(minimal, getter)()
    except (FileNotFoundError, KeyError, ValueError):
        return  # absent: raised
    assert frame is None or len(frame) == 0, f"{getter} returned data in minimal mode"


def test_minimal_on_disk_writes_the_essentials_and_nothing_else(cfg, tmp_path):
    out = tmp_path / "output"
    result = PythonEngine().run(_minimal(cfg), output_dir=out, seed=7)
    assert result.returncode == 0
    names = sorted(p.name for p in out.rglob("*.csv"))
    for kept in ("biomass", "abundance", "yield"):
        assert any(kept in n.lower() for n in names), f"no {kept} file in {names}"
    for skipped in ("diet", "ByAge", "BySize", "meanSize", "SSB", "ssb", "TL"):
        assert not any(skipped in n for n in names), (
            f"{skipped} file written in minimal mode: {names}"
        )
    assert isinstance(out, Path)


def test_baltic_calibration_runs_minimal_and_scores_the_same(cfg, monkeypatch):
    """calibrate_baltic.run_simulation reads only biomass and yield, so it runs minimal. The
    statistics it scores must be identical to a full run's; the overrides can turn the mode off."""
    import sys

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import calibrate_baltic as cal

    seen: list[str] = []
    real_run = PythonEngine.run

    def spy(self, config, *a, **k):
        seen.append(config.get("output.minimal.enabled", "unset"))
        return real_run(self, config, *a, **k)

    monkeypatch.setattr(PythonEngine, "run", spy)
    minimal = cal.run_simulation(dict(cfg), {}, n_years=1, seed=3)
    full = cal.run_simulation(dict(cfg), {"output.minimal.enabled": "false"}, n_years=1, seed=3)
    assert seen == ["true", "false"]
    assert minimal and minimal == full
    assert "calibrate_baltic" in sys.modules
