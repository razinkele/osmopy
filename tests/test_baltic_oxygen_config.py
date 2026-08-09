"""Adoption checks (Task 4, spec Phase 2a): the O2->benthos coupling is wired into the
production Baltic config, matches the certification-gate arm exactly, and is live (not inert)
when the demo config is staged and run through the real loading seam.

Phase 1's "construct ResourceState directly and inspect" pattern does NOT transfer here: the
wiring that matters lives in ``osmose.engine.simulate`` (``_load_oxygen_data`` + the
``ResourceState(..., oxygen=...)`` call inside ``simulate()``), not in a config dict a test
builds by hand. So these tests go through the same staging path
(``osmose.demo.osmose_demo("baltic", tmp)``) that ``scripts/baltic_stability_certify.py`` uses.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState
from osmose.engine.simulate import _load_oxygen_data

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVERLAY_CSV = _REPO_ROOT / "data" / "baltic" / "baltic_param-oxygen.csv"
_ARM_JSON = _REPO_ROOT / "data" / "baltic" / "calibration_results" / "o2_benthos_arm.json"
_MASTER_CSV = _REPO_ROOT / "data" / "baltic" / "baltic_all-parameters.csv"


def _raw_pairs(path: Path) -> dict[str, str]:
    """Parse a ``;``-separated OSMOSE param file into a key/value dict, ignoring ``#`` comments
    and blank lines. Independent of :class:`OsmoseConfigReader` on purpose -- this is a direct
    check that the overlay file's literal contents match the gate arm, not a check of how the
    full reader merges includes (that's covered separately below)."""
    pairs: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition(";")
        pairs[key.strip()] = value.strip()
    return pairs


def test_overlay_matches_gate_arm_exactly():
    # (a) the overlay's key/value pairs must reproduce the certification-gate arm byte-for-byte
    # (as strings) -- the whole point of "adoption" is deploying exactly what was gated, not a
    # hand-retyped approximation of it.
    arm = json.loads(_ARM_JSON.read_text())
    overlay = _raw_pairs(_OVERLAY_CSV)
    assert overlay == arm


def test_master_config_includes_oxygen_overlay():
    # (b) checked against the TRACKED repo file (not a staged copy) -- this is a statement about
    # the source config, independent of how osmose_demo() stages it.
    text = _MASTER_CSV.read_text()
    assert "osmose.configuration.oxygen;baltic_param-oxygen.csv" in text


@pytest.fixture
def _staged_baltic_config() -> dict:
    """Stage the real "baltic" demo (the same osmose_demo() call the certification script and
    the UI use) and read its merged config. Deliberately NOT reading data/baltic directly:
    osmose_demo() is what actually ends up running, and staging is a real seam a future change
    could break (e.g. an enumerated file list instead of copytree) without the tracked source
    files changing at all."""
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    return dict(OsmoseConfigReader().read(str(res["config_file"])))


def test_staged_demo_loads_oxygen_netcdf_with_24_frames(_staged_baltic_config):
    # (c) simulate._load_oxygen_data on the STAGED demo config (the config the certification
    # run and the UI actually consume) returns a live NetCDF-mode PhysicalData with 24 frames
    # (== simulation.time.ndtperyear, enforced by _load_oxygen_data's own frame-count guard) and
    # plausible bottom-O2 values.
    cfg = _staged_baltic_config
    assert cfg.get("ltl.oxygen.benthos.enabled") == "true"
    config_dir = Path(cfg["_osmose.config.dir"])
    data = _load_oxygen_data(cfg, config_dir)
    assert data is not None
    assert not data.is_constant
    assert data._data is not None
    assert data._data.shape[0] == 24
    finite = data._data[np.isfinite(data._data)]
    finite = finite[finite != 0.0]  # land cells are exact 0.0 (Task 1 convention)
    assert finite.min() >= 0.0
    assert finite.max() <= 650.0  # same plausibility bound as test_baltic_oxygen_forcing.py


def test_staged_demo_resource_state_coupling_is_live(_staged_baltic_config):
    # (d) a ResourceState built from the staged config + the loaded oxygen forcing shows a
    # non-degenerate (variance > 0) oxygen_factor_last after update(step=0) -- proof the
    # coupling actually fires end-to-end on the adopted config, not just that the keys parse.
    # A constant (all-ones or all-equal) factor row would pass a weaker "is not None" check
    # vacuously; variance > 0 forbids that.
    cfg = _staged_baltic_config
    config_dir = Path(cfg["_osmose.config.dir"])
    oxygen = _load_oxygen_data(cfg, config_dir)
    grid = Grid.from_netcdf(config_dir / "baltic_grid.nc")

    resources = ResourceState(config=cfg, grid=grid, oxygen=oxygen)
    resources.update(step=0)

    assert resources.oxygen_factor_last is not None
    assert np.var(resources.oxygen_factor_last) > 0.0
    # sanity: the coupling targets Benthos specifically (named-resource gate, not sp-index)
    assert "Benthos" in [r.name for r in resources.species]
