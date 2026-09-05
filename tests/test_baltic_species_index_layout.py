"""Guards that every hardcoded ``sp{N}`` index set in the Baltic calibrator still points at
the species block it names.

OSMOSE numbers focal, resource (LTL) and background species contiguously in one
``species.name.sp{idx}`` namespace, so inserting a focal species renumbers everything above
it. The cod disaggregation (cod -> cod_west sp0 + cod_east sp8) shifted the resource block
from sp8-sp13 to sp9-sp14 and the background block from sp14-sp15 to sp15-sp16. Index sets
written against the old layout keep resolving — to the wrong species — so nothing goes red:
the phase-14 FR predator set silently moved onto Benthos, and the LTL accessibility loops
silently moved onto cod_east.

These tests assert on ``species.type`` rather than on names, because names are exactly what a
disaggregation changes (cod -> cod_west). They read only the tracked config CSVs, so they run
in a clean checkout — deliberately NOT behind a ``tests/_data_guards.py`` skip, since a
skip-guard is why the original breakage went unnoticed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "data" / "baltic"
_PARAM_FILES = ("baltic_param-species.csv", "baltic_param-ltl.csv", "baltic_param-background.csv")


def _species_blocks() -> tuple[dict[int, str], dict[int, str]]:
    """(index -> name, index -> type) parsed from the tracked Baltic param CSVs."""
    names: dict[int, str] = {}
    types: dict[int, str] = {}
    for fname in _PARAM_FILES:
        for raw in (CONFIG_DIR / fname).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ";" not in line:
                continue
            key, _, value = line.partition(";")
            key, value = key.strip(), value.strip()
            for prefix, target in (("species.name.sp", names), ("species.type.sp", types)):
                if key.startswith(prefix) and key[len(prefix) :].isdigit():
                    target[int(key[len(prefix) :])] = value
    return names, types


def _indices_of_type(wanted: str) -> set[int]:
    _, types = _species_blocks()
    return {idx for idx, kind in types.items() if kind == wanted}


def _load_script(name: str):
    """Import a module from scripts/ (which has no __init__.py).

    scripts/ goes on sys.path because some of these modules import their siblings by
    bare name (evaluate_calibration_vs_ices -> validate_baltic_vs_ices_sag).
    """
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script = scripts_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibrate_baltic():
    return _load_script("calibrate_baltic")


@pytest.fixture(scope="module")
def evaluate_vs_ices():
    return _load_script("evaluate_calibration_vs_ices")


def test_config_layout_is_the_disaggregated_one():
    """Sanity check: these guards are meaningless if the config is not the 9-focal layout."""
    names, types = _species_blocks()
    assert types[8] == "focal" and names[8] == "cod_east"
    assert _indices_of_type("focal") == set(range(9))
    assert _indices_of_type("resource") == set(range(9, 15))
    assert _indices_of_type("background") == {15, 16}


def test_phase14_fr_halfsat_keys_target_focal_and_background_predators(calibrate_baltic):
    """The 4 phase-14 FR predators are 2 focal fish + the 2 background predators.

    A stale index here puts the half-saturation param on Benthos, and the run still completes.
    """
    keys, _bounds, _x0 = calibrate_baltic.get_phase14_params()
    idx = [int(k.rsplit(".sp", 1)[1]) for k in keys]
    _, types = _species_blocks()
    kinds = [types[i] for i in idx]
    assert kinds.count("focal") == 2, f"expected 2 focal predators, got {list(zip(idx, kinds))}"
    assert kinds.count("background") == 2, (
        f"expected 2 background predators, got {list(zip(idx, kinds))}"
    )
    assert "resource" not in kinds, f"FR halfsat landed on a resource: {list(zip(idx, kinds))}"


def test_fr_predator_set_matches_across_calibrator_and_evaluator(
    calibrate_baltic, evaluate_vs_ices
):
    """calibrate_baltic and evaluate_calibration_vs_ices document that these must stay in sync."""
    keys, _b, _x = calibrate_baltic.get_phase14_params()
    from_calibrator = tuple(int(k.rsplit(".sp", 1)[1]) for k in keys)
    assert from_calibrator == tuple(evaluate_vs_ices.FR_PREDATOR_SP)


def test_ltl_accessibility_indices_are_exactly_the_resource_block(calibrate_baltic):
    """Phases 1d-1g pre-fix `accessibility2fish` across "the LTL block".

    Asserting equality (not membership) catches both halves of a shift: cod_east wrongly
    included and Benthos wrongly dropped.
    """
    assert set(calibrate_baltic.LTL_RESOURCE_INDICES) == _indices_of_type("resource")


def test_phyto_and_zoo_indices_partition_the_resource_block(calibrate_baltic):
    """Phyto + zoo must together cover the resource block exactly, with no overlap.

    This is the assertion that catches the original bug directly: the zoo tuple had drifted
    to include a phytoplankton group and to drop Benthos off the top.
    """
    phyto = set(calibrate_baltic._PHYTO_RESOURCE_INDICES)
    zoo = set(calibrate_baltic._ZOO_RESOURCE_INDICES)
    assert not phyto & zoo, f"phyto and zoo overlap on {sorted(phyto & zoo)}"
    assert phyto | zoo == _indices_of_type("resource")


def test_zoo_regrowth_sentinel_expands_onto_resources_only(calibrate_baltic):
    """The grouped zoo-regrowth sentinel must expand to depletable resources, never to fish."""
    _, types = _species_blocks()
    kinds = {i: types[i] for i in calibrate_baltic._ZOO_RESOURCE_INDICES}
    assert set(kinds.values()) == {"resource"}, f"zoo regrowth hits a non-resource: {kinds}"


def test_a2_phytoplankton_regrowth_targets_resources(calibrate_baltic):
    """A2 fixes phytoplankton regrowth fast; a stale index sets a regrowth rate on a fish."""
    cfg = calibrate_baltic.enable_a2_base_config({})
    _, types = _species_blocks()
    touched = {
        int(k.rsplit(".sp", 1)[1]): v
        for k, v in cfg.items()
        if k.startswith("species.regrowth.rate.sp")
    }
    kinds = {i: types[i] for i in touched}
    assert set(kinds.values()) == {"resource"}, f"A2 regrowth hits a non-resource: {kinds}"
