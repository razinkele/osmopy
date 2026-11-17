"""Every demo the UI offers must pass the validation the UI runs on load and before a run.

The Run tab blocks on any validation error (``ui/pages/run.py``: "VALIDATION ERRORS (run
blocked)"), so a schema bound or a parsing rule that contradicts a shipped config stops the demo
from running at all. Three such cases, all found 2026-11-15:

- ``reproduction.rv.gate.ref`` had ``max_val=1.0`` while the Baltic demos run 150 (the RV series is
  in km^3), so all three Baltic demos were blocked since that value went in (2026-07-28).
- The validator split per-stage arrays on ';' only, but the engine also splits on ',', so
  Benguela's "15, 5" size ratios read as 12 bad numbers.
- Validating those parts then exposed sardine's adult-stage size ratio 4000 against a cap of 1000.

The check is the UI's own: ``summarize_config_validation`` with the demo's config directory, which
adds file-reference checks to the schema checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.config.validator import summarize_config_validation
from osmose.demo import list_demos, osmose_demo
from osmose.schema import build_registry


@pytest.mark.parametrize("demo", list_demos())
def test_demo_passes_the_ui_validation(demo, tmp_path):
    info = osmose_demo(demo, tmp_path)
    master = Path(info["config_file"])
    cfg = OsmoseConfigReader().read(master)
    errors, _warnings = summarize_config_validation(cfg, build_registry(), master.parent)
    assert errors == [], f"{demo}: {errors}"


# Shipped masters loadable on their own (via the Advanced tab's import, or as a run config).
# Excluded: baltic_a2 / baltic_depensation are overlays (only valid merged, as their demos above
# test), and baltic-fine's grid, maps and NetCDF files are generated locally, not tracked.
_DATA = Path(__file__).resolve().parent.parent / "data"
_STANDALONE = sorted(
    m
    for m in _DATA.glob("*/*all-parameters*.csv")
    if m.parent.name not in ("baltic_a2", "baltic_depensation", "baltic-fine")
)


@pytest.mark.parametrize("master", _STANDALONE, ids=lambda m: f"{m.parent.name}/{m.name}")
def test_shipped_master_passes_the_ui_validation(master):
    """Found 2026-11-15: data/examples_bioen's species.maturity.r (1.3-2.7, Java's bioen example)
    against a schema max of 1.0. r is a coefficient, not a fraction."""
    cfg = OsmoseConfigReader().read(master)
    errors, _warnings = summarize_config_validation(cfg, build_registry(), master.parent)
    assert errors == [], f"{master}: {errors}"


@pytest.mark.parametrize("demo", list_demos())
def test_every_focal_species_has_additional_mortality(demo, tmp_path):
    """Python reads a missing additional or larval mortality rate as 0 and Java refuses the config
    (no mortality scenario). baltic_a2's overlay predated the cod split and left cod_east (sp8)
    with neither for four months (#273). A rate counts if any scenario sets it (the plain key or a
    file variant such as `mortality.additional.rate.bytdt.file.sp{i}`)."""
    info = osmose_demo(demo, tmp_path)
    cfg = dict(OsmoseConfigReader().read(Path(info["config_file"])))
    n = int(cfg["simulation.nspecies"])

    def has(prefix: str, i: int) -> bool:
        return any(k.startswith(prefix) and k.endswith(f".sp{i}") for k in cfg)

    missing = [
        (i, cfg.get(f"species.name.sp{i}"), kind)
        for i in range(n)
        for kind, prefix in (
            ("adult", "mortality.additional.rate"),
            ("larval", "mortality.additional.larva.rate"),
        )
        if not has(prefix, i)
    ]
    assert not missing, f"{demo}: focal species without additional mortality: {missing}"


@pytest.mark.parametrize("demo", list_demos())
def test_no_resource_key_lands_on_a_focal_species(demo, tmp_path):
    """Resource parameters are indexed like species (resources follow the focal species), so a
    species added before them shifts every resource index. baltic_a2's depletion overlay kept its
    pre-split indices and gave cod_east (sp8) a plankton regrowth rate (#273)."""
    info = osmose_demo(demo, tmp_path)
    cfg = dict(OsmoseConfigReader().read(Path(info["config_file"])))
    n = int(cfg["simulation.nspecies"])
    on_focal = sorted(
        k for k in cfg if k.startswith("species.regrowth.rate.sp") and int(k.rsplit("sp", 1)[1]) < n
    )
    assert not on_focal, f"{demo}: resource keys on focal species: {on_focal}"
