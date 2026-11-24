"""The C# engine's per-fishery fishing equals the Python engine's, byte for byte (spec 2026-11-24).

EEC with all four per-fishery gaps switched on, captured mid-run just before mortality (the
test_csharp_prep machinery): one species also in a second fishery at catchability 0.5 (G1, G2), a
fishery with its own non-uniform map (G3), and a fishery scheduled on steps 0-5 only, captured at
step 30 (= step 6 of year 1) when it is off (G4). The C# PrepProbe's fishing rates and discard
fractions must equal _precompute_effective_rates exactly. Needs the built engine; skips without it
unless OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.fisheries import OFF, FisheryModel
from osmose.engine.processes.mortality import _precompute_effective_rates
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import csharp_engine_block_reason
from tests.test_csharp_prep import capture_state, write_state

ROOT = Path(__file__).resolve().parents[1]
STEP = 30


def _overrides(workdir: Path) -> dict[str, str]:
    raw = dict(
        OsmoseConfigReader().read(Path(osmose_demo("eec_full", workdir / "src")["config_file"]))
    )
    cfg_dir = Path(raw["_osmose.config.dir"])
    q = pd.read_csv(cfg_dir / raw["fisheries.catchability.file"], index_col=0).astype(float)
    sp0 = q.index[0]
    other = q.columns[1] if q.loc[sp0, q.columns[0]] > 0 else q.columns[0]
    q.loc[sp0, other] = 0.5  # G1 + G2
    (workdir / "q.csv").write_text(q.to_csv())
    base = pd.read_csv(cfg_dir / raw["fisheries.movement.file.map0"], sep=";", header=None)
    # G3, and mean ~2 over ocean so Java's normalisation (G5) genuinely rescales it
    grad = base.astype(float) * np.linspace(0.5, 3.5, base.shape[1])[None, :]
    grad.to_csv(workdir / "grad.csv", sep=";", header=False, index=False)
    # G4 on a fishery that is NOT the one carrying the 0.5 term, so that term is live at STEP
    gap = next(
        k
        for k in range(14)
        if raw.get(f"fisheries.movement.fishery.map{k}", "")
        .replace("_", "")
        .replace("-", "")
        .lower()
        != str(other).replace("_", "").replace("-", "").lower()
        and k != 0
    )
    return {
        "fisheries.catchability.file": str(workdir / "q.csv"),
        # the gradient on EVERY fishery map: on map0 alone it sat on a fishery that catches
        # nothing (the dogfish one), so it reached no rate and the G3/G5 checks were vacuous
        **{
            k.replace("fishery.map", "file.map"): str(workdir / "grad.csv")
            for k in raw
            if k.startswith("fisheries.movement.fishery.map")
        },
        f"fisheries.movement.steps.map{gap}": "0;1;2;3;4;5",  # G4: off on steps 6-23
        "_test.gap_map": str(gap),
        "_test.q_species": str(sp0),
        "_test.q_fishery": str(other),
    }


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    d = tmp_path_factory.mktemp("per_fishery")
    config, grid, raw, st, _ = capture_state("eec_full", STEP, d, _overrides(d))
    build_bundle(config, raw, grid, seed=11).save(d / "model.bin")
    write_state(st, d / "state.bin")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.PrepProbe.Run(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])"
    )
    args = [
        sys.executable,
        "-c",
        code,
        str(ROOT),
        str(d / "model.bin"),
        str(d / "state.bin"),
        str(STEP),
        str(d / "prep.bin"),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return config, raw, st, Bundle.load(d / "prep.bin")


def test_the_config_takes_the_per_fishery_path_and_csharp_accepts_it(probe):
    config, raw, _, _ = probe
    m = config.fishery_model
    assert isinstance(m, FisheryModel)
    assert csharp_engine_block_reason(raw) is None
    gap = raw["_test.gap_map"]
    fg = m.names.index(raw[f"fisheries.movement.fishery.map{gap}"])
    assert m.map_index[fg, STEP] == OFF  # G4 is live at the captured step


def test_fishing_rates_and_discards_equal_python_byte_for_byte(probe):
    config, _, st, prep = probe
    _, _, eff_fishing, discard = _precompute_effective_rates(
        st, config, config.mortality_subdt, STEP
    )
    got_f = np.asarray(prep.get("prep.eff_fishing"))
    got_d = np.asarray(prep.get("prep.discard"))
    assert got_f.tobytes() == np.asarray(eff_fishing, dtype=np.float64).tobytes()
    assert got_d.tobytes() == np.asarray(discard, dtype=np.float64).tobytes()
    assert (eff_fishing > 0).sum() > 50  # a real, fished population: not vacuous


def test_the_half_catchability_term_is_live_at_the_captured_step(probe):
    """Guards the byte test: removing the q = 0.5 term must change the rates, or the C#
    catchability product is never exercised (a mutation dropping it once survived)."""
    import copy

    import numpy as np

    config, raw, st, _ = probe
    m = config.fishery_model
    names = [config.species_names[i] for i in range(config.n_species)]
    sp = names.index(raw["_test.q_species"])
    f = list(pd.read_csv(raw["fisheries.catchability.file"], index_col=0).columns).index(
        raw["_test.q_fishery"]
    )
    q = m.catchability.copy()
    q[sp, f] = 0.0
    c2 = copy.copy(config)
    c2.fishery_model = type(m)(**{**m.__dict__, "catchability": q})
    with_term = _precompute_effective_rates(st, config, config.mortality_subdt, STEP)[2]
    without = _precompute_effective_rates(st, c2, config.mortality_subdt, STEP)[2]
    assert not np.array_equal(with_term, without)


def test_normalisation_genuinely_changes_the_python_rates(probe):
    """Guards the byte test for G5: the rates with normalised maps must differ from the rates
    with raw maps, or a C# build skipping normalisation would still pass (it once did twice:
    a 0.2-1.8 ramp with mean 1, then a ramp on a fishery that catches nothing)."""
    import copy

    config, _, st, _ = probe
    raw_cfg = copy.copy(config)
    raw_cfg.fishing_ocean_mask = None  # fishing_map then returns the raw maps
    raw_cfg.__dict__.pop("_fishing_map_cache", None)
    norm = _precompute_effective_rates(st, config, config.mortality_subdt, STEP)[2]
    raw = _precompute_effective_rates(st, raw_cfg, config.mortality_subdt, STEP)[2]
    assert not np.array_equal(norm, raw)


def _gradient_only(workdir: Path) -> dict[str, str]:
    """Only the gradient maps: catchability untouched, every map the same file -> the config
    stays species-reducible, so this exercises the PER-SPECIES path with a map normalisation
    genuinely rescales (shipped maps all have mean 1 and cannot)."""
    over = _overrides(workdir)
    return {k: v for k, v in over.items() if k.startswith("fisheries.movement.file.map")}


@pytest.fixture(scope="module")
def probe_species(tmp_path_factory):
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why})")
    d = tmp_path_factory.mktemp("per_species_gradient")
    config, grid, raw, st, _ = capture_state("eec_full", STEP, d, _gradient_only(d))
    build_bundle(config, raw, grid, seed=11).save(d / "model.bin")
    write_state(st, d / "state.bin")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.PrepProbe.Run(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])"
    )
    args = [
        sys.executable,
        "-c",
        code,
        str(ROOT),
        str(d / "model.bin"),
        str(d / "state.bin"),
        str(STEP),
        str(d / "prep.bin"),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return config, st, Bundle.load(d / "prep.bin")


def test_per_species_path_normalised_maps_equal_python_byte_for_byte(probe_species):
    import copy

    config, st, prep = probe_species
    assert config.fishery_model is None  # the per-species path
    want = _precompute_effective_rates(st, config, config.mortality_subdt, STEP)[2]
    assert (
        np.asarray(prep.get("prep.eff_fishing")).tobytes()
        == np.asarray(want, dtype=np.float64).tobytes()
    )
    raw_cfg = copy.copy(config)
    raw_cfg.fishing_ocean_mask = None
    raw_cfg.__dict__.pop("_fishing_map_cache", None)
    raw = _precompute_effective_rates(st, raw_cfg, config.mortality_subdt, STEP)[2]
    assert not np.array_equal(want, raw)  # normalisation is live: not vacuous
