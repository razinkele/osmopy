import sys
from collections import namedtuple
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import baltic_bistability_chunk0 as c0  # noqa: E402

Tgt = namedtuple("Tgt", "species target lower upper weight", defaults=(1.0,))
COD = dict(target=120000.0, lower=60000.0, upper=250000.0)


# ---------------------------------------------------------------- Task 1
def test_collapsed_wins_over_stationarity_sentinel():
    assert c0.classify_state(0.0, 10.0, 1.0, **COD) == "collapsed"
    assert c0.classify_state(3000.0, 10.0, 1.0, **COD) == "collapsed"


def test_classify_bands_and_stationarity():
    assert c0.classify_state(120000, 0.5, 0.01, **COD) == "undetermined"
    assert c0.classify_state(30000, 0.1, 0.01, **COD) == "low"
    assert c0.classify_state(120000, 0.1, 0.01, **COD) == "in_range"
    assert c0.classify_state(400000, 0.1, 0.01, **COD) == "overshoot"


def test_basins_differ():
    assert c0.basins_differ("in_range", "collapsed", 0.9) is True
    assert c0.basins_differ("collapsed", "collapsed", 0.9) is False
    assert c0.basins_differ("overshoot", "overshoot", 0.9) is False
    assert c0.basins_differ("in_range", "in_range", 0.8) is True
    assert c0.basins_differ("in_range", "in_range", 0.1) is False


def test_aggregate_states():
    assert c0.aggregate_states(["in_range", "in_range", "in_range"]) == "in_range"
    assert c0.aggregate_states(["in_range", "collapsed", "in_range"]) == "seed-split"
    assert c0.aggregate_states(["in_range", "in_range", "failed"]) == "in_range"
    assert c0.aggregate_states(["failed", "undetermined"]) == "undetermined"


# ---------------------------------------------------------------- Task 2
def _targets():
    return [
        Tgt("cod", 120000, 60000, 250000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
    ]


def _stats(**means):
    d = {}
    for sp, m in means.items():
        d[f"{sp}_mean"] = m
        d[f"{sp}_cv"] = 0.05 if m > 0 else 10.0
        d[f"{sp}_trend"] = 0.01
    return d


def test_partial_collapse_vetoes_relaxation():
    targets = _targets()
    base = c0.species_states(_stats(cod=120000, sprat=25_000_000, herring=1_500_000), targets)
    low = c0.species_states(_stats(cod=120000, sprat=300_000, herring=1_500_000), targets)
    t = c0.accessibility_transition(base, low, targets)
    assert t["new_undershoot"] == 1
    assert c0.accessibility_verdict(t)[0] is False
    assert "collapse" in c0.accessibility_verdict(t)[1].lower()


def test_genuine_relaxation_passes():
    targets = _targets()
    base = c0.species_states(_stats(cod=120000, sprat=25_000_000, herring=20_000_000), targets)
    low = c0.species_states(_stats(cod=120000, sprat=1_500_000, herring=1_500_000), targets)
    t = c0.accessibility_transition(base, low, targets)
    assert t["new_undershoot"] == 0
    ok, msg = c0.accessibility_verdict(t)
    assert ok is True and "real lever" in msg.lower()


def test_nonstationary_withholds_verdict():
    targets = _targets()
    drifting = _stats(cod=120000, sprat=25_000_000, herring=1_500_000)
    drifting["sprat_cv"] = 0.9
    base = c0.species_states(drifting, targets)
    low = c0.species_states(_stats(cod=120000, sprat=1_500_000, herring=1_500_000), targets)
    t = c0.accessibility_transition(base, low, targets)
    assert t["undetermined"] >= 1
    assert c0.accessibility_verdict(t)[0] is False
    assert "provisional" in c0.accessibility_verdict(t)[1].lower()


def test_seed_split_species_withholds_accessibility_verdict():
    targets = _targets()
    base = {"cod": "in_range", "sprat": "overshoot", "herring": "in_range"}
    low = {"cod": "in_range", "sprat": "seed-split", "herring": "in_range"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["undetermined"] >= 1
    assert c0.accessibility_verdict(t)[0] is False


def test_low_weight_species_does_not_gate():
    targets = _targets() + [Tgt("perch", 20000, 8000, 50000, 0.2)]
    base = {"cod": "in_range", "sprat": "overshoot", "herring": "overshoot", "perch": "overshoot"}
    low = {"cod": "in_range", "sprat": "in_range", "herring": "in_range", "perch": "low"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["gated_species"] == 3
    assert t["new_undershoot"] == 0
    assert c0.accessibility_verdict(t)[0] is True


def test_collapsed_stock_in_lowered_arm_blocks_real_lever():
    targets = _targets()
    base = {"cod": "collapsed", "sprat": "overshoot", "herring": "overshoot"}
    low = {"cod": "collapsed", "sprat": "in_range", "herring": "in_range"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["collapsed_lowered"] >= 1
    ok, msg = c0.accessibility_verdict(t)
    assert ok is False and "still broken" in msg.lower()


def test_medium_weight_collapse_blocks_real_lever():
    targets = _targets() + [Tgt("flounder", 50000, 20000, 100000, 0.5)]
    base = {"cod": "in_range", "sprat": "overshoot", "herring": "overshoot", "flounder": "in_range"}
    low = {"cod": "in_range", "sprat": "in_range", "herring": "in_range", "flounder": "collapsed"}
    t = c0.accessibility_transition(base, low, targets)
    assert t["collapsed_lowered"] >= 1
    assert c0.accessibility_verdict(t)[0] is False


# ---------------------------------------------------------------- Task 3
def test_ics_vary_only_cod():
    rich, poor = c0.cod_rich_seeding(), c0.cod_poor_seeding()
    assert set(rich) == {"population.seeding.biomass.sp0", "population.seeding.year.max"}
    assert set(poor) == set(rich)
    assert float(rich["population.seeding.biomass.sp0"]) > float(
        poor["population.seeding.biomass.sp0"]
    )
    assert rich["population.seeding.year.max"] == "4"


def test_accessibility_scope_and_safe_run():
    assert set(c0.accessibility_override(0.1)) == {
        f"species.accessibility2fish.sp{i}" for i in (8, 10, 11, 12)
    }
    assert c0.safe_run(lambda *a: {"cod_mean": 5.0}, {}, {}, 5, 0) == {"cod_mean": 5.0}
    assert c0.safe_run(lambda *a: {}, {}, {}, 5, 0)["_failed"] is True
    assert c0.safe_run(lambda *a: {"herring_mean": 1.0}, {}, {}, 5, 0)["_failed"] is True

    def boom(*a):
        raise RuntimeError("x")

    assert c0.safe_run(boom, {}, {}, 5, 0)["_failed"] is True


# ---------------------------------------------------------------- Task 4
def _bands():
    return {"target": 120000.0, "lower": 60000.0, "upper": 250000.0}


def _runner_bistable(config, overrides, n_years, seed):
    scale = float(overrides["mortality.additional.larva.rate.sp0"]) / 15.0
    seeded = float(overrides.get("population.seeding.biomass.sp0", "0"))
    if abs(scale - 0.3) < 1e-9:
        cod = 120000.0 if seeded >= 100000 else 0.0
    else:
        cod = 120000.0 if scale < 0.9 else 0.0
    cv = 0.05 if cod > 0 else 10.0
    return {"cod_mean": cod, "cod_cv": cv, "cod_trend": 0.01}


def test_point_detects_bistable_including_collapsed_basin():
    pt = c0.run_bistability_point(
        0.3, {}, {0: 15.0}, _bands(), [0, 1, 2], runner=_runner_bistable, n_years=15
    )
    assert pt["rich_state"] == "in_range"
    assert pt["poor_state"] == "collapsed"
    assert pt["outcome"] == "bistable"
    assert pt["established"] is True


def test_seed_split_outcome():
    def flaky(config, overrides, n_years, seed):
        seeded = float(overrides.get("population.seeding.biomass.sp0", "0"))
        cod = 120000.0 if (seeded >= 100000 and seed != 1) else 0.0
        return {"cod_mean": cod, "cod_cv": 0.05 if cod > 0 else 10.0, "cod_trend": 0.01}

    pt = c0.run_bistability_point(0.3, {}, {0: 15.0}, _bands(), [0, 1, 2], runner=flaky, n_years=15)
    assert pt["rich_state"] == "seed-split"
    assert pt["outcome"] == "seed-split"


def test_sweep_verdict_and_stable_persistence():
    seen = []
    out = c0.run_bistability_sweep(
        [0.1, 0.3, 1.0],
        {},
        {0: 15.0},
        _bands(),
        [0, 1, 2],
        runner=_runner_bistable,
        n_years=15,
        on_point=seen.append,
    )
    assert out["bistable"] is True and 0.3 in out["bistable_scales"]
    assert "conservative" in out["verdict"].lower()
    assert 0.0 <= out["establishment_fraction"] <= 1.0
    assert set(seen[-1]) >= {"points", "bistable", "verdict", "complete"}
    assert seen[0]["complete"] is False


# ---------------------------------------------------------------- Task 5
def test_ab_excludes_failed_and_flags_all_failed():
    targets = _targets()

    def low_crashes(config, overrides, n_years, seed):
        if "species.accessibility2fish.sp11" in overrides:
            raise RuntimeError("blowup")
        return _stats(cod=120000, sprat=1_500_000, herring=1_500_000)

    out = c0.run_accessibility_ab({}, targets, [0, 1, 2], runner=low_crashes, n_years=15)
    assert out["relaxed"] is False
    assert "instrument-failed" in out["verdict"].lower()
    assert "collapse" not in out["verdict"].lower()


def test_ab_real_relaxation():
    targets = _targets()

    def runner(config, overrides, n_years, seed):
        low = "species.accessibility2fish.sp11" in overrides
        if low:
            return _stats(cod=120000, sprat=1_500_000, herring=1_500_000)
        return _stats(cod=120000, sprat=25_000_000, herring=20_000_000)

    out = c0.run_accessibility_ab({}, targets, [0, 1], runner=runner, n_years=15)
    assert out["relaxed"] is True and out["n_failed"] == 0


# ---------------------------------------------------------------- Task 6
def test_loaders():
    cfg = {f"mortality.additional.larva.rate.sp{i}": str(i + 1) for i in range(8)}
    rates = c0.read_base_larva_rates(cfg)
    assert rates[0] == 1.0 and rates[7] == 8.0
    assert c0.read_cod_bands([Tgt("cod", 120000, 60000, 250000)]) == {
        "target": 120000.0,
        "lower": 60000.0,
        "upper": 250000.0,
    }


# ---------------------------------------------------------------- Task 1 (warm-start)
def test_warmstart_override():
    assert c0.warmstart_override(False) == {}
    assert c0.warmstart_override(True) == {"module.population.initialisation.enabled": "true"}


def test_regime_shift_ic_builders():
    cd = c0.cod_dominated_seeding()
    cl = c0.clupeid_dominated_seeding()
    # cod axis: cod high in the cod-dominated IC, a remnant in the clupeid-dominated IC
    assert float(cd["population.seeding.biomass.sp0"]) > float(cl["population.seeding.biomass.sp0"])
    # clupeid axis: herring (sp1) + sprat (sp2) high in clupeid-dominated, suppressed in cod-dominated
    assert float(cl["population.seeding.biomass.sp1"]) > float(cd["population.seeding.biomass.sp1"])
    assert float(cl["population.seeding.biomass.sp2"]) > float(cd["population.seeding.biomass.sp2"])
    # exact spec values
    assert cd["population.seeding.biomass.sp0"] == "250000"
    assert cl["population.seeding.biomass.sp2"] == "2500000"
    # both carry the (now-inert-under-warmstart) global seeding window key
    assert "population.seeding.year.max" in cd and "population.seeding.year.max" in cl


# ---------------------------------------------------------------- Task 2 (clupeid axis)
def _clup_targets():
    return [
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]


def test_clupeid_axis_valid_and_sum():
    runs = [
        _stats(herring=1_500_000, sprat=2_500_000),
        _stats(herring=1_500_000, sprat=2_500_000),
    ]
    biomass, valid = c0.clupeid_axis(runs, _clup_targets())
    assert valid is True
    assert biomass == 4_000_000


def test_clupeid_axis_nonstationary_is_invalid():
    drifting = _stats(herring=1_500_000, sprat=2_500_000)
    drifting["herring_cv"] = 0.9  # non-stationary -> herring 'undetermined'
    biomass, valid = c0.clupeid_axis([drifting, drifting], _clup_targets())
    assert valid is False


def test_clupeid_axis_seed_split_is_invalid():
    runs = [
        _stats(herring=1_500_000, sprat=2_500_000),  # in_range
        _stats(herring=100_000, sprat=2_500_000),  # herring 'collapsed' -> disagreement
    ]
    _, valid = c0.clupeid_axis(runs, _clup_targets())
    assert valid is False


def test_clupeid_axis_all_failed():
    biomass, valid = c0.clupeid_axis([{"_failed": True}], _clup_targets())
    assert biomass == 0.0 and valid is False


# ---------------------------------------------------------------- Task 3 (outcome helpers)
def test_cod_axis_outcome_extracted_logic():
    assert c0.cod_axis_outcome("in_range", "collapsed", 0.9) == "bistable"
    assert c0.cod_axis_outcome("seed-split", "in_range", 0.0) == "seed-split"
    assert c0.cod_axis_outcome("undetermined", "in_range", 0.0) == "undetermined"
    assert c0.cod_axis_outcome("in_range", "in_range", 0.1) == "same-basin"
    assert c0.cod_axis_outcome("in_range", "in_range", 0.8) == "bistable"  # gap-driven split


def test_regime_shift_outcome_both_axes_diverge():
    # cod persists in cod-dominated arm (a), collapses in clupeid-dominated arm (b);
    # clupeids boom in b (4.0M) vs suppressed in a (0.5M)
    assert (
        c0.regime_shift_outcome("in_range", "collapsed", 500_000.0, 4_000_000.0, True, True)
        == "regime-shift"
    )


def test_regime_shift_outcome_cod_only_is_partial():
    # cod diverges but clupeid gap is tiny (3.9M vs 4.0M)
    assert (
        c0.regime_shift_outcome("in_range", "collapsed", 3_900_000.0, 4_000_000.0, True, True)
        == "partial"
    )


def test_regime_shift_outcome_clupeid_only_is_partial():
    # clupeids diverge but cod persists in BOTH arms (no collapse in b)
    assert (
        c0.regime_shift_outcome("in_range", "in_range", 500_000.0, 4_000_000.0, True, True)
        == "partial"
    )


def test_regime_shift_outcome_neither_is_monostable():
    assert (
        c0.regime_shift_outcome("in_range", "in_range", 3_900_000.0, 4_000_000.0, True, True)
        == "same-basin"
    )


def test_regime_shift_outcome_withheld_when_undetermined_or_invalid():
    # cod arm undetermined -> provisional
    assert (
        c0.regime_shift_outcome("seed-split", "collapsed", 500_000.0, 4_000_000.0, True, True)
        == "provisional"
    )
    # clupeid arm invalid -> provisional
    assert (
        c0.regime_shift_outcome("in_range", "collapsed", 500_000.0, 4_000_000.0, False, True)
        == "provisional"
    )


# ---------------------------------------------------------------- Task 4 (generalized sweep)
def _runner_regime(config, overrides, n_years, seed):
    """Cod-dominated arm (cod seed >= 100k) -> cod in_range + clupeids 'low';
    clupeid-dominated arm -> cod collapsed + clupeids booming."""
    cod_seed = float(overrides.get("population.seeding.biomass.sp0", "0"))
    if cod_seed >= 100_000:
        return _stats(cod=120_000, herring=400_000, sprat=300_000)
    return _stats(cod=0, herring=1_500_000, sprat=2_500_000)


def test_point_regime_shift_records_clupeid_and_outcome():
    pt = c0.run_bistability_point(
        1.0,
        {},
        {0: 15.0},
        _bands(),
        [0, 1, 2],
        runner=_runner_regime,
        n_years=15,
        ic_a=c0.cod_dominated_seeding,
        ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift",
        clupeid_targets=_clup_targets(),
    )
    assert pt["rich_state"] == "in_range"  # cod persists in cod-dominated arm
    assert pt["poor_state"] == "collapsed"  # cod collapses in clupeid-dominated arm
    assert pt["b_clupeid_biomass"] > pt["a_clupeid_biomass"]
    assert pt["a_clupeid_valid"] is True and pt["b_clupeid_valid"] is True
    assert pt["outcome"] == "regime-shift"
    assert pt["regime_shift"] is True


def test_regime_shift_sweep_verdict_and_incremental():
    seen = []
    out = c0.run_bistability_sweep(
        [1.0, 0.3],
        {},
        {0: 15.0},
        _bands(),
        [0, 1, 2],
        runner=_runner_regime,
        n_years=15,
        on_point=seen.append,
        ic_a=c0.cod_dominated_seeding,
        ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift",
        clupeid_targets=_clup_targets(),
    )
    assert out["regime_shift"] is True
    assert 1.0 in out["regime_shift_scales"]
    assert "regime shift" in out["verdict"].lower()
    assert out["complete"] is True
    assert seen[0]["complete"] is False


def test_regime_shift_sweep_monostable_when_convergent():
    def convergent(config, overrides, n_years, seed):
        # both arms -> cod in_range + clupeids in_range: no divergence on either axis
        return _stats(cod=120_000, herring=1_500_000, sprat=1_500_000)

    out = c0.run_bistability_sweep(
        [1.0, 0.3],
        {},
        {0: 15.0},
        _bands(),
        [0, 1],
        runner=convergent,
        n_years=15,
        ic_a=c0.cod_dominated_seeding,
        ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift",
        clupeid_targets=_clup_targets(),
    )
    assert out["regime_shift"] is False
    assert "monostable" in out["verdict"].lower()


def test_warmstart_flag_injected_into_overrides():
    captured = []

    def spy(config, overrides, n_years, seed):
        captured.append(dict(overrides))
        return _stats(cod=120_000, herring=400_000, sprat=300_000)

    c0.run_bistability_point(
        1.0,
        {},
        {0: 15.0},
        _bands(),
        [0],
        runner=spy,
        n_years=5,
        warmstart=True,
        ic_a=c0.cod_dominated_seeding,
        ic_b=c0.clupeid_dominated_seeding,
        contrast="regime-shift",
        clupeid_targets=_clup_targets(),
    )
    assert captured  # both arms ran
    assert all(o.get("module.population.initialisation.enabled") == "true" for o in captured)


# ---------------------------------------------------------------- Task 5 (CLI + preflight)
def test_contrast_specs():
    tgts = [
        Tgt("cod", 120_000, 60_000, 250_000),
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]
    both = c0.contrast_specs("both", tgts)
    assert [s["label"] for s in both] == ["cod-axis", "regime-shift"]
    assert both[0]["clupeid_targets"] is None
    assert both[1]["ic_a"] is c0.cod_dominated_seeding
    assert both[1]["ic_b"] is c0.clupeid_dominated_seeding
    assert {t.species for t in both[1]["clupeid_targets"]} == {"herring", "sprat"}
    assert both[1]["out_name"] == "baltic_chunk0_warmstart_bistability_regime-shift.json"
    assert len(c0.contrast_specs("cod-axis", tgts)) == 1
    assert len(c0.contrast_specs("regime-shift", tgts)) == 1


def test_preflight_check():
    ok, msg = c0.preflight_check(_stats(cod=120_000, herring=800_000, sprat=600_000))
    assert ok is True and "ok" in msg.lower()
    assert c0.preflight_check({"_failed": True, "_error": "boom"})[0] is False
    nan_stats = {"cod_mean": float("nan"), "herring_mean": 1.0, "sprat_mean": 1.0}
    assert c0.preflight_check(nan_stats)[0] is False
    assert c0.preflight_check(_stats(cod=0, herring=0, sprat=0))[0] is False


def test_cli_warmstart_writes_both_contrasts(tmp_path, monkeypatch):
    tgts = [
        Tgt("cod", 120_000, 60_000, 250_000),
        Tgt("herring", 1_500_000, 800_000, 3_000_000),
        Tgt("sprat", 1_500_000, 800_000, 2_500_000),
    ]
    monkeypatch.setattr(c0, "read_base_config", lambda: {})
    monkeypatch.setattr(c0, "read_base_larva_rates", lambda cfg, n_focal=8: {0: 15.0})
    monkeypatch.setattr(c0, "_load_targets", lambda: tgts)
    monkeypatch.setattr(c0, "_default_runner", _runner_regime)
    monkeypatch.setattr(c0, "_DIAG_DIR", tmp_path)
    rc = c0.main(["--warmstart", "--contrast", "both", "--smoke"])
    assert rc == 0
    assert (tmp_path / "baltic_chunk0_warmstart_bistability_cod-axis.json").exists()
    assert (tmp_path / "baltic_chunk0_warmstart_bistability_regime-shift.json").exists()


def test_cli_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(c0, "read_base_config", lambda: {})
    monkeypatch.setattr(c0, "read_base_larva_rates", lambda cfg, n_focal=8: {0: 15.0})
    monkeypatch.setattr(c0, "_load_targets", lambda: [Tgt("cod", 120_000, 60_000, 250_000)])
    monkeypatch.setattr(c0, "_default_runner", _runner_regime)
    monkeypatch.setattr(c0, "_DIAG_DIR", tmp_path)
    rc = c0.main(["--preflight"])
    assert rc == 0  # _runner_regime cod-dominated arm returns a persisting stock


# ---------------------------------------------------------------- cod-axis verdict framing
def test_cod_axis_verdict_warmstart_reframes_text():
    pts = [
        {"scale": 0.1, "outcome": "same-basin", "established": True},
        {"scale": 1.0, "outcome": "same-basin", "established": True},
    ]
    egg = c0._cod_axis_verdict(pts, warmstart=False)
    ws = c0._cod_axis_verdict(pts, warmstart=True)
    # summary fields are identical (only the verdict prose differs)
    assert egg["bistable"] == ws["bistable"] is False
    assert egg["establishment_fraction"] == ws["establishment_fraction"] == 1.0
    # egg-only path keeps the v3 framing verbatim (parity)
    assert "egg-only" in egg["verdict"].lower() and "task 7" in egg["verdict"].lower()
    # warm-start path drops the egg-only / Task-7 framing (this run USED the primitive)
    assert "egg-only" not in ws["verdict"].lower() and "task 7" not in ws["verdict"].lower()
    assert "warm-start" in ws["verdict"].lower() and "monostable" in ws["verdict"].lower()
