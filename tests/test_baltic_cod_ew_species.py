"""The disaggregated Baltic config has cod_west (sp0) + cod_east (sp8) as two
focal species with distinct eastern/western life-history, and every unsplit
species is unchanged apart from its shifted sp index.

Eastern Baltic cod (cod.27.24-32) signatures encoded here (fidelity review):
impaired *condition* (leaner weight-at-length) — NOT a heritable slow-growth
trait, so von Bertalanffy Linf/K stay equal to western; larger, more buoyant
eggs (low-salinity deep-basin spawning); early maturation at small size; a
truncated lifespan. The collapse driver (elevated M + RV recruitment gate)
lands in Task 6, not here.
"""

from osmose.config import OsmoseConfigReader

CFG = "data/baltic/baltic_all-parameters.csv"


def _cfg():
    return dict(OsmoseConfigReader().read(CFG))


def test_nine_focal_species_cod_west_and_east():
    cfg = _cfg()
    assert cfg["simulation.nspecies"] == "9"
    assert cfg["species.name.sp0"] == "cod_west"
    assert cfg["species.name.sp8"] == "cod_east"
    assert cfg["species.type.sp8"] == "focal"


def test_cod_east_distinct_eastern_life_history():
    cfg = _cfg()
    # Larger, more buoyant eggs than western (low-salinity adaptation)
    assert float(cfg["species.egg.size.sp8"]) > float(cfg["species.egg.size.sp0"])
    # Early maturation at a smaller size (collapsed-stock signature)
    assert float(cfg["species.maturity.size.sp8"]) < float(cfg["species.maturity.size.sp0"])
    # Impaired condition: leaner weight-at-length
    assert float(cfg["species.length2weight.condition.factor.sp8"]) < float(
        cfg["species.length2weight.condition.factor.sp0"]
    )
    # Condition, NOT slow growth: von Bertalanffy Linf unchanged from western
    assert cfg["species.linf.sp8"] == cfg["species.linf.sp0"]
    # Truncated lifespan
    assert int(cfg["species.lifespan.sp8"]) < int(cfg["species.lifespan.sp0"])


def test_cod_east_has_full_scalar_param_set():
    cfg = _cfg()
    for key in (
        "species.k.sp8",
        "species.t0.sp8",
        "species.relativefecundity.sp8",
        "stock.recruitment.type.sp8",
        "reproduction.season.file.sp8",
        "predation.efficiency.critical.sp8",
        "mortality.additional.rate.sp8",
        "mortality.starvation.rate.max.sp8",
        "population.seeding.biomass.sp8",
        "movement.distribution.method.sp8",
        "output.cutoff.age.sp8",
        "simulation.nschool.sp8",
    ):
        assert key in cfg, f"cod_east missing {key}"
    assert cfg["stock.recruitment.type.sp8"] == "shepherd"


def test_unsplit_species_only_index_shifted():
    """Herring..stickleback keep their params; LTL/background shifted +1 with
    values intact."""
    cfg = _cfg()
    # focal sp1-7 unchanged
    assert cfg["species.name.sp1"] == "herring"
    assert cfg["species.name.sp7"] == "stickleback"
    # LTL shifted: Diatoms was sp8, now sp9; Benthos was sp13, now sp14
    assert cfg["species.name.sp9"] == "Diatoms"
    assert cfg["species.name.sp14"] == "Benthos"
    # background shifted: GreySeal sp14->sp15, Cormorant sp15->sp16
    assert cfg["species.name.sp15"] == "GreySeal"
    assert cfg["species.name.sp16"] == "Cormorant"


def test_cod_east_seasonality_file_exists_and_sums_to_one():
    import csv
    from pathlib import Path

    path = Path("data/baltic/reproduction/reproduction-seasonality-sp8.csv")
    assert path.exists()
    with path.open() as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    vals = [float(r[1].strip('"')) for r in rows[1:] if len(r) >= 2]
    assert abs(sum(vals) - 1.0) < 1e-3, f"seasonality sums to {sum(vals)}"
