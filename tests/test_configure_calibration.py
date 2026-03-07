"""Tests for configure_calibration auto-detection."""

from osmose.calibration.configure import configure_calibration


def test_detects_mortality_natural_rate():
    config = {"mortality.natural.rate.sp0": "0.2", "simulation.time.total": "100"}
    result = configure_calibration(config)
    params = result["params"]
    assert len(params) == 1
    p = params[0]
    assert p["key"] == "mortality.natural.rate.sp0"
    assert p["guess"] == 0.2
    assert p["lower"] == 0.001
    assert p["upper"] == 2.0


def test_detects_multiple_species():
    config = {
        "mortality.natural.rate.sp0": "0.3",
        "mortality.natural.rate.sp1": "0.5",
    }
    result = configure_calibration(config)
    assert len(result["params"]) == 2
    keys = [p["key"] for p in result["params"]]
    assert "mortality.natural.rate.sp0" in keys
    assert "mortality.natural.rate.sp1" in keys


def test_detects_species_k():
    config = {"species.k.sp0": "0.15"}
    result = configure_calibration(config)
    assert len(result["params"]) == 1
    p = result["params"][0]
    assert p["lower"] == 0.01
    assert p["upper"] == 1.0


def test_detects_species_linf():
    config = {"species.linf.sp0": "50.0"}
    result = configure_calibration(config)
    p = result["params"][0]
    assert p["guess"] == 50.0
    assert p["lower"] == 1.0
    assert p["upper"] == 300.0


def test_detects_predation_ingestion():
    config = {"predation.ingestion.rate.max.sp2": "3.5"}
    result = configure_calibration(config)
    p = result["params"][0]
    assert p["lower"] == 0.5
    assert p["upper"] == 10.0


def test_detects_predation_efficiency():
    config = {"predation.efficiency.critical.sp0": "0.5"}
    result = configure_calibration(config)
    p = result["params"][0]
    assert p["lower"] == 0.1
    assert p["upper"] == 0.9


def test_detects_population_seeding():
    config = {"population.seeding.biomass.sp0": "5000"}
    result = configure_calibration(config)
    p = result["params"][0]
    assert p["guess"] == 5000.0
    assert p["lower"] == 100
    assert p["upper"] == 1000000


def test_detects_mortality_larva():
    config = {"mortality.natural.larva.rate.sp0": "1.5"}
    result = configure_calibration(config)
    p = result["params"][0]
    assert p["lower"] == 0.001
    assert p["upper"] == 10.0


def test_detects_mortality_starvation():
    config = {"mortality.starvation.rate.max.sp0": "0.8"}
    result = configure_calibration(config)
    p = result["params"][0]
    assert p["lower"] == 0.001
    assert p["upper"] == 5.0


def test_ignores_non_calibratable_keys():
    config = {
        "simulation.time.total": "100",
        "grid.nx": "10",
        "output.dir.path": "/tmp",
    }
    result = configure_calibration(config)
    assert len(result["params"]) == 0


def test_empty_config():
    result = configure_calibration({})
    assert result == {"params": []}


def test_all_patterns_detected():
    config = {
        "mortality.natural.rate.sp0": "0.2",
        "mortality.natural.larva.rate.sp0": "1.0",
        "mortality.starvation.rate.max.sp0": "0.5",
        "species.k.sp0": "0.1",
        "species.linf.sp0": "50",
        "predation.ingestion.rate.max.sp0": "3.0",
        "predation.efficiency.critical.sp0": "0.4",
        "population.seeding.biomass.sp0": "10000",
    }
    result = configure_calibration(config)
    assert len(result["params"]) == 8
