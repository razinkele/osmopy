from osmose.engine.config import EngineConfig


def _base_cfg() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "1",
    }


def test_config_parses_yieldn_meansize_flags():
    cfg = EngineConfig.from_dict(
        {
            **_base_cfg(),
            "output.yield.abundance.enabled": "true",
            "output.size.enabled": "true",
            "output.yield.abundance.netcdf.enabled": "true",
            "output.size.netcdf.enabled": "true",
        }
    )
    assert cfg.output_yield_abundance is True
    assert cfg.output_mean_size is True
    assert cfg.output_yield_abundance_netcdf is True
    assert cfg.output_mean_size_netcdf is True


def test_config_yieldn_meansize_flags_default_false():
    cfg = EngineConfig.from_dict(_base_cfg())
    assert cfg.output_yield_abundance is False
    assert cfg.output_mean_size is False
    assert cfg.output_yield_abundance_netcdf is False
    assert cfg.output_mean_size_netcdf is False
