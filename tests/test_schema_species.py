from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from osmose.schema.base import ParamType


def test_simulation_fields_count():
    assert len(SIMULATION_FIELDS) >= 10


def test_simulation_ndtperyear():
    field = next(f for f in SIMULATION_FIELDS if "ndtperyear" in f.key_pattern)
    assert field.param_type == ParamType.INT
    assert field.default == 24
    assert field.category == "simulation"
    assert not field.indexed


def test_simulation_nspecies():
    field = next(f for f in SIMULATION_FIELDS if "nspecies" in f.key_pattern)
    assert field.param_type == ParamType.INT
    assert field.min_val >= 1


def test_species_fields_count():
    assert len(SPECIES_FIELDS) >= 25


def test_all_species_fields_indexed():
    # Per-species fields carry the {idx} placeholder and must be marked indexed.
    # Reproduction master-switch fields (e.g. reproduction.rv.gate.enabled,
    # reproduction.rv.spatial.enabled, reproduction.recruitment.ceiling.enabled)
    # live in the species schema module but are scalar, non-indexed switches.
    for f in SPECIES_FIELDS:
        if "{idx}" in f.key_pattern:
            assert f.indexed, f"Species field {f.key_pattern} should be indexed"
        else:
            assert not f.indexed, f"Non-{{idx}} field {f.key_pattern} must not be indexed"


def test_species_growth_params_present():
    patterns = [f.key_pattern for f in SPECIES_FIELDS]
    assert "species.linf.sp{idx}" in patterns
    assert "species.k.sp{idx}" in patterns
    assert "species.t0.sp{idx}" in patterns


def test_species_reproduction_params_present():
    patterns = [f.key_pattern for f in SPECIES_FIELDS]
    assert "species.maturity.size.sp{idx}" in patterns
    assert "species.relativefecundity.sp{idx}" in patterns
    assert "species.sexratio.sp{idx}" in patterns


def test_species_linf_metadata():
    linf = next(f for f in SPECIES_FIELDS if f.key_pattern == "species.linf.sp{idx}")
    assert linf.param_type == ParamType.FLOAT
    assert linf.unit == "cm"
    assert linf.min_val is not None
    assert linf.category == "growth"
