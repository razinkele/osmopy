"""Parity guards for two parity-preserving perf changes (2026-06-24).

Both must be BIT-EXACT with the prior implementations:
- ``EngineConfig.movement_is_random`` / ``movement_is_maps`` replace per-school
  list comprehensions over ``movement_method`` (the A1/A2 vectorization template):
  precompute a per-species bool array once, fancy-index by the school ``species_id``.
- ``aggregate_diet_by_species`` uses per-column ``np.bincount`` instead of
  ``np.add.at`` — the same summation order, so bit-identical.
"""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.output import aggregate_diet_by_species


def _movement_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
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
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "2",
    }


def _ref_aggregate(diet_matrix, species_id, n_pred):
    """Reference: the original np.add.at implementation."""
    result = np.zeros((n_pred, diet_matrix.shape[1]), dtype=np.float64)
    focal = species_id < n_pred
    if focal.any():
        np.add.at(result, species_id[focal], diet_matrix[focal])
    return result


def test_aggregate_diet_bincount_bit_exact_vs_addat():
    rng = np.random.default_rng(0)
    n, n_focal, n_cols = 300, 5, 8
    species_id = rng.integers(0, n_focal + 2, n).astype(np.int32)  # focal + background
    diet = rng.random((n, n_cols)) * 1e3
    got = aggregate_diet_by_species(diet, species_id, n_focal)
    ref = _ref_aggregate(diet, species_id, n_focal)
    assert np.array_equal(got, ref), "diet aggregation must be bit-exact vs np.add.at"


def test_aggregate_diet_all_background_is_zeros():
    species_id = np.array([5, 6, 7], dtype=np.int32)  # all >= n_pred
    diet = np.ones((3, 4))
    got = aggregate_diet_by_species(diet, species_id, 5)
    assert got.shape == (5, 4)
    assert not got.any()


def test_movement_method_bool_arrays_match_comprehension():
    cfg = EngineConfig.from_dict(_movement_config())  # movement_method == ["random"]
    assert cfg.movement_is_random.dtype == np.bool_
    assert cfg.movement_is_maps.dtype == np.bool_
    exp_random = np.array([m == "random" for m in cfg.movement_method])
    exp_maps = np.array([m == "maps" for m in cfg.movement_method])
    assert np.array_equal(cfg.movement_is_random, exp_random)
    assert np.array_equal(cfg.movement_is_maps, exp_maps)


def test_movement_bool_arrays_indexed_by_species_match_per_school_loop():
    cfg = EngineConfig.from_dict(_movement_config())
    sp = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    # the vectorized index must equal the old per-school comprehension exactly
    assert np.array_equal(
        cfg.movement_is_random[sp],
        np.array([cfg.movement_method[s] == "random" for s in sp]),
    )
    assert np.array_equal(
        cfg.movement_is_maps[sp],
        np.array([cfg.movement_method[s] == "maps" for s in sp]),
    )


def test_movement_mask_vectorization_equivalence_mixed_methods():
    """The precompute-and-fancy-index logic must equal the old per-school
    comprehension for a multi-species config mixing every method ("maps" species
    need map files so cannot be built via from_dict here — assert the exact numpy
    operations movement() relies on, which is what makes the vectorization safe)."""
    movement_method = ["random", "maps", "none", "random", "maps"]
    is_random = np.array([m == "random" for m in movement_method], dtype=np.bool_)
    is_maps = np.array([m == "maps" for m in movement_method], dtype=np.bool_)
    # a school species_id array hitting every species in arbitrary order
    sp = np.array([4, 0, 2, 1, 3, 0, 1, 2], dtype=np.int32)
    assert np.array_equal(is_random[sp], np.array([movement_method[s] == "random" for s in sp]))
    assert np.array_equal(is_maps[sp], np.array([movement_method[s] == "maps" for s in sp]))
    assert is_random[sp].dtype == np.bool_ and len(is_random[sp]) == len(sp)
