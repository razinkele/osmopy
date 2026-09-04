"""Unit tests for `per_fish_ingestion_cap`, the live bioen allometric ingestion cap.

Superseded from `bioen_ingestion_cap` (deleted, see osmose/engine/processes/bioen_predation.py
history): that per-species-scalar form was a post-hoc cap applied to a per-school total and is
no longer on the mortality path (`per_fish_ingestion_cap` is, applied per FISH and scaled by
instantaneous abundance at every predator visit -- Java `BioenPredationMortality`). These cases
carry over the same four scenarios the deleted function's tests covered (adult, larval additive
correction, vectorized mixed, theta=1 degenerate collapse), now against the array-per-species
signature. Java-parity coverage for the full multi-species / background-predator shape lives in
`tests/test_engine_bioen_mortality_parity.py` (Gate G section 1) -- these tests are the
module-local unit layer beneath it, and specifically restore the theta=1 degenerate case that
file does not cover.
"""

import numpy as np

from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap


class TestPerFishIngestionCap:
    def test_adult_ingestion_cap(self):
        """Adult (ageDt >= larvaeThresDt): Imax/ndt * w_g^beta / subdt, per fish."""
        cap = per_fish_ingestion_cap(
            weight=np.array([0.001]),
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),
            i_max_all=np.array([10.0]),
            beta=np.array([0.75]),
            larvae_thres_dt=np.array([1], dtype=np.int32),
            theta=np.array([2.0]),
            c_rate=np.array([1.0]),
            n_species=1,
            n_dt_per_year=24,
            n_subdt=10,
        )
        w_g = 0.001 * 1e6
        expected = (10.0 / 24) * w_g**0.75 / 10 * 1e-6
        np.testing.assert_allclose(cap, expected, rtol=1e-10)

    def test_larvae_additive_correction(self):
        """Larvae (ageDt < larvaeThresDt): (Imax + (theta-1)*c_rate)/ndt * w_g^beta / subdt."""
        cap = per_fish_ingestion_cap(
            weight=np.array([0.0001]),
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([0], dtype=np.int32),
            i_max_all=np.array([10.0]),
            beta=np.array([0.75]),
            larvae_thres_dt=np.array([1], dtype=np.int32),
            theta=np.array([3.0]),
            c_rate=np.array([2.0]),
            n_species=1,
            n_dt_per_year=24,
            n_subdt=10,
        )
        w_g = 0.0001 * 1e6
        expected = ((10.0 + (3.0 - 1) * 2.0) / 24) * w_g**0.75 / 10 * 1e-6
        np.testing.assert_allclose(cap, expected, rtol=1e-10)

    def test_vectorized_mixed(self):
        """Mix of adults and larvae in one array give different effective rates."""
        cap = per_fish_ingestion_cap(
            weight=np.array([0.001, 0.0001]),
            species_id=np.array([0, 0], dtype=np.int32),
            age_dt=np.array([10, 0], dtype=np.int32),  # adult, larva
            i_max_all=np.array([10.0]),
            beta=np.array([0.75]),
            larvae_thres_dt=np.array([1], dtype=np.int32),
            theta=np.array([2.0]),
            c_rate=np.array([1.0]),
            n_species=1,
            n_dt_per_year=24,
            n_subdt=10,
        )
        assert cap.shape == (2,)
        assert cap[1] != cap[0]

    def test_theta_one_no_correction(self):
        """When theta=1, larvae get the same effective rate as adults regardless of c_rate.

        Not covered by test_engine_bioen_mortality_parity.py's three per_fish_ingestion_cap
        cases -- restored here per the deleted bioen_ingestion_cap test of the same name.
        """
        cap_adult = per_fish_ingestion_cap(
            weight=np.array([0.001]),
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),  # >= larvae_thres_dt: adult branch
            i_max_all=np.array([10.0]),
            beta=np.array([0.75]),
            larvae_thres_dt=np.array([1], dtype=np.int32),
            theta=np.array([1.0]),
            c_rate=np.array([0.0]),
            n_species=1,
            n_dt_per_year=24,
            n_subdt=10,
        )
        cap_larva = per_fish_ingestion_cap(
            weight=np.array([0.001]),
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([0], dtype=np.int32),  # < larvae_thres_dt: larval branch
            i_max_all=np.array([10.0]),
            beta=np.array([0.75]),
            larvae_thres_dt=np.array([1], dtype=np.int32),
            theta=np.array([1.0]),
            c_rate=np.array([5.0]),  # large c_rate must have no effect when theta=1
            n_species=1,
            n_dt_per_year=24,
            n_subdt=10,
        )
        np.testing.assert_allclose(cap_adult, cap_larva)
