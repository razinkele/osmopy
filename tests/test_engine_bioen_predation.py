"""Tests for bioenergetic allometric ingestion cap."""
import numpy as np
from osmose.engine.processes.bioen_predation import bioen_ingestion_cap


class TestBioenIngestionCap:
    def test_adult_ingestion_cap(self):
        """Adult: I_max * w_g^beta / (n_dt * subdt)."""
        cap = bioen_ingestion_cap(weight=np.array([0.001]), i_max=10.0, beta=0.75,
            n_dt_per_year=24, n_subdt=10, is_larvae=np.array([False]))
        w_g = 0.001 * 1e6
        expected = 10.0 * w_g**0.75 / (24 * 10)
        np.testing.assert_allclose(cap, expected, rtol=1e-10)

    def test_larvae_additive_correction(self):
        """Larvae: (I_max + (theta-1)*c_rate) * w_g^beta / (n_dt * subdt)."""
        cap = bioen_ingestion_cap(weight=np.array([0.0001]), i_max=10.0, beta=0.75,
            n_dt_per_year=24, n_subdt=10, is_larvae=np.array([True]), theta=3.0, c_rate=2.0)
        w_g = 0.0001 * 1e6
        expected = (10.0 + (3.0-1)*2.0) * w_g**0.75 / (24*10)
        np.testing.assert_allclose(cap, expected, rtol=1e-10)

    def test_vectorized_mixed(self):
        """Mix of adults and larvae in one array."""
        w = np.array([0.001, 0.0001])
        is_larv = np.array([False, True])
        cap = bioen_ingestion_cap(w, 10.0, 0.75, 24, 10, is_larv, theta=2.0, c_rate=1.0)
        assert cap.shape == (2,)
        assert cap[1] != cap[0]  # different effective rates

    def test_theta_one_no_correction(self):
        """When theta=1, larvae get same rate as adults."""
        cap_adult = bioen_ingestion_cap(np.array([0.001]), 10.0, 0.75, 24, 10, np.array([False]))
        cap_larva = bioen_ingestion_cap(np.array([0.001]), 10.0, 0.75, 24, 10, np.array([True]), theta=1.0, c_rate=5.0)
        np.testing.assert_allclose(cap_adult, cap_larva)
