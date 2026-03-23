"""
Tests for gravitational time dilation and redshift functions.

Validation uses known results from the literature:
- Schwarzschild: dτ/dt = √(1 - 2M/r) = √(1 - r_s/r)
- Photon sphere r=3M: dτ/dt = 1/√3
- Minkowski limit r→∞: dτ/dt → 1
- Redshift: 1+z = √(-g_tt(emitter)) / √(-g_tt(observer))
"""

import numpy as np
import astropy.units as u
import pytest
from numpy.testing import assert_allclose

from einsteinpy import constant
from einsteinpy.coordinates import SphericalDifferential, BoyerLindquistDifferential
from einsteinpy.metric import Schwarzschild, Kerr
from einsteinpy.utils import proper_time_ratio, redshift_factor
from einsteinpy.utils.time_dilation import _position_to_x_vec

# Mass for which r_s = 1 m. r in meters then equals r/r_s (e.g. r=3 → 3 r_s).
MASS_SCHWARZSCHILD_RADIUS_1M = constant.Mass_Schwarzschild_Radius_1m


def _schwarzschild_metric(M=MASS_SCHWARZSCHILD_RADIUS_1M):
    """Create Schwarzschild metric with coords at r=100m (dummy)."""
    sph = SphericalDifferential(
        t=0 * u.s,
        r=100 * u.m,
        theta=np.pi / 2 * u.rad,
        phi=0 * u.rad,
        v_r=0 * u.m / u.s,
        v_th=0 * u.rad / u.s,
        v_p=0 * u.rad / u.s,
    )
    return Schwarzschild(coords=sph, M=M)


def _kerr_metric(M=MASS_SCHWARZSCHILD_RADIUS_1M, a=0.5):
    """Create Kerr metric with coords at r=100m."""
    bl = BoyerLindquistDifferential(
        t=0 * u.s,
        r=100 * u.m,
        theta=np.pi / 2 * u.rad,
        phi=0 * u.rad,
        v_r=0 * u.m / u.s,
        v_th=0 * u.rad / u.s,
        v_p=0 * u.rad / u.s,
    )
    return Kerr(coords=bl, M=M, a=a * u.one)


class TestProperTimeRatio:
    """Tests for proper_time_ratio."""

    def test_schwarzschild_at_photon_sphere(self):
        """At r=3M (photon sphere): dτ/dt = 1/√3 (exact)."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad  # Use metric's r_s for consistency
        r_photon = 1.5 * rs  # r = 3M, and 2M = r_s, so r = 1.5*r_s
        ratio = proper_time_ratio(metric, (r_photon, np.pi / 2))
        expected = 1 / np.sqrt(3)
        assert_allclose(ratio, expected, rtol=1e-10, atol=1e-12)

    def test_schwarzschild_minkowski_limit(self):
        """r→∞ ⇒ dτ/dt → 1. At r=1e10 r_s, 1 - r_s/r = 1e-10, sqrt ≈ 1 - 5e-11."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r_large = 1e10 * rs
        ratio = proper_time_ratio(metric, (r_large, np.pi / 2))
        assert_allclose(ratio, 1.0, rtol=1e-8, atol=1e-10)

    def test_schwarzschild_monotonic_in_r(self):
        """dτ/dt increases with r."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r1, r2 = 4 * rs, 10 * rs
        ratio1 = proper_time_ratio(metric, (r1, np.pi / 2))
        ratio2 = proper_time_ratio(metric, (r2, np.pi / 2))
        assert ratio2 - ratio1 > 1e-9
        assert ratio1 < ratio2

    def test_schwarzschild_explicit_formula(self):
        """dτ/dt = √(1 - r_s/r) at arbitrary r > r_s."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r = 5 * rs
        ratio = proper_time_ratio(metric, (r, np.pi / 2))
        expected = np.sqrt(1 - rs / r)
        assert_allclose(ratio, expected, rtol=1e-10, atol=1e-12)

    def test_kerr_equatorial_vs_pole_differs(self):
        """At same r, polar (θ=0) and equatorial (θ=π/2) observers differ."""
        metric = _kerr_metric(a=0.5)
        rs = metric.sch_rad
        r = 10 * rs
        ratio_equator = proper_time_ratio(metric, (r, np.pi / 2))
        ratio_pole = proper_time_ratio(metric, (r, 0.0))
        assert abs(ratio_equator - ratio_pole) > 1e-8

    def test_moving_observer_velocity_zero_matches_stationary(self):
        """velocity=(0,0,0) gives same result as velocity=None."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r = 10 * rs
        ratio_stationary = proper_time_ratio(metric, (r, np.pi / 2))
        ratio_moving = proper_time_ratio(metric, (r, np.pi / 2), velocity=(0.0, 0.0, 0.0))
        assert_allclose(ratio_stationary, ratio_moving, rtol=1e-10, atol=1e-12)

    def test_moving_observer_slower_than_stationary(self):
        """Non-zero velocity → dτ/dt smaller than stationary (kinematic dilation)."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r = 10 * rs
        ratio_stationary = proper_time_ratio(metric, (r, np.pi / 2))
        # v_phi = v/r (rad/s). Use v=100 m/s subrelativistic.
        v_phi = 100.0 / r
        ratio_moving = proper_time_ratio(
            metric, (r, np.pi / 2), velocity=(0.0, 0.0, v_phi)
        )
        assert ratio_moving < ratio_stationary
        assert ratio_moving > 0


class TestRedshiftFactor:
    """Tests for redshift_factor."""

    def test_emitter_closer_than_observer_redshift(self):
        """Emitter deeper in well → 1+z > 1 (redshift)."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        pos_emitter = (6 * rs, np.pi / 2)
        pos_observer = (50 * rs, np.pi / 2)
        z = redshift_factor(metric, pos_emitter, pos_observer)
        assert z > 1

    def test_redshift_consistent_with_proper_time_ratio(self):
        """1+z = (dτ/dt)_obs / (dτ/dt)_emit for stationary observers."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        pos_emitter = (6 * rs, np.pi / 2)
        pos_observer = (20 * rs, np.pi / 2)
        z = redshift_factor(metric, pos_emitter, pos_observer)
        dtau_emit = proper_time_ratio(metric, pos_emitter)
        dtau_obs = proper_time_ratio(metric, pos_observer)
        expected_z = dtau_obs / dtau_emit
        assert_allclose(z, expected_z, rtol=1e-8, atol=1e-10)

    def test_redshift_with_moving_observer(self):
        """redshift_factor accepts observer_velocity and emitter_velocity."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        pos_emitter = (6 * rs, np.pi / 2)
        pos_observer = (20 * rs, np.pi / 2)
        z = redshift_factor(
            metric,
            pos_emitter,
            pos_observer,
            emitter_velocity=(0.0, 0.0, 50.0 / (6 * rs)),
            observer_velocity=(0.0, 0.0, 0.0),
        )
        assert np.isfinite(z)
        assert z > 0


class TestPositionToXVec:
    """Tests for _position_to_x_vec helper."""

    def test_position_one_element(self):
        """(r,) → theta=π/2, phi=0."""
        result = _position_to_x_vec((10.0,))
        expected = np.array([0.0, 10.0, np.pi / 2, 0.0])
        assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_position_two_elements(self):
        """(r, theta) → phi=0."""
        result = _position_to_x_vec((10.0, np.pi / 4))
        expected = np.array([0.0, 10.0, np.pi / 4, 0.0])
        assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_position_three_elements(self):
        """(r, theta, phi) → all used."""
        result = _position_to_x_vec((10.0, np.pi / 4, 0.5))
        expected = np.array([0.0, 10.0, np.pi / 4, 0.5])
        assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_position_empty_raises(self):
        """Empty position raises ValueError."""
        with pytest.raises(ValueError, match="must have 1, 2, or 3 elements"):
            _position_to_x_vec([])

    def test_position_scalar_treated_as_one_element(self):
        """Scalar r becomes (r,)."""
        result = _position_to_x_vec(5.0)
        assert_allclose(result[1], 5.0, rtol=0, atol=1e-12)
        assert result[2] == np.pi / 2


class TestProperTimeRatioEdgeCases:
    """Tests for proper_time_ratio edge cases and branches."""

    def test_position_formats_equivalent(self):
        """(r,), (r, theta), (r, theta, phi) give same result at same location."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r, theta = 10 * rs, np.pi / 2
        r1 = proper_time_ratio(metric, (r,))
        r2 = proper_time_ratio(metric, (r, theta))
        r3 = proper_time_ratio(metric, (r, theta, 0.0))
        assert_allclose(r1, r2, rtol=1e-12, atol=1e-14)
        assert_allclose(r2, r3, rtol=1e-12, atol=1e-14)

    def test_callable_metric(self):
        """proper_time_ratio accepts callable returning covariant metric."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r = 10 * rs

        def mock_metric(x_vec):
            return metric.metric_covariant(x_vec)

        ratio = proper_time_ratio(mock_metric, (r, np.pi / 2))
        expected = proper_time_ratio(metric, (r, np.pi / 2))
        assert_allclose(ratio, expected, rtol=1e-10, atol=1e-12)

    def test_velocity_wrong_size_raises(self):
        """velocity with != 3 elements raises ValueError."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        with pytest.raises(ValueError, match="velocity must have 3 elements"):
            proper_time_ratio(metric, (10 * rs, np.pi / 2), velocity=(1.0, 2.0))
        with pytest.raises(ValueError, match="velocity must have 3 elements"):
            proper_time_ratio(
                metric, (10 * rs, np.pi / 2), velocity=(1.0, 2.0, 3.0, 4.0)
            )

    def test_velocity_invalid_returns_nan(self):
        """When v0 yields dt_dtau <= 0 or non-finite, proper_time_ratio returns nan."""
        from unittest import mock

        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r = 10 * rs
        # Patch v0 to return 0 (spacelike/invalid) -> dt_dtau <= 0 path
        with mock.patch(
            "einsteinpy.utils.time_dilation.v0",
            return_value=0.0,
        ):
            result = proper_time_ratio(
                metric, (r, np.pi / 2), velocity=(1e6, 0.0, 0.0)
            )
        assert np.isnan(result)

    def test_velocity_extreme_reduces_ratio(self):
        """Very high (but valid) tangential velocity further reduces dτ/dt."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r = 10 * rs
        # High but sub-luminal tangential velocity: v = r*v_phi < c
        c_si = constant.c.value
        v_phi_high = (c_si * 0.9) / r
        ratio_high = proper_time_ratio(
            metric, (r, np.pi / 2), velocity=(0.0, 0.0, v_phi_high)
        )
        ratio_stationary = proper_time_ratio(metric, (r, np.pi / 2))
        assert ratio_high < ratio_stationary
        assert 0 < ratio_high < 1


class TestRedshiftFactorEdgeCases:
    """Tests for redshift_factor edge cases."""

    def test_redshift_nan_when_emitter_inside_horizon(self):
        """redshift_factor returns nan when emitter inside horizon."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        pos_emitter = (0.5 * rs, np.pi / 2)
        pos_observer = (50 * rs, np.pi / 2)
        z = redshift_factor(metric, pos_emitter, pos_observer)
        assert np.isnan(z)

    def test_redshift_nan_when_observer_inside_horizon(self):
        """redshift_factor returns nan when observer inside horizon."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        pos_emitter = (10 * rs, np.pi / 2)
        pos_observer = (0.5 * rs, np.pi / 2)
        z = redshift_factor(metric, pos_emitter, pos_observer)
        assert np.isnan(z)

    def test_redshift_both_moving(self):
        """redshift_factor with both emitter and observer moving."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        pos_emitter = (6 * rs, np.pi / 2)
        pos_observer = (20 * rs, np.pi / 2)
        z = redshift_factor(
            metric,
            pos_emitter,
            pos_observer,
            emitter_velocity=(0.0, 0.0, 30.0 / (6 * rs)),
            observer_velocity=(0.0, 0.0, 10.0 / (20 * rs)),
        )
        assert np.isfinite(z)
        assert z > 0


class TestInvalidPositions:
    """Tests for invalid positions (inside horizon/ergosphere)."""

    def test_schwarzschild_inside_horizon(self):
        """Position inside horizon should raise ValueError or return nan."""
        metric = _schwarzschild_metric()
        rs = metric.sch_rad
        r_inner = 0.5 * rs
        try:
            result = proper_time_ratio(metric, (r_inner, np.pi / 2))
            assert np.isnan(result)
        except ValueError:
            pass
