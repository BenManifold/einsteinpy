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
