"""
This is a test module, that tests the non-coordinate conversion utilities in ``einsteinpy.coordinates.utils``.

"""
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from einsteinpy.coordinates import BoyerLindquistDifferential, SphericalDifferential
from einsteinpy.coordinates.utils import (
    bl_to_cartesian,
    bl_to_cartesian_fast,
    cartesian_to_bl,
    cartesian_to_bl_fast,
    cartesian_to_spherical,
    cartesian_to_spherical_fast,
    lorentz_factor,
    spherical_to_cartesian,
    spherical_to_cartesian_fast,
    v0,
)
from einsteinpy.metric import Schwarzschild, Kerr, KerrNewman

from einsteinpy import constant

_c = constant.c.value


def test_lorentz_factor():
    """
    Tests, if the value of Lorentz Factor, calculated using \
    ``einsteinpy.coordindates.utils.lorentz_factor()`` is the same as, \
    that calculated by brute force

    """
    # Calculated using v0()
    gamma = lorentz_factor(-0.1, -0.01, 0.05)

    # Calculated by brute force
    v_vec = np.array([-0.1, -0.01, 0.05])
    v_norm2 = v_vec[0]**2 + v_vec[1]**2 + v_vec[2]**2
    gamma_b = 1 / np.sqrt(1 - v_norm2 / _c ** 2)

    assert_allclose(gamma, gamma_b, rtol=1e-8)


@pytest.fixture
def sph():
    return SphericalDifferential(
        0. * u.s,
        1. * u.m,
        np.pi / 2 * u.rad,
        0.1 * u.rad,
        0. * u.m / u.s,
        0.1 * u.rad / u.s,
        951 * u.rad / u.s
    )


@pytest.fixture
def bl():
    return BoyerLindquistDifferential(
        0. * u.s,
        1. * u.m,
        np.pi / 2 * u.rad,
        0.1 * u.rad,
        0. * u.m / u.s,
        0.1 * u.rad / u.s,
        951 * u.rad / u.s
    )


def test_v0(sph, bl):
    """
    Tests, if the 4-Velocity in KerrNewman Metric is the same as that in Kerr Metric, \
    in the limit Q -> 0 and if it becomes the same as that in Schwarzschild \
    Metric, in the limits, a -> 0 & Q -> 0

    """
    M = 1e24 * u.kg

    ms = Schwarzschild(coords=sph, M=M)
    mk = Kerr(coords=bl, M=M, a=0.5 * u.one)
    mk0 = Kerr(coords=bl, M=M, a=0. * u.one)
    mkn = KerrNewman(coords=bl, M=M, a=0.5 * u.one, Q=0. * u.C)
    mkn0 = KerrNewman(coords=bl, M=M, a=0. * u.one, Q=0. * u.C)

    v4vec_s = sph.velocity(ms)
    v4vec_k = bl.velocity(mk)
    v4vec_k0 = bl.velocity(mk0)
    v4vec_kn = bl.velocity(mkn)
    v4vec_kn0 = bl.velocity(mkn0)

    assert_allclose(v4vec_s, v4vec_k0, rtol=1e-8)
    assert_allclose(v4vec_k, v4vec_kn, rtol=1e-8)
    assert_allclose(v4vec_kn0, v4vec_s, rtol=1e-8)


def test_compare_vt_schwarzschild():
    """
    Tests, if the value of timelike component of 4-Velocity in Schwarzschild spacetime, \
    calculated using ``einsteinpy.coordindates.utils.v0()`` is the same as that calculated by \
    brute force

    """
    # Calculated using v0()
    M = 1e24 * u.kg
    sph = SphericalDifferential(
        0. * u.s,
        1. * u.m,
        np.pi / 2 * u.rad,
        0.1 * u.rad,
        -0.1 * u.m / u.s,
        -0.01 * u.rad / u.s,
        0.05 * u.rad / u.s
    )

    ms = Schwarzschild(coords=sph, M=M)
    x_vec = sph.position()
    ms_mat = ms.metric_covariant(x_vec)
    v_vec = sph.velocity(ms)

    sph.v_t = (ms,)  # Setting v_t
    vt_s = sph.v_t  # Getting v_t

    # Calculated by brute force
    A = ms_mat[0, 0]
    C = ms_mat[1, 1] * v_vec[1]**2 + ms_mat[2, 2] * v_vec[2]**2 + ms_mat[3, 3] * v_vec[3]**2 - _c ** 2
    D = - 4 * A * C
    vt_sb = np.sqrt(D) / (2 * A)

    assert_allclose(vt_s.value, vt_sb, rtol=1e-8)


@pytest.fixture
def sph2():
    return SphericalDifferential(
        0. * u.s,
        1. * u.m,
        np.pi / 2 * u.rad,
        0.1 * u.rad,
        -0.1 * u.m / u.s,
        -0.01 * u.rad / u.s,
        0.05 * u.rad / u.s
    )


@pytest.fixture
def bl2():
    return BoyerLindquistDifferential(
        0. * u.s,
        1. * u.m,
        np.pi / 2 * u.rad,
        0.1 * u.rad,
        -0.1 * u.m / u.s,
        -0.01 * u.rad / u.s,
        0.05 * u.rad / u.s
    )


def test_compare_vt_schwarzschild_kerr_kerrnewman(sph2, bl2):
    """
    Tests, whether the timelike component of 4-Velocity in KerrNewman Metric is the same as that \
    in Kerr Metric, in the limit Q -> 0 and if it becomes the same as that in Schwarzschild \
    Metric, in the limits, a -> 0 & Q -> 0

    """
    M = 1e24 * u.kg

    ms = Schwarzschild(coords=sph2, M=M)
    mk = Kerr(coords=bl2, M=M, a=0.5 * u.one)
    mk0 = Kerr(coords=bl2, M=M, a=0. * u.one)
    mkn = KerrNewman(coords=bl2, M=M, a=0.5 * u.one, Q=0. * u.C)
    mkn0 = KerrNewman(coords=bl2, M=M, a=0. * u.one, Q=0. * u.C)

    v_vec_ms = sph2.velocity(ms)
    v_vec_mk = bl2.velocity(mk)
    v_vec_mk0 = bl2.velocity(mk0)
    v_vec_mkn = bl2.velocity(mkn)
    v_vec_mkn0 = bl2.velocity(mkn0)

    assert_allclose(v_vec_ms, v_vec_mk0, rtol=1e-8)
    assert_allclose(v_vec_mk, v_vec_mkn, rtol=1e-8)
    assert_allclose(v_vec_mkn0, v_vec_ms, rtol=1e-8)


class TestCoordinateConversionUtils:
    """Direct tests for coordinate conversion utility functions."""

    def test_cartesian_to_spherical_fast_without_velocities(self):
        """cartesian_to_spherical_fast without velocities uses novel path."""
        t, r, th, phi = cartesian_to_spherical_fast(
            0.0, 10.0, 0.0, 0.0, velocities_provided=False
        )
        assert_allclose(r, 10.0, rtol=1e-10)
        assert_allclose(th, np.pi / 2, rtol=1e-10)
        assert_allclose(phi, 0.0, rtol=1e-10)

    def test_cartesian_to_spherical_fast_with_velocities(self):
        """cartesian_to_spherical_fast with velocities uses full conversion."""
        result = cartesian_to_spherical_fast(
            0.0, 10.0, 0.0, 0.0,
            v_x=1.0, v_y=0.0, v_z=0.0,
            velocities_provided=True,
        )
        assert len(result) == 7
        t, r, th, phi, v_r, v_th, v_p = result
        assert_allclose(r, 10.0, rtol=1e-10)
        assert_allclose(v_r, 1.0, rtol=1e-10)

    def test_cartesian_to_spherical_roundtrip(self):
        """cartesian_to_spherical and spherical_to_cartesian are inverse."""
        x, y, z = 3.0, 4.0, 5.0
        vx, vy, vz = 0.1, -0.2, 0.05
        _, r, th, phi, v_r, v_th, v_p = cartesian_to_spherical(
            0.0, x, y, z, vx, vy, vz
        )
        _, x2, y2, z2, vx2, vy2, vz2 = spherical_to_cartesian(
            0.0, r, th, phi, v_r, v_th, v_p
        )
        assert_allclose([x, y, z], [x2, y2, z2], rtol=1e-10)
        assert_allclose([vx, vy, vz], [vx2, vy2, vz2], rtol=1e-8)

    def test_cartesian_to_bl_fast_without_velocities(self):
        """cartesian_to_bl_fast without velocities uses novel path."""
        alpha = 0.5
        result = cartesian_to_bl_fast(
            0.0, 10.0, 0.0, 0.0, alpha, velocities_provided=False
        )
        assert len(result) == 4  # (t, r, theta, phi)
        t, r, th, phi = result
        assert r > 0
        assert 0 <= th <= np.pi

    def test_cartesian_to_bl_fast_with_velocities(self):
        """cartesian_to_bl_fast with velocities uses full conversion."""
        alpha = 0.5
        result = cartesian_to_bl_fast(
            0.0, 10.0, 0.0, 0.0, alpha,
            v_x=0.1, v_y=0.0, v_z=0.0,
            velocities_provided=True,
        )
        assert len(result) == 7

    def test_spherical_to_cartesian_fast_without_velocities(self):
        """spherical_to_cartesian_fast without velocities uses novel path."""
        r, th, phi = 10.0, np.pi / 2, np.pi / 4
        result = spherical_to_cartesian_fast(
            0.0, r, th, phi, velocities_provided=False
        )
        x = 10 / np.sqrt(2)
        assert_allclose(result[1:4], [x, x, 0.0], rtol=1e-10, atol=1e-14)

    def test_spherical_to_cartesian_fast_with_velocities(self):
        """spherical_to_cartesian_fast with velocities uses full conversion."""
        result = spherical_to_cartesian_fast(
            0.0, 10.0, np.pi / 2, np.pi / 4,
            v_r=1.0, v_th=0.0, v_p=0.1,
            velocities_provided=True,
        )
        assert len(result) == 7

    def test_bl_to_cartesian_fast_without_velocities(self):
        """bl_to_cartesian_fast without velocities uses novel path."""
        alpha = 0.5
        result = bl_to_cartesian_fast(
            0.0, 10.0, np.pi / 2, 0.0, alpha, velocities_provided=False
        )
        assert len(result) == 4  # (t, x, y, z)

    def test_bl_to_cartesian_fast_with_velocities(self):
        """bl_to_cartesian_fast with velocities uses full conversion."""
        alpha = 0.5
        result = bl_to_cartesian_fast(
            0.0, 10.0, np.pi / 2, 0.0, alpha,
            v_r=0.1, v_th=0.0, v_p=0.1,
            velocities_provided=True,
        )
        assert len(result) == 7

    def test_bl_cartesian_roundtrip(self):
        """bl_to_cartesian and cartesian_to_bl are inverse."""
        r, th, phi = 10.0, np.pi / 2, 0.5
        vr, vth, vp = 0.1, 0.01, 0.2
        alpha = 0.75
        _, x, y, z, vx, vy, vz = bl_to_cartesian(
            0.0, r, th, phi, alpha, vr, vth, vp
        )
        _, r2, th2, phi2, vr2, vth2, vp2 = cartesian_to_bl(
            0.0, x, y, z, alpha, vx, vy, vz
        )
        assert_allclose([r, th, phi], [r2, th2, phi2], rtol=1e-8)
        assert_allclose([vr, vth, vp], [vr2, vth2, vp2], rtol=1e-6)

    def test_v0_direct_call(self):
        """v0 returns positive timelike component from metric and 4-velocity spatial parts."""
        # v0 expects spatial components of 4-velocity (u^1, u^2, u^3).
        # Use Schwarzschild metric and small spatial 4-velocity.
        M = 1e24 * u.kg
        sph = SphericalDifferential(
            0.0 * u.s, 10.0 * u.m, np.pi / 2 * u.rad, 0.0 * u.rad,
            -0.1 * u.m / u.s, -0.01 * u.rad / u.s, 0.05 * u.rad / u.s,
        )
        ms = Schwarzschild(coords=sph, M=M)
        x_vec = sph.position()
        g = ms.metric_covariant(x_vec)
        v_vec = sph.velocity(ms)
        vt = v0(g, float(v_vec[1]), float(v_vec[2]), float(v_vec[3]))
        assert vt > 0
        assert np.isfinite(vt)
